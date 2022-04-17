import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name


logger = logging.getLogger(__name__)

def main():

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    config_file = "j-rl-idpt-3.yml"
    output = "nb-rl-idpt-3"
    cfg, args = get_configs(config_file)
    set_seed(args["seed"])
    set_wandb_env_vars(cfg)

    datamodule = NERDataModule(cfg)

    for fold in range(cfg["k_folds"]):


        cfg, args = get_configs(config_file)
        cfg["fold"] = fold
        args["output_dir"] = f"{output}-f{fold}"

        args = TrainingArguments(**args)

        with accelerator.main_process_first():
            datamodule.prepare_datasets(fold=fold)

        # Callbacks
        wb_callback = NewWandbCB(cfg)
        save_callback = SaveCallback(
            min_score_to_save=cfg["min_score_to_save"], metric_name="eval_f1"
        )
        masking_callback = MaskingProbCallback(cfg["masking_prob"])

        callbacks = [wb_callback, save_callback, masking_callback]
        if cfg["use_swa"]:
            callbacks.append(BasicSWACallback(start_after=cfg["swa_start_after"], save_every=cfg["swa_save_every"]))


        train_dataset = datamodule.get_train_dataset()
        eval_dataset = datamodule.get_eval_dataset()

    data_collator = DataCollatorWithMasking(
            tokenizer=datamodule.tokenizer,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    model_config = AutoConfig.from_pretrained(cfg["model_name_or_path"])
    model_config.update(
        {
            "num_labels": 1,
            "hidden_dropout_prob": cfg["dropout"],
            "layer_norm_eps": cfg["layer_norm_eps"],
            "run_start": str(datetime.datetime.utcnow()),
            "use_crf": cfg.get("use_crf", False),
            "use_sift": cfg.get("use_sift", False),
            "use_focal_loss": cfg.get("use_focal_loss", False),
        }
    )

    model = get_pretrained(model_config, cfg["model_name_or_path"])

    reinit_model_weights(
        model, cfg["reinit_layers"], model_config
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("nbme-dataset-cartography", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            resume_step = None
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        if "epoch" in path:
            args.num_train_epochs -= int(path.replace("epoch_", ""))
        else:
            resume_step = int(path.replace("step_", ""))
            args.num_train_epochs -= resume_step // len(train_dataloader)
            resume_step = (args.num_train_epochs * len(train_dataloader)) - resume_step

    for epoch in range(args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        train_logits = None
        train_labels = None
        train_ids = None

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == 0 and step < resume_step:
                continue
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            if train_logits is None:
                train_logits = outputs.logits.detach().cpu().numpy()
                train_labels = batch["labels"].detach().cpu().numpy()
                train_ids = batch["id"].detach().cpu().numpy()
            else:
                train_logits = np.append(train_logits, outputs.logits.detach().cpu().numpy())
                train_labels = np.append(train_labels, batch["labels"].detach().cpu().numpy())
                train_ids = np.append(train_ids, batch["id"].detach().cpu().numpy())

            # Calculate loss


            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.sigmoid()


        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                },
            )
        # average 
        log_training_dynamics(output_dir=args.output_dir,
                              epoch=epoch,
                              train_ids=list(train_ids),
                              train_logits=list(train_logits),
                              train_labels=list(train_labels))

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)


if __name__ == "__main__":
    main()