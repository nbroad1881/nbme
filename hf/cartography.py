import json
import logging
import math
import os
import argparse
from datetime import datetime

import datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AdamW,
    AutoConfig,
    get_scheduler,
    TrainingArguments,
)

from utils import (
    set_wandb_env_vars,
    kaggle_metrics,
    DataCollatorWithMasking,
    reinit_model_weights,
    log_training_dynamics,
)
from config import get_configs
from data import NERDataModule
from callbacks import NewWandbCB, SaveCallback, MaskingProbCallback
from model import get_pretrained


logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)


def main():

    parser = get_parser()
    pargs = parser.parse_args()

    output = pargs["config_file"].split(".")[0]
    cfg, args = get_configs(pargs["config_file"])
    set_seed(args["seed"])
    set_wandb_env_vars(cfg)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(
        log_with=args["report_to"], logging_dir=args["output_dir"]
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    datamodule = NERDataModule(cfg)
    with accelerator.main_process_first():
        datamodule.prepare_datasets()

    for fold in range(cfg["k_folds"]):

        cfg, args = get_configs(pargs["config_file"])
        cfg["fold"] = fold
        args["output_dir"] = f"{output}-f{fold}"

        args = TrainingArguments(**args)

        # Callbacks
        wb_callback = NewWandbCB(cfg)
        save_callback = SaveCallback(
            min_score_to_save=cfg["min_score_to_save"], metric_name="eval_f1"
        )
        masking_callback = MaskingProbCallback(cfg["masking_prob"])

        callbacks = [wb_callback, save_callback, masking_callback]

        train_dataset = datamodule.get_train_dataset(fold=fold)
        eval_dataset = datamodule.get_eval_dataset(fold=fold)

        data_collator = DataCollatorWithMasking(
            tokenizer=datamodule.tokenizer,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

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

    reinit_model_weights(model, cfg["reinit_layers"], model_config)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_steps is None or args.max_steps == -1:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to initialize the trackers we use, and also store our configuration
    if "wandb" in args.report_to:
        experiment_config = vars({**cfg, **args.to_dict()})
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("nbme-dataset-cartography", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        if "wandb" in args.report_to:
            total_loss = 0

        train_logits = None
        train_labels = None
        train_ids = None

        for step, batch in enumerate(train_dataloader):

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            if train_logits is None:
                train_logits = outputs.logits.detach().cpu().numpy()
                train_labels = batch["labels"].detach().cpu().numpy()
                train_ids = batch["id"].detach().cpu().numpy()
            else:
                train_logits = np.append(
                    train_logits, outputs.logits.detach().cpu().numpy()
                )
                train_labels = np.append(
                    train_labels, batch["labels"].detach().cpu().numpy()
                )
                train_ids = np.append(train_ids, batch["id"].detach().cpu().numpy())

            # We keep track of the loss at each epoch
            if "wandb" in args.report_to:
                total_loss += loss.detach().float()

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.sigmoid()

        eval_metrics = kaggle_metrics([predictions], eval_dataset)

        if "wandb" in args.report_to:
            accelerator.log(
                {
                    "train_loss": total_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                },
            )

        # average
        log_training_dynamics(
            output_dir=args.output_dir,
            epoch=epoch,
            train_ids=list(train_ids),
            train_logits=list(train_logits),
            train_labels=list(train_labels),
        )


if __name__ == "__main__":
    main()
