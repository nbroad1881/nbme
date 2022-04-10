import os
import datetime

from datasets import load_metric

import wandb
import torch
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback

from data import MLMDataModule
from config import get_configs
from utils import set_wandb_env_vars
from callbacks import NewWandbCB

if __name__ == "__main__":

    config_file = "j-rl-mlm-0.yml"
    output = "nb-rl-mlm-0"
    cfg, args = get_configs(config_file)
    set_seed(args["seed"])
    set_wandb_env_vars(cfg)

    datamodule = MLMDataModule(cfg)

    fold = 0

    cfg["fold"] = fold
    args["output_dir"] = f"{output}-f{fold}"

    args = TrainingArguments(**args)

    with args.main_process_first(desc="dataset pre-processing"):
        datamodule.prepare_datasets(fold=fold)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    wb_callback = NewWandbCB(cfg)
    callbacks = [wb_callback]

    train_dataset = datamodule.get_train_dataset()
    eval_dataset = datamodule.get_eval_dataset()

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Eval dataset length: {len(eval_dataset)}")

    model_config = AutoConfig.from_pretrained(cfg["model_name_or_path"])
    model_config.update(
        {
            "hidden_dropout_prob": cfg["dropout"],
            "layer_norm_eps": cfg["layer_norm_eps"],
            "run_start": str(datetime.datetime.utcnow()),
        }
    )

    model = AutoModelForMaskedLM.from_pretrained(cfg["model_name_or_path"], config=model_config)

    data_collator = DataCollatorForLanguageModeling(
            tokenizer=datamodule.tokenizer,
            return_tensors="pt",
        )

    trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=datamodule.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

    trainer.remove_callback(WandbCallback)

    trainer.train()

    if cfg.get("use_swa"):
        trainer.model.load_state_dict(torch.load(os.path.join(args.output_dir, 'swa_weights.bin')))
        eval_results = trainer.evaluate()

    wandb.finish()

    torch.cuda.empty_cache()