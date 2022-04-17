import os
from functools import partial
import datetime

import wandb
import torch
from transformers import Trainer, TrainingArguments, AutoConfig
from transformers.trainer_utils import set_seed
from transformers.integrations import WandbCallback


from data import NERDataModule
from config import get_configs
from model import get_pretrained
from utils import (
    kaggle_metrics,
    DataCollatorWithMasking,
    set_wandb_env_vars,
    reinit_model_weights,
)
from callbacks import NewWandbCB, SaveCallback, MaskingProbCallback, BasicSWACallback
from sift import SiftTrainer

if __name__ == "__main__":

    config_file = "j-dv3l-1.yml"
    output = "nb-dv3l-1"
    cfg, args = get_configs(config_file)
    set_seed(args["seed"])
    set_wandb_env_vars(cfg)

    datamodule = NERDataModule(cfg)

    for fold in range(cfg["k_folds"]):


        cfg, args = get_configs(config_file)
        cfg["fold"] = fold
        args["output_dir"] = f"{output}-f{fold}"

        args = TrainingArguments(**args)

        with args.main_process_first(desc="dataset pre-processing"):
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
        
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Eval dataset length: {len(eval_dataset)}")
        compute_metrics = partial(kaggle_metrics, dataset=eval_dataset)

        model_config = AutoConfig.from_pretrained(cfg["model_name_or_path"], use_auth_token=True)
        model_config.update(
            {
                "num_labels": 1,
                # "hidden_dropout_prob": cfg["dropout"],
                # "layer_norm_eps": cfg["layer_norm_eps"],
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

        data_collator = DataCollatorWithMasking(
            tokenizer=datamodule.tokenizer,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )

        Trainer = SiftTrainer if cfg.get("use_sift") else Trainer
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
