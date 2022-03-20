from functools import partial
import datetime

import wandb
from transformers import Trainer, TrainingArguments, AutoConfig
from transformers.trainer_utils import set_seed
from transformers.integrations import WandbCallback


from data import DataModule
from config import get_configs
from model import get_pretrained
from utils import (
    kaggle_metrics,
    DataCollatorWithMasking,
    set_wandb_env_vars,
    reinit_model_weights,
)
from callbacks import NewWandbCB, SaveCallback, MaskingProbCallback

if __name__ == "__main__":

    config_file = "q-rob1.yml"
    output = "rob-l"
    cfg, args = get_configs(config_file)
    set_seed(cfg["seed"])
    set_wandb_env_vars(cfg)

    datamodule = DataModule(cfg)

    for fold in range(cfg["k_folds"]):


        cfg, args = get_configs(config_file)
        cfg["fold"] = fold
        args["output_dir"] = f"{output}_f{fold}"

        args = TrainingArguments(**args)

        with args.main_process_first(desc="dataset pre-processing"):
            datamodule.prepare_datasets(fold=fold)

        # Callbacks
        wb_callback = NewWandbCB(cfg)
        save_callback = SaveCallback(
            min_score_to_save=cfg["min_score_to_save"], metric_name="eval_f1"
        )
        masking_callback = MaskingProbCallback(cfg["masking_prob"])

        train_dataset = datamodule.get_train_dataset()
        eval_dataset = datamodule.get_eval_dataset()

        compute_metrics = partial(kaggle_metrics, dataset=eval_dataset)

        model_config = AutoConfig.from_pretrained(cfg["model_name_or_path"])
        model_config.update(
            {
                "num_labels": 1,
                "hidden_dropout_prob": cfg["dropout"],
                # "layer_norm_eps": cfg["layer_norm_eps"],
                "run_start": str(datetime.datetime.utcnow()),
            }
        )

        model = get_pretrained(cfg["model_name_or_path"], model_config)

        if cfg["reinit_layers"] > 0:
            backbone_name = model.backbone_name
            reinit_model_weights(
                getattr(model, backbone_name), cfg["reinit_layers"], model_config
            )

        data_collator = DataCollatorWithMasking(
            tokenizer=datamodule.tokenizer,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=datamodule.tokenizer,
            data_collator=data_collator,
            callbacks=[wb_callback, save_callback, masking_callback],
        )

        trainer.remove_callback(WandbCallback)

        trainer.train()
        wandb.finish()
