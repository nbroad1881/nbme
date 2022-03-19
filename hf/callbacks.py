import os

from transformers import TrainingArguments
from transformers.integrations import WandbCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.utils import logging
from transformers.file_utils import is_torch_tpu_available

logger = logging.get_logger(__name__)


class NewWandbCB(WandbCallback):
    def __init__(self, run_config):
        super().__init__()
        self.run_config = run_config

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also override the following environment
        variables:
        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training. Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to disable gradient logging or `"all"` to
                log gradients and parameters.
            WANDB_PROJECT (`str`, *optional*, defaults to `"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (`bool`, *optional*, defaults to `False`):
                Whether or not to disable wandb entirely. Set *WANDB_DISABLED=true* to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict(), **self.run_config}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            run_name = os.getenv("WANDB_NAME", args.run_name)

            if self._wandb.run is None:
                tags = os.getenv("WANDB_TAGS", None)
                save_code = os.getenv("WANDB_DISABLE_CODE", None)

                # environment variables get priority
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    group=os.getenv("WANDB_RUN_GROUP"),
                    notes=os.getenv("WANDB_NOTES", None),
                    entity=os.getenv("WANDB_ENTITY", None),
                    id=os.getenv("WANDB_RUN_ID", None),
                    dir=os.getenv("WANDB_DIR", None),
                    tags=tags if tags is None else tags.split(","),
                    job_type=os.getenv("WANDB_JOB_TYPE", None),
                    mode=os.getenv("WANDB_MODE", None),
                    anonymous=os.getenv("WANDB_ANONYMOUS", None),
                    save_code=bool(save_code) if save_code is not None else save_code,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

            # # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps),
                )


class SaveCallback(TrainerCallback):
    def __init__(self, min_score_to_save, metric_name) -> None:
        """
        After evaluation, if the `metric_name` value is higher than
        `min_score_to_save` the model will get saved.
        If `metric_name` value > `min_score_to_save`, then
        `metric_name` value becomes the new `min_score_to_save`.
        """
        super().__init__()

        self.min_score_to_save = min_score_to_save
        self.metric_name = metric_name

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        metrics = kwargs.get("metrics")
        if metrics is None:
            raise ValueError("No metrics found for SaveCallback")

        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            raise KeyError(f"{self.metric_name} not found in metrics")

        if metric_value > self.min_score_to_save:
            control.should_save = True
            logger.info(f"Saving model.")
            self.min_score_to_save = metric_value
            kwargs["model"].config.update({"best_cv_f1": metric_value})
        else:
            logger.info("Not saving model.")

            
