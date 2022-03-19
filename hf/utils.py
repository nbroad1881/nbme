import os
from itertools import chain
from dataclasses import dataclass

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import DataCollatorForTokenClassification
from transformers.utils import logging

logger = logging.get_logger(__name__)


def set_wandb_env_vars(cfg):
    os.environ["WANDB_ENTITY"] = cfg["entity"]
    os.environ["WANDB_PROJECT"] = cfg["project"]
    os.environ["WANDB_RUN_GROUP"] = cfg["group"]
    os.environ["WANDB_JOB_TYPE"] = cfg["job_type"]
    os.environ["WANDB_NOTES"] = cfg["notes"]
    os.environ["WANDB_TAGS"] = cfg["tags"]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_location_predictions(preds, dataset):
    """
    Finds the prediction indexes at the character level.
    """
    all_predictions = []
    for pred, offsets, seq_ids, text in zip(
        preds, dataset["offset_mapping"], dataset["sequence_ids"], dataset["pn_history"]
    ):
        pred = sigmoid(pred)
        start_idx = None
        current_preds = []
        for p, o, s_id in zip(pred, offsets, seq_ids):
            if s_id is None or s_id == 0:
                continue

            if p > 0.5:
                if start_idx is None:
                    start_idx = o[0]
                end_idx = o[1]
            elif start_idx is not None:
                current_preds.append((start_idx, end_idx))
                start_idx = None

        if start_idx is not None:
            current_preds.append((start_idx, end_idx))

        all_predictions.append(current_preds)

    return all_predictions


def kaggle_metrics(eval_prediction, dataset):
    """
    For `compute_metrics`

    Use partial for the args and kwargs to pass other data
    into the `compute_metrics` function.
    """

    preds = get_location_predictions(eval_prediction.predictions, dataset)

    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels, text in zip(
        eval_prediction.predictions,
        dataset["offset_mapping"],
        dataset["sequence_ids"],
        dataset["labels"],
        dataset["text"],
    ):

        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros((num_chars), dtype=bool)

        for (tok_start, tok_end), s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0:  # ignore question part of input
                continue
            if int(label) == 1:

                char_labels[tok_start:tok_end] = 1
                if (
                    text[tok_start].isspace()
                    and tok_start > 0
                    and not char_labels[tok_start - 1]
                ):
                    char_labels[tok_start] = 0

        char_preds = np.zeros((num_chars))

        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1
            if (
                text[start_idx].isspace()
                and start_idx > 0
                and not char_preds[start_idx - 1]
            ):
                char_preds[start_idx] = 0

        all_labels.extend(char_labels)
        all_preds.extend(char_preds)

    results = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    return {"precision": results[0], "recall": results[1], "f1": results[2]}


@dataclass
class DataCollatorWithMasking(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Have to modify to make label tensors float and not int.
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100
    return_tensors = "pt"
    masking_prob: float = None

    def torch_call(self, features):
        batch = super().torch_call(features)
        label_name = "label" if "label" in features[0].keys() else "labels"

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float32)

        if self.masking_prob is not None:
            batch = self.mask_tokens(batch)

        return batch

    def mask_tokens(self, batch):
        """
        Mask the inputs at `masking_prob` probability.
        The loss from masked tokens will still be included.
        """

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in batch["input_ids"].tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix = torch.full(batch["input_ids"].shape, self.masking_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        batch["input_ids"][masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return batch


def reinit_model_weights(model, n_layers, config):

    if config.model_type == "bart":
        encoder_layers = model.encoder.layers
        decoder_layers = model.decoder.layers

        reinit_layers(encoder_layers, n_layers, config)
        reinit_layers(decoder_layers, n_layers, config)
    else:
        encoder_layers = model.encoder.layer
        reinit_layers(encoder_layers, n_layers, config)
 
 
def reinit_layers(layers, n_layers, config):

    if config.model_type == "bart":
        std = config.init_std    
    else:
        std = config.initializer_range

    logger.info(f"Reinitializing last {n_layers} layers in the encoder.")
    for layer in layers[-n_layers:]:
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


def layerwise_learning_rate(model, lr=3e-5, wd=0.01, alpha=0.8):
    model_type = model.backbone_name

    layers = (
        [getattr(model, model_type).embeddings]
        + [getattr(model, model_type).encoder.layer]
        + [model.output]
    )
    layers.reverse()

    optimizer_grouped_parameters = []

    no_decay = ["bias", "LayerNorm.weight"]
    for layer in layers:
        lr *= alpha
        optimizer_grouped_parameters += [
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in layer.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


def uniform_learning_rate(model, wd=0.01):

    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": wd,
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