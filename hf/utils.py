import os
import itertools
from dataclasses import dataclass

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
from transformers import DataCollatorForTokenClassification
from transformers.utils import logging

logger = logging.get_logger(__name__)


def set_wandb_env_vars(cfg):
    os.environ["WANDB_ENTITY"] = cfg.get("entity", "")
    os.environ["WANDB_PROJECT"] = cfg.get("project", "")
    os.environ["WANDB_RUN_GROUP"] = cfg.get("group", "")
    os.environ["WANDB_JOB_TYPE"] = cfg.get("job_type", "")
    os.environ["WANDB_NOTES"] = cfg.get("notes", "")
    os.environ["WANDB_TAGS"] = ",".join(cfg.get("tags", ""))


# From https://www.kaggle.com/theoviel/evaluation-metric-folds-baseline

def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)

def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score

def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results

def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions

def get_location_predictions(preds, dataset):
    """
    Finds the prediction indexes at the character level.
    """
    all_predictions = []
    for pred, offsets, seq_ids in zip(
        preds, dataset["offset_mapping"], dataset["sequence_ids"]
    ):
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

    pred_idxs = get_location_predictions(eval_prediction.predictions, dataset)

    all_labels = []
    all_preds = []
    for preds, locations, text in zip(
        pred_idxs,
        dataset["locations"],
        dataset["pn_history"],
    ):

        num_chars = len(text)
        char_labels = np.zeros((num_chars), dtype=bool)

        for start, end in locations:
            char_labels[start:end] = 1

        char_preds = np.zeros((num_chars), dtype=bool)

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
    This must be used with the MaskingProbCallback that sets the environment
    variable at the beginning and end of the training step. This callback ensures
    that there is no masking done during evaluation.
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100
    return_tensors = "pt"

    def torch_call(self, features):
        batch = super().torch_call(features)
        label_name = "label" if "label" in features[0].keys() else "labels"

        batch[label_name] = batch[label_name].type(torch.float32)

        masking_prob = os.getenv("MASKING_PROB")
        if masking_prob is not None and masking_prob != "0":
            batch = self.mask_tokens(batch, float(masking_prob))

        return batch

    def mask_tokens(self, batch, masking_prob):
        """
        Mask the inputs at `masking_prob` probability.
        The loss from masked tokens will still be included.
        """

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in batch["input_ids"].tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix = torch.full(batch["input_ids"].shape, masking_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        batch["input_ids"][masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return batch


def reinit_model_weights(model, n_layers, config):

    backbone = model.backbone
    if config.model_type == "bart":
        std = config.init_std
    else:
        std = config.initializer_range

    if config.model_type == "bart":
        encoder_layers = backbone.encoder.layers
        decoder_layers = backbone.decoder.layers

        reinit_layers(encoder_layers, n_layers, std)
        reinit_layers(decoder_layers, n_layers, std)
    else:
        encoder_layers = backbone.encoder.layer
        reinit_layers(encoder_layers, n_layers, std)

    reinit_modules([model.output], std)

def reinit_layers(layers, n_layers, std):
    for layer in layers[-n_layers:]:
        reinit_modules(layer.modules(), std)

def reinit_modules(modules, std, reinit_embeddings=False):
    for module in modules:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif reinit_embeddings and isinstance(module, torch.nn.Embedding):
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

    for i, layer in enumerate(layers):
        # This keeps top layer = lr
        if i > 0:
            lr *= alpha
        optimizer_grouped_parameters += uniform_learning_rate(layer, wd)

    return optimizer_grouped_parameters


def uniform_learning_rate(model, lr, wd=0.01):

    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
