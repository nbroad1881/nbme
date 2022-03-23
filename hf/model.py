# @torch.inference_mode()

from typing import Any
from dataclasses import dataclass

import torch
from torch import nn
from transformers.file_utils import ModelOutput
from transformers import logging

from focal_loss import FocalLoss

logger = logging.get_logger(__name__)


@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before crf).
        crf (crf output)
    """

    loss: Any = None
    logits: Any = None
    crf: Any = None


def __init__(self, config):
    super(self.PreTrainedModel, self).__init__(config)

    kwargs = {"add_pooling_layer": False}
    if config.model_type not in {"bert", "roberta"}:
        kwargs = {}
    setattr(self, self.backbone_name, self.ModelClass(config, **kwargs))

    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.dropout1 = nn.Dropout(0.1)
    self.dropout2 = nn.Dropout(0.2)
    self.dropout3 = nn.Dropout(0.3)
    self.dropout4 = nn.Dropout(0.4)
    self.dropout5 = nn.Dropout(0.5)
    self.output = nn.Linear(config.hidden_size, config.num_labels)

    if self.run_config and self.run_config.use_crf:
        from torchcrf import CRF

        self.crf = CRF(self.run_config.num_classes, batch_first=True)

    if self.run_config and self.run_config.use_focal_loss:
        self.loss_fn = FocalLoss()


def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    labels=None,
):

    # Funky alert
    outputs = getattr(self, self.backbone_name)(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
    )

    sequence_output = self.dropout(outputs.last_hidden_state)

    # if labels, then we are training
    loss = None
    crf_output = None
    if labels is not None:

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        if self.run_config and self.run_config.use_crf:

            mask = labels > -1
            labels = labels * mask
            mask[:, 0] = 1
            attention_mask = attention_mask * mask

            loss = -self.crf(logits, labels, attention_mask.bool())
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
            loss1 = loss_fct(
                logits1.view(-1, self.config.num_labels),
                labels.view(-1, self.config.num_labels),
            )
            loss2 = loss_fct(
                logits2.view(-1, self.config.num_labels),
                labels.view(-1, self.config.num_labels),
            )
            loss3 = loss_fct(
                logits3.view(-1, self.config.num_labels),
                labels.view(-1, self.config.num_labels),
            )
            loss4 = loss_fct(
                logits4.view(-1, self.config.num_labels),
                labels.view(-1, self.config.num_labels),
            )
            loss5 = loss_fct(
                logits5.view(-1, self.config.num_labels),
                labels.view(-1, self.config.num_labels),
            )
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            loss = torch.masked_select(
                loss, labels.view(-1, self.config.num_labels) > -1
            ).mean()
    # otherwise, doing inference
    else:
        logits = self.output(sequence_output)
        if self.run_config and self.run_config.use_crf:
            crf_output = self.crf.decode(logits, attention_mask.bool())

    return TokenClassifierOutput(
        loss=loss,
        logits=logits.sigmoid(),
        crf=crf_output,
    )


def get_model(config, init=False):
    model_type = type(config).__name__[: -len("config")]
    if model_type == "Bart":
        name = f"{model_type}PretrainedModel"
    else:
        name = f"{model_type}PreTrainedModel"
    PreTrainedModel = getattr(__import__("transformers", fromlist=[name]), name)
    name = f"{model_type}Model"
    ModelClass = getattr(__import__("transformers", fromlist=[name]), name)

    model = type(
        "CustomModel",
        (PreTrainedModel,),
        {"__init__": __init__, "forward": forward},
    )

    model._keys_to_ignore_on_load_unexpected = [r"pooler"]
    model._keys_to_ignore_on_load_missing = [r"position_ids"]

    model.PreTrainedModel = PreTrainedModel
    model.ModelClass = ModelClass
    model.backbone_name = config.model_type

    # changes deberta-v2 --> deberta
    if "deberta" in model.backbone_name:
        model.backbone_name = "deberta"

    if init:
        return model(config)
    return model


def get_pretrained(model_name_or_path, config, **kwargs):

    model = get_model(config, init=False)

    return model.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        **kwargs,
    )
