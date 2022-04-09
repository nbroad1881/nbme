# @torch.inference_mode()

from typing import Any
from dataclasses import dataclass

import torch
from torch import nn
from transformers.file_utils import ModelOutput
from transformers import PreTrainedModel, logging, AutoModel
from transformers.activations import ACT2FN

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


class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropouts = [nn.Dropout(d / 10) for d in range(1, 6)]
        self.output = nn.Linear(config.hidden_size, config.num_labels)

        if config.to_dict().get("use_crf"):
            from torchcrf import CRF

            self.crf = CRF(config.num_labels, batch_first=True)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        if config.to_dict().get("use_focal_loss"):
            self.loss_fn = FocalLoss()
        if config.to_dict().get("use_gated"):
            self.output = GatedDense(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        outputs = self.backbone(
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

            all_logits = [
                self.output(self.dropouts[i](sequence_output)) for i in range(5)
            ]

            logits = sum(all_logits) / len(self.dropouts)

            if self.config and self.config.to_dict().get("use_crf"):

                mask = labels > -1
                labels = labels * mask
                mask[:, 0] = 1
                attention_mask = attention_mask * mask

                loss = -self.crf(logits, labels, attention_mask.bool())
            elif self.config.to_dict().get("use_focal_loss"):
                pass
            else:
                all_losses = [
                    self.loss_fn(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1, self.config.num_labels),
                    )
                    for logits in all_logits
                ]
                loss = sum(all_losses) / len(self.dropouts)
                loss = torch.masked_select(
                    loss, labels.view(-1, self.config.num_labels) > -1
                ).mean()

        # otherwise, doing inference
        else:
            logits = self.output(sequence_output)
            if self.config and self.config.to_dict().get("use_crf"):
                crf_output = self.crf.decode(logits, attention_mask.bool())

        return TokenClassifierOutput(
            loss=loss,
            logits=logits.sigmoid(),
            crf=crf_output,
        )


def get_pretrained(config, model_path):
    model = CustomModel(config)

    if model_path.endswith("pytorch_model.bin"):
        model.load_state_dict(torch.load(model_path))
    else:
        model.backbone = AutoModel.from_pretrained(model_path)

    return model


class GatedDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_size, 512, bias=False)
        self.wi_1 = nn.Linear(config.hidden_size, 512, bias=False)
        self.wo = nn.Linear(512, config.num_labels, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = ACT2FN[config.gated_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act_fn(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
