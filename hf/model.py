# @torch.inference_mode()
import os
from typing import Any, Optional, Union, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from transformers.file_utils import ModelOutput
from transformers import PreTrainedModel, logging, AutoModel
from transformers.activations import ACT2FN
from transformers.models.deberta.modeling_deberta import (
    DebertaPreTrainedModel,
    DebertaModel,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
)
from transformers.modeling_outputs import MaskedLMOutput

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

        self.dropout = nn.Dropout(config.output_dropout)
        self.dropouts = [nn.Dropout(0.5) for d in range(8)]
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

        sequence_output = outputs.last_hidden_state

        # if labels, then we are training
        loss = None
        crf_output = None
        if labels is not None:

            all_logits = torch.stack(
                [
                    self.output(drpt(sequence_output))
                    for drpt in self.dropouts
                ],
                dim=0,
            )

            logits = torch.mean(all_logits, dim=0)

            if self.config and self.config.to_dict().get("use_crf"):

                mask = labels > -1
                labels = labels * mask
                mask[:, 0] = 1
                attention_mask = attention_mask * mask

                loss = -self.crf(logits, labels, attention_mask.bool())
            elif self.config.to_dict().get("use_focal_loss"):
                pass
            else:
                all_losses = torch.stack([
                    self.loss_fn(
                        lgts.view(-1, self.config.num_labels),
                        labels.view(-1, self.config.num_labels),
                    )
                    for lgts in all_logits
                ], dim=0)
                loss = torch.mean(all_losses, dim=0)
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
            logits=logits,
            crf=crf_output,
        )


def get_pretrained(config, model_path):
    model = CustomModel(config)

    if model_path.endswith("pytorch_model.bin"):
        model.load_state_dict(torch.load(model_path))
    else:
        model.backbone = AutoModel.from_pretrained(
            model_path, use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True)
        )

    return model


class LSTMCharHead(nn.Module):
    def __init__(
        self,
        len_voc,
        use_msd=True,
        embed_dim=64,
        lstm_dim=64,
        char_embed_dim=32,
        sent_embed_dim=32,
        ft_lstm_dim=32,
        n_models=1,
    ):
        super().__init__()
        self.use_msd = use_msd

        self.proba_lstm = nn.LSTM(
            n_models * 2, ft_lstm_dim, batch_first=True, bidirectional=True
        )

        self.lstm = nn.LSTM(
            char_embed_dim + ft_lstm_dim * 2 + sent_embed_dim,
            lstm_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            lstm_dim * 2, lstm_dim, batch_first=True, bidirectional=True
        )

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 4, lstm_dim),
            nn.ReLU(),
            nn.Linear(lstm_dim, 2),
        )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, tokens, sentiment, token_probas):
        bs, T = tokens.size()

        probas_fts, _ = self.proba_lstm(token_probas)

        char_fts = self.char_embeddings(tokens)

        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))

        features = torch.cat([char_fts, sentiment_fts, probas_fts], -1)
        features, _ = self.lstm(features)
        features2, _ = self.lstm2(features)

        features = torch.cat([features, features2], -1)

        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding="same",
        use_bn=True,
    ):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2 * dilation

        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                ),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.conv(x)


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


class DebertaForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.lm_predictions = DebertaOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        pass

    def set_output_embeddings(self, new_embeddings):
        pass

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # note that the input embeddings must be passed as an argument
        prediction_scores = self.lm_predictions(sequence_output, self.get_input_embeddings())

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = DebertaLMPredictionHead(config)

    # note that the input embeddings must be passed as an argument
    def forward(self, sequence_output, embeds):
        prediction_scores = self.lm_head(sequence_output, embeds)
        return prediction_scores

class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))


    # note that the input embeddings must be passed as an argument
    def forward(self, hidden_states, embeds):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = torch.matmul(hidden_states, embeds.weight.t()) + self.bias
        return hidden_states

class DebertaV2ForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.lm_predictions = DebertaOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        pass

    def set_output_embeddings(self, new_embeddings):
        pass

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # note that the input embeddings must be passed as an argument
        prediction_scores = self.lm_predictions(sequence_output, self.get_input_embeddings())

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )