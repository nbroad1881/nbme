from itertools import chain
from dataclasses import dataclass

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import DataCollatorForTokenClassification


def kaggle_metrics(eval_prediction, dataset):
    """
    For `compute_metrics`

    Use partial for the args and kwargs to pass other data
    into the `compute_metrics` function.
    """
    all_labels = []
    all_preds = []
    for preds, offsets, seq_ids, labels, text in zip(
        eval_prediction.predictions, 
        dataset["offset_mapping"], 
        dataset["sequence_ids"], 
        dataset["labels"], 
        dataset["text"]
    ):
        
        num_chars = max(list(chain(*offsets)))
        char_labels = np.zeros((num_chars), dtype=bool)
        
        for (tok_start, tok_end), s_id, label in zip(offsets, seq_ids, labels):
            if s_id is None or s_id == 0: # ignore question part of input
                continue
            if int(label) == 1:
                
                char_labels[tok_start:tok_end] = 1
                if text[tok_start].isspace() and tok_start>0 and not char_labels[tok_start-1]:
                    char_labels[tok_start] = 0
                    
        char_preds = np.zeros((num_chars))
        
        for start_idx, end_idx in preds:
            char_preds[start_idx:end_idx] = 1
            if text[start_idx].isspace() and start_idx>0 and not char_preds[start_idx-1]:
                char_preds[start_idx] = 0
            
        all_labels.extend(char_labels)
        all_preds.extend(char_preds)
        
    results = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    
    return {
            "precision": results[0],
            "recall": results[1],
            "f1": results[2]
        } 


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
