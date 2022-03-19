from pathlib import Path
from ast import literal_eval
from functools import partial
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


def create_folds(df, kfolds=8):
    """
    To have good CV, the data should be split
    so that no `pn_num` is in multiple folds.

    args:
        df should be merge of `train.csv`, `features.csv`, and `patient_notes.csv`
    """
    gkf = GroupKFold(n_splits=kfolds)
    groups = df["pn_num"]
    df["fold"] = -1
    for fold, (val_idx, _) in enumerate(gkf.split(df, y=df["location"], groups=groups)):
        df.loc[val_idx, "fold"] = fold

    return df


def fix_annotations(df):
    """
    Nearly the same as https://www.kaggle.com/yasufuminakama/nbme-deberta-base-baseline-train?scriptVersionId=87264998&cellId=17

    args:
        df should be merge of `train.csv`, `features.csv`, and `patient_notes.csv`
    """

    df.loc[338, "annotation"] = '["father heart attack"]'
    df.loc[338, "location"] = '["764 783"]'

    df.loc[621, "annotation"] = '["for the last 2-3 months", "over the last 2 months"]'
    df.loc[621, "location"] = '["77 100", "398 420"]'

    df.loc[655, "annotation"] = '["no heat intolerance", "no cold intolerance"]'
    df.loc[655, "location"] = '["285 292;301 312", "285 287;296 312"]'

    df.loc[1262, "annotation"] = '["mother thyroid problem"]'
    df.loc[1262, "location"] = '["551 557;565 580"]'

    df.loc[1265, "annotation"] = "['felt like he was going to \"pass out\"']"
    df.loc[1265, "location"] = '["131 135;181 212"]'

    df.loc[1396, "annotation"] = '["stool , with no blood"]'
    df.loc[1396, "location"] = '["259 280"]'

    df.loc[1591, "annotation"] = '["diarrhoe non blooody"]'
    df.loc[1591, "location"] = '["176 184;201 212"]'

    df.loc[1615, "annotation"] = '["diarrhea for last 2-3 days"]'
    df.loc[1615, "location"] = '["249 257;271 288"]'

    df.loc[1664, "annotation"] = '["no vaginal discharge"]'
    df.loc[1664, "location"] = '["822 824;907 924"]'

    df.loc[1714, "annotation"] = '["started about 8-10 hours ago"]'
    df.loc[1714, "location"] = '["101 129"]'

    df.loc[1929, "annotation"] = '["no blood in the stool"]'
    df.loc[1929, "location"] = '["531 539;549 561"]'

    df.loc[2134, "annotation"] = '["last sexually active 9 months ago"]'
    df.loc[2134, "location"] = '["540 560;581 593"]'

    df.loc[2191, "annotation"] = '["right lower quadrant pain"]'
    df.loc[2191, "location"] = '["32 57"]'

    df.loc[2553, "annotation"] = '["diarrhoea no blood"]'
    df.loc[2553, "location"] = '["308 317;376 384"]'

    df.loc[3124, "annotation"] = '["sweating"]'
    df.loc[3124, "location"] = '["549 557"]'

    df.loc[
        3858, "annotation"
    ] = '["previously as regular", "previously eveyr 28-29 days", "previously lasting 5 days", "previously regular flow"]'
    df.loc[
        3858, "location"
    ] = '["102 123", "102 112;125 141", "102 112;143 157", "102 112;159 171"]'

    df.loc[4373, "annotation"] = '["for 2 months"]'
    df.loc[4373, "location"] = '["33 45"]'

    df.loc[4763, "annotation"] = '["35 year old"]'
    df.loc[4763, "location"] = '["5 16"]'

    df.loc[4782, "annotation"] = '["darker brown stools"]'
    df.loc[4782, "location"] = '["175 194"]'

    df.loc[4908, "annotation"] = '["uncle with peptic ulcer"]'
    df.loc[4908, "location"] = '["700 723"]'

    df.loc[6016, "annotation"] = '["difficulty falling asleep"]'
    df.loc[6016, "location"] = '["225 250"]'

    df.loc[6192, "annotation"] = '["helps to take care of aging mother and in-laws"]'
    df.loc[6192, "location"] = '["197 218;236 260"]'

    df.loc[
        6380, "annotation"
    ] = '["No hair changes", "No skin changes", "No GI changes", "No palpitations", "No excessive sweating"]'
    df.loc[
        6380, "location"
    ] = '["480 482;507 519", "480 482;499 503;512 519", "480 482;521 531", "480 482;533 545", "480 482;564 582"]'

    df.loc[
        6562, "annotation"
    ] = '["stressed due to taking care of her mother", "stressed due to taking care of husbands parents"]'
    df.loc[6562, "location"] = '["290 320;327 337", "290 320;342 358"]'

    df.loc[6862, "annotation"] = '["stressor taking care of many sick family members"]'
    df.loc[6862, "location"] = '["288 296;324 363"]'

    df.loc[
        7022, "annotation"
    ] = '["heart started racing and felt numbness for the 1st time in her finger tips"]'
    df.loc[7022, "location"] = '["108 182"]'

    df.loc[7422, "annotation"] = '["first started 5 yrs"]'
    df.loc[7422, "location"] = '["102 121"]'

    df.loc[8876, "annotation"] = '["No shortness of breath"]'
    df.loc[8876, "location"] = '["481 483;533 552"]'

    df.loc[
        9027, "annotation"
    ] = '["recent URI", "nasal stuffines, rhinorrhea, for 3-4 days"]'
    df.loc[9027, "location"] = '["92 102", "123 164"]'

    df.loc[
        9938, "annotation"
    ] = '["irregularity with her cycles", "heavier bleeding", "changes her pad every couple hours"]'
    df.loc[9938, "location"] = '["89 117", "122 138", "368 402"]'

    df.loc[9973, "annotation"] = '["gaining 10-15 lbs"]'
    df.loc[9973, "location"] = '["344 361"]'

    df.loc[10513, "annotation"] = '["weight gain", "gain of 10-16lbs"]'
    df.loc[10513, "location"] = '["600 611", "607 623"]'

    df.loc[11551, "annotation"] = '["seeing her son knows are not real"]'
    df.loc[11551, "location"] = '["386 400;443 461"]'

    df.loc[11677, "annotation"] = '["saw him once in the kitchen after he died"]'
    df.loc[11677, "location"] = '["160 201"]'

    df.loc[12124, "annotation"] = '["tried Ambien but it didnt work"]'
    df.loc[12124, "location"] = '["325 337;349 366"]'

    df.loc[
        12279, "annotation"
    ] = '["heard what she described as a party later than evening these things did not actually happen"]'
    df.loc[12279, "location"] = '["405 459;488 524"]'

    df.loc[
        12289, "annotation"
    ] = '["experienced seeing her son at the kitchen table these things did not actually happen"]'
    df.loc[12289, "location"] = '["353 400;488 524"]'

    df.loc[13238, "annotation"] = '["SCRACHY THROAT", "RUNNY NOSE"]'
    df.loc[13238, "location"] = '["293 307", "321 331"]'

    df.loc[
        13297, "annotation"
    ] = '["without improvement when taking tylenol", "without improvement when taking ibuprofen"]'
    df.loc[13297, "location"] = '["182 221", "182 213;225 234"]'

    df.loc[13299, "annotation"] = '["yesterday", "yesterday"]'
    df.loc[13299, "location"] = '["79 88", "409 418"]'

    df.loc[13845, "annotation"] = '["headache global", "headache throughout her head"]'
    df.loc[13845, "location"] = '["86 94;230 236", "86 94;237 256"]'

    df.loc[14083, "annotation"] = '["headache generalized in her head"]'
    df.loc[14083, "location"] = '["56 64;156 179"]'

    return df

def location_to_ints(location):
    to_return = []
    
    for loc_str in location:
        loc_strs = loc_str.split(";")
        
        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))
        
    return to_return

def process_feature_text(text):
    text = text.replace("-OR-", ";")
    return text.replace("-", " ")

def tokenize(example, tokenizer, max_seq_length, padding):
    
    tokenized_inputs = tokenizer(
        example["feature_text"],
        example["pn_history"],
        truncation="only_second",
        max_length=max_seq_length,
        padding=padding,
        return_offsets_mapping=True
    )
    
    # labels should be float
    labels = [0.0]*len(tokenized_inputs["input_ids"])
    tokenized_inputs["locations"] = location_to_ints(example["location"])
    tokenized_inputs["sequence_ids"] = tokenized_inputs.sequence_ids()
    
    for idx, (seq_id, offsets) in enumerate(zip(tokenized_inputs["sequence_ids"], tokenized_inputs["offset_mapping"])):
        if seq_id is None or seq_id == 0:
            labels[idx] = -100.0 # don't calculate loss on question part or special tokens
            continue
            
        exit = False
        token_start, token_end = offsets
        for label_start, label_end in tokenized_inputs["locations"]:
            if exit: 
                break
            if token_start <= label_start < token_end or token_start < label_end <= token_end or label_start <= token_start < label_end:
                labels[idx] = 1.0 # labels should be float
                exit = True
            
    
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs
    

@dataclass
class DataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        data_dir = Path(self.cfg["data_dir"])

        features_df = pd.read_csv(data_dir / "features.csv")
        notes_df = pd.read_csv(data_dir / "patient_notes.csv")
        train_df = pd.read_csv(data_dir / "train.csv")

        train_df = train_df.merge(features_df, on=["feature_num", "case_num"], how="left")
        train_df = train_df.merge(notes_df, on=["pn_num", "case_num"], how="left")
        train_df = fix_annotations(train_df)

        train_df = create_folds(train_df)

        train_df["annotation"] = [literal_eval(x) for x in train_df.annotation]
        train_df["location"]  = [literal_eval(x) for x in train_df.location]

        train_df["feature_text"] = [process_feature_text(x) for x in train_df["feature_text"]]
        self.train_df = train_df[train_df["annotation"].map(len)!=0].copy().reset_index(drop=True)


        if (
            "deberta-v2" in self.run_config.tokenizer_path
            or "deberta-v3" in self.run_config.tokenizer_path
        ):
            from transformers.models.deberta_v2.tokenization_deberta_v2_fast import (
                DebertaV2TokenizerFast,
            )

            self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
                self.run_config.tokenizer_path
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.run_config.tokenizer_path
            )

    def prepare_datasets(self, fold):

        self.dataset = DatasetDict()

        self.dataset["train"] = Dataset.from_pandas(self.train_df[self.train_df["fold"]!=fold].copy().reset_index(drop=True))
        self.dataset['validation'] = Dataset.from_pandas(self.train_df[self.train_df["fold"]==fold].copy().reset_index(drop=True))

        
        self.dataset["train"] =self.dataset["train"].map(
            partial(
                tokenize,
                tokenizer=self.tokenizer,
                max_seq_length=self.cfg["max_seq_length"], 
                padding=self.cfg["padding"],
            ),
            batched=False,
            num_proc=self.cfg["num_proc"],
            remove_columns=[],
        )

        self.dataset['validation'] = self.dataset['validation'].map(
            partial(
                tokenize,
                tokenizer=self.tokenizer,
                max_seq_length=self.cfg["max_seq_length"], 
                padding=self.cfg["padding"],
            ),
            batched=False,
            num_proc=self.cfg["num_proc"],
            remove_columns=[],
        )

    def get_train_dataset(self):
        return self.dataset["train"]

    def get_eval_dataset(self):
        return self.dataset["validation"]