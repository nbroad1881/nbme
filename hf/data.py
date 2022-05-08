import os
import re
from pathlib import Path
from ast import literal_eval
from functools import partial
from itertools import chain
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict


def create_folds(df, kfolds=8, groups_col="pn_num"):
    """
    To have good CV, the data should be split
    so that no `pn_num` is in multiple folds.

    args:
        df should be merge of `train.csv`, `features.csv`, and `patient_notes.csv`
    """
    gkf = GroupKFold(n_splits=kfolds)
    fold_idxs = [val_idx for _, val_idx in gkf.split(df, groups=df[groups_col])]
    return fold_idxs


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

    # df.loc[7528, 'location'] = '["840 871"]'
    df.loc[7528, "location"] = '["916 933","938 946"]'
    df.loc[5172, "location"] = '["25 27","50 54"]'
    df.loc[9085, "location"] = '["6 10"]'
    # double check df.loc[11130, 'location'] = '["141 158","213 235"]'
    df.loc[9838, "location"] = '["236 243","251 262"]'
    df.loc[8431, "location"] = '["0 2","23 24"]'
    df.loc[8447, "location"] = '["16 22"]'
    df.loc[6696, "location"] = '["279 287","289 311"]'
    df.loc[5540, "location"] = '["0 2","30 33"]'
    df.loc[6979, "location"] = '["502 508","529 542","857 870","875 887"]'
    df.loc[8140, "location"] = '["680 733"]'
    df.loc[8129, "location"] = "[]"  # this one has a span
    df.loc[3010, "location"] = '["769 772"]'
    df.loc[11921, "location"] = "[]"
    # df.loc[11921, 'location'] = '["15 22"]'
    df.loc[2837, "location"] = '["0 5"]'

    df.loc[7948, "location"] = '["86 98"]'
    df.loc[6891, "location"] = '["93 104","618 626","157 168","207 226","619 626"]'

    df.loc[12022, "location"] = '["379 415"]'
    df.loc[4040, "location"] = '["117 121","131 144","146 152","160 174","180 217"]'

    df.loc[3940, "location"] = '["255 263","324 341","255 263"]'
    df.loc[9270, "location"] = '["64 74"]'
    df.loc[2831, "location"] = '["306 326"]'
    df.loc[8053, "location"] = '["11 12"]'
    df.loc[7172, "location"] = '["448 467", "635 641", "673 681"]'
    df.loc[1424, "location"] = '["74 88", "109 129", "109 129"]'
    df.loc[7777, "location"] = '["288 291", "292 306"]'
    df.loc[9195, "location"] = '["245 255", "311 327"]'
    df.loc[6718, "location"] = '["543 561"]'
    df.loc[
        3938, "location"
    ] = '["114 121", "125 130", "140 164", "166 186", "196 222", "113 222", "166 186", "125 164"]'
    df.loc[3937, "location"] = '["239 250"]'
    df.loc[7671, "location"] = '["404 411", "565 572"]'
    df.loc[8061, "location"] = '["45 74"]'
    df.loc[9580, "location"] = '["148 167"]'
    df.loc[9199, "location"] = "[]"
    df.loc[8280, "location"] = '["89 105"]'
    df.loc[1054, "location"] = '["827 846"]'
    df.loc[1058, "location"] = "[]"

    df.loc[10756, "location"] = '["376 387"]'
    # df.loc[10756, 'location'] = '["544 566"]'

    df.loc[3787, "location"] = '["83 124"]'
    df.loc[7278, "location"] = '["113 130"]'
    # df.loc[7278, 'location'] = '["132 153"]'

    df.loc[2829, "location"] = '["6 7"]'
    df.loc[7243, "location"] = '["14 20"]'
    df.loc[3930, "location"] = '["359 374"]'
    df.loc[7957, "location"] = '["295 303", "316 337"]'
    df.loc[7945, "location"] = '["45 46"]'
    df.loc[11402, "location"] = '["31 38"]'
    df.loc[8068, "location"] = '["217 230", "246 258", "209 230"]'

    df.loc[2825, "location"] = '["533 548", "904 919"]'
    df.loc[3920, "location"] = '["344 355", "344 355"]'

    df.loc[3939, "location"] = '["44 65"]'
    df.loc[7941, "location"] = '["269 290"]'
    df.loc[11780, "location"] = '["81 92"]'
    df.loc[6825, "location"] = '["703 717"]'
    df.loc[13962, "location"] = '["55 64"]'
    df.loc[11690, "location"] = '["28 35", "67 80"]'
    df.loc[3927, "location"] = '["444 468"]'
    df.loc[3019, "location"] = '["493 504"]'
    df.loc[8065, "location"] = '["192 203"]'
    df.loc[7246, "location"] = '["99 111"]'
    df.loc[3021, "location"] = '["97 141", "35 78"]'
    df.loc[7242, "location"] = '["113 124", "113 148"]'
    df.loc[3931, "location"] = '["27 43"]'

    df.loc[3174, "location"] = '["45 48", "55 59"]'
    df.loc[972, "location"] = '["61 81"]'
    df.loc[9587, "location"] = '["220 233"]'
    df.loc[3011, "location"] = '["724 739"]'
    df.loc[9998, "location"] = '["103 111", "606 614"]'
    df.loc[7814, "location"] = '["171 184"]'
    df.loc[8056, "location"] = '["111 131", "17 29", "94 109"]'
    df.loc[3920, "location"] = '["344 355", "344 355"]'
    df.loc[12685, "location"] = '[ "251 273", "232 246", "259 273"]'
    df.loc[1265, "location"] = '["181 212"]'
    df.loc[2356, "location"] = '["231 239"]'
    df.loc[7255, "location"] = '["380 393"]'
    df.loc[7632, "location"] = '["342 358"]'
    df.loc[7022, "location"] = '["133 182"]'
    df.loc[3610, "location"] = '["300 349", "356 401"]'
    df.loc[9352, "location"] = '["205 233", "332 360"]'
    df.loc[1535, "location"] = '["284 291", "568 578"]'
    df.loc[
        3326, "location"
    ] = '["83 104", "109 141", "143 174", "234 238", "177 187", "212 229"]'
    df.loc[8253, "location"] = '["58 77"]'
    df.loc[2717, "location"] = "[]"
    # df.loc[2717, 'location'] = '["310 326"]'
    df.loc[3020, "location"] = '["198 219", "225 259", "178 196"]'
    df.loc[8447, "location"] = '["16 22"]'
    df.loc[740, "location"] = '["25 29"]'
    df.loc[4073, "location"] = '["536 547"]'
    df.loc[1026, "location"] = '["6 10"]'
    df.loc[6737, "location"] = '["24 30"]'
    df.loc[9171, "location"] = "[]"
    df.loc[9172, "location"] = '["341 351", "366 380"]'
    df.loc[1992, "location"] = '["341 349"]'
    df.loc[7639, "location"] = "['11 12']"
    df.loc[1910, "location"] = "['0 2']"
    df.loc[11990, "location"] = "['28 29']"
    df.loc[8578, "location"] = '["130 144"]'
    df.loc[8579, "location"] = '["69 124"]'
    df.loc[8340, "location"] = '["70 81"]'
    # df.loc[8340, 'location'] = '["70 81"]'
    df.loc[8349, "location"] = '["91 122"]'
    df.loc[11905, "location"] = '["99 139"]'
    df.loc[11908, "location"] = '["446 526"]'
    df.loc[3015, "location"] = '["277 303"]'
    df.loc[9199, "location"] = '["83 105"]'
    df.loc[7814, "location"] = '["171 184"]'

    # investigate 13099, 5453, 8578, 2453, 6060, 9380, 3032, 658, 9199, 3937, 10055, 6000
    # 5453, 9184, 4679, 8776, 7633, 473, 8515, 8480, 6300, 9187, 8219, 9316, 9228, 9187
    # not guessing punctuation midway through (comma in 11461)
    # not getting punctuation at end (7271, 4836)
    # n/v for nausea vomiting
    # presents for 8263
    # No blood in stool, no hemochezia
    # FHx of depression or Family history of depression, mdd
    # Heavy sweating (also reports?)
    # No premenstrual symptoms (has not had)
    # Recent upper respiratory symptoms
    # loss of interest (endorses?)
    # heavy periods or irregular periods (now?)
    # hot flashes (endorses?)
    # viral symptoms or rhinorrhea or scratchy throat (ST)
    #  8 to 10 hours of acute pain (abdominal?)
    # Duration x 1 day (pain?)

    # Intermittent symptoms (episodes?)
    return df


def location_to_ints(location):
    to_return = []

    for loc_str in location:
        loc_strs = loc_str.split(";")

        for loc in loc_strs:
            start, end = loc.split()
            to_return.append((int(start), int(end)))

    return to_return


def process_feature_text(text, use_custom_features=False):

    if use_custom_features:
        separator = " or "
        if "beers" in text:
            text = separator.join([text, "drink alcohol", "etoh", "occasional"])
        elif text in {
            "45-year",
            "67-year",
            "44-year",
            "26-year",
            "20-year",
            "17-year",
            "35-year",
        }:
            text = separator.join([text + " y o year old"])
        elif "IUD" in text:
            text = separator.join([text, "intrauterine device"])
        elif text == "Unprotected-Sex":
            text = separator.join(
                [
                    text,
                    "no contraception",
                    "no condoms",
                    "no protection",
                    "no barrier",
                    "no ocp",
                ]
            )
        elif text == "1-day-duration-OR-2-days-duration":
            text = separator.join([text, "yesterday", "day ago"])
        elif text == "Prior-episodes-of-diarrhea":
            text = separator.join(
                [text, "loose stool", "diarrhea days ago", "soft", "watery stools"]
            )
        elif text == "Irregular-flow-OR-Irregular-frequency-OR-Irregular-intervals":
            text = separator.join(
                [
                    text,
                    "variable blood or flow",
                    "menses",
                    "use pads tampons",
                    "last days",
                    "heavy and light flow",
                    "no pattern periods",
                ]
            )
        elif text == "Episodes-of-heart-racing":
            text = separator.join(
                [text, "palpitations", "heart pounding", "heart beating fast"]
            )
        elif text == "Epigastric-discomfort":
            text = separator.join(["Epigastric or stomach discomfort"])
        elif text == "Chest-pain":
            text = separator.join([text, "CP"])
        elif text == "Vaginal-dryness":
            text = separator.join([text, "uses lubrication"])
        elif text == "Diminished-energy-or-feeling-drained":
            text = separator.join([text, "fatigue"])
        elif (
            text
            == "Right-sided-LQ-abdominal-pain-or-Right-lower-quadrant-abdominal-pain"
        ):
            text = separator.join([text, "RLQ pain"])
        elif text == "Global-headache-or-diffuse-headache":
            text = separator.join([text, "HA"])
        elif text == "Infertility-HX-or-Infertility-history":
            text = separator.join([text, "P0"])
        elif text == "Vomiting":
            text = separator.join([text, "can't keep down food"])
        elif text == "No-hair-changes-OR-no-nail-changes-OR-no-temperature-intolerance":
            text = separator.join(
                [
                    "no hair or nail changes",
                    "no hot/cold/temperature intolerance",
                    "denies sweating",
                ]
            )
        elif text == "Intermittent-symptoms":
            text = separator.join([text, "episodes"])
        elif text == "Intermittent":
            text = separator.join([text, "comes and goes"])
        elif text == "No-chest-pain":
            text = separator.join([text, "no angina"])
        # only feature num 208 has g2p2 for "Female", g2p2 only shows up when no other indicator

    text = text.replace("-OR-", " or ")
    return text.replace("-", " ")


def tokenize(example, tokenizer, max_seq_length, padding):

    tokenized_inputs = tokenizer(
        example["feature_text"],
        example["pn_history"],
        truncation="only_second",
        max_length=max_seq_length,
        padding=padding,
        return_offsets_mapping=True,
    )

    # labels should be float
    labels = [0.0] * len(tokenized_inputs["input_ids"])
    tokenized_inputs["locations"] = location_to_ints(example["location"])
    tokenized_inputs["sequence_ids"] = tokenized_inputs.sequence_ids()

    if len(tokenized_inputs["locations"]) > 0:
        for idx, (seq_id, offsets) in enumerate(
            zip(tokenized_inputs["sequence_ids"], tokenized_inputs["offset_mapping"])
        ):
            if seq_id is None or seq_id == 0:
                # don't calculate loss on question part or special tokens
                labels[idx] = -100.0
                continue

            token_start, token_end = offsets
            for label_start, label_end in tokenized_inputs["locations"]:
                if (
                    token_start <= label_start < token_end
                    or token_start < label_end <= token_end
                    or label_start <= token_start < label_end
                ):
                    labels[idx] = 1.0  # labels should be float
    else:
        for idx, seq_id in enumerate(tokenized_inputs["sequence_ids"]):
            if seq_id is None or seq_id == 0:
                # don't calculate loss on question part or special tokens
                labels[idx] = -100.0

    tokenized_inputs["labels"] = labels
    

    return tokenized_inputs


def tokenize_with_newline_replacement(
    example,
    tokenizer,
    max_seq_length,
    padding,
    space_id,
    newline_id,
):
    """
    For deberta v2, space_id ("▁") is 250.
    For deberta v3, space_id ("▁") is 507.
    Both have newline_id = 128001
    """

    tokenized_inputs = tokenize(example, tokenizer, max_seq_length, padding)

    pattern = "[\n\r]+"

    matches = [x for x in re.finditer(pattern, example["pn_history"])]

    if len(matches) == 0:
        return tokenized_inputs

    new_ids = []
    new_tok_types = []
    new_mask = []
    new_labels = []
    new_offsets = []
    new_sequence_ids = []

    keys = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "labels",
        "offset_mapping",
        "sequence_ids",
    ]

    zipped = zip(*[tokenized_inputs[k] for k in keys])

    match_idx = 0
    num_matches = len(matches)
    for id_, t_id, mask, lab, (o1, o2), s_id in zipped:

        # additional offset if there is a match
        additional = 0

        if match_idx < num_matches and s_id == 1:
            m = matches[match_idx]
            if o1 >= m.start() and o2 > m.end():
                new_ids.extend([space_id, newline_id])
                new_tok_types.extend([t_id, t_id])
                new_mask.extend([mask, mask])
                new_labels.extend([lab, lab])
                new_offsets.extend(
                    [(m.start(), m.start() + 1), (m.start() + 1, m.end())]
                )
                new_sequence_ids.extend([s_id, s_id])

                match_idx += 1
                additional += len(m.group(0))

        new_ids.append(id_)
        new_tok_types.append(t_id)
        new_mask.append(mask)
        new_labels.append(lab)
        new_offsets.append((o1 + additional, o2))
        new_sequence_ids.append(s_id)

    return {
        "input_ids": new_ids,
        "token_type_ids": new_tok_types,
        "attention_mask": new_mask,
        "labels": new_labels,
        "offset_mapping": new_offsets,
        "sequence_ids": new_sequence_ids,
        **{k: v for k, v in tokenized_inputs.items() if k not in keys},
    }


def substitute_for_newline(text, repl=" [n] ", return_matches=True):
    pattern = "[\n\r]+"

    matches = [x for x in re.finditer(pattern, text)]

    new_text = re.sub(pattern, repl, text)

    if return_matches:
        return new_text, matches
    # Don't need matches for MLM
    return new_text


@dataclass
class NERDataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        data_dir = Path(self.cfg["data_dir"])

        features_df = pd.read_csv(data_dir / "features.csv")
        features_df.loc[27, "feature_text"] = "Last-Pap-smear-1-year-ago"
        notes_df = pd.read_csv(data_dir / "patient_notes.csv")
        train_df = pd.read_csv(data_dir / "train.csv")

        train_df = train_df.merge(
            features_df, on=["feature_num", "case_num"], how="left"
        )
        train_df = train_df.merge(notes_df, on=["pn_num", "case_num"], how="left")
        train_df = fix_annotations(train_df)

        train_df["annotation"] = [literal_eval(x) for x in train_df.annotation]
        train_df["location"] = [literal_eval(x) for x in train_df.location]

        train_df["feature_text"] = [
            process_feature_text(
                x, use_custom_features=self.cfg.get("use_custom_features")
            )
            for x in train_df["feature_text"]
        ]

        if self.cfg.get("use_lowercase"):
            train_df["pn_history"] = train_df["pn_history"].str.lower()

        self.train_df = train_df.sample(frac=1, random_state=42)
        if self.cfg["DEBUG"]:
            self.train_df = self.train_df.sample(n=1000)


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_name_or_path"],
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )

        if (
            self.cfg.get("newline_replacement")
            and self.cfg.get("newline_replacement").strip() not in self.tokenizer.vocab
        ):
            self.tokenizer.add_tokens([self.cfg.get("newline_replacement").strip()])

        if self.cfg.get("make_pseudolabels"):
            self.unlabeled_df = (
                pd.read_csv(data_dir / "train.csv", usecols=["id", "case_num", "feature_num"]) 
                .merge(notes_df, on=["case_num"], how="right")
                .merge(features_df, on=["feature_num"], how="left")
            )

            self.unlabeled_df = self.unlabeled_df.loc[
                ~self.unlabeled_df.pn_num.isin(self.train_df.pn_num), :
            ]

            self.unlabeled_df = self.unlabeled_df.groupby('feature_num', group_keys=False).apply(lambda x: x.sample(min(len(x), 500)))
            self.unlabeled_df["location"] = [[]]*len(self.unlabeled_df)
            
        self.train_df["temp_id"] = list(range(len(self.train_df)))
        self.train_df[["id", "temp_id"]].to_csv("id2id.csv", index=False)
        self.train_df["id"] = self.train_df["temp_id"]
        self.fold_idxs = create_folds(self.train_df, kfolds=self.cfg["k_folds"])
        if self.cfg.get("use_pseudolabels"):
            pl_df = pd.read_csv(data_dir/self.cfg.get("use_pseudolabels"))
            pl_df["annotation"] = [literal_eval(x) for x in pl_df.annotation]
            pl_df["location"] = [literal_eval(x) for x in pl_df.location]
            pl_df = pl_df[~pl_df.pn_num.isin(self.train_df.pn_num)]
            pl_df = pl_df.groupby('feature_num', group_keys=False).apply(lambda x: x.sample(min(len(x), 400)))
            
            
            train_df_size = len(self.train_df)
            self.train_df = pd.concat([self.train_df, pl_df], axis=0, ignore_index=True)
            self.fold_idxs.append(list(range(train_df_size, len(self.train_df))))

    def prepare_datasets(self, cfg=None):
        
        if cfg:
            self.cfg = cfg

        if self.cfg.get("make_pseudolabels"):
            self.dataset = Dataset.from_pandas(self.unlabeled_df)
        else:
            self.dataset = Dataset.from_pandas(self.train_df)

        if self.cfg.get("newline_replacement"):
            vocab = self.tokenizer.vocab
            newline_id = vocab[self.cfg["newline_replacement"].strip()]
            space_id = vocab[self.cfg["space_token"]]

            self.dataset = self.dataset.map(
                partial(
                    tokenize_with_newline_replacement,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.cfg["max_seq_length"],
                    padding=self.cfg["padding"],
                    space_id=space_id,
                    newline_id=newline_id,
                ),
                batched=False,
                num_proc=self.cfg["num_proc"],
                remove_columns=[],
            )
        else:
            self.dataset = self.dataset.map(
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

    def get_train_dataset(self, fold):
        idxs = list(chain(*[i for f, i in enumerate(self.fold_idxs) if f != fold]))
        return self.dataset.select(idxs)

    def get_eval_dataset(self, fold):
        return self.dataset.select(self.fold_idxs[fold])


@dataclass
class MLMDataModule:

    cfg: dict = None

    def __post_init__(self):
        if self.cfg is None:
            raise ValueError

        data_dir = Path(self.cfg["data_dir"])

        train_df = pd.read_csv(data_dir / "patient_notes.csv")

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model_name_or_path"])

        if self.cfg.get("newline_replacement"):
            repl = self.cfg.get("newline_replacement")
            train_df["pn_history"] = [
                substitute_for_newline(t, repl, return_matches=False)
                for t in train_df["pn_history"]
            ]
            self.tokenizer.add_tokens([repl.strip()])

        if self.cfg.get("use_lowercase"):
            train_df["pn_history"] = train_df["pn_history"].str.lower()

        self.train_df = train_df.sample(frac=1, random_state=42)
        if self.cfg["DEBUG"]:
            self.train_df = self.train_df.sample(n=2500)

        self.fold_idxs = create_folds(
            self.train_df.reset_index(drop=True), kfolds=self.cfg["k_folds"]
        )

    def prepare_datasets(self, fold):

        self.dataset = DatasetDict()

        train_idxs = list(
            chain(*[i for f, i in enumerate(self.fold_idxs) if f != fold])
        )

        self.dataset["train"] = Dataset.from_pandas(
            self.train_df.reset_index(drop=True).loc[train_idxs]
        )
        self.dataset["validation"] = Dataset.from_pandas(
            self.train_df.reset_index(drop=True).loc[self.fold_idxs[fold]]
        )

        self.dataset = self.dataset.map(
            lambda x: self.tokenizer(x["pn_history"], return_special_tokens_mask=True),
            batched=True,
            num_proc=self.cfg["num_proc"],
            remove_columns=self.dataset["train"].column_names,
        )

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= self.cfg["max_seq_length"]:
                total_length = (total_length // self.cfg["max_seq_length"]) * self.cfg[
                    "max_seq_length"
                ]
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + self.cfg["max_seq_length"]]
                    for i in range(0, total_length, self.cfg["max_seq_length"])
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        self.dataset = self.dataset.map(
            group_texts,
            batched=True,
            num_proc=self.cfg["num_proc"],
            remove_columns=self.dataset["train"].column_names,
        )

    def get_train_dataset(self):
        return self.dataset["train"]

    def get_eval_dataset(self):
        return self.dataset["validation"]
