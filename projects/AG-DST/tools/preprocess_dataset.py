#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocess dataset."""

import argparse
from collections import defaultdict
import json
import os
import re

from tqdm import tqdm

from utils import get_logger


logger = get_logger(__name__)


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    return args


def main(args):
    """Main function."""
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    process_dataset(args.data_path, args.mapping_file, args.save_path)
    divide_data(args.data_path, args.save_path)


def process_dataset(data_path, mapping_file, save_path):
    """Process dataset: schema highlight and normalization."""
    data_file = os.path.join(data_path, "data.json")
    with open(data_file, "r") as fin:
        data = json.load(fin)
        logger.info(f"Load dataset from `{data_file}`")
    # token mapping for normalization.
    token_mapping = []
    with open(mapping_file, "r") as fin:
        for line in fin:
            origin_token, new_token = line.strip().split("\t")
            token_mapping.append((" " + origin_token + " ", " " + new_token + " "))

    for dial_id, dial in tqdm(data.items(), desc="Dialogue"):
        for turn in dial["log"]:
            highlight_text = schema_highlight(turn)
            normalized_text = normalize(highlight_text, token_mapping)
            normalize_ds(turn)
            turn["processed_text"] = normalized_text

    out_data_file = os.path.join(save_path, "processed_data.json")
    with open(out_data_file, "w") as fout:
        json.dump(data, fout, indent=2)
        logger.info(f"Save dataset to `{out_data_file}`")


def schema_highlight(turn):
    """
    Highlight some slot values in the utterance.

    Args:
        turn(dict): A dictionary of the dialogue turn.

    Returns:
        highlight_text(str): The dialogue utterance after highlighting.
    """
    highlight_slots = ("name",)
    span_info = sorted(turn["span_info"], key=lambda x: x[3])
    text = turn["text"]
    highlight_text = ""
    prev_end = 0
    for _, slot, val, start, end in span_info:
        if slot not in highlight_slots or val == "dontcare":
            continue
        highlight_text += text[prev_end: start]
        highlight_text += f"<{slot}/> {text[start: end]} </{slot}>"
        prev_end = end
    highlight_text += text[prev_end:]
    return highlight_text


def normalize(text, token_mapping):
    """
    Normalize utterance.
    Similar to the `normalize` in https://github.com/budzianowski/multiwoz/blob/master/utils/nlp.py
    """
    def insert_space(text, token):
        sidx = 0
        while True:
            sidx = text.find(token, sidx)
            if sidx == -1:
                break
            if sidx + 1 < len(text) and re.match("[0-9]", text[sidx - 1]) and \
                    re.match("[0-9]", text[sidx + 1]):
                sidx += 1
                continue
            if text[sidx - 1] != " ":
                text = text[:sidx] + " " + text[sidx:]
                sidx += 1
            if sidx + len(token) < len(text) and text[sidx + len(token)] != " ":
                text = text[:sidx + 1] + " " + text[sidx + 1:]
            sidx += 1
        return text

    text = text.lower().strip()

    # from TripPy: normalize time
    # https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/blob/master/dataset_multiwoz21.py
    text = re.sub("(\d{1})(a\.?m\.?|p\.?m\.?)", r"\1 \2", text)  # am/pm without space
    text = re.sub("(^| )(\d{1,2}) (a\.?m\.?|p\.?m\.?)", r"\1\2:00 \3", text)  # am/pm short to long form
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2}) ?(\d{2})([^0-9]|$)", r"\1\2 \3:\4\5", text)
    text = re.sub("(^| )(\d{2})[;.,](\d{2})", r"\1\2:\3", text)  # Wrong separator
    text = re.sub("(^| )(at|from|by|until|after) ?(\d{1,2})([;., ]|$)", r"\1\2 \3:00\4", text)
    text = re.sub("(^| )(\d{1}:\d{2})", r"\g<1>0\2", text)  # Add missing leading 0
    # Map 12 hour times to 24 hour times
    text = re.sub("(\d{2})(:\d{2}) ?p\.?m\.?", lambda x: str(int(x.groups()[0]) + 12 if int(x.groups()[0]) < 12 \
        else int(x.groups()[0])) + x.groups()[1], text)
    text = re.sub("(^| )24:(\d{2})", r"\g<1>00:\2", text)  # Correct times that use 24 as hour

    # from TripPy: normalize text
    # https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public/-/blob/master/dataset_multiwoz21.py
    text = re.sub("n't", " not", text)
    text = re.sub("(^| )zero(-| )star([s.,? ]|$)", r"\g<1>0 star\3", text)
    text = re.sub("(^| )one(-| )star([s.,? ]|$)", r"\g<1>1 star\3", text)
    text = re.sub("(^| )two(-| )star([s.,? ]|$)", r"\g<1>2 star\3", text)
    text = re.sub("(^| )three(-| )star([s.,? ]|$)", r"\g<1>3 star\3", text)
    text = re.sub("(^| )four(-| )star([s.,? ]|$)", r"\g<1>4 star\3", text)
    text = re.sub("(^| )five(-| )star([s.,? ]|$)", r"\g<1>5 star\3", text)
    text = re.sub("archaelogy", "archaeology", text)  # Systematic typo
    text = re.sub("guesthouse", "guest house", text)  # Normalization
    text = re.sub("(^| )b ?& ?b([.,? ]|$)", r"\1bed and breakfast\2", text)  # Normalization
    text = re.sub("bed & breakfast", "bed and breakfast", text)  # Normalization

    # normalize phone number
    match_res = re.finditer(r"\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})", text)
    tmp_text = ""
    prev_end = 0
    for match_obj in match_res:
        start, end = match_obj.start(), match_obj.end()
        tmp_text += text[prev_end:start] + "".join(match_obj.groups())
        prev_end = end
    tmp_text += text[prev_end:]
    text = tmp_text

    # normalize postcode
    match_res = re.finditer(r"((cb|c\.b|c b|pe|p\.e|p e)[\. ]?\d{1,2}[, ]+\d{1}[\. ]?"
                            r"[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})",
                            text)
    tmp_text = ""
    prev_end = 0
    for match_obj in match_res:
        start, end = match_obj.start(), match_obj.end()
        tmp_text += text[prev_end:start] + re.sub(r"[,\. ]", "", match_obj[0])
        prev_end = end
    tmp_text += text[prev_end:]
    text = tmp_text

    # replace some tokens
    replacements = (
        ((";",), ","),
        (("-", "' ",), " "),
        (("\u2018", "\u2019",), "'"),
        (("\"", "@", "(", ")",), ""),
    )
    for origins, new in replacements:
        for origin in origins:
            text = text.replace(origin, new)

    # replace tokens according to the multiwoz token mapping
    for origin, new in token_mapping:
        text = " " + text + " "
        text = text.replace(origin, new)[1:-1]

    # insert white space before and after several tokens
    for token in ("?", ".", ",", "!", ":", "'s", "'ve"):
        text = insert_space(text, token)

    # remove multiple spaces
    text = " ".join(text.split())

    # concatenate numbers
    text = re.sub(r" (\d+) (\d+)", r" \1\2", text)

    return text


def normalize_ds(turn):
    """Normalize dialogue state."""
    processed_metadata = defaultdict(lambda: defaultdict(dict))
    slot_map = {"ticket": "people"}
    for dom, dom_ds in turn["metadata"].items():
        for slot_type, slot_vals in dom_ds.items():
            for slot, vals in slot_vals.items():
                slot = slot_map.get(slot, slot).lower()
                processed_metadata[dom][slot_type][slot] = vals
    turn["processed_metadata"] = processed_metadata


def divide_data(data_path, save_path):
    """Divide dataset into train / dev / test."""
    with open(os.path.join(save_path, "processed_data.json"), "r") as fin:
        data = json.load(fin)
    data = filter_dialogue(data)

    for data_tag in ("train", "dev", "test"):
        with open(os.path.join(data_path, f"{data_tag}_dial_ids.txt"), "r") as fin:
            dial_ids = list(map(lambda x: x.strip(), fin.readlines()))

        dial_ids = set(dial_ids)
        out_data = dict()
        for dial_id, dial in data.items():
            if dial_id in dial_ids:
                out_data[dial_id] = dial

        out_data_file = os.path.join(save_path, f"{data_tag}_data.json")
        with open(out_data_file, "w") as fout:
            json.dump(out_data, fout, indent=2)
            logger.info(f"Save {data_tag} set to `{out_data_file}`")


def filter_dialogue(data):
    """
    Filter dialogues that contain various types of text and annotation errors.
    Similar to the `analyze_dialogue` in https://github.com/budzianowski/multiwoz/blob/master/create_delex_data.py
    """
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)

    out_data = dict()
    for dial_id, dial in data.items():
        if len(dial["log"]) % 2 != 0:
            logger.warning(f"Odd # of turns: {dial_id}")
            continue
        for turn_idx, turn in enumerate(dial["log"]):
            if len(turn["text"].split()) > 60:
                logger.warning(f"Utterance is too long: {dial_id}-{turn_idx}")
                break
            if not is_ascii(turn["text"]):
                logger.warning(f"Utterance is not ascii: {dial_id}-{turn_idx}")
                break
        else:
            out_data[dial_id] = dial
    return out_data


if __name__ == "__main__":
    args = setup_args()
    main(args)
