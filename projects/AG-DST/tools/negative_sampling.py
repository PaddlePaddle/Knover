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
"""Negative sampling of dialogue state."""

import argparse
from collections import defaultdict
import json
import math
import os
import random

from tqdm import tqdm

from utils import get_logger, get_schema


logger = get_logger(__name__)
random.seed(1007)


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="multiwoz", choices=["multiwoz", "woz"])
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    return args


def main(args):
    """Main function."""
    with open(args.data_file, "r") as fin:
        data = json.load(fin)
        logger.info(f"Load dataset from `{args.data_file}`")

    schema = get_schema(args.dataset)
    # NOTE: for WOZ 2.0, `restaurant_db.json` is mocked for convenience
    candidate_vals = get_candidate_vals(args.db_path, schema)

    for dial_id, dial in tqdm(data.items(), desc="Dialogue"):
        for turn_idx, turn in enumerate(dial["log"]):
            if turn_idx % 2 == 0:
                # user turn
                continue
            if turn_idx - 2 >= 0:
                prev_metadata = dial["log"][turn_idx - 2]["processed_metadata"]
            else:
                prev_metadata = None
            cur_metadata = turn["processed_metadata"]
            ds_diff = get_ds_diff(cur_metadata, schema, prev_metadata)
            neg_metadata = neg_sample(ds_diff, cur_metadata, candidate_vals)
            turn["negative_metadata"] = neg_metadata

    with open(args.out_file, "w") as fout:
        json.dump(data, fout, indent=2)
        logger.info(f"Save dataset to `{args.out_file}`")


def get_candidate_vals(db_path, schema):
    """Get candidate slot values"""
    # get semi slot values from DB file
    db_semi_vals = {}
    for dir, _, files in os.walk(db_path):
        for file_name in files:
            if not file_name.endswith("_db.json"):
                continue
            with open(os.path.join(dir, file_name), "r") as fin:
                db_slot_vals = json.load(fin)
                db_slot_vals = [{s.lower(): v for s, v in db_item.items()} for db_item in db_slot_vals]
            dom = file_name.split("_")[0]
            db_semi_vals[dom] = db_slot_vals

    candidate_vals = defaultdict(dict)
    for dom, dom_ds in schema.items():
        for slot_type, slots in dom_ds.items():
            for slot in slots:
                if slot_type == "semi":
                    # semi
                    if dom == "taxi":
                        if slot in ("arriveby", "leaveat"):
                            vals = []
                            for hour in range(6, 24):
                                for minute in range(0, 60, 15):
                                    vals.append(f"{hour:02d}:{minute:02d}")
                        elif slot in ("departure", "destination"):
                            vals = set()
                            for d in ("attraction", "hotel", "restaurant"):
                                for db_item in db_semi_vals.get(d, []):
                                    vals.add(db_item["name"])
                            vals = list(vals)
                    else:
                        vals = [db_item[slot] for db_item in db_semi_vals[dom]]
                        vals = list(set(vals))
                else:
                    # book
                    if slot == "day":
                        vals = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                    elif slot in ("people", "stay"):
                        vals = ["1", "2", "3", "4", "5", "6", "7", "8"]
                    elif slot == "time":
                        vals = []
                        for hour in range(6, 24):
                            for minute in range(0, 60, 15):
                                vals.append(f"{hour:02d}:{minute:02d}")
                candidate_vals[dom][slot] = vals

    return candidate_vals


def get_ds_diff(cur_metadata, schema, prev_metadata=None):
    """Get the difference from previous metadata to current metadata."""
    if prev_metadata is None:
        prev_metadata = {}

    val_map = {
        "don't care": "dont care",
        "do n't care": "dont care",
        "do nt care": "dont care",
        "dontcare": "dont care",
        "none": "not mentioned",
        "": "not mentioned"
    }
    ds_diff = defaultdict(lambda: defaultdict(dict))
    for dom, dom_ds in schema.items():
        for slot_type, slots in dom_ds.items():
            for slot in slots:
                prev_vals = prev_metadata.get(dom, {}).get(slot_type, {}).get(slot, [])
                cur_vals = cur_metadata.get(dom, {}).get(slot_type, {}).get(slot, [])
                prev_val = val_map.get(prev_vals[0], prev_vals[0]) if len(prev_vals) else "not mentioned"
                cur_val = val_map.get(cur_vals[0], cur_vals[0]) if len(cur_vals) else "not mentioned"

                if prev_val != cur_val:
                    ds_diff[dom][slot_type][slot] = [cur_val]

    return ds_diff


def neg_sample(ds_diff, cur_metadata, candidate_vals):
    """Negative sampling for dialogue state."""
    neg_ds = defaultdict(lambda: defaultdict(dict))
    for dom, dom_ds_diff in ds_diff.items():
        diff_slots = list(dom_ds_diff["book"].keys()) + list(dom_ds_diff["semi"].keys())
        if "booked" in diff_slots:
            diff_slots.remove("booked")
        cnt_neg_slots = int(math.sqrt(len(diff_slots)))
        neg_slots = random.sample(diff_slots, cnt_neg_slots)

        for slot_type, slot_vals in cur_metadata[dom].items():
            for slot, vals in slot_vals.items():
                if slot not in neg_slots:
                    continue
                val = vals[0] if len(vals) > 0 else "not mentioned"

                act = ""
                # exchange confusing slots
                if slot == "leaveat" and len(slot_vals["arriveby"]) > 0 and slot_vals["arriveby"][0] != val:
                    act = random.choices(["exchange", ""], [0.6, 0.4], k=1)[0]
                    exchange_vals = slot_vals["arriveby"]
                elif slot == "arriveby" and len(slot_vals["leaveat"]) > 0 and slot_vals["leaveat"][0] != val:
                    act = random.choices(["exchange", ""], [0.6, 0.4], k=1)[0]
                    exchange_vals = slot_vals["leaveat"]
                elif slot == "departure" and len(slot_vals["destination"]) > 0 and slot_vals["destination"][0] != val:
                    act = random.choices(["exchange", ""], [0.6, 0.4], k=1)[0]
                    exchange_vals = slot_vals["destination"]
                elif slot == "destination" and len(slot_vals["departure"]) > 0 and slot_vals["departure"][0] != val:
                    act = random.choices(["exchange", ""], [0.6, 0.4], k=1)[0]
                    exchange_vals = slot_vals["departure"]
                elif slot == "area":
                    for other_dom in ["hotel", "restaurant", "attraction"]:
                        if other_dom not in cur_metadata:
                            continue
                        if len(cur_metadata[other_dom]["semi"]["area"]) > 0 and \
                            cur_metadata[other_dom]["semi"]["area"][0] != val:
                            act = random.choices(["exchange", ""], [0.6, 0.4], k=1)[0]
                            exchange_vals = cur_metadata[other_dom]["semi"]["area"]

                if not act:
                    # random replacement
                    if val in ["not mentioned", "none", ""]:
                        act = random.choices(["dc", "wrong"], [0.7, 0.3], k=1)[0]
                    elif val in ["dont care", "don't care", "do n't care", "do nt care", "dontcare"]:
                        act = random.choices(["nm", "wrong"], [0.8, 0.2], k=1)[0]
                    else:
                        act = random.choices(["nm", "dc", "wrong"], [0.6, 0.2, 0.2], k=1)[0]

                if act == "nm":
                    neg_ds[dom][slot_type][slot] = []
                elif act == "dc":
                    neg_ds[dom][slot_type][slot] = ["dont care"]
                elif act == "wrong":
                    wrong_val = val
                    sample_cnt = 0
                    while wrong_val == val or sample_cnt < 5:
                        wrong_val = random.choice(candidate_vals[dom][slot])
                        sample_cnt += 1
                    if wrong_val == val:
                        neg_ds[dom][slot_type][slot] = []
                    else:
                        neg_ds[dom][slot_type][slot] = [wrong_val]
                elif act == "exchange":
                    neg_ds[dom][slot_type][slot] = exchange_vals

    # fill other slots
    neg_metadata = defaultdict(lambda: defaultdict(dict))
    for dom, dom_ds in cur_metadata.items():
        for slot_type, slot_vals in dom_ds.items():
            for slot, vals in slot_vals.items():
                neg_vals = neg_ds.get(dom, {}).get(slot_type, {}).get(slot)
                if neg_vals is not None:
                    neg_metadata[dom][slot_type][slot] = neg_vals
                else:
                    neg_metadata[dom][slot_type][slot] = vals

    return neg_metadata


if __name__ == "__main__":
    args = setup_args()
    main(args)
