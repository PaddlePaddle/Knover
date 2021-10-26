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
"""DST utils."""

from collections import defaultdict
import logging


def get_logger(name, level=10):
    """Get logger."""
    logger = logging.getLogger(name)
    logger.propagate = 0
    logger.setLevel(level)
    header = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s %(filename)s[%(lineno)d]: %(message)s")
    header.setFormatter(formatter)
    logger.addHandler(header)
    return logger


def get_schema(dataset):
    """Get schema for a dataset."""
    if dataset == "multiwoz":
        schema = {
            "attraction": {
                "book": [],
                "semi": ["area", "name", "type"]
            },
            "hotel": {
                "book": ["day", "people", "stay"],
                "semi": ["area", "internet", "name", "parking", "pricerange", "stars", "type"]
            },
            "restaurant": {
                "book": ["day", "people", "time"],
                "semi": ["area", "food", "name", "pricerange"]
            },
            "taxi": {
                "book": [],
                "semi": ["arriveby", "departure", "destination", "leaveat"]
            },
            "train": {
                "book": ["people"],
                "semi": ["arriveby", "day", "departure", "destination", "leaveat"]
            }
        }
    elif dataset == "woz":
        schema = {
            "restaurant": {
                "book": [],
                "semi": ["area", "food", "pricerange"]
            }
        }
    else:
        raise ValueError(f"Unknown dataset: `{dataset}`")
    return schema


def flatten_ds(ds_dict, schema):
    """
    Flatten dialogue state from dict to sequence.

    Args:
        ds_dict(dcit): The dialogue state dict.
        schema(dict): The schema of the current dataset.

    Returns:
        ds_seq(list): The sequence of dialogue state after flattening.
    """
    ds_seq = []
    for dom in schema:
        for slot_type in schema[dom]:
            for slot in schema[dom][slot_type]:
                slot_tag = f"<{dom}-{slot_type}-{slot}>"
                vals = ds_dict.get(dom, {}).get(slot_type, {}).get(slot, ["not mentioned"])
                out_vals = []
                for val in vals:
                    if val in ["dont care", "don't care", "do n't care", "do nt care", "dontcare"]:
                        val = "<dc>"
                    elif val in ["not mentioned", "none", ""]:
                        val = "<nm>"
                    out_vals.append(val)

                ds_seq.extend([slot_tag, out_vals[0]])
    ds_seq = list(map(lambda x: x.lower(), ds_seq))
    return ds_seq


def parse_ds(ds_seq, schema, remove_nm=True, convert_specials=True):
    """
    Parse dialogue state from sequence to dict.

    Args:
        ds_seq(str): The sequence of dialogue state.
        schema(dict): Domains and slots of the current dataset.
        remove_nm(bool): If true, remove the `<nm>` slot value.
        convert_specials(bool): If true, convert special tokens `<nm>` and `<dc>` to `not mentioned` and `dontcare`.

    Returns:
        ds_dict(defaultdict(dict)): The DS dict after parsing.
    """
    ds_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    ds_seq = ds_seq.strip().split()
    cur_dom, cur_slot_type, cur_slot, cur_val = "", "", "", []
    for token in ds_seq:
        if token in ("<ds/>", "</ds>"):
            continue
        if token[0] == "<" and token[-1] == ">":
            if token.count("-") == 2:
                # dom-slot_type-slot special token
                # Save previous slot and values
                if cur_dom and cur_slot_type and cur_slot and cur_val:
                    ds_dict[cur_dom][cur_slot_type][cur_slot] = " ".join(cur_val)
                cur_dom, cur_slot_type, cur_slot = token[1:-1].split("-")
                cur_val = []
            elif token in ("<nm>", "<dc>"):
                cur_val.append(token)
        else:
            cur_val.append(token)

    if all([cur_dom, cur_slot_type, cur_slot, cur_val]):
        ds_dict[cur_dom][cur_slot_type][cur_slot] = " ".join(cur_val)

    out_ds_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for dom in ds_dict:
        for slot_type in ds_dict[dom]:
            for slot, slot_val in ds_dict[dom][slot_type].items():
                if remove_nm and slot_val == "<nm>":
                    continue
                if convert_specials:
                    if slot_val == "<nm>":
                        slot_val = "not mentioned"
                    elif slot_val == "<dc>":
                        slot_val = "dontcare"
                out_ds_dict[dom][slot_type][slot] = [slot_val]

    return out_ds_dict
