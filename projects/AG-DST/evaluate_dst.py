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
"""Main evaluation program for dialogue state tracking."""

import argparse
from collections import defaultdict
import json
import os
import re

from utils import get_logger, get_schema


logger = get_logger(__name__)


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser(description="Main evaluation program for dialogue state tracking.")
    parser.add_argument("--inference_labels", type=str, required=True)
    parser.add_argument("--ground_labels", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="multiwoz", choices=["multiwoz", "woz"])
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    return args


def evaluate_dst(args):
    """Evaluation main function."""
    with open(args.inference_labels, "r") as fin_infer, open(args.ground_labels, "r") as fin_ground:
        infer_labels = json.load(fin_infer)
        ground_labels = json.load(fin_ground)
        assert len(infer_labels) == len(ground_labels), "The length of prediction and ground-trurh must be same!"
        logger.info(f"Load inference labels from `{args.inference_labels}`")
        logger.info(f"Load ground labels from `{args.ground_labels}`")

    metric = Metric(args.dataset)

    for ground_label, infer_label in zip(ground_labels, infer_labels):
        metric.update(ground_label, infer_label)

    # save results
    scores = metric.score()
    score_file = os.path.join(args.save_path, "scores.json")
    with open(score_file, "w") as fout:
        json.dump(scores, fout, indent=2)
        logger.info(f"Save scores to `{score_file}`")
    bad_cases = metric.bad_cases
    bad_case_file = os.path.join(args.save_path, "bad_cases.txt")
    with open(bad_case_file, "w") as fout:
        fout.write("\n".join(bad_cases))
        logger.info(f"Save bad cases to `{bad_case_file}`")


class Metric(object):
    """Metric for evaluating dialogue state tracking."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.schema = get_schema(dataset)
        self.reset()

    def reset(self):
        self._num_samples = 0
        self._num_matched = 0
        self._num_slots = 0
        self._num_matched_slots = 0
        self._slot_status = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))

        self._bad_cases = set()

    def update(self, ground_label, infer_label):
        joint_flag = True

        for dom in self.schema:
            for slot_type, slots in self.schema[dom].items():
                for slot in slots:
                    self._num_slots += 1
                    self._slot_status[dom][slot_type][slot]["num_slots"] += 1

                    if dom in ground_label and slot_type in ground_label[dom] and slot in ground_label[dom][slot_type]:
                        ground_val = set(ground_label[dom][slot_type][slot])
                        ground_val = normalize_slot_val(ground_val, self.dataset)
                    else:
                        ground_val = None

                    if dom in infer_label and slot_type in infer_label[dom] and slot in infer_label[dom][slot_type]:
                        infer_val = infer_label[dom][slot_type][slot][0]
                        infer_val = normalize_slot_val(infer_val, self.dataset)
                    else:
                        infer_val = None

                    if (isinstance(ground_val, set) and infer_val in ground_val) or ground_val == infer_val:
                        self._num_matched_slots += 1
                        self._slot_status[dom][slot_type][slot]["num_matched"] += 1
                    else:
                        # logger.debug(f"{dom}-{slot_type}-{slot}: {ground_val} <==> {infer_val}")
                        self._bad_cases.add(f"{dom}-{slot_type}-{slot}: {ground_val} <==> {infer_val}")
                        joint_flag = False

        if joint_flag:
            self._num_matched += 1
        self._num_samples += 1

    def score(self):
        joint_acc = self._num_matched / self._num_samples
        slot_acc = self._num_matched_slots / self._num_slots

        slot_status = defaultdict(lambda: defaultdict(dict))
        for dom in self.schema:
            for slot_type, slots in self.schema[dom].items():
                for slot in slots:
                    slot_status[dom][slot_type][slot] = \
                        self._slot_status[dom][slot_type][slot]["num_matched"] / \
                            self._slot_status[dom][slot_type][slot]["num_slots"]

        scores = {
            "joint_acc": joint_acc,
            "slot_acc": slot_acc,
            "slot_status": slot_status
        }
        return scores

    @property
    def bad_cases(self):
        return sorted(list(self._bad_cases))


def normalize_slot_val(slot_val, dataset):
    """Normalize slot values."""
    def normalize_val(val):
        # time before 12
        if re.match(r"\d:\d\d", val):
            val = "0" + val
        # inconsistent annotation
        val = val.replace("'", "")
        if val.startswith("the "):
            val = val[4:]
        if val.endswith(" restaurant"):
            val = val[:-11]
        label_token_map = {
            "cafe uno": "caffee uno",
            "hut fen ditton": "hut fenditton",
        }
        for token1, token2 in label_token_map.items():
            val = val.replace(token1, token2)
        if dataset == "woz":
            label_token_map = {
                "bbq": "barbeque",
                "singapore": "singaporean",
                "asian": "asian oriental",
                "modern belgian": "belgian",
                "aussie": "australian",
                "african oriental": "asian oriental",
            }
            for token1, token2 in label_token_map.items():
                val = val.replace(token1, token2)

        return val

    if isinstance(slot_val, set):
        norm_slot_val = set()
        norm_slot_val.update(slot_val)
        for v in slot_val:
            norm_v = normalize_val(v)
            norm_slot_val.add(norm_v)
    elif isinstance(slot_val, str):
        norm_slot_val = normalize_val(slot_val)
    else:
        raise TypeError

    return norm_slot_val


if __name__ == "__main__":
    args = setup_args()
    evaluate_dst(args)
