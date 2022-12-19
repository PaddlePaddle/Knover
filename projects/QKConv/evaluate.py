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

"""Evaluate generated response."""

import argparse
import json

from utils import DATASET_ZOO


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["smd", "qrecc", "wow"])
    parser.add_argument("--pred_file", type=str, required=True, help="Prediction")
    parser.add_argument("--entity_file", type=str, help="Entity file for smd dataset", default=None)
    parser.add_argument("--save_file", type=str, required=True)
    args = parser.parse_args()
    return args


def evaluate(args):
    """Main evaluation function."""
    dataset = DATASET_ZOO[args.dataset](args.pred_file, entity_file=args.entity_file)
    
    preds = []
    refs = []
    for dial in dataset.data:
        if dial["response"] != "":
            preds.append(dial["generated_response"])
            refs.append(dial["response"])
    assert len(preds) == len(refs), f"{len(preds)} != {len(refs)}"

    results = {
        "dataset": args.dataset,
        "file": args.pred_file,
        "metrics": dataset.evaluate(preds, refs)
    }

    print(json.dumps(results, indent=2))
    with open(args.save_file, "a") as fout:
        json.dump(results, fout, indent=2)
    return


if __name__ == "__main__":
    args = setup_args()
    evaluate(args)