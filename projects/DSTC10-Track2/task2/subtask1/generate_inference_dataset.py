#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Convert json format logs -> task1 dataset."""

import argparse
import json
import re

from tqdm import tqdm

re_punc = re.compile(r"[!\"()+,-./;<=>?@\\^`{|}~]")


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--label_file", type=str, default=None)
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    logs = json.load(open(args.log_file))
    if args.label_file is not None:
        labels = json.load(open(args.label_file))
    else:
        labels = [None for _ in logs]

    with open(args.out_file, "w") as out_f:
        if labels[0] is None:
            out_f.write("src\n")
        else:
            out_f.write("src\tlabel\n")
        for log, label in zip(tqdm(logs, desc="Generate task1 input dataset with context"), labels):
            dialog = []
            for turn in log:
                turn_text = turn["text"].replace("\t", " ").replace("\n", " ").lower()
                turn_text = re_punc.sub(" ", turn_text)
                dialog.append(f"{turn_text}\x01{1 if turn['speaker'] == 'U' else 0}")
            dialog_str = " [SEP] ".join(dialog)

            dialog_str = dialog_str.replace("[ent_start]", "[ENT_START]")
            dialog_str = dialog_str.replace("[ent_end]", "[ENT_END]")
            dialog_str = dialog_str.replace("[loc_start]", "[LOC_START]")
            dialog_str = dialog_str.replace("[loc_end]", "[LOC_END]")
            
            if label is None:
                out_f.write(f"{dialog_str}\n")
            else:
                out_f.write(f"{dialog_str}\t{1 if label['target'] else 0}\n")


if __name__ == "__main__":
    args = setup_args()
    main(args)
