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
"""Minimal tokenize utterances in log.json file."""

import argparse
import json

from tqdm import tqdm

from minimal_tokenizer import tokenize

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_logs", type=str, required=True)
    parser.add_argument("--output_logs", type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    """Main function."""
    input_logs = json.load(open(args.input_logs))
    
    for log in tqdm(input_logs):
        for turn in log:
            ori_src = turn["text"]
            turn["text"] = tokenize(ori_src.lower().replace("\t", " ").replace("\n", " "))

    print("done!", flush=True)   
    json.dump(input_logs, open(args.output_logs, "w"), indent=2)

if __name__ == "__main__":
    args = setup_args()
    main(args)
