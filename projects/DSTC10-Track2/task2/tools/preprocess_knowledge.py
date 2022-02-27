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
"""Minimal tokenize texts in knowledge.json file."""

import argparse
import copy
import json

from tqdm import tqdm

from minimal_tokenizer import tokenize

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_knowledge", type=str, required=True)
    parser.add_argument("--output_knowledge", type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    """Main function."""
    knowledge = json.load(open(args.input_knowledge))
    input_knowledge = copy.deepcopy(knowledge)
    for domain in tqdm(knowledge):
        for entity_id in knowledge[domain]:
            entity = input_knowledge[domain][entity_id]
            if entity["city"] == "San Francisco":
                entity["name"] = tokenize(entity["name"])
                for doc_id in entity["docs"]:
                    doc = entity["docs"][doc_id]
                    doc["title"] = tokenize(doc["title"])
                    doc["body"] = tokenize(doc["body"])
            else:
                input_knowledge[domain].pop(entity_id)

    print("done!", flush=True)   
    json.dump(input_knowledge, open(args.output_knowledge, "w"), indent=2)

if __name__ == "__main__":
    args = setup_args()
    main(args)
