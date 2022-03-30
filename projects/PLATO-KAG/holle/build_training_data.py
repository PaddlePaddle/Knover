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
"""Build training data."""

import argparse
import json
from tqdm import tqdm
import random

TOKEN_NOCHOSEN = "no_passages_used"


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_knowledge", type=int, default=32)
    parser.add_argument("--do_lower", action="store_true")
    args = parser.parse_args()
    return args

def convert_to_tgt(title, sentence):
    """Get formated knowledge."""
    field_names = ["[TITLE]", "[BODY]"]
    field_values = [title, sentence]

    if args.do_lower:
        field_values = [x.lower() for x in field_values]

    fields = [name + " " + value for name, value in zip(field_names, field_values)]
    return " ".join(fields)

def sample_knowledge(knowledge_dict):
    """Sample knowledge."""
    sample_list = []
    while len(sample_list) == 0:
        title = random.choice(list(knowledge_dict.keys()))
        sample_list = list(knowledge_dict[title])
    sentence = random.choice(sample_list)
    return convert_to_tgt(title, sentence)

def main(args):
    """Main function."""
    total_example = 0
    data = json.load(open(args.data_file))
    print("total dialogs:", len(data))

    null_doc = convert_to_tgt(TOKEN_NOCHOSEN, TOKEN_NOCHOSEN)

    with open(args.out_file, "w") as out_f:
        out_f.write(f"src\tknowledge\ttgt\n")
        for _ in range(args.num_epochs):
            for episode_idx, raw_episode in tqdm(enumerate(data)):
                chosen_topic = raw_episode[0]["title"]
                if args.do_lower:
                    chosen_topic = chosen_topic.lower()
                context_by_now = []
                # check
                skip = False
                for item in raw_episode:
                    if len(item["text"]) == 0 or len(item["labels"][0]) == 0:
                        skip = True
                if skip:
                    continue
                for example_idx in range(len(raw_episode)):
                    current_item = raw_episode[example_idx]
                    current_turn = current_item["text"].replace("\t", " ").replace("\n", " ")

                    if args.do_lower:
                        current_turn = current_turn.lower()
                    context_by_now.append(f"{current_turn}\x01{1}")

                    knowledge_sentences = current_item["knowledge"]

                    checked_sentence = knowledge_sentences[0]
                    title = TOKEN_NOCHOSEN if checked_sentence == TOKEN_NOCHOSEN else chosen_topic
                    tgt = convert_to_tgt(title, checked_sentence)

                    knowledge_dict = {TOKEN_NOCHOSEN:[TOKEN_NOCHOSEN], chosen_topic:[]}

                    for k in knowledge_sentences:
                        if k not in knowledge_dict[chosen_topic]:
                            knowledge_dict[chosen_topic].append(k)

                    knowledges = []
                    for title, passages in knowledge_dict.items():
                        for p in passages:
                            knowledges.append(convert_to_tgt(title, p))

                    for gt_idx, k in enumerate(knowledges):
                        if k == tgt:
                            break
                    else:
                        gt_idx = None
                    if gt_idx is not None:
                        del knowledges[gt_idx]

                    if args.max_knowledge > 0:
                        random.shuffle(knowledges)
                        knowledges = knowledges[:args.max_knowledge - 1]

                        if null_doc != tgt and null_doc not in knowledges:
                            knowledges = [null_doc] + knowledges[:-1]

                    knowledges = [tgt] + knowledges

                    all_knowledge = " [SEP] ".join(knowledges)

                    if len(context_by_now) == 0:
                        dialog = ""
                    else:
                        dialog = " [SEP] ".join(context_by_now)

                    response = current_item["labels"][0]

                    if args.do_lower:
                        response = response.lower()

                    out_f.write(f"{dialog}\t{all_knowledge}\t{response}\1{0}\n")
                    context_by_now.append(f"{response}\x01{0}")
                    total_example += 1

    print("total_examples:", total_example)

if __name__ == "__main__":
    args = setup_args()
    main(args)