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
"""Build inference data for selection."""

import argparse
import json
import random
from tqdm import tqdm

TOKEN_NOCHOSEN = "no_passages_used"

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_src_file", type=str, required=True)
    parser.add_argument("--out_doc_file", type=str, required=True)
    parser.add_argument("--out_desc_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--do_lower", action="store_true")
    args = parser.parse_args()
    return args

# The three following functions are adapted from Meta ParlAI:
# https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/wizard_of_wikipedia/agents.py
def first_val(dictionary):
    """Get the first value."""
    vals = list(dictionary.values())
    if len(vals) > 0:
        return vals[0]
    return ""

def first_key(dictionary):
    """Get the first key."""
    keys = list(dictionary.keys())
    if len(keys) > 0:
        return keys[0]
    return ""

def get_chosen_title_and_sent(wizard_entry, k_dict):
    """
    Return a nicely extracted title and chosen sentence.
    :return: pair (title, sentence)
    """
    title_dict = wizard_entry.get("checked_passage", "none")
    sentence_dict = wizard_entry.get("checked_sentence", {})
    title = None
    sentence = None
    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ""
            # cand_title1 is the title from the `checked_passage`
            cand_title1 = first_val(title_dict) if title_dict else ""
            # cand_title2 is the extracted title of the passage from the
            #   sentence dict, which is e.g. `self_Vermont_Syrup_0`
            cand_title2 = " ".join(first_key(sentence_dict).split("_")[1:-1])
            if (
                cand_title1
                and cand_title1 in k_dict
                and sentence in k_dict[cand_title1]
            ):
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:  # neither candidate title is the right one
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence

def convert_to_tgt(title, sentence):
    """Get formated knowledge."""
    field_names = ["[TITLE]", "[BODY]"]
    field_values = [title, sentence]

    if args.do_lower:
        field_values = [x.lower() for x in field_values]

    fields = [name + " " + value for name, value in zip(field_names, field_values)]
    return " ".join(fields)

def main(args):
    """Main function."""
    data = json.load(open(args.data_file))
    print("total diglogs:", len(data))

    total_turn = 0
    total_example = 0
    out_knowledge = {}
    knowledge_desc = []

    out_knowledge[TOKEN_NOCHOSEN] = [TOKEN_NOCHOSEN]

    # save src
    with open(args.out_src_file, "w") as out_src_f:
        out_src_f.write(f"topic\tsrc\n")
        for d in tqdm(data):
            count_turn = 0
            wizard_first = "Wizard" in d["dialog"][0]["speaker"]
            chosen_topic = d.get("chosen_topic", "")
            chosen_topic_passages = d["chosen_topic_passage"]

            context_by_now = []
            for idx in range(len(d["dialog"])):
                is_wizard_turn = "Wizard" in d["dialog"][idx]["speaker"]
                total_turn += 1
                # current is wizard
                if is_wizard_turn:
                    add = True
                    wizard_entry = d["dialog"][idx]

                    # first, get knowledge
                    apprentice_ret_passages = wizard_ret_passages = {}
                    if not wizard_first or idx != 0:
                        apprentice_entry = d["dialog"][idx - 1]
                        apprentice_ret_passages = apprentice_entry["retrieved_passages"]

                    if idx - 2 >= 0:
                        wizard_prev_entry = d["dialog"][idx - 2]
                        wizard_ret_passages = wizard_prev_entry["retrieved_passages"]

                    knowledge_dict = {chosen_topic: chosen_topic_passages}
                    for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
                        for passage in ret_passes:
                            for k, v in passage.items():
                                if k not in knowledge_dict.keys():
                                    knowledge_dict[k] = v

                    # gt
                    title, sentence = get_chosen_title_and_sent(wizard_entry, knowledge_dict)

                    if title not in out_knowledge.keys():
                        out_knowledge[title] = []
                    if sentence not in out_knowledge[title]:
                        out_knowledge[title].append(sentence)

                    for title, passages in knowledge_dict.items():
                        if title not in out_knowledge:
                            out_knowledge[title] = []
                        for p in passages:
                            if p not in out_knowledge[title]:
                                out_knowledge[title].append(p)

                    if len(context_by_now) == 0:
                        dialog = ""
                    else:
                        dialog = " [SEP] ".join(context_by_now)

                    topic = chosen_topic
                    if args.do_lower:
                        topic = chosen_topic.lower()

                    total_example += 1
                    out_src_f.write(f"{topic}\t{dialog}\n")

                current_turn = d["dialog"][idx]["text"].replace("\t", " ").replace("\n", " ")
                if args.do_lower:
                    current_turn = current_turn.lower()
                context_by_now.append(f"{current_turn}\x01{0 if is_wizard_turn else 1}")

    print("total_turns:", total_turn)
    print("total_example:", total_example)

    # save tgt
    all_knowledges = []
    with open(args.out_doc_file, "w") as out_cands_f:
        out_cands_f.write(f"tgt\n")
        for title in out_knowledge.keys():
            for p in out_knowledge[title]:
                item = convert_to_tgt(title, p)
                all_knowledges.append(item)
                out_cands_f.write(f"{item}\n")
    print("total k =", len(all_knowledges))

    # get item index
    for d in tqdm(data):
        wizard_first = "Wizard" in d["dialog"][0]["speaker"]
        chosen_topic = d.get("chosen_topic", "")
        chosen_topic_passages = d["chosen_topic_passage"]

        for idx in range(len(d["dialog"])):
            is_wizard_turn = "Wizard" in d["dialog"][idx]["speaker"]
            # current is wizard
            if is_wizard_turn:
                wizard_entry = d["dialog"][idx]

                # first, get knowledge
                apprentice_ret_passages = wizard_ret_passages = {}
                if not wizard_first or idx != 0:
                    apprentice_entry = d["dialog"][idx - 1]
                    apprentice_ret_passages = apprentice_entry["retrieved_passages"]

                if idx - 2 >= 0:
                    wizard_prev_entry = d["dialog"][idx - 2]
                    wizard_ret_passages = wizard_prev_entry["retrieved_passages"]

                knowledge_dict = {chosen_topic: chosen_topic_passages}
                for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
                    for passage in ret_passes:
                        for k, v in passage.items():
                            if k not in knowledge_dict.keys():
                                knowledge_dict[k] = v

                # gt
                title, sentence = get_chosen_title_and_sent(wizard_entry, knowledge_dict)
                gt = convert_to_tgt(title, sentence)

                knowledges = []
                for title, passages in knowledge_dict.items():
                    for p in passages:
                        knowledges.append(convert_to_tgt(title, p))

                knowledges = [gt] + [convert_to_tgt(TOKEN_NOCHOSEN, TOKEN_NOCHOSEN)] + knowledges
                # remove another gt which is not at the idx 0
                for gt_idx, k in enumerate(knowledges[1:]):
                    if k == gt:
                        break
                else:
                    gt_idx = None
                if gt_idx is not None:
                    del knowledges[gt_idx + 1]

                item = {}
                item["gt"] = all_knowledges.index(gt)
                item["candidates"] = []
                for k in knowledges:
                    item["candidates"].append(all_knowledges.index(k))
                knowledge_desc.append(item)

    json.dump(knowledge_desc, open(args.out_desc_file, "w"), indent=2)


if __name__ == "__main__":
    args = setup_args()
    main(args)