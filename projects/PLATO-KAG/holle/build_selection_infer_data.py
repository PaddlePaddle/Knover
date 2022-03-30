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
    parser.add_argument("--multi_ref_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--do_lower", action="store_true")
    args = parser.parse_args()
    return args

# The two following functions are adapted from SKT:
# https://github.com/bckim92/sequential-knowledge-transformer/blob/master/data/holle.py
def _f1_score(true_set, pred_set, eps=1e-12):
    """Get F1 score."""
    precision = len(true_set.intersection(pred_set)) / (float(len(pred_set)) + eps)
    recall = len(true_set.intersection(pred_set)) / (float(len(true_set)) + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    return f1_score

def _get_best_match_idx(gt_span, label_candidates, response):
    """Get best match idx."""
    gt_span_words = set(gt_span.split())
    response_words = set(response.split())
    label_words_candidates = [
        set(x.split()) for x in label_candidates
    ]

    f1_scores = []
    for label_words_candidate in label_words_candidates:
        f1_scores.append(_f1_score(gt_span_words, label_words_candidate))

    if sum(f1_scores) == 0.0:
        f1_scores = []
        for label_words_candidate in label_words_candidates:
            f1_scores.append(_f1_score(response_words, label_words_candidate))

    max_idx = f1_scores.index(max(f1_scores))

    return max_idx

def convert_to_knowledge(title, sentence):
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
    print("total dialogs:", len(data))

    total_turn = 0
    total_example = 0
    out_knowledge = {}
    knowledge_desc = []

    out_knowledge[TOKEN_NOCHOSEN] = [TOKEN_NOCHOSEN]

    # save src
    with open(args.out_src_file, "w") as out_src_f:
        out_src_f.write(f"src\n")
        for episode_idx, raw_episode in tqdm(enumerate(data)):
            chosen_topic = raw_episode[0]["title"]
            if args.do_lower:
                chosen_topic = chosen_topic.lower()
            context_by_now = []

            for example_idx in range(len(raw_episode)):
                current_item = raw_episode[example_idx]
                current_turn = current_item["text"].replace("\t", " ").replace("\n", " ")

                if args.do_lower:
                    current_turn = current_turn.lower()
                context_by_now.append(f"{current_turn}\x01{1}")

                knowledge_sentences = current_item["knowledge"]

                checked_sentence = knowledge_sentences[0]
                title = TOKEN_NOCHOSEN if checked_sentence == TOKEN_NOCHOSEN else chosen_topic
                tgt = convert_to_knowledge(title, checked_sentence)

                knowledge_dict = {TOKEN_NOCHOSEN:[TOKEN_NOCHOSEN], chosen_topic:[]}

                for k in knowledge_sentences:
                    if k not in knowledge_dict[chosen_topic]:
                        knowledge_dict[chosen_topic].append(k)

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

                response = current_item["labels"][0]

                if args.do_lower:
                    response = response.lower()

                out_src_f.write(f"{dialog}\n")
                context_by_now.append(f"{response}\x01{0}")
                total_example += 1

    print("total_examples:", total_example)

    # save tgt
    all_knowledges = []
    with open(args.out_doc_file, "w") as out_cands_f:
        out_cands_f.write(f"tgt\n")
        for title in out_knowledge.keys():
            for p in out_knowledge[title]:
                item = convert_to_knowledge(title, p)
                all_knowledges.append(item)
                out_cands_f.write(f"{item}\n")
    print("total k =", len(all_knowledges))

    multi_responses = json.load(open(args.multi_ref_file))
    # get item index
    for episode_idx, raw_episode in tqdm(enumerate(data)):
        multi_cnt = 0
        chosen_topic = raw_episode[0]["title"]
        context_by_now = []
        for example_idx in range(len(raw_episode)):
            current_item = raw_episode[example_idx]
            knowledge_sentences = current_item["knowledge"]

            checked_sentence = knowledge_sentences[0]
            title = TOKEN_NOCHOSEN if checked_sentence == TOKEN_NOCHOSEN else chosen_topic
            tgt = convert_to_knowledge(title, checked_sentence)

            knowledge_dict = {TOKEN_NOCHOSEN:[TOKEN_NOCHOSEN], chosen_topic:[]}

            for k in knowledge_sentences:
                if k != TOKEN_NOCHOSEN and k not in knowledge_dict[chosen_topic]:
                    knowledge_dict[chosen_topic].append(k)

            knowledges = []
            for title, passages in knowledge_dict.items():
                for p in passages:
                    knowledges.append(convert_to_knowledge(title, p))

            for gt_idx, k in enumerate(knowledges):
                if k == tgt:
                    break
            else:
                gt_idx = None
            if gt_idx is not None:
                del knowledges[gt_idx]

            knowledges = [tgt] + knowledges
            item = {}
            item["gt"] = all_knowledges.index(tgt)
            item["candidates"] = []
            for k in knowledges:
                item["candidates"].append(all_knowledges.index(k))

            item["multi_gts"] = [item["gt"]]

            response = current_item["labels"][0]
            if multi_cnt < len(raw_episode):
                if f"ts_{episode_idx}_{multi_cnt}" in multi_responses.keys():
                    multi_response_id = f"ts_{episode_idx}_{multi_cnt}"
                    for multi_idx in range(len(multi_responses[multi_response_id]["responses"])):
                        raw_multi_response = multi_responses[multi_response_id]["responses"][multi_idx]
                        raw_multi_span = multi_responses[multi_response_id]["spans"][multi_idx]

                        if raw_multi_response != response:
                            multi_response = raw_multi_response
                            multi_span = raw_multi_span
                            if isinstance(multi_span, int):
                                multi_span = str(multi_span)
                            multi_knowledge_sentences = knowledge_sentences
                            multi_knowledge_idx = _get_best_match_idx(multi_span, multi_knowledge_sentences, multi_response)

                            checked_sentence = knowledge_sentences[multi_knowledge_idx]
                            title = TOKEN_NOCHOSEN if checked_sentence == TOKEN_NOCHOSEN else chosen_topic
                            tgt = convert_to_knowledge(title, checked_sentence)

                            item["multi_gts"].append(all_knowledges.index(tgt))
                    multi_cnt += 1
            knowledge_desc.append(item)

    json.dump(knowledge_desc, open(args.out_desc_file, "w"), indent=2)


if __name__ == "__main__":
    args = setup_args()
    main(args)