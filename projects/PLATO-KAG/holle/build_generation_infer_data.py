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
"""Build inference data for generation."""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--doc_file", type=str, required=True)
    parser.add_argument("--desc_file", type=str, required=True)
    parser.add_argument("--multi_ref_file", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--do_lower", action="store_true")
    args = parser.parse_args()
    return args

def recall_at(targets, scores, k):
    """Calculation for recall at k."""
    for target in targets:
        if target in scores[:k]:
            return 1.0
    return 0.0


def main(args):
    """Main function."""
    src = np.load(os.path.join(args.input_folder, "src_emb.npy"))
    cands = np.load(os.path.join(args.input_folder, "doc_emb.npy"))
    inner_product = np.dot(src, cands.T)

    item_knowledge = json.load(open(args.desc_file))
    num_samples = len(item_knowledge)
    cands_list = []
    with open(args.doc_file) as cands_f:
        while True:
            try:
                cands_list.append(next(cands_f).strip())
            except:
                print("finish")
                break
    cands_list = cands_list[1:]

    cnt_recall_1 = 0
    cnt_multi_recall_1 = 0
    selected_k = []

    for i in range(num_samples):
        candidates = item_knowledge[i]["candidates"]
        multi_gts = item_knowledge[i]["multi_gts"]
        scores = []
        for j in candidates:
            try:
                score = inner_product[i][j]
                scores.append(float(score))
            except:
                print(score)

        scores -= np.max(scores)
        exp_s = np.exp(scores)
        scores = exp_s / np.sum(exp_s)
        target = scores[0]

        multi_targets = []
        for j in range(len(multi_gts)):
            multi_targets.append(scores[candidates.index(multi_gts[j])])

        index = np.argsort(-scores)
        scores = scores[index]

        recall_1 = recall_at(multi_targets, scores, 1)
        cnt_multi_recall_1 += recall_1

        recall_1 = recall_at([target], scores, 1)
        cnt_recall_1 += recall_1

        selected_k.append(cands_list[candidates[index[0]]])

    print("\nSingle Reference:")
    print(f"Recall@1: {cnt_recall_1/num_samples :.3f}")
    print("\nMultiple References:")
    print(f"Recall@1: {cnt_multi_recall_1/num_samples :.3f}")
    with open(os.path.join(args.output_folder, "selection_metric.txt"), "w") as out_f:
        out_f.write(f"Single Reference:\n")
        out_f.write(f"Recall@1: {cnt_recall_1/num_samples :.3f}\n\n")
        out_f.write(f"Multiple References:\n")
        out_f.write(f"Recall@1: {cnt_multi_recall_1/num_samples :.3f}\n")

    data = json.load(open(args.data_file))
    k_idx = 0

    # single ref
    with open(os.path.join(args.output_folder, "infer_data.tsv"), "w") as out_f:
        out_f.write(f"src\tknowledge\ttgt\n")
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

                if len(context_by_now) == 0:
                    dialog = ""
                else:
                    dialog = " [SEP] ".join(context_by_now)

                response = current_item["labels"][0]

                if args.do_lower:
                    response = response.lower()

                out_f.write(f"{dialog}\t{selected_k[k_idx]}\t{response}\1{0}\n")
                k_idx += 1
                context_by_now.append(f"{response}\x01{0}")

    print("total_examples:", k_idx)

    # multi ref
    k_idx = 0
    cnt_ref = 0
    multi_responses = json.load(open(args.multi_ref_file))
    with open(os.path.join(args.output_folder, "multi_ref_infer_data.tsv"), "w") as out_f:
        out_f.write(f"src\tknowledge\ttgt\n")
        for episode_idx, raw_episode in tqdm(enumerate(data)):
            multi_cnt = 0
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

                if len(context_by_now) == 0:
                    dialog = ""
                else:
                    dialog = " [SEP] ".join(context_by_now)

                response = current_item["labels"][0].replace("\t", " ").replace("\n", " ")
                if args.do_lower:
                    response = response.lower()

                cnt_ref += 1
                cur_k = selected_k[k_idx].replace("\t", " ").replace("\n", " ")
                out_f.write(f"{dialog}\t{cur_k}\t{response}\1{0}\n")

                if multi_cnt < len(raw_episode):
                    if f"ts_{episode_idx}_{multi_cnt}" in multi_responses.keys():
                        multi_response_id = f"ts_{episode_idx}_{multi_cnt}"
                        for multi_idx in range(len(multi_responses[multi_response_id]["responses"])):
                            cur_response = multi_responses[multi_response_id]["responses"][multi_idx].replace("\t", " ").replace("\n", " ")
                            if args.do_lower:
                                cur_response = cur_response.lower()
                            if cur_response != response:
                                cnt_ref += 1
                                if not cur_k or not cur_response:
                                    cur_response = " "
                                out_f.write(f"{dialog}\t{cur_k}\t{cur_response}\1{0}\n")
                        multi_cnt += 1

                k_idx += 1

                context_by_now.append(f"{response}\x01{0}")

    print("total_examples:", cnt_ref)


if __name__ == "__main__":
    args = setup_args()
    main(args)