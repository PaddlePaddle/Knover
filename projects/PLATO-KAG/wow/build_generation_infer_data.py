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
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--do_lower", action="store_true")
    args = parser.parse_args()
    return args

def recall_at(target, scores, k):
    """Calculation for recall at k."""
    if target in scores[:k]:
        return 1.0
    else:
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

    cnt_recall_1 = 0.0
    selected_k = []

    for i in range(num_samples):
        candidates = item_knowledge[i]["candidates"]
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
        index = np.argsort(-scores)
        scores = scores[index]

        recall_1 = recall_at(target, scores, 1)
        cnt_recall_1 += recall_1

        selected_k.append(cands_list[candidates[index[0]]])

    print(f"\nrecall@1 = {cnt_recall_1/num_samples :.3f}")
    with open(os.path.join(args.output_folder, "selection_metric.txt"), "w") as out_f:
        out_f.write(f"recall@1 = {cnt_recall_1/num_samples :.3f}\n")

    data = json.load(open(args.data_file))
    k_idx = 0

    with open(os.path.join(args.output_folder, "infer_data.tsv"), "w") as out_f:
        out_f.write(f"topic\tsrc\tknowledge\ttgt\n")
        for d in tqdm(data):
            wizard_first = "Wizard" in d["dialog"][0]["speaker"]
            chosen_topic = d.get("chosen_topic", "")
            chosen_topic_passages = d["chosen_topic_passage"]

            context_by_now = []
            for idx in range(len(d["dialog"])):
                is_wizard_turn = "Wizard" in d["dialog"][idx]["speaker"]
                # current is wizard
                if is_wizard_turn:
                    if len(context_by_now) == 0:
                        dialog = ""
                    else:
                        dialog = " [SEP] ".join(context_by_now)

                    response = d["dialog"][idx]["text"].replace("\t", " ").replace("\n", " ")
                    if args.do_lower:
                        response = response.lower()
                        chosen_topic = chosen_topic.lower()

                    out_f.write(f"{chosen_topic}\t{dialog}\t{selected_k[k_idx]}\t{response}\1{0}\n")
                    k_idx += 1

                current_turn = d["dialog"][idx]["text"].replace("\t", " ").replace("\n", " ")
                if args.do_lower:
                    current_turn = current_turn.lower()
                context_by_now.append(f"{current_turn}\x01{0 if is_wizard_turn else 1}")

    print("total examples: ", k_idx)


if __name__ == "__main__":
    args = setup_args()
    main(args)