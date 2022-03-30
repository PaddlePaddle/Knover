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
"""Evaluate multi-reference generation."""

import argparse
from collections import Counter
import json
import math
import random
import re

import numpy as np
from tqdm import tqdm

re_art = re.compile(r"\b(a|an|the)\b")
re_punc = re.compile(r"[!\"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']")

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_output_file", type=str, required=True)
    parser.add_argument("--gen_output_file", type=str, required=True)
    parser.add_argument("--infer_input_file", type=str, required=True)
    args = parser.parse_args()
    return args

# The four following functions are adapted from Meta ParlAI:
# https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/metrics.py
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return re_punc.sub(" ", text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return scores[0]


def f1_metric(hypothesis, references):
    """calculate f1 metric"""
    pre = []
    re = []
    f1 = []
    for hyp, ref in zip(hypothesis, references):
        (_pre, _re, _f1) = _f1_score(hyp, [ref])
        f1.append(_f1)
        pre.append(_pre)
        re.append(_re)
    return np.mean(pre), np.mean(re), np.mean(f1)


def main(args):
    """Main function."""
    pred_items = []
    prev_src = None
    cur_item = None
    with open(args.eval_output_file, "r") as eval_f, open(args.gen_output_file, "r") as gen_f, open(args.infer_input_file, "r") as input_f:
        header = next(input_f)
        while True:
            try:
                src, selected_k, tgt = next(input_f).strip().split("\t")
                # new line
                if src != prev_src:
                    if cur_item:
                        pred_items.append(cur_item)
                    prev_src = src

                    cur_item = dict()
                    cur_item["selected_k"] = selected_k

                    pred = next(gen_f).strip()
                    cur_item["pred"] = pred

                    eval_output = next(eval_f).strip().split("\t")
                    if len(eval_output) == 2:
                        eval_output += [""]
                    token_num, token_loss, tgt = eval_output
                    cur_item["labels"] = [tgt]
                    cur_item["token_num"] = [int(token_num)]
                    cur_item["token_loss"] = [float(token_loss)]
                else:
                    eval_output = next(eval_f).strip().split("\t")
                    if len(eval_output) == 2:
                        eval_output += [""]
                    token_num, token_loss, tgt = eval_output
                    cur_item["labels"].append(tgt)
                    cur_item["token_num"].append(int(token_num))
                    cur_item["token_loss"].append(float(token_loss))
            except Exception as e:
                print("finish")
                break

    pred_items.append(cur_item)
    num_sample = len(pred_items)
    print(num_sample)

    total_loss = 0
    total_num = 0
    f1_sum = 0
    kf1_sum = 0

    for item in pred_items:
        item_min_idx = item["token_loss"].index(min(item["token_loss"]))
        total_loss += item["token_num"][item_min_idx] * item["token_loss"][item_min_idx]
        total_num += item["token_num"][item_min_idx]

        item_f1_list = []
        for gt in item["labels"]:
            _, _, f1 = f1_metric([item["pred"]], [gt])
            item_f1_list.append(f1)
        max_f1 = max(item_f1_list)
        f1_sum += max_f1

        kf1_sum += f1_metric([item["pred"]], [item["selected_k"]])[-1]

    multi_gen_loss = total_loss / total_num
    print(f"\nPPL: {math.exp(multi_gen_loss):.3f}")
    print(f"Unigram F1: {f1_sum/num_sample:.3f}")
    print(f"Knowledge F1: {kf1_sum/num_sample:.3f}")

if __name__ == "__main__":
    args = setup_args()
    main(args)