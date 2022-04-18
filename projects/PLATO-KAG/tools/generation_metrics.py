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
"""Automatic metrics for generation task."""

import argparse
from collections import Counter
import re

import numpy as np

re_art = re.compile(r"\b(a|an|the)\b")
re_punc = re.compile(r"[!\"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']")

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--refer_file", type=str, required=True)
    parser.add_argument("--hypo_file", type=str, required=True)
    parser.add_argument("--include_topic", action="store_true")
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
    gt_list = []
    selected_k_list = []
    with open(args.refer_file) as gt_f:
        while True:
            try:
                line = next(gt_f).strip()
                if args.include_topic:
                    topic, src, selected_k, response = line.split("\t")
                else:
                    src, selected_k, response = line.split("\t")
                if len(response.split("\1")) == 2:
                    response, role_id = response.split("\1")
                selected_k_list.append(selected_k)
                gt_list.append(response)
            except Exception as e:
                print("finish loading reference file")
                break
    gt_list = gt_list[1:]
    selected_k_list = selected_k_list[1:]

    gen_list = []
    with open(args.hypo_file) as infer_file:
        while True:
            try:
                gen_list.append(next(infer_file).strip())
            except:
                print("finish loading hypothesis file")
                break

    f1 = f1_metric(gen_list, gt_list)
    print(f"\nUnigram F1: {f1[-1]:.3f}")
    k_f1 = f1_metric(gen_list, selected_k_list)
    print(f"Knowledge F1: {k_f1[-1]:.3f}")


if __name__ == "__main__":
    args = setup_args()
    main(args)
