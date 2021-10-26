#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Create serialized data for dialogue state tracking."""

import argparse
from collections import  defaultdict
import json
import os
import random

from tqdm import tqdm

from utils import flatten_ds, get_logger, get_schema


logger = get_logger(__name__)
random.seed(1007)

GREETINGS = ["hello .",
             "what can i do for you ?",
             "hello , what can i do for you ?",
             "is there anything i can do for you ?"]


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="multiwoz", choices=["multiwoz", "woz"])
    parser.add_argument("--data_type", type=str, choices=["train", "dev", "test"], required=True)
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    return args


def main(args):
    """Main function."""
    with open(args.data_file, "r") as fin:
        data = json.load(fin)
        logger.info(f"Load dataset from `{args.data_file}`")

    schema = get_schema(args.dataset)

    empty_ds_seq = "<ds/> " + " ".join(flatten_ds({}, schema)) + " </ds>"
    ds_labels = []
    ds_seqs = []
    for dial_id, dial in tqdm(data.items(), desc="Dialogue"):
        dial_utt_seqs = process_utt(dial)
        (dial_ds_labels, dial_ds_seqs), (_, dial_neg_ds_seqs) = process_ds(dial, schema)
        ds_labels.extend(dial_ds_labels)
        if args.data_type in ("train", "dev"):
            # concatenate utterance and dialogue state for training: cur_utt + prev_ds -> cur_ds
            for idx, (turn_utt_seq, turn_ds_seq, turn_neg_ds_seq) in \
                enumerate(zip(dial_utt_seqs, dial_ds_seqs, dial_neg_ds_seqs)):
                # basic generation
                if idx == 0:
                    prev_turn_ds_seq = empty_ds_seq
                ds_seqs.append(f"<gen/> {turn_utt_seq} [SEP] {prev_turn_ds_seq} </gen>\x010\t{turn_ds_seq}\x010")
                # amending generation
                ds_seqs.append(f"<amend/> {turn_utt_seq} [SEP] {turn_neg_ds_seq} </amend>\x010\t{turn_ds_seq}\x010")
                prev_turn_ds_seq = turn_ds_seq
        elif args.data_type == "test":
            # save utterance for inference
            for idx, turn_utt_seq in enumerate(dial_utt_seqs):
                ds_seqs.append(f"{dial_id}\t{2 * idx}\t{turn_utt_seq}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    label_path = os.path.join(args.save_path, f"{args.data_type}_labels.json")
    with open(label_path, "w") as fout:
        json.dump(ds_labels, fout, indent=2)
        logger.info(f"Save DS labels to `{label_path}`")

    seq_path = os.path.join(args.save_path, f"{args.data_type}_seq.tsv")
    with open(seq_path, "w") as fout:
        if args.data_type in ("train", "dev"):
            ds_seqs.insert(0, "src\ttgt")
        else:
            ds_seqs.insert(0, "dial_id\tturn_idx\tutts")
        fout.write("\n".join(ds_seqs))
        logger.info(f"Save sequence to `{seq_path}`")


def process_utt(dial):
    """Convert utterance into sequence."""
    greeting = random.choice(GREETINGS)
    dial_utts = [greeting] + [turn["processed_text"] for turn in dial["log"]]
    dial_utt_seqs = []
    for idx in range(len(dial_utts)):
        if idx % 2 != 0:
            # user turn
            last_two_utts = dial_utts[idx - 1: idx + 1]
            dial_utt_seqs.append(f"<utt/> <sys> {last_two_utts[0]}\x010 [SEP] <user> {last_two_utts[1]} </utt>\x011")
    return dial_utt_seqs


def process_ds(dial, schema):
    """Convert dialogue state into sequence."""
    dial_ds_labels, dial_neg_ds_labels = extract_ds(dial)
    dial_ds_seqs = []
    dial_neg_ds_seqs = []

    # ground DS
    for turn_ds_label in dial_ds_labels:
        turn_ds_seq_ls = flatten_ds(turn_ds_label, schema)
        # add special token
        turn_ds_seq = "<ds/> " + " ".join(turn_ds_seq_ls) + " </ds>"
        dial_ds_seqs.append(turn_ds_seq)
    # negative DS
    for turn_neg_ds_label in dial_neg_ds_labels:
        turn_neg_ds_seq_ls = flatten_ds(turn_neg_ds_label, schema)
        # add special token
        turn_neg_ds_seq = "<ds/> " + " ".join(turn_neg_ds_seq_ls) + " </ds>"
        dial_neg_ds_seqs.append(turn_neg_ds_seq)

    return (dial_ds_labels, dial_ds_seqs), (dial_neg_ds_labels, dial_neg_ds_seqs)


def extract_ds(dial):
    """Extract dialogue state."""
    dial_ds_labels = []
    dial_neg_ds_labels = []

    for turn_idx, turn in enumerate(dial["log"]):
        if turn_idx % 2 == 0:
            # user turn
            continue
        # ground DS
        turn_ds_labels = defaultdict(lambda: defaultdict(dict))
        for dom, dom_ds in turn["processed_metadata"].items():
            for slot_type, slot_vals in dom_ds.items():
                for slot, vals in slot_vals.items():
                    if slot != "booked" and len(vals) > 0:
                        turn_ds_labels[dom][slot_type][slot] = vals
        dial_ds_labels.append(turn_ds_labels)
        # negative DS
        if "negative_metadata" in turn:
            turn_neg_ds_labels = defaultdict(lambda: defaultdict(dict))
            for dom, dom_ds in turn["negative_metadata"].items():
                for slot_type, slot_vals in dom_ds.items():
                    for slot, vals in slot_vals.items():
                        if slot != "booked" and len(vals) > 0:
                            turn_neg_ds_labels[dom][slot_type][slot] = vals
            dial_neg_ds_labels.append(turn_neg_ds_labels)

    return dial_ds_labels, dial_neg_ds_labels


if __name__ == "__main__":
    args = setup_args()
    main(args)
