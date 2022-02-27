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
"""Preprocess DSTC10 test set."""

import argparse
from collections import defaultdict
import json
import os
import random

import utils


GREETINGS = [
    "hello .",
    "what can i do for you ?",
    "hello , what can i do for you ?",
    "is there anything i can do for you ?"
]
logger = utils.get_logger(__name__)


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    return args


def merge_session(data_path, save_path):
    """Merge turn samples to dialogue sessions."""
    with open(os.path.join(data_path, "logs.json"), "r") as fin:
        logs = json.load(fin)
        logger.debug(f"#sample: {len(logs)}")

    # record the number of sample turns
    dial_nodes = defaultdict(list)
    for idx, dial in enumerate(logs):
        dial_len = len(dial)
        dial_seq = "###".join([turn["text"] for turn in dial])
        dial_node = BiLinkNode(index=idx, value=dial_seq)
        dial_nodes[dial_len].append(dial_node)

    # merge dialogue session
    dial_lens = sorted(list(dial_nodes.keys()), reverse=True)
    for i in range(len(dial_lens) - 1):
        for dial_i in dial_nodes[dial_lens[i]]:
            success_flag = False
            for j in range(i + 1, len(dial_lens)):
                for dial_j in dial_nodes[dial_lens[j]]:
                    if dial_i.value.startswith(dial_j.value):
                        if dial_i.prev is not None:
                            logger.warning(f"dial_i: {dial_i.value}")
                            logger.warning(f"dial_i.prev: {dial_i.prev.value}")
                        if dial_j.next is not None:
                            logger.warning(f"dial_j: {dial_j.value}")
                            logger.warning(f"dial_j.next: {dial_j.next.value}")
                            continue
                        dial_i.prev = dial_j
                        dial_j.next = dial_i
                        success_flag = True
                        break
                if success_flag:
                    break

    # output
    session_to_sample_indices = []
    dial_lens.reverse()
    for i in range(len(dial_lens)):
        for dial_i in dial_nodes[dial_lens[i]]:
            prev_idx = dial_i.get_prev_idx(session_to_sample_indices)
            if prev_idx is None:
                session_to_sample_indices.append([dial_i.index])
            else:
                session_to_sample_indices[prev_idx].append(dial_i.index)
    
    session_logs = []
    for session_indices in session_to_sample_indices:
        turns = []
        for sample_idx in session_indices:
            turns.append(logs[sample_idx])
        session_logs.append(turns)
    with open(os.path.join(save_path, "session_logs.json"), "w") as fout_logs, \
        open(os.path.join(save_path, "session_to_sample_mapping.txt"), "w") as fout_map:
        logger.debug(f"#dialogue: len(session_logs)")
        json.dump(session_logs, fout_logs, indent=2)
        session_to_sample_mapping = []
        for session_indices in session_to_sample_indices:
            session_to_sample_mapping.extend(list(map(str, session_indices)))
            session_to_sample_mapping.append("")
        fout_map.write("\n".join(session_to_sample_mapping))

    return session_logs


class BiLinkNode(object):
    """Bi-link node calss."""
    def __init__(self, index=None, value=None, prev=None, next=None):
        self.index = index
        self.value = value
        self.prev = prev
        self.next = next

    def get_prev_idx(self, session_to_sample_indices):
        """Get previous turn sample index."""
        if self.prev is None:
            return None
        for idx, node_ls in enumerate(session_to_sample_indices):
            if self.prev.index == node_ls[-1]:
                return idx
        return None


def create_knover_format(session_logs, save_path):
    """Create Knover data format.""" 
    hist_len = 2
    target_content = ["dial_index\tturn_index\tcontext"]

    for dial_idx, dial_logs in enumerate(session_logs):
        for turn_idx, turn_logs in enumerate(dial_logs):
            turn_len = len(turn_logs)
            if turn_idx == 0:
                prev_log_len = 0
            cur_logs = turn_logs[max(prev_log_len - 2 * hist_len, 0):]
            if cur_logs[0]["speaker"] != "S":
                cur_logs.insert(0, {"speaker": "S", "text": random.choice(GREETINGS)})

            # merge utt
            context_ls = ["<utt/> "]
            for idx, log in enumerate(cur_logs):
                role = "user" if idx % 2 != 0 else "sys"
                cur_utt = utils.pre_process_utt(log["text"], role=role)
                if idx == len(cur_logs) - 1:
                    cur_utt += " </utt>"
                if log["speaker"] == "U":
                    cur_utt = f"<user> {cur_utt}\x011"
                elif log["speaker"] == "S":
                    cur_utt = f"<sys> {cur_utt}\x010"
                context_ls.append(cur_utt)
                if idx != len(cur_logs) - 1:
                    context_ls.append(" [SEP] ")

            context = "".join(context_ls)
            target_content.append(f"{dial_idx}\t{turn_len}\t{context}")
            prev_log_len = turn_len

    with open(os.path.join(save_path, "test.knover.tsv"), "w") as fout:
        fout.write("\n".join(target_content))


if __name__ == "__main__":
    args = setup_args()
    session_logs = merge_session(args.data_path, args.save_path)
    create_knover_format(session_logs, args.save_path)
