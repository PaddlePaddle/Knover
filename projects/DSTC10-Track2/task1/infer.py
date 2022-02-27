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
"""Main inference program."""

import argparse
from collections import defaultdict, namedtuple
import json
import os

import paddle
import paddle.fluid as fluid

import knover.models as models
from knover.tasks.dialog_generation import DialogGeneration
from knover.utils import check_cuda, parse_args, str2bool, Timer
from utils import get_logger, flatten_ds, parse_ds, PostProcess


logger = get_logger(__name__)


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser(description="Main dynamic inference program.")
    parser.add_argument("--infer_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--db_file", type=str, required=True)
    parser.add_argument("--session_to_sample_mapping_file", type=str, required=True)
    parser.add_argument("--dial_batch_size", type=int, default=8)
    parser.add_argument("--normalization", type=str2bool, default=True)
    parser.add_argument("--db_guidance", type=str2bool, default=True)

    models.add_cmdline_args(parser)
    DialogGeneration.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.run_infer = True  # only build infer program
    args.display()
    return args


def infer(args):
    """Main inference function."""
    place = fluid.CUDAPlace(0)

    task = DialogGeneration(args)
    model = models.create_model(args, place)
    task.debug()
    
    empty_ds_seq = "<ds/> " + " ".join(flatten_ds({})) + " </ds>"
    post_process = PostProcess(args.db_file, normalization=args.normalization, db_guidance=args.db_guidance)

    # record original order and init status
    output_order = []
    # {"dial_id": {"prev_ds": "", "turns": [], "cur_turn_idx": 0}}
    dial_status = defaultdict(dict)
    with open(args.infer_file, "r") as fin:
        next(fin)
        for line in fin:
            dial_id, turn_idx, utt = line.strip().split("\t")
            output_order.append(f"{dial_id}-{turn_idx}")
            if dial_id not in dial_status:
                dial_status[dial_id]["prev_ds"] = empty_ds_seq
                dial_status[dial_id]["turns"] = []
                dial_status[dial_id]["cur_turn_idx"] = 0
            dial_status[dial_id]["turns"].append({"utts": utt, "turn_idx": turn_idx})
    dial_ids = sorted(list(dial_status.keys()))

    # batch inference
    outputs = {}
    timer = Timer()
    batch_idx = 0
    while len(dial_ids) > 0:
        logger.info(f"Batch index: {batch_idx}")
        batch_idx += 1
        timer.start()
        cur_dial_ids = dial_ids[:args.dial_batch_size]

        cur_inputs = {}
        for cur_dial_id in cur_dial_ids:
            cur_dial_turn = dial_status[cur_dial_id]["turns"][dial_status[cur_dial_id]["cur_turn_idx"]]
            cur_utt = cur_dial_turn["utts"]
            prev_ds = dial_status[cur_dial_id]["prev_ds"]
            src = f"{cur_utt} [SEP] {prev_ds}\x010"
            cur_inputs[f"{cur_dial_id}-{cur_dial_turn['turn_idx']}"] = src
        cur_outputs = generate(cur_inputs, model, task)
        time_cost_infer = timer.pass_time
        logger.debug(f"Time cost (prediction): {time_cost_infer}")

        # post process
        cur_outputs_postprocess = {}
        for dial_turn_tag, pred_ds in cur_outputs.items():
            dial_id, _ = dial_turn_tag.split("-")
            cur_dial_turn = dial_status[dial_id]["turns"][dial_status[dial_id]["cur_turn_idx"]]
            cur_utt_ls = cur_dial_turn["utts"].split("[SEP]")
            postprocessed_pred_ds = post_process.run(
                pred_ds, prev_ds=dial_status[dial_id]["prev_ds"], utt_list=cur_utt_ls
            )
            cur_outputs_postprocess[dial_turn_tag] = postprocessed_pred_ds
        outputs.update(cur_outputs_postprocess)
        time_cost_postprocess = timer.pass_time - time_cost_infer
        logger.debug(f"Time cost (postprocess): {time_cost_postprocess}")

        # update `cur_turn_idx` and `prev_ds`
        for dial_turn_tag in cur_outputs:
            dial_id, _ = dial_turn_tag.split("-")
            dial_status[dial_id]["cur_turn_idx"] += 1
            if dial_status[dial_id]["cur_turn_idx"] >= len(dial_status[dial_id]["turns"]):
                dial_ids.remove(dial_id)
            else:
                dial_status[dial_id]["prev_ds"] = outputs[dial_turn_tag]
        timer.reset()
    
    # reorder and output
    sample_indices = []
    with open(args.session_to_sample_mapping_file, "r") as fin:
        for line in fin:
            line = line.strip()
            if line:
                sample_indices.append(int(line))
    pred_seqs = [outputs[dial_turn_tag] for dial_turn_tag in output_order]
    pred_sample_labels = [None] * len(pred_seqs)
    for pred_ds_seq, sample_idx in zip(pred_seqs, sample_indices):
        pred_ds_dict = parse_ds(pred_ds_seq, date_prefix="$")
        pred_sample_labels[sample_idx] = pred_ds_dict

    out_seq_file = os.path.join(args.save_path, "inference_output.txt")
    out_sample_label_file = os.path.join(args.save_path, "inference_labels.json")
    with open(out_seq_file, "w") as fout_seq, open(out_sample_label_file, "w") as fout_label:
        fout_seq.write("\n".join(pred_seqs))
        json.dump(pred_sample_labels, fout_label, indent=2)
    logger.info(f"Save inference sequences to `{out_seq_file}`")
    logger.info(f"Save inference sample labels to `{out_sample_label_file}`")


def generate(inputs, model, task):
    """Generation from sequence to sequence."""
    data_tags = list(inputs.keys())
    records = []
    for data_id, data_tag in enumerate(data_tags):
        Example = namedtuple("Example", ["src", "data_id"])
        example = Example(src=inputs[data_tag], data_id=data_id)
        task.reader.features[data_id] = example
        record = task.reader._convert_example_to_record(example, is_infer=True)
        records.append(record)
    data = task.reader._pad_batch_records(records, is_infer=True)
    predictions = task.infer_step(model, data)
    outputs = {}
    for pred in predictions:
        outputs[data_tags[pred["data_id"]]] = pred["response"]
    return outputs


if __name__ == "__main__":
    paddle.enable_static()
    args = setup_args()
    check_cuda(True)
    infer(args)
