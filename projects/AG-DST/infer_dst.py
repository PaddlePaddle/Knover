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
"""Main inference program for dialogue state tracking."""

import argparse
from collections import defaultdict, namedtuple
import json
import os

import paddle
import paddle.distributed.fleet as fleet
import paddle.fluid as fluid
from tqdm import tqdm

import knover.models as models
import knover.tasks as tasks
from knover.utils import check_cuda, parse_args, Timer

from utils import flatten_ds, get_logger, get_schema, parse_ds


logger = get_logger(__name__)


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser(description="Main inference program for dialogue state tracking.")
    parser.add_argument("--infer_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="multiwoz", choices=["multiwoz", "woz"])
    parser.add_argument("--dial_batch_size", type=int, default=32)

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.run_infer = True  # only build infer program
    args.display()
    return args


def infer_dst(args):
    """Inference main function."""
    if args.is_distributed:
        fleet.init(is_collective=True)

        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        trainers_num = fleet.worker_num()
        trainer_id = fleet.worker_index()
        phase = "distributed_test"
    else:
        dev_count = 1
        gpu_id = 0
        trainers_num = 1
        trainer_id = 0
        phase = "test"
    place = fluid.CUDAPlace(gpu_id)

    task = tasks.create_task(args)
    model = models.create_model(args, place)
    # task.debug()

    schema = get_schema(args.dataset)
    empty_ds_seq = "<ds/> " + " ".join(flatten_ds({}, schema)) + " </ds>"

    # record original order and init status
    output_order = []
    # {"dial_id": {"prev_ds": "", "turns": [{"utts": utts, "turn_idx": turn_idx}], "cur_idx": 0}}
    dial_status = defaultdict(dict)
    with open(args.infer_file, "r") as fin:
        next(fin)
        for line in fin:
            dial_id, turn_idx, utts = line.strip().split("\t")
            output_order.append(f"{dial_id}-{turn_idx}")
            if dial_id not in dial_status:
                dial_status[dial_id]["prev_ds"] = empty_ds_seq
                dial_status[dial_id]["turns"] = []
                dial_status[dial_id]["cur_idx"] = 0
            dial_status[dial_id]["turns"].append({"utts": utts, "turn_idx": turn_idx})
    dial_ids = list(dial_status.keys())

    # batch inference
    outputs = {}
    timer = Timer()
    while len(dial_ids) > 0:
        timer.start()
        cur_dial_ids = dial_ids[:args.dial_batch_size]
        logger.info(f"Sampled dialogue ids: {cur_dial_ids}")

        # 1st: basic generation
        basic_inputs = {}
        for cur_dial_id in cur_dial_ids:
            cur_idx = dial_status[cur_dial_id]["cur_idx"]
            cur_dial_turn = dial_status[cur_dial_id]["turns"][cur_idx]
            cur_utts = cur_dial_turn["utts"]
            prev_ds = dial_status[cur_dial_id]["prev_ds"]
            src = f"<gen/> {cur_utts} [SEP] {prev_ds} </gen>\x010"
            basic_inputs[f"{cur_dial_id}-{cur_dial_turn['turn_idx']}"] = src
        basic_outputs = generate(basic_inputs, model, task)

        # 2nd: amending generation
        amending_inputs = {}
        for cur_dial_id in cur_dial_ids:
            cur_idx = dial_status[cur_dial_id]["cur_idx"]
            cur_dial_turn = dial_status[cur_dial_id]["turns"][cur_idx]
            cur_utts = cur_dial_turn["utts"]
            basic_ds = basic_outputs[f"{cur_dial_id}-{cur_dial_turn['turn_idx']}"]
            src = f"<amend/> {cur_utts} [SEP] {basic_ds} </amend>\x010"
            amending_inputs[f"{cur_dial_id}-{cur_dial_turn['turn_idx']}"] = src
        amending_outputs = generate(amending_inputs, model, task)

        outputs.update(amending_outputs)
        time_cost_infer = timer.pass_time
        logger.info(f"Time cost: {time_cost_infer}")

        # debug info
        for dial_turn_tag in basic_inputs:
            logger.debug(f"[basic input]: {basic_inputs[dial_turn_tag]}")
            logger.debug(f"[basic output]: {basic_outputs[dial_turn_tag]}")
            logger.debug(f"[amending input]: {amending_inputs[dial_turn_tag]}")
            logger.debug(f"[amending output]: {amending_outputs[dial_turn_tag]}")

        # update dial_status
        for dial_turn_tag in amending_outputs:
            dial_id, _ = dial_turn_tag.split("-")
            dial_status[dial_id]["cur_idx"] += 1
            if dial_status[dial_id]["cur_idx"] >= len(dial_status[dial_id]["turns"]):
                dial_ids.remove(dial_id)
            else:
                dial_status[dial_id]["prev_ds"] = outputs[dial_turn_tag]
        timer.reset()

    # reorder and output
    if gpu_id == 0:
        pred_seqs = []
        pred_labels = []
        for dial_turn_tag in output_order:
            pred_seqs.append(outputs[dial_turn_tag])
            pred_label = parse_ds(outputs[dial_turn_tag], schema)
            pred_labels.append(pred_label)

        out_seq_file = os.path.join(args.save_path, "inference_output.txt")
        out_label_file = os.path.join(args.save_path, "inference_labels.json")
        with open(out_seq_file, "w") as fout_seq, open(out_label_file, "w") as fout_label:
            fout_seq.write("\n".join(pred_seqs))
            json.dump(pred_labels, fout_label, indent=2)
        logger.info(f"Save inference sequences to `{out_seq_file}`")
        logger.info(f"Save inference labels to `{out_label_file}`")


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
    infer_dst(args)
