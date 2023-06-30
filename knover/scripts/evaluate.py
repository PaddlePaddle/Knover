#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Evaluation main program."""

import argparse
from collections import defaultdict
import json
import os
import subprocess
import time

import paddle
import paddle.fluid as fluid

import knover.models as models
import knover.tasks as tasks
from knover.scripts.train import evaluate as evaluate_dataset
from knover.utils import check_cuda, parse_args, str2bool, Timer


def setup_args():
    """Setup evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_distributed", type=str2bool, default=False,
                        help="Whether to run distributed evaluation.")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Whether to run evaluation in debug mode.")
    parser.add_argument("--save_path", type=str, default="output",
                        help="The path where to save temporary files.")
    parser.add_argument("--eval_file", type=str, required=True,
                        help="The evaluation dataset: file / filelist. "
                        "See more details in `docs/usage.md`: `file_format`.")
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Display evaluation log information every X steps.")

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    args.display()
    return args


def evaluate(args):
    """Evaluation main function."""
    if args.is_distributed:
        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        phase = "distributed_test"
    else:
        dev_count = 1
        gpu_id = 0
        phase = "test"
    place = fluid.CUDAPlace(gpu_id)

    # setup task and model
    task = tasks.create_task(args)
    model = models.create_model(args, place)

    # setup dataset
    eval_generator = task.get_data_loader(
        model,
        input_file=args.eval_file,
        num_part=model.topo.data_info.size,
        part_id=model.topo.data_info.rank,
        phase=phase
    )
    if model.topo.pp_info.size != 1:
        raise ValueError("Cannot support pipeline in evaluation now!")
    if model.topo.world.size > dev_count:
        raise ValueError("Cannot support evaluation on multiple nodes now!")

    evaluate_dataset(
        task,
        model,
        eval_generator,
        args,
        dev_count,
        gpu_id,
        training_step=0,
        tag="test"
    )
    return


if __name__ == "__main__":
    paddle.enable_static()
    args = setup_args()
    check_cuda(True)
    evaluate(args)
