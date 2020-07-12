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
"""Inference main program."""

import argparse
from collections import defaultdict
import json
import os
import subprocess
import time

import numpy as np
import paddle.fluid as fluid

import models
import tasks
from utils import check_cuda
from utils.args import parse_args, str2bool


def setup_args():
    """
    Setup arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_distributed", type=str2bool, default=False)
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--infer_file", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)

    parser.add_argument("--skip_steps", type=int, default=1)

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.run_infer = True # only build infer program
    print(json.dumps(args, indent=2))
    return args


def infer(args):
    """
    Inference main function.
    """
    if args.is_distributed:
        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        phase = "distributed_test"
    else:
        dev_count = 1
        gpu_id = 0
        phase = "test"
    place = fluid.CUDAPlace(gpu_id)

    task = tasks.create_task(args)
    model = models.create_model(args, place)
    infer_generator = task.reader.data_generator(
        input_file=args.infer_file,
        phase=phase,
        is_infer=True
    )

    # run inference
    begin = time.time()
    infer_out = {}
    for steps, data in enumerate(infer_generator(), 1):
        predictions = task.infer_step(model, data)
        for info in predictions:
            infer_out[info["data_id"]] = info
        if steps % args.skip_steps == 0:
            time_cost = time.time() - begin
            print(f"[infer] steps: {steps}, time: {time_cost:.3f}, "
                  f"speed: {steps / time_cost:.3f} steps/s")

    time_cost = time.time() - begin
    print(f"[infer] steps: {steps} time cost: {time_cost}, "
          f"speed: {steps / time_cost} steps/s")

    if args.is_distributed:
        # merge inference outputs in distributed mode.
        part_file = os.path.join(args.save_path, f"inference_output.part_{gpu_id}")
        with open(part_file, "w") as fp:
            json.dump(predictions, fp, ensure_ascii=False)
        part_finish_file = os.path.join(args.save_path, f"inference_output.part_{gpu_id}.finish")
        with open(part_finish_file, "w"):
            pass

        if gpu_id == 0:
            part_files = f"inference_output.part_*.finish"
            while True:
                ret = subprocess.getoutput(f"find {args.save_path} -maxdepth 1 -name {part_files}")
                num_completed = len(ret.split("\n"))
                if num_completed != dev_count:
                    time.sleep(1)
                    continue
                infer_out = {}
                for dev_id in range(dev_count):
                    part_file = os.path.join(args.save_path, f"inference_output.part_{dev_id}")
                    with open(part_file, "r") as fp:
                        part_infer_out = json.load(fp)
                        for data_id in part_infer_out:
                            infer_out[data_id] = part_infer_out[data_id]
                break
            subprocess.getoutput("rm " + os.path.join(args.save_path, f"inference_output.part*"))

    if gpu_id == 0:
        # save inference outputs
        inference_output = os.path.join(args.save_path, "inference_output.txt")
        with open(inference_output, "w") as f:
            for data_id in sorted(infer_out.keys(), key=lambda x: int(x)):
                f.write(infer_out[data_id][args.output_name] + "\n")

    return


if __name__ == "__main__":
    args = setup_args()
    check_cuda(True)
    infer(args)
