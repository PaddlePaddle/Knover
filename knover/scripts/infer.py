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

import paddle
import paddle.fluid as fluid

import knover.models as models
import knover.tasks as tasks
from knover.utils import check_cuda, Timer
from knover.utils.args import parse_args, str2bool


def setup_args():
    """Setup inference arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_distributed", type=str2bool, default=False)
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--infer_file", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)

    parser.add_argument("--log_steps", type=int, default=1)

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.run_infer = True # only build infer program
    args.display()
    return args


def infer(args):
    """Inference main function."""
    if args.is_distributed:
        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        # gpu_id = int(0)
        phase = "distributed_test"
    else:
        dev_count = 1
        # gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        gpu_id = int(0)
        phase = "test"
    place = fluid.CUDAPlace(gpu_id)
    task = tasks.create_task(args)
    model = models.create_model(args, place)
    infer_generator = task.get_data_loader(
        model,
        input_file=args.infer_file,
        num_part=dev_count,
        part_id=gpu_id,
        phase=phase,
        is_infer=True
    )

    # run inference
    timer = Timer()
    timer.start()
    infer_out = {}
    step = 0
    for step, data in enumerate(infer_generator(), 1):
        predictions = task.infer_step(model, data)
        for pred in predictions:
            infer_out[pred["data_id"]] = pred
        if step % args.log_steps == 0:
            time_cost = timer.pass_time
            print(f"\tstep: {step}, time: {time_cost:.3f}, "
                  f"queue size: {infer_generator._queue.size()}, "
                  f"speed: {step / time_cost:.3f} steps/s")

    time_cost = timer.pass_time
    print(f"[infer] steps: {step} time cost: {time_cost}, "
          f"speed: {step / time_cost} steps/s")

    if args.is_distributed:
        # merge inference outputs in distributed mode.
        part_file = os.path.join(args.save_path, f"inference_output.part_{gpu_id}")
        with open(part_file, "w") as fp:
            json.dump(infer_out, fp, ensure_ascii=False, indent=2)
        part_finish_file = os.path.join(args.save_path, f"inference_output.part_{gpu_id}.finish")
        with open(part_finish_file, "w"):
            pass

    # Only run on master GPU in each node
    if gpu_id != 1:
        return

    if args.is_distributed:
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

    # save inference outputs
    inference_output = os.path.join(args.save_path, "inference_output.txt")
    with open(inference_output, "w") as f:
        for data_id in sorted(infer_out.keys(), key=lambda x: int(x)):
            f.write("\t".join(map(str, [infer_out[data_id][name] for name in args.output_name.split(",")])) + "\n")
    print(f"save inference result into: {inference_output}")

    return

#from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
#import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
if __name__ == "__main__":
    if hasattr(paddle, "enable_static"):
        paddle.enable_static()
    # if args.is_distributed:
    #role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    
    fleet.init(is_collective=True)
    dev_count = fluid.core.get_cuda_device_count()
    args = setup_args()
    if args.is_distributed:      
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    else:
        gpu_id = int(0)
    # gpu_id = int(0)
    trainers_num = fleet.worker_num()
    trainer_id = fleet.worker_index()
    check_cuda(True)
    infer(args)
