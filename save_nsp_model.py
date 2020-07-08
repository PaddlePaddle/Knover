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
"""Save NSP model."""

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

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    parser.add_argument("--nsp_inference_model_path", type=str, required=True)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.run_infer = True # only build infer program
    print(json.dumps(args, indent=2))
    return args


def save(args):
    """
    Inference main function.
    """
    dev_count = 1
    gpu_id = 0
    place = fluid.CUDAPlace(gpu_id)

    task = tasks.create_task(args)
    model = models.create_model(args, place)
    model.save_infer_model(args.nsp_inference_model_path)
    return


if __name__ == "__main__":
    args = setup_args()
    print(json.dumps(args, indent=2))
    check_cuda(True)
    save(args)
