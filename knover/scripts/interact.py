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
"""Interaction main program."""

import argparse
from collections import namedtuple

from termcolor import colored, cprint
import paddle.fluid as fluid

import knover.models as models
from knover.tasks.dialog_generation import DialogGeneration
from knover.utils import check_cuda, parse_args


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()

    models.add_cmdline_args(parser)
    DialogGeneration.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.run_infer = True # only build infer program
    args.display()
    return args


def interact(args):
    """Interaction main function."""
    dev_count = 1
    gpu_id = 0
    place = fluid.CUDAPlace(gpu_id)

    task = DialogGeneration(args)
    model = models.create_model(args, place)

    task.debug()

    Example = namedtuple("Example", ["src", "data_id"])
    context = []
    start_info = "Enter [EXIT] to quit the interaction, [NEXT] to start a new conversation."
    cprint(start_info, "yellow", attrs=["bold"])
    while True:
        user_utt = input(colored("[Human]: ", "red", attrs=["bold"])).strip()
        if user_utt == "[EXIT]":
            break
        elif user_utt == "[NEXT]":
            context = []
            cprint(start_info, "yellow", attrs=["bold"])
        else:
            context.append(user_utt)
            example = Example(src=" [SEP] ".join(context), data_id=0)
            task.reader.features[0] = example
            record = task.reader._convert_example_to_record(example, is_infer=True)
            data = task.reader._pad_batch_records([record], is_infer=True)
            pred = task.infer_step(model, data)[0]
            bot_response = pred["response"]
            print(colored("[Bot]:", "blue", attrs=["bold"]), colored(bot_response, attrs=["bold"]))
            context.append(bot_response)

    return


if __name__ == "__main__":
    args = setup_args()
    check_cuda(True)
    interact(args)
