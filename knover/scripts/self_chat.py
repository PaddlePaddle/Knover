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
"""Self-chat main program."""

import argparse
from collections import namedtuple
import json

import paddle
import paddle.fluid as fluid
from termcolor import colored

import knover.models as models
from knover.tasks.dialog_generation import DialogGeneration
from knover.utils import check_cuda, parse_args, str2bool


def setup_args():
    """Setup self-chat arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Whether to run self-chat in debug mode.")
    parser.add_argument("--in_file", type=str, default=None,
                        help="If given, the input file contains the first utterance in each self-chat episode.")
    parser.add_argument("--out_file", type=str, default=None,
                        help="If given, the self-chat result will save in the output file in json format.")
    parser.add_argument("--num_episode", type=int, default=10,
                        help="If the input file is not given, self-chat will start with 'Hi!' and "
                        "run self-chat the given number of epsidoe repeatedly.")
    parser.add_argument("--num_turn", type=int, default=10,
                        help="The number of turn in each episode.")

    models.add_cmdline_args(parser)
    DialogGeneration.add_cmdline_args(parser)

    args = parse_args(parser)
    args.run_infer = True # only build infer program
    args.display()
    return args


def self_chat(args):
    """Self-chat main function."""
    place = fluid.CUDAPlace(0)

    task = DialogGeneration(args)
    model = models.create_model(args, place)

    def run_batch_self_chat(context_list):
        Example = namedtuple("Example", ["src", "data_id"])
        for i in range(args.num_turn * 2):
            batch_records = []
            for j, context in enumerate(context_list):
                src = []
                for i, utt in enumerate(context):
                    if args.use_role:
                        if (len(context) - i) % 2 == 0:
                            src.append(f"{utt}\x01{0}")
                        else:
                            src.append(f"{utt}\x01{1}")
                    else:
                        src.append(utt)
                src = " [SEP] ".join(src)
                example = Example(src=src, data_id=j)
                task.reader.features[j] = example
                try:
                    record = task.reader._convert_example_to_record(example, is_infer=True)
                except ValueError as e:
                    print(f"[FATAL] {e}")
                    raise e
                batch_records.append(record)
            data = task.reader._pad_batch_records(batch_records, is_infer=True)
            preds = task.infer_step(model, data)
            for context, pred in zip(context_list, preds):
                context.append(pred["response"])
        return

    if args.in_file is not None:
        with open(args.in_file) as in_f:
            context_list = []
            for line in in_f:
                context_list.append([line.strip()])
    else:
        context_list = [["Hi!"] for _ in range(args.num_episode)]

    for i in range(0, len(context_list), args.batch_size):
        run_batch_self_chat(context_list[i:i + args.batch_size])

    if args.out_file is not None:
        with open(args.out_file, "w") as out_f:
            print(f"save self-chat result into: {args.out_file}")
            json.dump(context_list, out_f, indent=2, ensure_ascii=False)
    else:
        for conv_id, context in enumerate(context_list, 1):
            print(colored(f"Conv {conv_id}", "yellow", attrs=["bold"]))
            print(colored("[Start]:", "yellow", attrs=["bold"]), colored(context[0], attrs=["bold"]))
            for i, utt in enumerate(context[1:], 0):
                if i % 2 == 0:
                    print(colored("[Bot1]:", "blue", attrs=["bold"]), colored(utt, attrs=["bold"]))
                else:
                    print(colored("[Bot2]:", "red", attrs=["bold"]), colored(utt, attrs=["bold"]))
            print()

    return


if __name__ == "__main__":
    paddle.enable_static()
    args = setup_args()
    check_cuda(True)
    self_chat(args)
