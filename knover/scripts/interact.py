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
import json
import os
import socket
import threading
import urllib.request

import paddle
import paddle.fluid as fluid
from termcolor import colored, cprint

import knover.models as models
from knover.tasks.dialog_generation import DialogGeneration
from knover.utils import check_cuda, parse_args, str2bool


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_distributed", type=str2bool, default=False,
                        help="Whether to run distributed inference.")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Whether to run server in debug mode.")
    parser.add_argument("--port", type=int, default=18123,
                        help="Launch sockets start from the given port. User inputs are transfered by socket.")

    models.add_cmdline_args(parser)
    DialogGeneration.add_cmdline_args(parser)

    args = parse_args(parser)
    args.run_infer = True # only build infer program
    args.display()
    return args


def interact(args):
    """Interaction main function."""
    if args.is_distributed:
        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))

    else:
        dev_count = 1
        gpu_id = 0
    place = fluid.CUDAPlace(gpu_id)

    task = DialogGeneration(args)
    model = models.create_model(args, place)

    if model.topo.pp_info.size != 1:
        raise ValueError("Cannot support pipeline in inference now!")
    if model.topo.sharding_info.size != 1:
        raise ValueError("Cannot support sharding in inference now!")
    if model.topo.world.size > dev_count:
        raise ValueError("Cannot support evaluation on multiple nodes now!")

    if args.is_distributed:
        if gpu_id > 0:
            Example = namedtuple("Example", ["src", "data_id"])
            context = []
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                host, port = "127.0.0.1", args.port + gpu_id
                s.bind((host, port))
                s.listen()
                while True:
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data.decode("utf8") == "[EXIT]":
                            break
                        example = Example(src=data.decode("utf8"), data_id=0)
                        task.reader.features[0] = example
                        try:
                            record = task.reader._convert_example_to_record(example, is_infer=True)
                        except ValueError as e:
                            print(f"[FATAL] {e}")
                            raise e
                        data = task.reader._pad_batch_records([record], is_infer=True)
                        pred = task.infer_step(model, data)[0]
                        bot_response = pred["response"]
                        context.append(bot_response)
            return
        else:
            def send_request(dst_id, src):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    host, port = "127.0.0.1", args.port + dst_id
                    s.connect((host, port))
                    data = src.encode("utf8")
                    s.sendall(data)

    Example = namedtuple("Example", ["src", "data_id"])
    context = []
    start_info = "Enter [EXIT] to quit the interaction, [NEXT] to start a new conversation."
    cprint(start_info, "yellow", attrs=["bold"])
    while True:
        if args.is_distributed:
            print(colored("[Human]:", "red", attrs=["bold"]))
            user_utt = input().strip()
        else:
            user_utt = input(colored("[Human]: ", "red", attrs=["bold"])).strip()
        if user_utt == "[EXIT]":
            if args.is_distributed:
                threads = []
                for i in range(1, dev_count):
                    thread = threading.Thread(target=send_request, args=(i, "[EXIT]"))
                    thread.start()
                    threads.append(thread)
            break
        elif user_utt == "[NEXT]":
            context = []
            cprint(start_info, "yellow", attrs=["bold"])
        else:
            context.append(user_utt)
            src = " [SEP] ".join(context)

            if args.is_distributed:
                threads = []
                for i in range(1, dev_count):
                    thread = threading.Thread(target=send_request, args=(i, src))
                    thread.start()
                    threads.append(thread)

            example = Example(src=src, data_id=0)
            task.reader.features[0] = example
            try:
                record = task.reader._convert_example_to_record(example, is_infer=True)
            except ValueError as e:
                print(f"[FATAL] {e}")
                raise e
            data = task.reader._pad_batch_records([record], is_infer=True)
            pred = task.infer_step(model, data)[0]
            bot_response = pred["response"]
            if args.is_distributed:
                print(colored("[Bot]:", "blue", attrs=["bold"]))
                print(colored(bot_response, attrs=["bold"]))
            else:
                print(colored("[Bot]:", "blue", attrs=["bold"]), colored(bot_response, attrs=["bold"]))
            context.append(bot_response)

            if args.is_distributed:
                for thread in threads:
                    thread.join()

    return


if __name__ == "__main__":
    paddle.enable_static()
    args = setup_args()
    check_cuda(True)
    interact(args)
