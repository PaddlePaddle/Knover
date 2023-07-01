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
"""Run dialogue generation inference server."""

import argparse
from collections import namedtuple
import json
import os
import random
import threading
from typing import Any, List, Optional

from fastapi import FastAPI
import paddle
import paddle.fluid as fluid
from termcolor import colored, cprint
import requests
from pydantic import BaseModel
import uvicorn

import knover.models as models
from knover.tasks.dialog_generation import DialogGeneration
from knover.utils import check_cuda, parse_args, str2bool


class Request(BaseModel):
    context: List[str]
    context_role: List[int] = []
    knowledge: List[str] = []
    knowledge_role: List[int] = []
    extra_infos: Optional[List[Any]]


class Response(BaseModel):
    name: str
    reply: str
    extra_info: Optional[Any]


def setup_args():
    """Setup dialogue generation inference server arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--is_distributed", type=str2bool, default=False,
                        help="Whether to run distributed inference.")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Whether to run server in debug mode.")
    parser.add_argument("--port", type=int, default=8233,
                        help="Launch servers starts from the given port. The master server will transmit the request to "
                        "each sub server.")
    parser.add_argument("--api_name", type=str, default="chitchat",
                        help="The API name of inference service. The uri is `/api/{api_name}`.")
    parser.add_argument("--bot_name", type=str, default="Knover",
                        help="The name of bot server.")

    models.add_cmdline_args(parser)
    DialogGeneration.add_cmdline_args(parser)

    args = parse_args(parser)
    args.run_infer = True # only build infer program
    args.display()
    return args


def run_server(args):
    """Run inference server main function."""
    if args.is_distributed:
        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    else:
        dev_count = 1
        gpu_id = 0
    place = fluid.CUDAPlace(gpu_id)

    task = DialogGeneration(args)
    model = models.create_model(args, place)

    Example = namedtuple("Example", ["src", "knowledge", "data_id"])

    app = FastAPI()

    lock = threading.Lock()

    @app.post(f"/api/{args.api_name}")
    def inference(req: Request) -> Response:
        """Inference service API."""
        lock.acquire()
        try:
            if args.is_distributed and gpu_id == 0:
                def __send_request(dst_id):
                    url = f"http://127.0.0.1:{args.port + dst_id}/api/{args.api_name}"
                    requests.post(url, json=req.dict())

                threads = []
                for dst_id in range(1, dev_count):
                    thread = threading.Thread(target=__send_request, args=(dst_id,))
                    thread.start()
                    threads.append(thread)

            data_id = random.randint(0, 2 ** 31 - 1)
            src = req.context
            if args.use_role:
                if len(req.context_role) > 0:
                    assert len(src) == len(req.context_role), "The number of item in context_role mismatchs the number of item in context."
                    src = [
                        f"{s}\x01{role_id}"
                        for role_id, s in zip(req.context_role, src)
                    ]
                else:
                    src = [
                        f"{s}\x01{(len(src) - i) % 2}"
                        for i, s in enumerate(src)
                    ]
            src = " [SEP] ".join(src)
            if args.use_role and len(req.knowledge_role) > 0:
                assert len(req.knowledge) == len(req.knowledge_role), "The number of item in knowledge_role mismatchs the number of item in knowledge."
                req.knowledge = [
                    f"{k}\x01{role_id}"
                    for role_id, k in zip(req.knowledge_role, req.knowledge)
                ]
            example = Example(
                src=src,
                knowledge=" [SEP] ".join(req.knowledge),
                data_id=data_id)
            task.reader.features[data_id] = example
            record = task.reader._convert_example_to_record(example, is_infer=True)
            data = task.reader._pad_batch_records([record], is_infer=True)
            pred = task.infer_step(model, data)[0]
            bot_response = pred["response"]
            print(colored("[Bot]:", "blue", attrs=["bold"]), colored(bot_response, attrs=["bold"]))
            task.reader.features.pop(data_id)
            ret = {
                "error_code": 0,
                "error_msg": "ok"
                "name": args.bot_name,
                "reply": bot_response,
                "extra_info": "candidates\n" + "\n".join(
                    f"{cand['response']} -- {cand['score']}" for cand in pred["candidates"]
                )
            }

            if args.is_distributed and gpu_id == 0:
                for thread in threads:
                    thread.join()
        except Exception as e:
            import traceback
            traceback.print_exc()
            ret = {
                "error_code": 1,
                "error_msg": f"[ERROR] {type(e)} {e}",
                "name": args.bot_name,
                "reply": "",
            }
        lock.release()
        return ret

    if gpu_id == 0:
        os.system(f"echo http://`hostname -i`:{args.port}/api/{args.api_name}")
    uvicorn.run(app, host="0.0.0.0", port=args.port + gpu_id)
    return


if __name__ == "__main__":
    paddle.enable_static()
    args = setup_args()
    check_cuda(True)
    run_server(args)
