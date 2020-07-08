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
"""Define task."""

from .task_base import Task

TASK_REGISTRY = {}


__all__ = [
    "TASK_REGISTRY",
    "register_task",
    "create_task",
    "add_cmdline_args"
]


def register_task(name):
    """
    Register a new task class.
    """
    def __wrapped__(cls):
        if name in TASK_REGISTRY:
            raise ValueError(f"Cannot register duplicate task ({name})")
        if not issubclass(cls, Task):
            raise ValueError(f"Task ({name}: {cls.__name__}) must extend Task")
        TASK_REGISTRY[name] = cls
        return cls

    return __wrapped__


def create_task(args) -> Task:
    """
    Create a task.
    """
    return TASK_REGISTRY[args.task](args)


def add_cmdline_args(parser):
    """
    Add cmdline argument of Task.
    """
    group = parser.add_argument_group("Task")
    group.add_argument("--task", type=str, required=True)

    args, _ = parser.parse_known_args()
    if args.task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task type: {args.task}")
    TASK_REGISTRY[args.task].add_cmdline_args(parser)
    return group


import tasks.dialog_generation
import tasks.next_sentence_prediction
