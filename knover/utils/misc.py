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
"""Miscellaneous utility."""

import time


class Timer(object):

    def __init__(self):
        self._pass_time = 0
        self._start_time = None
        return

    def start(self):
        self._start_time = time.time()

    def pause(self):
        self._pass_time += time.time() - self._start_time
        self._start_time = None

    def reset(self):
        self._pass_time = 0

    @property
    def pass_time(self):
        if self._start_time is None:
            return self._pass_time
        else:
            return self._pass_time + time.time() - self._start_time


ERROR_MESSAGE="\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
    Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
def check_cuda(use_cuda, err=ERROR_MESSAGE):
    """Check CUDA."""
    try:
        if use_cuda and not fluid.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass
