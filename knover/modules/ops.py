#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Many useful operators."""

import numpy as np
import paddle
import paddle.static as static


global_rng = np.random.RandomState(0)
sampling_seed = 0
sampling_rng = None


def reset_state():
    global sampling_seed
    sampling_seed = global_rng.randint(0, 2**32 - 1)
    reset_rng(sampling_seed)
    return sampling_seed


def reset_rng(random_seed):
    global sampling_rng
    sampling_rng = np.random.RandomState(random_seed)
    return


def _sampling_id(probs_list):
    probs_list = np.array(probs_list)
    global sampling_rng
    indices = []
    for probs in probs_list:
        indices.append(sampling_rng.choice(len(probs), 1, p=probs / np.sum(probs))[0])
    indices = np.array(indices)
    return indices


def sampling_id(probs):
    prog = static.default_main_program()
    sampling_ids = prog.current_block().create_var(name="sampling_ids", dtype="int64", shape=[-1])
    static.py_func(func=_sampling_id, x=probs, out=sampling_ids)
    return sampling_ids
