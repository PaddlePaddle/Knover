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
"""Tensor utility."""

from itertools import chain

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

from knover.modules.transformer_block import MultiHeadAttention


try:
    if paddle.version.cuda() < "11.0" or paddle.version.cudnn() < "7.6.3":
        TENSOR_CORE_MULTI = 8
    else:
        TENSOR_CORE_MULTI = 1
except:
    print("You can upgarde PaddlePaddle >= 2.2.0 for better AMP performance.")
    TENSOR_CORE_MULTI = 1


def to_optimized_size(sz):
    return (sz + TENSOR_CORE_MULTI - 1) // TENSOR_CORE_MULTI * TENSOR_CORE_MULTI


def get_tensor(tensor_name, to_np=True):
    """Get tensor by name."""
    var = fluid.global_scope().find_var(tensor_name)
    if var is None:
        return None
    tensor = var.get_tensor()
    if tensor is None:
        return None
    if to_np:
        return np.array(tensor)
    else:
        return tensor


def pad_batch_data(insts, pad_id=0):
    """Pad the instances to the max sequence length in batch. """
    max_len = to_optimized_size(max(map(len, insts)))
    inst_data = np.array([list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])
    return inst_data.astype("int64").reshape([-1, max_len])


def repeat(x, times):
    """Repeate tensor."""
    if isinstance(x, dict):
        return {k: repeat(v, times) for k, v in x.items()}
    elif isinstance(x, list):
        return [repeate(v, times) for v in x]
    elif isinstance(x, paddle.Tensor):
        return paddle.tile(x, [times] + [1] * (len(x.shape) - 1))
    else:
        return x


def gather(x, index):
    """Gather data by 1D index."""
    if isinstance(x, MultiHeadAttention.Cache):
        return MultiHeadAttention.Cache(gather(x.k, index), gather(x.v, index))
    elif isinstance(x, MultiHeadAttention.StaticCache):
        return MultiHeadAttention.StaticCache(gather(x.k, index), gather(x.v, index))
    elif isinstance(x, dict):
        return {k: gather(v, index) for k, v in x.items()}
    elif isinstance(x, list):
        return [gather(v, index) for v in x]
    elif isinstance(x, paddle.Tensor):
        if x.dtype == paddle.bool:
            x = paddle.cast(x, "float32")
            x = paddle.gather(x, index)
            x = paddle.cast(x, "bool")
            return x
        else:
            return paddle.gather(x, index)
    else:
        return x
