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

from collections import defaultdict
import copy

import numpy as np
import paddle
import paddle.static as static

from knover.utils import rindex


global_rng = None
sampling_seed = 0
sampling_rng = None


def reset_state(generate_seed):
    global global_rng
    if global_rng is None:
        global_rng = np.random.RandomState(generate_seed)

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


_ngram = None
_bos_id = None
_eos_id = None
ngram_stat_list = None
cur_ngram_list = None


def _init_ngram_blocking(token_ids):
    token_ids = np.array(token_ids)
    global ngram_stat_list, cur_ngram_list
    ngram_stat_list = [defaultdict(set) for _ in range(token_ids.shape[0])]
    cur_ngram_list = [[] for _ in range(token_ids.shape[0])]
    for ids, ngram_state in zip(token_ids[:, :, 0], ngram_stat_list):
        last_idx = rindex(ids.tolist(), _eos_id)
        cur_ids = []
        for i in range(last_idx + 1):
            if ids[i] in [_bos_id, _eos_id]:
                cur_ids = []
            else:
                cur_ids.append(ids[i])
            if len(cur_ids) >= _ngram:
                k = tuple(cur_ids[-_ngram:-1])
                ngram_state[k].add(ids[i])


def init_ngram_blocking(token_ids, ngram, bos_id, eos_id):
    """Initalize N-gram blocking related data."""
    global _ngram, _bos_id, _eos_id
    _ngram, _bos_id, _eos_id = ngram, bos_id, eos_id
    assert _ngram >= 1
    static.py_func(func=_init_ngram_blocking, x=token_ids, out=None)


def _apply_ngram_blocking(logits, is_finished):
    logits = np.array(logits) # shape: [B, V]
    is_finished = np.array(is_finished) # shape: [B, 1]
    global ngram_stat_list, cur_ngram_list
    for i in range(logits.shape[0]):
        if is_finished[i]:
            continue
        if len(cur_ngram_list[i]) >= _ngram - 1:
            k = tuple(cur_ngram_list[i][-_ngram + 1:])
            if k in ngram_stat_list[i]:
                for v in ngram_stat_list[i][k]:
                    logits[i][v] -= 1e9
    return logits


def apply_ngram_blocking(logits, is_finished):
    """Update logits by N-gram blocking strategy."""
    prog = static.default_main_program()
    new_logits = prog.current_block().create_var(name="out_logits", dtype=logits.dtype, shape=logits.shape)
    static.py_func(func=_apply_ngram_blocking, x=(logits, is_finished), out=new_logits)
    return new_logits


def _gather_ngram_stat(parent_idx):
    parent_idx = np.array(parent_idx) # shape: [B]
    global ngram_stat_list, cur_ngram_list
    new_ngram_stat_list, new_cur_ngram_list = [], []
    for idx in parent_idx:
        new_ngram_stat_list.append(copy.deepcopy(ngram_stat_list[idx]))
        new_cur_ngram_list.append(copy.deepcopy(cur_ngram_list[idx]))
    ngram_stat_list, cur_ngram_list = new_ngram_stat_list, new_cur_ngram_list


def _update_ngram_blocking(pred, is_finished):
    pred = np.array(pred) # shape: [B, 1]
    is_finished = np.array(is_finished) # shape: [B, 1]
    global ngram_stat_list, cur_ngram_list
    assert(len(ngram_stat_list) == len(cur_ngram_list) == pred.shape[0] == is_finished.shape[0])
    for ngram_stat, cur_ngram, x, flag in zip(ngram_stat_list, cur_ngram_list, pred[:, 0], is_finished[:, 0]):
        if flag:
            continue
        cur_ngram.append(x)
        if len(cur_ngram) >= _ngram:
            k = tuple(cur_ngram[-_ngram:-1])
            ngram_stat[k].add(x)


def update_ngram_blocking(pred, is_finished, parent_idx=None):
    """Update N-gram blocking strategy data."""
    if parent_idx is not None:
        static.py_func(func=_gather_ngram_stat, x=parent_idx, out=None)
    static.py_func(func=_update_ngram_blocking, x=(pred, is_finished), out=None)
