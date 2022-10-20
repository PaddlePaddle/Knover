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
    """Reset the global rng.

    Global rng is used to generate sampling seed to initialize sampling rng.
    """
    global global_rng
    if global_rng is None:
        global_rng = np.random.RandomState(generate_seed)

    global sampling_seed
    sampling_seed = global_rng.randint(0, 2 ** 32 - 1)
    reset_rng(sampling_seed)
    return


def reset_rng(sampling_seed):
    """Reset sampling rng which is used in sampling_id.

    After reset_rng with the same sampling_seed, you will get the same sampling result.

    Args:
        sampling_seed: a int value which is the seed of sampling rng.
    """
    global sampling_rng
    sampling_rng = np.random.RandomState(sampling_seed)
    return


def _sampling_id(probs_list):
    """Sampling from probability distributions in numpy."""
    probs_list = np.array(probs_list)
    global sampling_rng
    indices = []
    for probs in probs_list:
        indices.append(sampling_rng.choice(len(probs), 1, p=probs / np.sum(probs))[0])
    indices = np.array(indices)
    return indices


def sampling_id(probs):
    """Sampling from probability distributions in PaddlePaddle.

    Args:
        probs: represents the probability distributions, shape is [batch_size, num_outputs]

    Retrun:
        The sampled indices, shape is [batch_size]
    """
    prog = static.default_main_program()
    sampling_ids = prog.current_block().create_var(name="sampling_ids", dtype="int64", shape=[-1])
    static.py_func(func=_sampling_id, x=probs, out=sampling_ids)
    return sampling_ids


class NGramBlockingProcessor(object):
    """N-gram blocking strategy."""

    def __init__(self, ngram, bos_id, eos_id):
        self.ngram = ngram
        self.bos_id = bos_id
        self.eos_id = eos_id

    def init(self, token_ids):
        """Initalize N-gram blocking strategy related data."""
        def __wrapper__(token_ids):
            token_ids = np.array(token_ids)
            self.ngram_stat_list = [defaultdict(set) for _ in range(token_ids.shape[0])]
            self.cur_ngram_list = [[] for _ in range(token_ids.shape[0])]
            for ids, ngram_state in zip(token_ids[:, :, 0], self.ngram_stat_list):
                last_idx = rindex(ids.tolist(), self.eos_id)
                cur_ids = []
                for i in range(last_idx + 1):
                    if ids[i] in [self.bos_id, self.eos_id]:
                        cur_ids = []
                    else:
                        cur_ids.append(ids[i])
                    if len(cur_ids) >= self.ngram:
                        k = tuple(cur_ids[-self.ngram:-1])
                        ngram_state[k].add(ids[i])

        static.py_func(func=__wrapper__, x=token_ids, out=None)

    def apply(self, logits, is_finished):
        """Post process logits by N-gram blocking strategy."""
        def __wrapper__(logits, is_finished):
            logits = np.array(logits) # shape: [B, V]
            is_finished = np.array(is_finished) # shape: [B, 1]
            for i in range(logits.shape[0]):
                if is_finished[i]:
                    continue
                if len(self.cur_ngram_list[i]) >= self.ngram - 1:
                    k = tuple(self.cur_ngram_list[i][-self.ngram + 1:])
                    if k in self.ngram_stat_list[i]:
                        for v in self.ngram_stat_list[i][k]:
                            logits[i][v] -= 1e9
            return logits

        prog = static.default_main_program()
        new_logits = prog.current_block().create_var(name="out_logits", dtype=logits.dtype, shape=logits.shape)
        static.py_func(func=__wrapper__, x=(logits, is_finished), out=new_logits)
        return new_logits

    def update(self, pred, is_finished, parent_idx=None):
        """Update N-gram blocking strategy related data."""
        def __gather__(parent_idx):
            parent_idx = np.array(parent_idx) # shape: [B]
            new_ngram_stat_list, new_cur_ngram_list = [], []
            for idx in parent_idx:
                new_ngram_stat_list.append(copy.deepcopy(self.ngram_stat_list[idx]))
                new_cur_ngram_list.append(copy.deepcopy(self.cur_ngram_list[idx]))
            self.ngram_stat_list, self.cur_ngram_list = new_ngram_stat_list, new_cur_ngram_list

        def __wrapper__(pred, is_finished):
            pred = np.array(pred) # shape: [B, 1]
            is_finished = np.array(is_finished) # shape: [B, 1]
            assert(len(self.ngram_stat_list) == len(self.cur_ngram_list) == pred.shape[0] == is_finished.shape[0])
            for ngram_stat, cur_ngram, x, flag in zip(
                self.ngram_stat_list, self.cur_ngram_list, pred[:, 0], is_finished[:, 0]
            ):
                if flag:
                    continue
                cur_ngram.append(x)
                if len(cur_ngram) >= self.ngram:
                    k = tuple(cur_ngram[-self.ngram:-1])
                    ngram_stat[k].add(x)

        if parent_idx is not None:
            static.py_func(func=__gather__, x=parent_idx, out=None)
        static.py_func(func=__wrapper__, x=(pred, is_finished), out=None)
