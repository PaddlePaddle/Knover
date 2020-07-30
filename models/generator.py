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
"""Generator class"""

import numpy as np
import paddle.fluid.layers as layers

from utils.args import str2bool


class Generator(object):
    """
    Generator class

    Use generator in inference phase.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = parser.add_argument_group("Generator")
        group.add_argument("--min_dec_len", type=int, default=1)
        group.add_argument("--max_dec_len", type=int, default=64)

        group.add_argument("--decoding_strategy", type=str, default="topk_sampling",
                           choices=["beam_search", "topk_sampling", "topp_sampling"])
        group.add_argument("--temperature", type=float, default=1.)
        group.add_argument("--ignore_unk", type=str2bool, default=True)

        # multi sampling
        group.add_argument("--num_samples", type=int, default=None)

        # top-k sampling
        group.add_argument("--topk", type=int, default=10)

        # top-p sampling
        group.add_argument("--topp", type=float, default=0.9)

        # beam search
        group.add_argument("--beam_size", type=int, default=10)
        group.add_argument("--length_average", type=str2bool, default=True)
        group.add_argument("--length_penalty", type=float, default=0.0)

        return group

    def __init__(self, args):
        self.min_dec_len = args.min_dec_len
        self.max_dec_len = args.max_dec_len
        self.eos_id = args.eos_id
        self.unk_id = args.unk_id
        self.mask_id = args.mask_id
        self.vocab_size = args.vocab_size

        # model related

        # basic settings
        self.decoding_strategy = args.decoding_strategy
        self.ignore_unk = args.ignore_unk
        self.continuous_position = args.continuous_position
        self.temperature = args.temperature

        # reranking
        self.num_samples = args.num_samples

        # top-k sampling
        self.topk = args.topk

        # top-p sampling
        self.topp = args.topp

        # beam search
        self.beam_size = args.beam_size
        self.length_penalty = args.length_penalty
        self.length_average = args.length_average
        return

    def inference(self, model, inputs, outputs):
        """
        Run inference.

        Args:
            inputs(dict): Its key is input name(str) and its value is a Variable.
            model(object): A generate model. Need to implement `_generation_network` and `_calc_logits`.

        Returns:
            dict(str:Variable): Its key is output name(str) and its value is a Variable.
        """
        # prepare while loop
        max_len = layers.fill_constant(
            shape=[1], dtype="int64", value=self.max_dec_len, force_cpu=True)
        min_len = layers.fill_constant(
            shape=[1], dtype="int64", value=self.min_dec_len, force_cpu=True)
        step_idx = layers.fill_constant(
            shape=[1], dtype="int64", value=0, force_cpu=True)

        ids = layers.array_write(layers.reshape(inputs["tgt_ids"], (-1, 1)), step_idx)
        pos_biases = layers.array_write(layers.reshape(inputs["tgt_pos"], (-1, 1)), step_idx)
        scores = layers.array_write(inputs["init_score"], step_idx)
        tgt_generation_mask = layers.array_write(inputs["tgt_generation_mask"], step_idx)
        parent_idx = inputs["parent_idx"]

        if self.decoding_strategy == "beam_search":
            beam_size = self.beam_size
        else:
            beam_size = 1

        eos_penalty = np.zeros(self.vocab_size, dtype="float32")
        eos_penalty[self.eos_id] = -1e9
        eos_penalty = layers.assign(eos_penalty)

        token_penalty = np.zeros(self.vocab_size, dtype="float32")
        token_penalty[self.unk_id] = -1e9
        if self.mask_id >= 0:
            token_penalty[self.mask_id] = -1e9
        token_penalty = layers.assign(token_penalty)

        # start while loop
        cond = layers.less_than(x=step_idx, y=max_len)
        while_op = layers.While(cond)
        with while_op.block():
            pre_ids = layers.array_read(array=ids, i=step_idx)
            pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
            pre_scores = layers.array_read(array=scores, i=step_idx)
            pos_bias = layers.array_read(array=pos_biases, i=step_idx)
            pos_bias = layers.gather(input=pos_bias, index=parent_idx)

            tmp_tgt_generation_mask = layers.array_read(tgt_generation_mask, i=step_idx)
            dtype = tmp_tgt_generation_mask.dtype

            append_mask = layers.fill_constant_batch_size_like(
                    input=pre_ids,
                    value=1.0,
                    shape=[-1, 1, 1],
                    dtype=dtype)
            tmp_tgt_generation_mask = layers.concat([tmp_tgt_generation_mask, append_mask], axis=2)
            pre_mask = tmp_tgt_generation_mask = layers.gather(input=tmp_tgt_generation_mask, index=parent_idx)

            pre_sent = layers.fill_constant_batch_size_like(
                    input=pre_mask,
                    value=1,
                    shape=[-1, 1, 1],
                    dtype=pre_ids.dtype)

            if self.continuous_position:
                pre_pos = layers.elementwise_mul(
                    x=layers.fill_constant_batch_size_like(
                        input=pre_mask,
                        value=1,
                        shape=[-1, 1, 1],
                        dtype=pre_ids.dtype), y=step_idx, axis=0) + pos_bias
            else:
                pre_pos = layers.elementwise_mul(
                    x=layers.fill_constant_batch_size_like(
                        input=pre_mask,
                        value=1,
                        shape=[-1, 1, 1],
                        dtype=pre_ids.dtype), y=step_idx, axis=0)

            dec_out, _ = model._generation_network(
                token_ids=pre_ids,
                type_ids=pre_sent,
                pos_ids=pre_pos,
                generation_mask=tmp_tgt_generation_mask,
                gather_idx=parent_idx)
            logits = model._calc_logits(dec_out)

            # ignore unk and mask token
            if self.ignore_unk:
                logits = layers.elementwise_add(logits, token_penalty, axis=1)

            # min dec length
            min_len_cond = layers.less_than(x=step_idx, y=min_len)
            def min_len_penalty():
                """Plus minimum length penalty."""
                return layers.elementwise_add(logits, eos_penalty, axis=1)
            def no_penalty():
                """No penalty."""
                return logits
            logits = layers.case([(min_len_cond, min_len_penalty)], default=no_penalty)

            # get probs
            probs = layers.softmax(logits / self.temperature)

            if self.decoding_strategy == "beam_search":
                topk_scores, topk_indices = layers.topk(
                    input=probs, k=beam_size)
            else:
                if self.decoding_strategy.startswith("sampling"):
                    sampling_ids = layers.sampling_id(probs, dtype="int")
                elif self.decoding_strategy.startswith("topk_sampling"):
                    topk_probs, _ = layers.topk(input=probs, k=self.topk)
                    ge_cond = layers.cast(
                        layers.greater_equal(
                            probs,
                            layers.unsqueeze(topk_probs[:, -1], [1])),
                        "float32")
                    old_probs = probs
                    probs = probs * ge_cond / layers.reduce_sum(topk_probs, dim=-1, keep_dim=True)
                    sampling_ids = layers.sampling_id(probs, dtype="int")
                    probs = old_probs
                elif self.decoding_strategy.startswith("topp_sampling"):
                    sorted_probs, sorted_idx = layers.argsort(probs, descending=True)
                    cum_sorted_probs = layers.cumsum(sorted_probs, axis=1, exclusive=True)
                    lt_cond = layers.cast(
                        layers.less_than(
                            cum_sorted_probs,
                            layers.fill_constant_batch_size_like(
                                cum_sorted_probs,
                                cum_sorted_probs.shape,
                                cum_sorted_probs.dtype,
                                self.topp)
                        ),
                        "float32"
                    )
                    old_probs = probs
                    candidate_probs = sorted_probs * lt_cond
                    probs = candidate_probs / layers.reduce_sum(candidate_probs, dim=-1, keep_dim=True)
                    sampling_ids = layers.sampling_id(probs, dtype="int")
                    sampling_ids = layers.index_sample(sorted_idx, layers.unsqueeze(sampling_ids, [1]))
                    sampling_ids = layers.squeeze(sampling_ids, [1])
                    probs = old_probs
                else:
                    raise ValueError(self.decoding_strategy)

                sampling_scores = layers.one_hot(
                    layers.unsqueeze(sampling_ids, [1]), probs.shape[1]
                )
                sampling_scores = sampling_scores * probs - (1 - sampling_scores) * 1e3
                topk_scores, topk_indices = layers.topk(
                    input=sampling_scores, k=1)

            pre_len = layers.cast(step_idx, "float32")
            layers.increment(x=step_idx, value=1.0, in_place=True)
            cur_len = layers.cast(step_idx, "float32")

            # update scores
            if self.length_average:
                accu_scores = layers.elementwise_add(
                    x=layers.log(topk_scores), y=pre_scores * pre_len, axis=0) / cur_len
            elif self.length_penalty > 0:
                pre_lp = layers.pow((5 + pre_len) / 6, self.length_penalty)
                cur_lp = layers.pow((5 + cur_len) / 6, self.length_penalty)
                accu_scores = layers.elementwise_add(
                    x=layers.log(topk_scores), y=pre_scores * pre_lp, axis=0) / cur_lp
            else:
                accu_scores = layers.elementwise_add(
                    x=layers.log(topk_scores), y=pre_scores, axis=0)
            topk_indices = layers.lod_reset(topk_indices, pre_ids)
            accu_scores = layers.lod_reset(accu_scores, pre_ids)
            selected_ids, selected_scores, gather_idx = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=beam_size,
                end_id=self.eos_id,
                return_parent_idx=True)

            layers.array_write(selected_ids, i=step_idx, array=ids)
            layers.array_write(selected_scores, i=step_idx, array=scores)
            layers.array_write(pre_mask, i=step_idx, array=tgt_generation_mask)
            layers.array_write(pos_bias, i=step_idx, array=pos_biases)

            layers.assign(gather_idx, parent_idx)

            length_cond = layers.less_than(x=step_idx, y=max_len)
            finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
            layers.logical_and(x=length_cond, y=finish_cond, out=cond)

        finished_ids, finished_scores = layers.beam_search_decode(
            ids, scores, beam_size=beam_size, end_id=self.eos_id)

        predictions = {
            "finished_ids": finished_ids,
            "finished_scores": finished_scores,
            "token_ids": inputs["token_ids"],
            "data_id": inputs["data_id"]
        }
        return predictions
