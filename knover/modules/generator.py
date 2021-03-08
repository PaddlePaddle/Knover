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

import math

import numpy as np
import paddle
import paddle.nn.functional as F

from knover.utils import str2bool
from knover.utils import gather, repeat


GENERATOR_REGISTRY = {}


def register_generator(name):
    """Register a new model class."""

    def __wrapped__(cls):
        if name in GENERATOR_REGISTRY:
            raise ValueError(f"Cannot register duplicate generator ({name})")
        if not issubclass(cls, Generator):
            raise ValueError(f"Generator ({name}: {cls.__name__}) must extend Generator")
        GENERATOR_REGISTRY[name] = cls
        return cls

    return __wrapped__


class Generator(object):
    """Generator class.

    Use generator in generation phase.
    """

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = parser.add_argument_group("Generator")
        group.add_argument("--decoding_strategy", type=str, default="sampling",
                           choices=list(GENERATOR_REGISTRY.keys()),
                           help="The decoding strategy.")
        group.add_argument("--min_dec_len", type=int, default=1,
                           help="The minimum length of decoded sequence.")
        group.add_argument("--max_dec_len", type=int, default=64,
                           help="The maximum length of decoded sequence.")
        group.add_argument("--temperature", type=float, default=1.,
                           help="The temperature in each generation step.")
        group.add_argument("--ignore_unk", type=str2bool, default=True,
                           help="Whether to ignore UNK token in generation.")
        group.add_argument("--length_average", type=str2bool, default=True,
                           help="The hyper-parameter in beam search.")
        group.add_argument("--length_penalty", type=float, default=None,
                           help="The hyper-parameter in beam search.")

        args, _ = parser.parse_known_args()
        GENERATOR_REGISTRY[args.decoding_strategy].add_cmdline_args(parser)
        return group

    def __init__(self, args):
        self.bos_id = args.bos_id
        self.eos_id = args.eos_id
        self.unk_id = args.unk_id
        self.mask_id = args.mask_id
        self.vocab_size = args.vocab_size
        self.min_dec_len = args.min_dec_len
        self.max_dec_len = args.max_dec_len
        self.temperature = args.temperature
        self.ignore_unk = args.ignore_unk
        assert 1 <= self.min_dec_len <= self.max_dec_len

        self.length_average = args.length_average
        self.length_penalty = args.length_penalty
        if self.length_average and self.length_penalty is not None:
            print(f"[WARMING] Using length_average only and ignore length_penalty settings ({self.length_penalty}).")

        # model related
        self.use_role = args.use_role

    def _initialize_state(self, model, inputs, outputs):
        """Initialize the state of decoding algorithm."""
        state = {}

        # model related input
        bsz = inputs["tgt_ids"].shape[0]
        state["token_ids"] = inputs["tgt_ids"]
        state["type_ids"] = paddle.full([bsz, 1], 1, dtype="int64")
        state["pos_ids"] = inputs["tgt_pos"]
        if self.use_role:
            state["role_ids"] = paddle.full([bsz, 1], 0, dtype="int64")
        state["tgt_generation_mask"] = paddle.concat(
            [inputs["tgt_generation_mask"], paddle.full([bsz, 1, 1], 1.0)], axis=2)

        state["parent_idx"] = paddle.arange(bsz, dtype="int32")
        state["score"] = paddle.zeros([bsz, 1], dtype="float32")
        state["predictions"] = state["token_ids"]
        state["is_finished"] = paddle.full([bsz, 1], 0, dtype="bool")
        state["step"] = 0
        state["batch_size"] = bsz
        return state

    def _update_state(self, state, probs):
        """Update decoding state after one step prediction.

        Main implement of decoding strategy.
        """
        raise NotImplementedError

    def _process_final_state(self, state):
        return state

    def __call__(self, model, inputs, outputs):
        """Run generation.

        Args:
            model: A generation model. Need to implement `_generation_step` which inputs in a state,
                and return stepwise logits.
            inputs: A dict mapping input variable names to corresponding Variables.
            outputs: A dict mapping output variable name to corresponding Variables.

        Returns:
            predictions: A dict mapping keys to corresponding predictions.
        """
        self.logits_after_finished = paddle.zeros([1, self.vocab_size])
        self.logits_after_finished[:, self.eos_id] = 1e9

        state = self._initialize_state(model, inputs, outputs)
        while state["step"] < self.max_dec_len:
            # calculate generation probability distribution.
            logits = model._generation_step(state)

            # pre-process logits
            logits = logits / self.temperature
            logits[:, self.bos_id] = -1e9
            logits[:, self.mask_id] = -1e9
            if self.ignore_unk:
                logits[:, self.unk_id] = -1e9
            if state["step"] < self.min_dec_len:
                logits[:, self.eos_id] = -1e9
            # force the predicted id is [EOS] when sequence is completed.
            is_finished = state["is_finished"]
            logits = paddle.where(
                paddle.expand_as(is_finished, logits),
                paddle.expand_as(self.logits_after_finished, logits),
                logits
            )

            # update state
            probs = F.softmax(logits)
            state = self._update_state(state, probs)

            # all sequences are completed.
            if paddle.all(state["is_finished"]):
                break
            state["step"] += 1

        state = self._process_final_state(state)
        predictions = {
            "finished_ids": state["predictions"],
            "finished_score": state["score"][:, 0],
            "token_ids": inputs["token_ids"],
            "data_id": inputs["data_id"]
        }
        return predictions


@register_generator("sampling")
class Sampling(Generator):
    """Naive Sampling."""

    @classmethod
    def add_cmdline_args(cls, group):
        """Add cmdline arguments."""
        group.add_argument("--num_samples", type=int, default=1,
                           help="The number of candidates will be generated.")
        return group

    def __init__(self, args):
        super(Sampling, self).__init__(args)
        self.num_samples = args.num_samples

    def _sampling(self, probs):
        """Sampling function.

        Args:
            probs: the probability distribution of next token, shape is [batch_size, vocab_size].

        Returns:
            pred: the sampled next token, shape is [batch_size, 1].
        """
        return paddle.multinomial(probs)

    def _initialize_state(self, model, inputs, outputs):
        """Initialize the state of decoding algorithm."""
        state = super(Sampling, self)._initialize_state(model, inputs, outputs)
        state = repeat(state, self.num_samples)
        inputs["token_ids"] = repeat(inputs["token_ids"], self.num_samples)
        inputs["data_id"] = repeat(inputs["data_id"], self.num_samples)
        return state

    def _update_state(self, state, probs):
        step = state["step"]
        is_finished = state["is_finished"]

        pred = self._sampling(probs)
        probs = paddle.sum(F.one_hot(paddle.squeeze(pred, [1]), self.vocab_size) * probs, axis=1, keepdim=True)

        # model related input
        state["token_ids"] = pred
        state["pos_ids"] = state["pos_ids"] + 1
        state["tgt_generation_mask"] = paddle.concat([
            state["tgt_generation_mask"],
            paddle.unsqueeze(1 - paddle.cast(is_finished, "float32"), axis=[2]),
        ], axis=2)

        if self.length_average:
            next_score = (paddle.log(probs) + state["score"] * step) / (1 + step)
        elif self.length_penalty is not None:
            pre_w = math.pow((5 + step) / 6, self.length_penalty)
            cur_w = math.pow((6 + step) / 6, self.length_penalty)
            next_score = (paddle.log(probs) + state["score"] * pre_w) / cur_w
        else:
            next_score = paddle.log(probs) + state["score"]

        # keep finised sequences' score
        state["score"] = paddle.where(is_finished, state["score"], next_score)

        state["predictions"] = paddle.concat([state["predictions"], pred], axis=1)
        state["is_finished"] = pred == self.eos_id
        # Pop parent_idx !!!
        state.pop("parent_idx", None)
        return state


@register_generator("topk_sampling")
class TopkSampling(Sampling):

    @classmethod
    def add_cmdline_args(cls, group):
        """Add cmdline arguments."""
        Sampling.add_cmdline_args(group)
        # top-k sampling
        group.add_argument("--topk", type=int, default=10,
                           help="The hyper-parameter in top-k sampling..")
        return group

    def __init__(self, args):
        super(TopkSampling, self).__init__(args)
        self.topk = args.topk

    def _sampling(self, probs):
        """Top-k sampling function.

        Args:
            probs: the probability distribution of next token, shape is [batch_size, vocab_size].

        Returns:
            pred: the sampled next token from the top-k probabilities, shape is [batch_size, 1].
        """
        topk_probs, _ = paddle.topk(probs, k=self.topk)
        ge_cond = paddle.cast(probs >= topk_probs[:, -1:], "float32")
        probs = probs * ge_cond / paddle.sum(topk_probs, axis=-1, keepdim=True)
        return paddle.multinomial(probs)


@register_generator("topp_sampling")
class ToppSampling(Sampling):

    @classmethod
    def add_cmdline_args(cls, group):
        """Add cmdline arguments."""
        Sampling.add_cmdline_args(group)
        # top-p sampling
        group.add_argument("--topp", type=float, default=0.9,
                           help="The hyper-parameter in top-p sampling.")
        return group

    def __init__(self, args):
        super(ToppSampling, self).__init__(args)
        self.topp = args.topp

    def _sampling(self, probs):
        """Top-p sampling function.

        Args:
            probs: the probability distribution of next token, shape is [batch_size, vocab_size].

        Returns:
            pred: the sampled next token from the top-p probabilities, shape is [batch_size, 1].
        """
        sorted_idx = paddle.argsort(probs, descending=True)
        sorted_probs = paddle.index_sample(probs, sorted_idx)
        # exclusive cumsum
        cum_sorted_probs = paddle.cumsum(sorted_probs, axis=1)
        cum_sorted_probs[:, 1:] = cum_sorted_probs[:, :-1].clone()
        cum_sorted_probs[:, 0] = 0
        lt_cond = paddle.cast(cum_sorted_probs < self.topp, "float32")
        candidate_probs = sorted_probs * lt_cond
        probs = candidate_probs / paddle.sum(candidate_probs, axis=-1, keepdim=True)
        sampled_id = paddle.multinomial(probs)
        return paddle.index_sample(sorted_idx, sampled_id)


@register_generator("beam_search")
class BeamSearch(Generator):

    @classmethod
    def add_cmdline_args(cls, group):
        """Add cmdline arguments."""
        group.add_argument("--beam_size", type=int, default=10,
                           help="The hyper-parameter in beam search.")
        return group

    def __init__(self, args):
        super(BeamSearch, self).__init__(args)
        self.beam_size = args.beam_size
        return

    def _update_state(self, state, probs):
        bsz = state["batch_size"]
        step = state["step"]
        is_finished = state["is_finished"]

        # calcuate next scores
        if self.length_average:
            next_scores = (paddle.log(probs) + paddle.unsqueeze(state["score"], [1]) * step) / (1 + step)
        elif self.length_penalty is not None:
            pre_w = math.pow((5 + step) / 6, self.length_penalty)
            cur_w = math.pow((6 + step) / 6, self.length_penalty)
            next_scores = (paddle.log(probs) + paddle.unsqueeze(state["score"], [1]) * pre_w) / cur_w
        else:
            next_scores = paddle.log(probs) + state["score"]

        # keep finished sequences' score
        next_score = paddle.where(
            paddle.expand_as(is_finished, next_score),
            paddle.expand_as(state["score"], next_scores) + paddle.expand_as(self.logits_after_finished - 1e9, next_scores),
            next_scores,
        )

        if step > 0:
            next_scores = paddle.reshape(next_scores, [-1, self.beam_size * self.vocab_size])
            num_beams = self.beam_size
        else:
            num_beams = 1

        topk_scores, topk_indices = paddle.topk(next_scores, self.beam_size)
        beam_idx = topk_indices // self.vocab_size
        pred = topk_indices % self.vocab_size

        topk_scores = paddle.reshape(topk_scores, [-1])
        pred = paddle.reshape(pred, [-1])
        pred = paddle.cast(pred, "int")
        pred = paddle.unsqueeze(pred, [1])

        parent_idx = paddle.unsqueeze(paddle.arange(bsz, dtype="int64"), [1]) * num_beams + \
            paddle.unsqueeze(beam_idx, [0])
        parent_idx = paddle.reshape(parent_idx, [-1])

        # model related input
        state = gather(state, parent_idx)
        state["token_ids"] = pred
        state["pos_ids"] = state["pos_ids"] + 1
        state["tgt_generation_mask"] = paddle.concat([
            state["tgt_generation_mask"],
            paddle.unsqueeze(1 - paddle.cast(is_finished, "float32"), axis=[2])
        ], axis=2)

        state["score"] = topk_scores
        state["predictions"] = paddle.concat([state["predictions"], pred], axis=1)
        state["is_finished"] = pred == self.eos_id
        state["parent_idx"] = parent_idx
        return state

    def _process_final_state(self, state):
        bsz = state["batch_size"]
        # only return the best sequence
        state["predictions"] = paddle.reshape(state["predictions"], [bsz, self.beam_size, -1])
        state["score"] = paddle.reshape(state["score"], [bsz, self.beam_size])
        return state
