#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

"""Main inference program."""

import argparse
import json
import os

import paddle
from drqa import retriever
from paddlenlp.transformers import generation_utils, AutoTokenizer, AutoModelForConditionalGeneration
from tqdm import tqdm, trange

from utils import DATASET_ZOO


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["smd", "qrecc", "wow"])
    parser.add_argument("--infer_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--save_file", type=str, required=True)
    args = parser.parse_args()
    return args


def infer(args):
    """Main inference function."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForConditionalGeneration.from_pretrained(args.model_path)
    model.eval()
    model.to("gpu:0")
    print(f"Load model from: {args.model_path}")
    dataset = DATASET_ZOO[args.dataset](args.infer_file)

    query_generation(dataset, tokenizer, model, args.batch_size, args.beam_size)
    dataset.knowledge_selection()
    response_generation(dataset, tokenizer, model, args.batch_size, args.beam_size)

    if not os.path.exists(os.path.dirname(args.save_file)):
        os.makedirs(os.path.dirname(args.save_file))
    with open(args.save_file, "w") as fout:
        json.dump(dataset.data, fout, indent=2)
        print(f"Save inference output to: {args.save_file}")
    return


def query_generation(dataset, tokenizer, model, batch_size, beam_size):
    """Generate query by dialogue context."""
    # prepare input samples
    samples = dataset.generate_query_input()

    # call Q-TOD model
    generated_queries = generate(samples, tokenizer, model, batch_size, beam_size)

    # save generated query into data
    for dial, query in zip(dataset.data, generated_queries):
        dial["generated_query"] = query
    return


def response_generation(dataset, tokenizer, model, batch_size, beam_size):
    """Generate system response by dialogue context and retrieved knowledge."""
    # prepare input samples
    samples = dataset.generate_response_input()

    # call Q-TOD model
    generated_responses = generate(samples, tokenizer, model, batch_size, beam_size)

    # save generated response into data
    for dial, response in zip(dataset.data, generated_responses):
        dial["generated_response"] = response
    return


@paddle.no_grad()
def generate(samples, tokenizer, model, batch_size, beam_size):
    """Call Q-TOD model generation."""
    outputs = []
    for idx in trange(0, len(samples), batch_size, desc="Generation"):
        batch = samples[idx: idx + batch_size]
        tokenized_batch = tokenizer(
            batch,
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pd"
        )
        bart_generate_kwargs = {
            "no_repeat_ngram_size": 3,
            "forced_bos_token_id": 0,
            "early_stopping": True
        }
        batch_out, _ = model.generate(
            tokenized_batch["input_ids"],
            decode_strategy="beam_search",
            num_beams=beam_size,
            max_length=128,
            length_penalty=1,
            attention_mask=tokenized_batch.get("attention_mask"),
            no_repeat_ngram_size=3 if "bart" in model.full_name() else None,
            forced_bos_token_id=0 if "bart" in model.full_name() else None,
            early_stopping=True if "bart" in model.full_name() else None
        )
        batch_pred = tokenizer.batch_decode(batch_out, skip_special_tokens=True)

        outputs.extend(batch_pred)
    return outputs


class FixedBeamHypotheses:
    """Fix length penalty difference."""

    def __init__(self, num_beams, length_penalty, early_stopping):
        """Initialize n-best list of hypotheses."""
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """Number of hypotheses in the list."""
        return len(self.beams)

    def add(self, hyp, sum_logprobs, origin_len=0):
        """Add a new hypothesis to the list."""
        score = sum_logprobs / hyp.shape[-1] ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len, origin_len=0):
        """
        If there are enough hypotheses and that none of the hypotheses being 
        generated can become better than the worst one in the heap, then we 
        are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / (cur_len ** self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret


if __name__ == "__main__":
    generation_utils.BeamHypotheses = FixedBeamHypotheses
    args = setup_args()
    infer(args)