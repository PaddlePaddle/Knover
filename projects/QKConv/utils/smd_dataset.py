#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import string

import paddle
from paddlenlp.metrics import BLEU
from tqdm import tqdm, trange

from .base_dataset import BaseDataset
from .metrics import bleu_score, entityf1_score

class SMDDataset(BaseDataset):
    """SMD dataset"""

    def __init__(self, data_path, entity_file=None, **kwargs):
        super().__init__(data_path)
        
        if "dialogue" in self.data[0]:
            data = []
            for dial in self.data:
                context = []
                for turn in dial["dialogue"]:
                    utt = self.preprocess_text(turn["data"]["utterance"])
                    if turn["turn"] == "assistant":
                        item = {}
                        item["context"] = context.copy()
                        item["response"] = turn["data"]["utterance"].lower()
                        item["scenario"] = dial["scenario"]
                        data.append(item)
                    context.append(utt)
            self.data = data
        self.entity_file = entity_file

    def knowledge_selection(self):
        """Retrieve relevant knowledge by generated query."""
        import rocketqa
        model = rocketqa.load_model(model="v2_marco_ce", use_cuda=True, device_id=0, batch_size=128)
        for dial in tqdm(self.data, desc="Retrieval"):
            fields = dial["scenario"]["kb"]["column_names"]
            knowledge_records = dial["scenario"]["kb"]["items"] or []
            knowledge_seqs = [self.linearize_knowledge_record(k, fields) for k in knowledge_records]

            if len(knowledge_records) == 0:
                dial["selected_knowledge"] = []
            else:
                queries = [dial["generated_query"]] * len(knowledge_records)
                scores = model.matching(query=queries, para=knowledge_seqs)
                _, sorted_records = zip(*sorted(zip(scores, knowledge_records), key=lambda x: x[0], reverse=True))
                dial["selected_knowledge"] = list(sorted_records)[:3]

        # RocketQA uses PaddlePaddle static mode but PaddleNLP uses dynamic mode
        paddle.disable_static()
        return

    def generate_response_input(self):
        """Generate system response by dialogue context and retrieved knowledge."""
        # prepare input samples
        samples = []
        for dial in self.data:
            fields = dial["scenario"]["kb"]["column_names"]
            context = dial["context"]

            retrieved_knowledge_seq = self.linearize_knowledge(dial["selected_knowledge"], fields)
            src = "generate system response based on knowledge and dialogue context : knowledge : " + \
                retrieved_knowledge_seq + " ; dialogue context : " + " | ".join(context)
            samples.append(src)
        return samples

    def preprocess_text(self, text):
        """Preprocess utterance and table value."""
        text = text.strip().replace("\t", " ").lower()
        for p in string.punctuation:
            text = text.replace(p, f" {p} ")
        text = " ".join(text.split())
        return text

    def linearize_knowledge_record(self, knowledge_record, fields):
        """Convert a knowledge record into a flatten sequence with special symbols."""
        knowledge_seq = []
        for f in fields:
            value = self.preprocess_text(str(knowledge_record.get(f, "")))
            knowledge_seq.append(f.replace("_", " ") + " : " + value)
        return " | ".join(knowledge_seq)

    def linearize_knowledge(self, knowledge, fields):
        """Convert knowledge into a flatten sequecen with special symbols."""
        knowledge_seq = []
        knowledge_seq.append("col : " + " | ".join(map(lambda x: x.replace("_", " "), fields)))
        for idx, record in enumerate(knowledge):
            values = []
            for f in fields:
                v = self.preprocess_text(str(record.get(f, "")))
                values.append(v)

            record_seq = " | ".join(values)
            knowledge_seq.append(f"row {idx} : {record_seq}")
        return " || ".join(knowledge_seq)

    def evaluate(self, preds, refs):
        """Evaluate response generation and knowledge selection"""
        bleu = bleu_score(preds, refs)
        entity_f1 = entityf1_score(preds, refs, self.entity_file)
        results = {
            "BLEU": bleu,
            "Entity-F1": entity_f1
        }
        return results