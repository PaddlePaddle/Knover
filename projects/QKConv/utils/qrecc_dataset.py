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

import numpy as np
from pyserini.search.lucene import LuceneSearcher
import paddle
from tqdm import tqdm, trange

from .base_dataset import BaseDataset
from .metrics import f1_score, exact_score, recall_score


class QReCCDataset(BaseDataset):
    """QReCC dataset"""

    def __init__(self, data_path, **kwargs):
        super().__init__(data_path)
        if "Context" in self.data[0]:
            data = []
            for dial in self.data:
                item = {}
                item["context"] = dial["Context"] + [dial["Question"]]
                item["response"] = self.preprocess_text(dial["Answer"])
                item["gold_knowledge"] = dial["Passages"]
                data.append(item)
            self.data = data

    def knowledge_selection(self):
        """Retrieve relevant knowledge by generated query."""
        import rocketqa
        searcher = LuceneSearcher('models/index-paragraph')
        model = rocketqa.load_model(model="v2_marco_ce", use_cuda=True, device_id=0, batch_size=128)

        for dial in tqdm(self.data, desc="Retrieval"):
            hits = searcher.search(dial["generated_query"], k=50)

            knowledge_seqs = []
            knowledge_ids = []
            retrieve_scores = []
            for i in range(len(hits)):
                retrieve_scores.append(hits[i].score)
                knowledge_seqs.append(hits[i].lucene_document.toString().split("\n")[2][16:-1])
                knowledge_ids.append(hits[i].docid)
            retrieve_scores = self.normalize_retrieve_scores(retrieve_scores)

            if len(knowledge_seqs) == 0:
                dial["selected_knowledge"] = []
            else:
                queries = [dial["generated_query"]] * len(knowledge_seqs)
                rerank_scores = model.matching(query=queries, para=knowledge_seqs)
                _, _, sorted_records, sorted_ids = zip(*sorted(
                    zip(rerank_scores, retrieve_scores, knowledge_seqs, knowledge_ids),
                    key=lambda x: x[0] + x[1], reverse=True
                ))
                dial["selected_knowledge_id"] = list(sorted_ids)[:1]
                dial["selected_knowledge"] = list(sorted_records)[:1]

        # RocketQA uses PaddlePaddle static mode but PaddleNLP uses dynamic mode
        paddle.disable_static()
        return

    def preprocess_text(self, text):
        """Preprocess utterance and table value."""
        text = text.lower()
        return text

    def evaluate(self, preds, refs):
        """Evaluate response generation and knowledge selection"""
        em = exact_score(preds, refs)
        f1 = f1_score(preds, refs)
        recall = recall_score(self.data)
        results = {
            "EM": em,
            "F1": f1,
            "Recall@1": recall
        }
        return results