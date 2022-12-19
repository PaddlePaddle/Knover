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

import jsonlines
import paddle
from tqdm import tqdm, trange
from drqa import retriever

from .base_dataset import BaseDataset
from .metrics import f1_score, rougel_score, recall_score

class KILTWoWDataset(BaseDataset):
    """KILT WoW dataset"""

    def __init__(self, data_path, **kwargs):
        if "jsonl" in data_path:
            with jsonlines.open(data_path, "r") as reader:
                print(f"Load inference file from: {data_path}")
                self.data = []
                for dial in reader:
                    item = {}
                    item["id"] = dial["id"]
                    item["context"] = dial["input"].split("\n")
                    item["response"] = self.preprocess_text(dial["output"][0]["answer"])
                    item["gold_knowledge"] = [p["wikipedia_id"] for p in dial["output"][0]["provenance"]]
                    item["gold_knowledge2"] = [p["start_character"] for p in dial["output"][0]["provenance"]]
                    self.data.append(item)
        else:
            super().__init__(data_path)

    def knowledge_selection(self):
        """Retrieve relevant knowledge by generated query."""
        import rocketqa
        wiki_dict = {}
        with jsonlines.open("models/kilt_kb.jsonl", "r") as reader:
            for jline in reader:
                wiki_dict[jline["wikipedia_id"]] = jline["text"][1:3]

        searcher = retriever.get_class("tfidf")(tfidf_path="models/kilt_db_simple.npz")
        model = rocketqa.load_model(model="v2_marco_ce", use_cuda=True, device_id=0, batch_size=128)

        for dial in tqdm(self.data, desc="Retrieval"):
            knowledge_id, retrieve_score = searcher.closest_docs(dial["generated_query"], 50)

            knowledge_seqs = []
            knowledge_ids = []
            retrieve_scores = []
            for docid, s in zip(knowledge_id, retrieve_score):
                for i, doc in enumerate(wiki_dict[docid]):
                    retrieve_scores.append(s)
                    knowledge_seqs.append(doc)
                    knowledge_ids.append(docid + f"_{1+i}")
            retrieve_scores = self.normalize_retrieve_scores(retrieve_scores)

            if len(knowledge_seqs) == 0:
                dial["selected_knowledge"] = []
            else:
                queries = [dial["generated_query"]] * len(knowledge_seqs)
                rerank_scores = model.matching(query=queries, para=knowledge_seqs)
                _, _, sorted_records, sorted_id = zip(*sorted(
                    zip(rerank_scores, retrieve_scores, knowledge_seqs, knowledge_ids),
                    key=lambda x: x[0] + x[1], reverse=True
                ))
                dial["selected_knowledge_id"] = list(sorted_id)[:1]
                dial["selected_knowledge_id2"] = [pid.split("_")[1] for pid in dial["selected_knowledge_id"]]
                dial["selected_knowledge_id"] = [pid.split("_")[0] for pid in dial["selected_knowledge_id"]]
                dial["selected_knowledge"] = list(sorted_records)[:1]

        # RocketQA uses PaddlePaddle static mode but PaddleNLP uses dynamic mode
        paddle.disable_static()
        return
    
    def evaluate(self, preds, refs):
        """Evaluate response generation and knowledge selection"""
        rouge = rougel_score(preds, refs, avg=False)
        f1 = f1_score(preds, refs, avg=False)
        recall = recall_score(self.data, avg=False)
        kilt_rouge = sum([x * y for x, y in zip(rouge, recall)]) / len(recall)
        kilt_f1 = sum([x * y for x, y in zip(f1, recall)]) / len(recall)
        results = {
            "Rouge-L": sum(rouge) / len(rouge),
            "F1": sum(f1) / len(f1),
            "Recall@1": sum(recall) / len(recall),
            "KILT-Rouge-L": kilt_rouge,
            "KILT-F1": kilt_f1
        }
        return results
