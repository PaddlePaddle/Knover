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

"""Basic Dataset"""
import json
import os


class BaseDataset:
    """Basic dataset"""

    def __init__(self, data_path, **kwargs):
        with open(data_path, "r") as fin:
            self.data = json.load(fin)
            print(f"Load inference file from: {data_path}")

    def generate_query_input(self):
        """Generate query by dialogue context.
        
        Returns:
            list: Input texts for query generation
        """
        # prepare input samples
        samples = []
        for dial in self.data:
            context = dial["context"]
            src = "translate dialogue context to query : " + " | ".join(context)
            samples.append(self.preprocess_text(src))
        return samples

    def knowledge_selection(self):
        """Perform knowledge selection

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def generate_response_input(self):
        """Generate system response by dialogue context and retrieved knowledge.

        Returns:
            list: Input texts for response generation
        """
        # prepare input samples
        samples = []
        for dial in self.data:
            context = dial["context"]
            retrieved_knowledge_seq = dial["selected_knowledge"][0].strip() if dial["selected_knowledge"] else ""
            src = "generate system response based on knowledge and dialogue context : knowledge : " + \
                retrieved_knowledge_seq + " ; dialogue context : " + " | ".join(context)
            samples.append(self.preprocess_text(src))
        return samples

    def preprocess_text(self, text):
        """Preprocess utterance or table value.
        
        Returns:
            str: A processed text.
        """
        return text

    def normalize_retrieve_scores(self, retrieve_scores):
        """Normalize retrieve scores
        
        Returns:
            list: Retrieval scores after normalization
        """
        return [s * 0.01 for s in retrieve_scores]

    def evaluate(self, preds, refs):
        """Evaluate response generation and knowledge selection

        Args:
            preds (list): Prediction sentences
            refs (list): Reference sentences

        Raises:
            NotImplementedError: This abstract method needs to be implemented in subclasses.

        Returns:
            dict: metrics dictionary
        """
        raise NotImplementedError