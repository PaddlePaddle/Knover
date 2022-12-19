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

"""Metrics for response generation evaluation and retrival evaluation"""

import json
from paddlenlp.metrics import BLEU
from paddlenlp.metrics.squad import compute_exact, compute_f1
from rouge import Rouge


class EntityMetric(object):
    """Entity-F1 Metric for Response on the SMD dataset."""

    def __init__(self, entity_file):
        self.entities = self._load_entities(entity_file)

    def evaluate(self, preds, refs):
        extracted_preds_entities = []
        extracted_refs_entities = []
        for pred, ref in zip(preds, refs):
            pred_entities = self._extract_entities(pred)
            ref_entities = self._extract_entities(ref)
            extracted_preds_entities.append(pred_entities)
            extracted_refs_entities.append(ref_entities)
        entity_f1 = self._compute_entity_f1(extracted_preds_entities, extracted_refs_entities)
        return entity_f1

    def _load_entities(self, entities_file):
        with open(entities_file, "r") as fin:
            raw_entities = json.load(fin)
        entities = set()

        for slot, values in raw_entities.items():
            for val in values:
                if slot == "poi":
                    entities.add(val["address"].replace(" ", "_"))
                    entities.add(val["poi"].replace(" ", "_"))
                    entities.add(val["type"].replace(" ", "_"))
                elif slot == "distance":
                    entities.add(f"{val} miles")
                elif slot == "temperature":
                    entities.add(f"{val}f")
                else:
                    entities.add(val)

        # add missing entities
        missed_entities = ["yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist",
                            "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                            "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "jill", "jack", " hr "]
        for missed_entity in missed_entities:
            entities.add(missed_entity)
        # special handle of "hr"
        entities.remove("HR")
        entities.add(" HR ")

        processed_entities = []
        for val in entities:
            processed_entities.append(val.lower())
        processed_entities.sort(key=lambda x: len(x), reverse=True)
        return processed_entities

    def _extract_entities(self, response):
        def _is_sub_str(str_list, sub_str):
            for str_item in str_list:
                if sub_str in str_item:
                    return True
            return False

        response = f" {response} ".lower()
        extracted_entities = []

        # preprocess response
        for h in range(0, 13):
            response = response.replace(f"{h} am", f"{h}am")
            response = response.replace(f"{h} pm", f"{h}pm")
        for low_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
            for high_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                response = response.replace(f"{low_temp}-{high_temp}f", f"{low_temp}f-{high_temp}f")

        for entity in self.entities:
            if entity in response and not _is_sub_str(extracted_entities, entity):
                    extracted_entities.append(entity.replace(' ', '_'))

        return list(set(extracted_entities))

    def _compute_entity_f1(self, preds, refs):
        """Compute Entity-F1."""
        def _count(pred, ref):
            tp, fp, fn = 0, 0, 0
            if len(ref) != 0:
                for g in ref:
                    if g in pred:
                        tp += 1
                    else:
                        fn += 1
                for p in set(pred):
                    if p not in ref:
                        fp += 1
            return tp, fp, fn

        tp_all, fp_all, fn_all = 0, 0, 0
        for pred, ref in zip(preds, refs):
            tp, fp, fn = _count(pred, ref)
            tp_all += tp
            fp_all += fp
            fn_all += fn

        precision = tp_all / float(tp_all + fp_all) if (tp_all + fp_all) != 0 else 0
        recall = tp_all / float(tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return f1


def entityf1_score(preds, refs, entity_file):
    """Compute the Entity-F1 score

    Args:
        preds (list[str]): list of prediction
        refs (list[str]): list of reference
        entity_file (str): local path of entity file
    """
    entity_metric = EntityMetric(entity_file)
    entity_f1 = entity_metric.evaluate(preds, refs)
    return entity_f1


def bleu_score(preds, refs):
    """Compute the BLEU-4 score

    Args:
        preds (list[str]): list of prediction
        refs (list[str]): list of reference
    """
    metric = BLEU()

    preds = [pred.strip().lower().split(" ") for pred in preds]
    refs = [[ref.strip().lower().split(" ")] for ref in refs]
    for pred, ref in zip(preds, refs):
        metric.add_inst(pred, ref)
    scores = metric.score()
    return scores


def rougel_score(preds, refs, avg=True):
    """Compute the Rouge-L score

    Args:
        preds (list[str]): list of prediction
        refs (list[str]): list of reference
        avg (bool, optional): whether to return the average score. Defaults to True.
    """
    rouge = Rouge()
    scores  = []
    for pred, ref in zip(preds, refs):
        try:
            score = rouge.get_scores(pred, ref, avg=True)["rouge-l"]["f"]
        except ValueError:  # "Hypothesis is empty."
            socre = 0
        scores.append(score)
    return sum(scores) / len(scores) if avg else scores


def f1_score(preds, refs, avg=True):
    """Compute F1 scores

    Args:
        preds (list[str]): list of prediction
        refs (list[str]): list of reference
        avg (bool, optional): whether to return the average score. Defaults to True.
    """
    scores = []
    for pred, ref in zip(preds, refs):
        scores.append(compute_f1(ref, pred))
    return sum(scores) / len(scores) if avg else scores


def exact_score(preds, refs, avg=True):
    """Compute Rouge-L scores

    Args:
        preds (list[str]): list of prediction
        refs (list[str]): list of reference
        avg (bool, optional): whether to return the average score. Defaults to True.
    """
    scores = []
    for pred, ref in zip(preds, refs):
        scores.append(compute_exact(ref, pred))
    return sum(scores) / len(scores) if avg else scores


def recall_score(data, avg=True, k=1):
    """Compute the knowledge selection accuracy of query

    Args:
        data (BaseDataset): an instance of dataset containing knowledge selection results
        avg (bool, optional): Whether to return the average score. Defaults to True.
        k (int, optional): compute top-k accuracy. Defaults to 1.
    """
    recall = []
    for dial in data:
        if dial["response"] != "" and dial["gold_knowledge"] != []:
            gold_pid = dial["gold_knowledge"]
            recall.append(0)
            for i in range(k):
                pid = dial["selected_knowledge_id"][i]
                if pid in gold_pid:
                    recall[-1] = 1
                    break
    return sum(recall) / len(recall) if avg else recall