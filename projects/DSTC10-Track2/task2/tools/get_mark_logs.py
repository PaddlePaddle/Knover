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
"""Mark entities and locations in the original logs."""

import argparse
from difflib import SequenceMatcher
import functools
import json

from nltk import ngrams
from tqdm import tqdm

mark_start = "[ENT_START]"
mark_end = "[ENT_END]"

loc_start = "[LOC_START]"
loc_end = "[LOC_END]"

max_loc_gram = 6
max_n_gram = 11

def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--locations", type=str, required=True)
    parser.add_argument("--entities", type=str, required=True)
    parser.add_argument("--input_logs", type=str, required=True)
    parser.add_argument("--output_logs", type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    """Main function."""
    input_logs = json.load(open(args.input_logs))
    locations = json.load(open(args.locations))
    entities = json.load(open(args.entities))

    @functools.lru_cache(maxsize=None)
    def get_matching_result(ori_src, role):
        """Get fuzzy matching results of the utterance."""
        if role == "S":
            all_n_grams = get_all_n_grams(ori_src)
            match_n_grams = get_match_n_grams(all_n_grams, entities)
            merged_match = match_n_grams
            result = get_mark_src(ori_src, merged_match)

            match_loc_grams = get_match_location_n_grams(all_n_grams, locations)
            result = get_location_mark_src(result, merged_match, match_loc_grams)
        else:
            result = ori_src
        return result
    
    for log in tqdm(input_logs):
        for turn in log:
            ori_src = turn["text"]
            turn["text"] = get_matching_result(ori_src, turn["speaker"])

    print("done!", flush=True)   
    json.dump(input_logs, open(args.output_logs, "w"), indent=2)


def is_substr(sub, mylist):
    """Check substring."""
    return any(sub in mystring and sub != mystring for mystring in mylist)

def get_all_n_grams(src):    
    """Get all n-gram results from an utterance."""
    n_gram_dict = dict()    
    src_split = src.split()
    n_gram_dict[1] = src_split    
    for n in range(2, max_n_gram + 1):
        n_gram_dict[n] = []
        n_grams = ngrams(src_split, n)
        for g in n_grams:
            g = " ".join(g)
            n_gram_dict[n].append(g)
    
    return n_gram_dict

def remove_substr(comp_str, str_list):
    """Remove substring."""
    ret_list = []
    for s in str_list:
        if s in comp_str:
            continue
        ret_list.append(s)    
    return ret_list

def get_match_location_n_grams(all_n_grams, locations):
    """Get all fuzzy matching results of locations."""
    match_n_grams = set()

    for n in range(max_loc_gram, 0, -1):
        for n_g in all_n_grams[n]:
            if str(n) in locations.keys():
                for ke in locations[str(n)]:
                    if n == 1:
                        if n_g == ke:
                            match_n_grams.add(n_g)
                    else:
                        seq = SequenceMatcher(None, n_g, ke)
                        if seq.ratio() > 0.95:
                            match_n_grams.add(n_g)

                            for sub_n in range(1, n):
                                all_n_grams[sub_n] = remove_substr(n_g, all_n_grams[sub_n])
                            break
    if "in san francisco" in match_n_grams:
        match_n_grams.remove("in san francisco")
    return match_n_grams

def get_match_n_grams(all_n_grams, entities):
    """Get all fuzzy matching results of entities."""
    match_n_grams = set()

    for n in range(max_n_gram, 0, -1):
        for n_g in all_n_grams[n]:
            for ke in entities[str(n)]:
                seq = SequenceMatcher(None, n_g, ke)
                if seq.ratio() > 0.95:
                    match_n_grams.add(n_g)

                    for sub_n in range(1, n):
                        all_n_grams[sub_n] = remove_substr(n_g, all_n_grams[sub_n])
                    break
    if "in san francisco" in match_n_grams:
        match_n_grams.remove("in san francisco")
    return match_n_grams

def get_mark_src(src, merged_match):
    """Mark fuzzy matching results of entities."""
    mark_src = src.lower()
    for m in merged_match:
        mark_src = mark_src.replace(m, f"{mark_start} {m} {mark_end}")
    return mark_src

def get_location_mark_src(src, merged_match, loc_match):
    """Mark fuzzy matching results of locations."""
    mark_src = src
    for m in loc_match:
        is_sub = False
        for ent in merged_match:
            if m in ent:
                is_sub = True
                break
        if not is_sub:
            mark_src = mark_src.replace(m, f"{loc_start} {m} {loc_end}")
    return mark_src


if __name__ == "__main__":
    args = setup_args()
    main(args)
