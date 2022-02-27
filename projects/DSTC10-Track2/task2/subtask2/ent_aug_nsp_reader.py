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
"""NSP Reader for input with entity-augmentaton."""

from collections import namedtuple

from knover.data.nsp_reader import NSPReader

class EntAugNSPReader(NSPReader):
    """NSP Reader for input with entity-augmentaton."""
    def __init__(self, args):
        super(EntAugNSPReader, self).__init__(args)
        return
    
    def _update_type_ids(self, token_ids):
        """Mark entity spans in the input."""
        type_ids = [0] * len(token_ids)
        ent_start_idx = []
        ent_end_idx = []
        loc_start_idx = []
        loc_end_idx = []

        for idx, token in enumerate(token_ids):
            if token == 8007:
                ent_start_idx.append(idx)
            elif token == 8008:
                ent_end_idx.append(idx)
            elif token == 8009:
                loc_start_idx.append(idx)
            elif token == 8010:
                loc_end_idx.append(idx)
        if len(ent_start_idx) == len(ent_end_idx):
            for i in range(len(ent_start_idx)):
                start = ent_start_idx[i]
                end = ent_end_idx[i] + 1
                type_ids[start:end] = [3] * (end - start)
        
        if len(loc_start_idx) == len(loc_end_idx):
            for i in range(len(loc_start_idx)):
                start = loc_start_idx[i]
                end = loc_end_idx[i] + 1
                type_ids[start:end] = [4] * (end - start)
        
        return type_ids

    def _parse_src(self, src):
        """Parse source sequence and return corresponding fields."""
        field_values = super(EntAugNSPReader, self)._parse_src(src)
        field_values["type_ids"] = self._update_type_ids(field_values["token_ids"])
        return field_values