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
"""Convert raw data to numerical data."""

import argparse

from tqdm import tqdm

from ent_aug_nsp_reader import EntAugNSPReader
from knover.utils import parse_args, str2bool


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()

    EntAugNSPReader.add_cmdline_args(parser)
    parser.add_argument("--use_role", type=str2bool, default=False, help="Whether use role embeddings.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    
    args = parse_args(parser)
    return args


def main(args):
    """Numericalization main process."""
    reader = EntAugNSPReader(args)
    reader.data_id = 0
    generator = reader._read_file(args.input_file, phase="numericalize", is_infer=True)
    with open(args.output_file, "w") as fp:
        for record in tqdm(generator(), desc="Numericalizing"):
            cols = [" ".join(map(str, getattr(record, field_name)))
                    for field_name in record._fields
                    if isinstance(getattr(record, field_name), list)]
            line = ";".join(cols)
            fp.write(line + "\n")


if __name__ == "__main__":
    args = setup_args()
    main(args)
