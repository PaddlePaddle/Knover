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
"""Convert raw data to tokenized data."""

import argparse

from utils.args import parse_args
from utils.tokenization import SentencePieceTokenizer


def setup_args():
    parser = argparse.ArgumentParser()
    SentencePieceTokenizer.add_cmdline_args(parser)

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parse_args(parser)
    return args


def main(args):
    tokenizer = SentencePieceTokenizer(args)
    with open(args.input_file) as fp, open(args.output_file, "w") as output_fp:
        next(fp)
        output_fp.write("src\ttgt\n")
        for line in fp:
            src, tgt = line.strip().split("\t")
            src = " ".join(tokenizer.tokenize(src))
            output_fp.write(src + "\t" + " ".join(tokenizer.tokenize(tgt)) + "\n")


if __name__ == "__main__":
    args = setup_args()
    main(args)
