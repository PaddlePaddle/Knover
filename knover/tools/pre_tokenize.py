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

from tqdm import tqdm

from knover.utils import parse_args
import knover.utils.tokenization as tokenization


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, default="SentencePieceTokenizer")
    args, _ = parser.parse_known_args()
    tokenizer_cls = getattr(tokenization, args.tokenizer)
    tokenizer_cls.add_cmdline_args(parser)

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parse_args(parser)
    return args


def main(args):
    """Tokenization main process."""
    tokenizer_cls = getattr(tokenization, args.tokenizer)
    tokenizer = tokenizer_cls(args)
    tokenized_fields = ["src", "tgt", "knowledge"]
    with open(args.input_file) as fp, open(args.output_file, "w") as output_fp:
        headers = next(fp).rstrip("\n").split("\t")
        output_fp.write("\t".join(headers) + "\n")
        for line in tqdm(fp, desc="Tokenizing"):
            cols = line.rstrip("\n").split("\t")
            assert len(cols) == len(headers)
            for i, (name, field) in enumerate(zip(headers, cols)):
                if name in tokenized_fields:
                    utts = field.split(" [SEP] ")
                    for j, utt in enumerate(utts):
                        if "\x01" in utt:
                            utt, role_id = utt.split("\x01")
                            utts[j] = " ".join(tokenizer.tokenize(utt)) + "\x01" + role_id
                        else:
                            utts[j] = " ".join(tokenizer.tokenize(utt))
                    cols[i] = " [SEP] ".join(utts)
            output_fp.write("\t".join(cols) + "\n")


if __name__ == "__main__":
    args = setup_args()
    main(args)
