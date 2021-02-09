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
"""Tokenization classes."""

import collections
import json
import re
import six
import unicodedata

import sentencepiece as spm

from knover.utils import str2bool


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    text = text.replace(u"“", u'"')\
        .replace(u'”', u'"')\
        .replace(u'‘', "'")\
        .replace(u'’', u"'")\
        .replace(u'—', u'-')

    output = []
    for char in text:
        if _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(spm_model, text, return_unicode=True, sample=False):
    """Convert sentences into word pieces."""
    text = clean_text(text)

    if not sample:
        pieces = spm_model.EncodeAsPieces(text)
    else:
        pieces = spm_model.SampleEncodeAsPieces(text, 64, 0.1)

    return pieces


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.rstrip().split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


class SentencePieceTokenizer(object):
    """Runs end-to-end tokenziation."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = parser.add_argument_group("Tokenizer")
        group.add_argument("--vocab_path", type=str, required=True)
        group.add_argument("--specials_path", type=str, default="")
        group.add_argument("--do_lower_case", type=str2bool, default=False)
        group.add_argument("--spm_model_file", type=str, required=True)
        return group

    def __init__(self, args):
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(args.spm_model_file)
        self.vocab = load_vocab(args.vocab_path)
        self.do_lower_case = args.do_lower_case
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        pat_str = ""
        if args.specials_path != "":
            self.specials = load_vocab(args.specials_path)
            for special in self.specials:
                pat_str += "(" + re.escape(special) + ")|"
        else:
            self.specials = {}
        pat_str += "([a-zA-Z0-9\S]+)"
        self.pat = re.compile(pat_str)

    # Speedup tokenization.
    cached = {}

    def preprocess(self, text):
        text = preprocess_text(text, lower=self.do_lower_case)
        return text

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self.preprocess(text)
        if text in self.cached:
            return self.cached[text]
        tokens = []
        for match in self.pat.finditer(text):
            part_text = match.group(0)
            if part_text in self.specials:
                tokens.append(part_text)
                continue
            part_tokens = encode_pieces(self.spm_model, part_text, return_unicode=True)
            tokens.extend(part_tokens)
        self.cached[text] = tokens
        return tokens

    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to ids."""
        ret = []
        unk_id = self.vocab["<unk>"]
        for token in tokens:
            if token in self.vocab:
                ret.append(self.vocab[token])
            else:
                ret.append(unk_id)
        return ret

    def convert_ids_to_tokens(self, ids):
        """Convert ids to tokens."""
        return convert_by_vocab(self.inv_vocab, ids)

    def merge_subword(self, tokens):
        """Merge subword."""
        ret = []
        for token in tokens:
            if token.startswith(u"▁"):
                ret.append(token[1:])
            elif token in self.specials:
                ret.append(token)
            else:
                if len(ret):
                    ret[-1] += token
                else:
                    ret.append(token)

        ret = [token for token in ret if token]
        return ret

    def convert_ids_to_str(self, ids):
        """Convert ids to string."""
        tokens = self.convert_ids_to_tokens(ids)
        tokens = self.merge_subword(tokens)
        res = " ".join(tokens).replace("<s>", "")
        res = res.replace("</s>", "\n").replace("\n ", "\n").strip()
        return res


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False
