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
"""DenseEmbedding Reader."""

from collections import namedtuple

import numpy as np

from knover.data.dialog_reader import DialogReader
from knover.utils import pad_batch_data, str2bool

class DenseEmbeddingReader(DialogReader):
    """Dense Embedding Reader."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = DialogReader.add_cmdline_args(parser)
        group.add_argument("--max_topic_len", type=int, default=0)
        group.add_argument("--embedding_type", type=str, default="src",
                           choices=["src", "knowledge"])
        return group

    def __init__(self, args):
        super(DenseEmbeddingReader, self).__init__(args)
        self.max_topic_len = args.max_topic_len
        assert args.max_src_len + args.max_topic_len <= args.max_seq_len, \
            "args.max_src_len + args.max_topic_len > args.max_seq_len"

        self.embedding_type = args.embedding_type

        self.fields = ["token_ids", "type_ids", "pos_ids"]
        if args.use_role:
            self.fields.append("role_ids")
        self.num_numerical_fields = len(self.fields)
        self.fields += ["data_id"]
        self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))
        return

    def _parse_topic(self, topic):
        """Parse topic sequence and return corresponding fields."""
        # process topic
        topic = topic.strip()

        if self.data_format == "tokenized":
            topic_tokens = topic.split(" ")
        else:
            topic_tokens = self.tokenizer.tokenize(topic)

        topic_token_ids = self.tokenizer.convert_tokens_to_ids(topic_tokens)
        topic_token_ids.append(self.eos_id)

        # trim topic
        topic_token_ids = topic_token_ids[:self.max_topic_len - 1]

        topic_token_ids = [self.bos_id] + topic_token_ids

        field_values = {
            "token_ids": topic_token_ids,
            "type_ids": [3] * len(topic_token_ids),
            "pos_ids": list(range(len(topic_token_ids)))
        }
        if self.use_role:
            field_values["role_ids"] = [0] * len(topic_token_ids)

        return field_values

    def _convert_example_to_record(self, example, is_infer):
        """convert example to record"""
        if self.embedding_type == "knowledge":
            field_values = self._parse_knowledge(example.tgt)
        else:
            # in wow dataset, some samples don't have context
            ori_src_field_values = self._parse_src(example.src) if len(example.src) > 0 else {}
            topic_field_values = self._parse_topic(example.topic) if self.max_topic_len > 0 else {}

            if len(topic_field_values) > 0:
                field_values = {
                            k: topic_field_values[k] + ori_src_field_values.get(k, [])
                            for k in topic_field_values
                }
            elif len (ori_src_field_values) > 0:
                field_values = ori_src_field_values
            else:
                raise ValueError(f"Invalid example at {example.data_id}")

            if self.position_style == "relative":
                ctx_len = len(field_values["token_ids"])
                field_values["pos_ids"] = [
                    self.max_tgt_len + ctx_len - i - 1
                    for i in range(ctx_len)
                ]

            if self.position_style == "continuous":
                field_values["pos_ids"] = list(range(len(field_values["token_ids"])))

        field_values["data_id"] = example.data_id

        record = self.Record(**field_values)

        return record

    def _read_numerical_file(self, fp, delimiter=";"):
        # dense embedding task does not support `numerical` data_format now.
        raise NotImplementedError

    def _pad_batch_records(self, batch_records, is_infer, phase=None):
        """pad batch records and mask"""
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        if self.use_role:
            batch_role_ids = [record.role_ids for record in batch_records]

        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=self.pad_id)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=self.pad_id)

        if self.use_role:
            batch["role_ids"] = pad_batch_data(batch_role_ids, pad_id=self.pad_id)

        batch["attention_mask"] = self._gen_self_attn_mask(batch_token_ids, is_unidirectional=False)

        batch_data_id = [record.data_id for record in batch_records]
        batch["data_id"] = np.array(batch_data_id).astype("int64").reshape([-1, 1])

        return batch
