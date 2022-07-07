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
"""Classification Reader."""

from collections import namedtuple

import numpy as np

from knover.data.dialog_reader import DialogReader
from knover.utils import pad_batch_data, str2bool


class ClassificationReader(DialogReader):
    """Classification Reader."""

    def __init__(self, args):
        super(ClassificationReader, self).__init__(args)
        self.fields.append("label")
        self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))
        return

    def _convert_example_to_record(self, example, is_infer):
        """Convert example to record which can be used as the model's input."""
        field_values = self._parse_src(example.src)
        if len(field_values["token_ids"]) == 1:
            raise ValueError(f"Invalid example: context too long / no context - {example}")

        if self.max_knowledge_len > 0:
            knowledge_field_values = self._parse_knowledge(example.knowledge)
            field_values = {
                k: field_values[k] + knowledge_field_values[k]
                for k in field_values
            }

        tgt_start_idx = len(field_values["token_ids"])

        if self.position_style == "relative":
            ctx_len = len(field_values["token_ids"])
            field_values["pos_ids"] = [
                self.max_tgt_len + ctx_len - i - 1
                for i in range(ctx_len)
            ]

        if self.position_style == "continuous":
            field_values["pos_ids"] = list(range(len(field_values["token_ids"])))

        field_values["tgt_start_idx"] = tgt_start_idx
        field_values["data_id"] = example.data_id

        if not is_infer:
            field_values["label"] = int(example.label)

        record = self.Record(**field_values)
        return record

    def _read_numerical_file(self, fp, phase, is_infer, delimiter=";"):
        """Read a file which contains numerical data and yield records."""
        if is_infer:
            return super(ClassificationReader, self)._read_numerical_file(fp, phase, is_infer, delimiter)
        else:
            # Classification task does not support `numerical` data_format during training now.
            raise NotImplementedError

    def _pad_batch_records(self, batch_records, is_infer, phase=None):
        """Padding a batch of records and construct model's inputs."""
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=0)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=0)
        if self.use_role:
            batch_role_ids = [record.role_ids for record in batch_records]
            batch["role_ids"] = pad_batch_data(batch_role_ids, pad_id=0)

        attention_mask = self._gen_self_attn_mask(batch_token_ids, is_unidirectional=False)
        batch["attention_mask"] = attention_mask

        if not is_infer:
            batch_label = [record.label for record in batch_records]
            batch["label"] = np.array(batch_label).astype("int64").reshape([-1, 1])
        else:
            batch_data_id = [record.data_id for record in batch_records]
            batch["data_id"] = np.array(batch_data_id).astype("int64").reshape([-1, 1])

        return batch
