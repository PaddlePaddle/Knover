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
"""Diamante Reader."""

from collections import namedtuple

import numpy as np

from knover.data.dialog_reader import DialogReader
from knover.utils import mask, pad_batch_data


class DiamanteReader(DialogReader):
    """Diamante Reader."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = DialogReader.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        super(DiamanteReader, self).__init__(args)
        self.fields.append("label")
        self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))

        self.use_turn = args.use_turn
        self.do_generation = args.get("do_generation", False)
        return

    def _convert_example_to_record(self, example, is_infer):
        """Convert example to record which can be used as the model's input."""
        record = super(DiamanteReader, self)._convert_example_to_record(example, is_infer)
        if "label" in example._fields:
            record = record._replace(label=int(example.label))
        return record

    def _pad_batch_records(self, batch_records, is_infer, phase=None):
        """Padding a batch of records and construct model's inputs."""
        batch_size = len(batch_records)
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        if is_infer and self.do_generation:
            offset = -1
        else:
            offset = None
        batch["token_ids"] = [record.token_ids[:offset] for record in batch_records]
        batch["type_ids"] = pad_batch_data([record.type_ids[:offset] for record in batch_records], pad_id=0)
        batch["pos_ids"] = pad_batch_data([record.pos_ids[:offset] for record in batch_records], pad_id=0)
        if self.use_role:
            batch["role_ids"] = pad_batch_data([record.role_ids[:offset] for record in batch_records], pad_id=0)
        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]
        batch["generation_mask"] = self._gen_self_attn_mask(batch["token_ids"], batch_tgt_start_idx)

        if is_infer:
            if self.do_generation:
                batch["tgt_ids"] = np.array(
                    [record.token_ids[-1] for record in batch_records], dtype="int64").reshape(-1, 1, 1).tolist()
                batch["tgt_type"] = np.array(
                    [record.type_ids[-1] for record in batch_records], dtype="int64").reshape(-1, 1)
                batch["tgt_pos"] = np.array(
                    [record.pos_ids[-1] for record in batch_records], dtype="int64").reshape(-1, 1)
                if self.use_role:
                    batch["tgt_role"] = np.array(
                        [record.role_ids[-1] for record in batch_records], dtype="int64").reshape(-1, 1)
                batch["tgt_generation_mask"] = self._gen_tgt_attn_mask(batch["token_ids"])
            else:
                batch["tgt_label"], batch["tgt_idx"], batch["label_idx"] = mask(
                    batch_tokens=batch_token_ids,
                    vocab_size=self.vocab_size,
                    tgt_starts=batch_tgt_start_idx,
                    bos_id=self.bos_id,
                    labels=[1] * batch_size,
                    is_unidirectional=True)

            batch["data_id"] = np.array([record.data_id for record in batch_records], dtype="int64")
        else:
            batch_label = [record.label for record in batch_records]
            batch["tgt_label"], batch["tgt_idx"], batch["label_idx"] = mask(
                batch_tokens=batch_token_ids,
                vocab_size=self.vocab_size,
                tgt_starts=batch_tgt_start_idx,
                bos_id=self.bos_id,
                labels=batch_label,
                is_unidirectional=True)

            batch["label"] = np.array(batch_label, dtype="int64").reshape([-1, 1])

        batch["token_ids"] = pad_batch_data(batch["token_ids"], pad_id=self.pad_id)
        return batch
