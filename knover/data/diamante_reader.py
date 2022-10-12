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
from knover.utils import mask, pad_batch_data, str2bool


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
        record = super(DiamanteReader, self)._convert_example_to_record(example, False)
        if "label" in example._fields:
            record = record._replace(label=int(example.label))
        return record

    def _pad_batch_records(self, batch_records, is_infer, phase=None):
        """Padding a batch of records and construct model's inputs."""
        batch_size = len(batch_records)
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        if self.use_role:
            batch_role_ids = [record.role_ids for record in batch_records]
        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]

        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=0)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=0)
        if self.use_role:
            batch["role_ids"] = pad_batch_data(batch_role_ids, pad_id=0)

        batch["generation_mask"] = self._gen_self_attn_mask(batch_token_ids, batch_tgt_start_idx)

        if is_infer:
            if self.do_generation:
                tgt_ids = np.array([[[self.bos_id]]] * len(batch_token_ids), dtype="int64")
                if self.position_style == "continuous":
                    tgt_pos = np.array(batch_tgt_start_idx, dtype="int64")
                else:
                    tgt_pos = np.zeros_like(batch_tgt_start_idx, dtype="int64")
                tgt_pos = tgt_pos.reshape(-1, 1, 1)
                batch["init_score"] = np.zeros_like(tgt_ids, dtype="float32").reshape(-1, 1).tolist()
                batch["tgt_ids"] = tgt_ids.tolist()
                batch["tgt_pos"] = tgt_pos.tolist()
                batch["parent_idx"] = np.array(range(batch_size), dtype="int32")

                batch["tgt_generation_mask"] = batch["generation_mask"][:, 0:1, :]
            else:
                batch["tgt_label"], batch["tgt_idx"], batch["label_idx"] = mask(
                    batch_tokens=batch_token_ids,
                    vocab_size=self.vocab_size,
                    bos_id=self.bos_id,
                    tgt_starts=batch_tgt_start_idx,
                    labels=[1]*batch_size,
                    is_unidirectional=True)

            batch_data_id = [record.data_id for record in batch_records]
            batch["data_id"] = np.array(batch_data_id).astype("int64").reshape([-1, 1])
        else:
            batch_label = [record.label for record in batch_records]
            batch["tgt_label"], batch["tgt_idx"], batch["label_idx"] = mask(
                batch_tokens=batch_token_ids,
                vocab_size=self.vocab_size,
                bos_id=self.bos_id,
                tgt_starts=batch_tgt_start_idx,
                labels=batch_label,
                is_unidirectional=True)

            batch_label = np.array(batch_label).astype("int64").reshape([-1, 1])
            batch["label"] = batch_label

        return batch
