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
"""Plato Reader."""

from collections import namedtuple

import numpy as np

from knover.data.dialog_reader import DialogReader
from knover.utils import mask, pad_batch_data


class PlatoReader(DialogReader):
    """The implement of PlatoReader"""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = DialogReader.add_cmdline_args(parser)
        group.add_argument("--neg_pool_size", type=int, default=2 ** 16,
                           help="The size of random negative pool.")
        return group

    def __init__(self, args):
        super(PlatoReader, self).__init__(args)
        self.use_bow = args.use_bow
        self.use_nsp = args.use_nsp
        self.neg_pool_size = args.neg_pool_size
        if self.use_nsp:
            self.fields.extend(["neg_token_ids", "neg_type_ids", "neg_pos_ids"])
            if self.use_role:
                self.fields.append("neg_role_ids")
            self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))
        return

    def sort_key(self, record):
        """The key of record.

        We will apply sorting before batching. It can decrease the number of padding and
        speedup training.
        """
        if self.use_nsp:
            return [2 * len(record.token_ids), len(record.neg_token_ids)]
        else:
            return super().sort_key(record)

    def _mix_negative_sample(self, reader, neg_pool_size=2 ** 16):
        """Mix random negative samples into dataset."""
        def _gen_from_pool(pool):
            """Generate negative sample related fields from pool."""
            num_samples = len(pool)
            if num_samples == 1:
                # it is impossible to generate negative sample when the pool has only one sample
                return
            self.global_rng.shuffle(pool)
            for i in range(num_samples):
                j = (i + 1) % num_samples
                idx_i = pool[i].tgt_start_idx
                idx_j = pool[j].tgt_start_idx
                # add negative sample fields
                neg_fields = {}
                neg_fields["neg_token_ids"] = pool[i].token_ids[:idx_i] + pool[j].token_ids[idx_j:]
                neg_fields["neg_type_ids"] = pool[i].type_ids[:idx_i] + pool[j].type_ids[idx_j:]
                if self.position_style == "continuous":
                    neg_fields["neg_pos_ids"] = list(range(len(neg_fields["neg_token_ids"])))
                else:
                    neg_fields["neg_pos_ids"] = pool[i].pos_ids[:idx_i] + pool[j].pos_ids[idx_j:]
                if self.use_role:
                    neg_fields["neg_role_ids"] = pool[i].role_ids[:idx_i] + pool[j].role_ids[idx_j:]
                pool[i] = pool[i]._replace(**neg_fields)
            self.global_rng.shuffle(pool)
            for record in pool:
                yield record

        def __wrapper__():
            pool = []
            for record in reader():
                pool.append(record)
                if len(pool) == neg_pool_size:
                    for record in _gen_from_pool(pool):
                        yield record
                    pool = []
            if len(pool) > 0:
                for record in _gen_from_pool(pool):
                    yield record
        return __wrapper__

    def _batch_reader(self, reader, phase=None, is_infer=False):
        """Construct a batch reader from a record reader."""
        if self.use_nsp and not is_infer:
            reader = self._mix_negative_sample(reader, self.neg_pool_size)
        return super(PlatoReader, self)._batch_reader(
            reader,
            phase=phase,
            is_infer=is_infer)

    def _pad_batch_records(self, batch_records, is_infer, **kwargs):
        """Padding a batch of records and construct model's inputs."""
        batch_size = len(batch_records)
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch["token_ids"] = [record.token_ids[:-1] for record in batch_records]
        batch["type_ids"] = pad_batch_data([record.type_ids[:-1] for record in batch_records], pad_id=0)
        batch["pos_ids"] = pad_batch_data([record.pos_ids[:-1] for record in batch_records], pad_id=0)
        if self.use_role:
            batch["role_ids"] = pad_batch_data([record.role_ids[:-1] for record in batch_records], pad_id=0)
        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]
        batch["generation_mask"] = self._gen_self_attn_mask(
            batch["token_ids"],
            batch_tgt_start_idx=batch_tgt_start_idx,
            is_unidirectional=True,
            num_aux_token=1)

        if is_infer:
            batch["tgt_ids"] = np.array(
                [record.token_ids[-1] for record in batch_records], dtype="int64").reshape(-1, 1, 1).tolist()
            batch["tgt_type"] = np.array(
                [record.type_ids[-1] for record in batch_records], dtype="int64").reshape(-1, 1)
            batch["tgt_pos"] = np.array(
                [record.pos_ids[-1] for record in batch_records], dtype="int64").reshape(-1, 1)
            if self.use_role:
                batch["tgt_role"] = np.array(
                    [record.role_ids[-1] for record in batch_records], dtype="int64").reshape(-1, 1)
            batch["latent_id"] = np.zeros([batch_size], dtype="int32")
            batch["tgt_generation_mask"] = self._gen_tgt_attn_mask(batch["token_ids"], num_aux_token=1)

            batch["data_id"] = np.array([record.data_id for record in batch_records], dtype="int64")
        else:
            batch["rec_mask"] = self._gen_self_attn_mask(batch["token_ids"], is_unidirectional=False, num_aux_token=1)

            if self.use_nsp:
                # NOTE: remove the last token
                batch["neg_token_ids"] = [record.neg_token_ids[:-1] for record in batch_records]
                batch["neg_type_ids"] = pad_batch_data([record.neg_type_ids[:-1] for record in batch_records], pad_id=0)
                batch["neg_pos_ids"] = pad_batch_data([record.neg_pos_ids[:-1] for record in batch_records], pad_id=0)
                if self.use_role:
                    batch["neg_role_ids"] = pad_batch_data(
                        [record.neg_role_ids[:-1] for record in batch_records], pad_id=0)

                batch["neg_rec_mask"] = self._gen_self_attn_mask(
                    batch["neg_token_ids"], is_unidirectional=False, num_aux_token=1)
                batch["neg_token_ids"] = pad_batch_data(batch["neg_token_ids"], pad_id=self.pad_id)

            mask_return_list = mask(
                batch_tokens=batch_token_ids,
                vocab_size=self.vocab_size,
                tgt_starts=batch_tgt_start_idx,
                is_unidirectional=True,
                use_latent=True,
                use_bow=self.use_bow)
            batch["tgt_label"] = mask_return_list[0]
            batch["tgt_idx"] = mask_return_list[1]
            if self.use_bow:
                batch["bow_label"] = mask_return_list[2]
                batch["bow_idx"] = mask_return_list[3]

        batch["token_ids"] = pad_batch_data(batch["token_ids"], pad_id=self.pad_id)
        return batch
