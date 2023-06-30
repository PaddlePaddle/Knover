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
"""KAG Reader."""

from collections import namedtuple
import copy

import numpy as np

from knover.data.dialog_reader import DialogReader
from knover.utils import mask, pad_batch_data, to_optimized_size

class KAGReader(DialogReader):
    """KAG Reader."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = DialogReader.add_cmdline_args(parser)
        group.add_argument("--max_topic_len", type=int, default=0)
        return group

    def __init__(self, args):
        super(KAGReader, self).__init__(args)
        self.max_topic_len = args.max_topic_len
        assert args.max_knowledge_len + args.max_src_len + args.max_tgt_len + args.max_topic_len <= args.max_seq_len, \
            "args.max_knowledge_len + args.max_src_len + args.max_tgt_len + args.max_tpoic_len > args.max_seq_len"

        self.do_kag_training = args.do_kag_training
        if self.do_kag_training:
            self._parse_fields_for_kag_training(args)
        return

    def _parse_fields_for_kag_training(self, args):
        """Parse and return corresponding fields for KAG training."""
        self.max_knowledge_num = args.max_knowledge_num

        base_fields_names = ["token_ids", "type_ids", "pos_ids"]
        if args.use_role:
            base_fields_names.append("role_ids")

        self.fields = []
        for name in base_fields_names:
            self.fields.append("dual_src_" + name)

        self.num_numerical_fields = len(self.fields)

        self.fields += ["single_item_list", "dual_knowledge_list", "exact_k_len"]
        self.fields += ["tgt_start_idx", "data_id"]
        self.fields += ["tgt_mask_pos"]

        self.sort_key = lambda record: [len(record.dual_src_token_ids)]
        self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))


    def _parse_topic(self, topic):
        """Parse topic sequence and return corresponding fields."""
        topic = topic.strip()

        if self.data_format == "tokenized":
            topic_tokens = topic.split(" ")
        else:
            topic_tokens = self.tokenizer.tokenize(topic)

        topic_token_ids = self.tokenizer.convert_tokens_to_ids(topic_tokens)
        topic_token_ids.append(self.eos_id)
        topic_token_ids = [self.bos_id] + topic_token_ids

        # trim topic
        topic_token_ids = topic_token_ids[:self.max_topic_len]

        field_values = {
            "token_ids": topic_token_ids,
            "type_ids": [3] * len(topic_token_ids),
            "pos_ids": list(range(len(topic_token_ids)))
        }
        if self.use_role:
            field_values["role_ids"] = [0] * len(topic_token_ids)

        return field_values

    def _parse_knowledge_list(self, knowledge):
        """Parse knowledge list and return corresponding fields."""
        knowledge_list = []
        if self.knowledge_style == "original":
            ks = knowledge.split("[SEP]")
        elif self.knowledge_style == "reversed":
            ks = reversed(knowledge.split("[SEP]"))

        for k in ks:
            k_pos_ids = [0]
            k = k.strip()
            if self.data_format == "tokenized":
                k_tokens = k.split(" ")
            else:
                k_tokens = self.tokenizer.tokenize(k)

            k_token_ids = [self.bos_id] + self.tokenizer.convert_tokens_to_ids(k_tokens) + [self.eos_id]
            k_pos_ids = list(range(len(k_token_ids)))

            if len(k_token_ids) > self.max_knowledge_len:
                if self.knowledge_style == "original":
                    k_token_ids = k_token_ids[:self.max_knowledge_len]
                    k_pos_ids = k_pos_ids[:self.max_knowledge_len]
                else:
                    k_token_ids = k_token_ids[-self.max_knowledge_len:]
                    k_pos_ids = k_pos_ids[-self.max_knowledge_len:]

            field_values = {
                "token_ids": k_token_ids,
                "type_ids": [2] * len(k_token_ids),
                "pos_ids": k_pos_ids
            }
            if self.use_role:
                field_values["role_ids"] = [0] * len(k_token_ids)

            knowledge_list.append(field_values)

        k_num = len(knowledge_list)

        if k_num < self.max_knowledge_num:
            pad_k_item = {
                "token_ids": [self.pad_id],
                "type_ids": [0],
                "pos_ids": [0]
            }
            if self.use_role:
                pad_k_item["role_ids"] = [0]

            knowledge_list = knowledge_list + [pad_k_item] * (self.max_knowledge_num - k_num)
        else:
            knowledge_list = knowledge_list[:self.max_knowledge_num]

        return knowledge_list, k_num

    def _get_field_values_for_training(self, example):
        """Get field values for KAG training."""
        # in wow dataset, some samples don't have context
        ori_src_field_values = self._parse_src(example.src) if len(example.src) > 0 else {}
        topic_field_values = self._parse_topic(example.topic) if self.max_topic_len > 0 else {}

        if len(topic_field_values) > 0:
            src_field_values = {
                k: topic_field_values[k] + ori_src_field_values.get(k, [])
                for k in topic_field_values
            }
        elif len (ori_src_field_values) > 0:
            src_field_values = ori_src_field_values
        else:
            raise ValueError(f"Invalid example at {example.data_id}")

        knowledge_list, exact_k_len = self._parse_knowledge_list(example.knowledge)

        tgt_field_values = self._parse_tgt(example.tgt)

        field_values = {
            "dual_src_" + k: src_field_values[k]
            for k in src_field_values
        }

        field_values = {
            "dual_src_" + k: src_field_values[k]
            for k in src_field_values
        }

        tgt_label = tgt_field_values["token_ids"][1:]
        tgt_idx = [i for i in range(len(tgt_label))]
        tgt_label = np.array(tgt_label, dtype="int64").reshape([-1, 1])
        tgt_idx = np.array(tgt_idx, dtype="int64").reshape([-1, 1])

        tgt_mask_pos = []
        for i in range(len(tgt_idx)):
            tgt_mask_pos.append(tgt_idx[i][0])

        field_values["tgt_mask_pos"] = tgt_mask_pos

        field_values["tgt_start_idx"] = []
        field_values["single_item_list"] = []
        field_values["dual_knowledge_list"] = []

        for knowledge in knowledge_list:
            # build knowledge list for dual tower
            field_values["dual_knowledge_list"].append(knowledge)

            # build items for single tower
            # concatenate knowledge in single tower
            if self.knowledge_position == "pre_src":
                item_field_values = {
                    k: knowledge[k] + src_field_values[k]
                    for k in knowledge
                }
            else:
                item_field_values = {
                    k: src_field_values[k] + knowledge[k]
                    for k in knowledge
                }

            field_values["tgt_start_idx"].append(len(item_field_values["token_ids"]))

            if self.position_style == "relative":
                ctx_len = len(item_field_values["token_ids"])
                item_field_values["pos_ids"] = [
                    self.max_tgt_len + ctx_len - i - 1
                    for i in range(ctx_len)
                ]

            # concatenate tgt in single tower
            item_field_values = {
                k: item_field_values[k] + tgt_field_values[k]
                for k in item_field_values
            }

            if self.position_style == "continuous":
                item_field_values["pos_ids"] = list(range(len(item_field_values["token_ids"])))

            field_values["single_item_list"].append(item_field_values)

        if self.position_style == "relative":
            ctx_len = len(src_field_values["token_ids"])
            field_values["dual_src_pos_ids"] = [
                self.max_knowledge_len + ctx_len - i - 1
                for i in range(ctx_len)
            ]

        if self.position_style == "continuous":
            field_values["dual_src_pos_ids"] = list(range(len(field_values["dual_src_pos_ids"])))

        field_values["data_id"] = example.data_id
        field_values["exact_k_len"] = exact_k_len

        return field_values

    def _get_field_values_for_generation(self, example, is_infer):
        """Get field values for KAG generation."""
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
            raise ValueError(f"Invalid example: {example}")

        if self.max_knowledge_len > 0:
            # add knowledge
            knowledge_field_values = self._parse_knowledge(example.knowledge)
            if self.knowledge_position == "pre_src":
                field_values = {
                    k: knowledge_field_values[k] + field_values.get(k, [])
                    for k in knowledge_field_values
                }
            else:
                field_values = {
                    k: field_values.get(k, []) + knowledge_field_values[k]
                    for k in knowledge_field_values
                }

        tgt_start_idx = len(field_values["token_ids"])

        if self.position_style == "relative":
            ctx_len = len(field_values["token_ids"])
            field_values["pos_ids"] = [
                self.max_tgt_len + ctx_len - i - 1
                for i in range(ctx_len)
            ]

        tgt_field_values = self._parse_tgt(
            "" if is_infer and self.do_generation and "tgt" not in example._fields else example.tgt,
            add_eos=not is_infer or not self.do_generation)
        field_values = self._merge_field_values(field_values, tgt_field_values)

        if self.position_style == "continuous":
            field_values["pos_ids"] = list(range(len(field_values["token_ids"])))

        field_values["tgt_start_idx"] = tgt_start_idx
        field_values["data_id"] = example.data_id

        return field_values

    def _convert_example_to_record(self, example, is_infer):
        """Convert example to record."""
        if self.do_kag_training:
            field_values = self._get_field_values_for_training(example)
        else:
            field_values = self._get_field_values_for_generation(example, is_infer)

        record = self.Record(**field_values)

        return record

    def _read_numerical_file(self, fp, delimiter=";"):
        # KAG task does not support `numerical` data_format now.
        raise NotImplementedError

    def _get_batch_knowledge_ids(self, batch_records, ids_type):
        """Get batch knowledge ids."""
        # [b * k, l, 1]
        batch_k_ids = []
        for record in batch_records:
            for knowledge in record.dual_knowledge_list:
                batch_k_ids.append(knowledge[ids_type])
        return batch_k_ids

    def _get_batch_single_item(self, batch_records, ids_type):
        """Get batch single item."""
        # [b * k, l, 1]
        batch_single_ids = []
        for record in batch_records:
            for single_item in record.single_item_list:
                batch_single_ids.append(single_item[ids_type])
        return batch_single_ids

    def _mask_batch_as_list_for_topk_gen(self,
            batch_size,
            batch_tokens,
            vocab_size,
            batch_mask_start_pos,
            batch_tgt_mask_pos,
            exact_k_lens,
            bos_id=1,
            eos_id=2,
            mask_id=3):
        """Add mask for batch_tokens, return out, mask_label, mask_pos;

        Note: mask_pos responding the batch_tokens after padded;
        """
        batch_tokens = copy.deepcopy(batch_tokens)
        max_len = max(map(len, batch_tokens))
        mask_label_list = []
        mask_pos_list = []

        # for batch_size
        for i in range(batch_size):
            exact_k_len = exact_k_lens[i]
            mask_start_pos = batch_mask_start_pos[i]
            tgt_mask_pos = batch_tgt_mask_pos[i]
            # for knowledge
            for k in range(self.max_knowledge_num):
                sent_index = i * self.max_knowledge_num + k
                sent = batch_tokens[sent_index]
                mask_start_idx = mask_start_pos[k]
                mask_label = []
                mask_pos = []

                if k < exact_k_len:
                    for idx in range(len(tgt_mask_pos)):
                        offset = tgt_mask_pos[idx]
                        mask_label.append(sent[mask_start_idx + offset + 1])
                        mask_pos.append([i, k, mask_start_idx + offset])
                else:
                    # pad
                    for idx in range(len(tgt_mask_pos)):
                        offset = tgt_mask_pos[idx]
                        mask_label.append(self.bos_id)
                        mask_pos.append([i, k, mask_start_idx + offset])

                mask_label = np.array(mask_label, dtype="int64").reshape([-1, 1])
                mask_pos = np.array(mask_pos, dtype="int64").reshape([-1, 3])
                mask_label_list.append(mask_label)
                mask_pos_list.append(mask_pos)

        return_list = [mask_label_list, mask_pos_list]
        return return_list

    def _pad_batch_data_to_len(self, insts, pad_id=0, given_len=0):
        """Pad the instances to a given length in batch."""
        max_len = to_optimized_size(max(map(len, insts)))
        if given_len < max_len:
            raise ValueError(f"given_len = {given_len}, max_len = {max_len}, given_len should be larger than max_len in batch data.")
        inst_data = np.array([list(inst) + [pad_id] * (given_len - len(inst)) for inst in insts], dtype="int64")
        return inst_data.reshape([-1, given_len, 1])

    def _pad_batch_data_to_len_for_topk(self, insts, pad_id=0, given_len=0):
        """Pad the instances to a given length in batch."""
        max_len = to_optimized_size(max(map(len, insts)))
        if given_len < max_len:
            raise ValueError(f"given_len = {given_len}, max_len = {max_len}, given_len should be larger than max_len in batch data.")
        inst_data = []
        for inst in insts:
            first = inst[0]
            b = first[0]
            k = first[1]
            cur_len = len(inst)
            pad_item = [b, k, pad_id]
            for i in range(cur_len):
                inst_data.append(inst[i])
            for i in range(given_len - cur_len):
                inst_data.append(pad_item)

        inst_data = np.array(inst_data, dtype="int64")
        # 4d
        return inst_data.reshape([-1, self.max_knowledge_num, given_len, 3])

    def _pad_batch_records_for_training(self, batch_records):
        """Pad batch records and mask for KAG training."""
        batch = {}

        batch["data_id"] = np.array([record.data_id for record in batch_records], dtype="int64")

        # [n, k, len, 1]
        single_batch_token_ids = self._get_batch_single_item(batch_records, "token_ids")

        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]

        batch_size = len(batch_records)
        tgt_label, tgt_idx = self._mask_batch_as_list_for_topk_gen(
            batch_size=batch_size,
            exact_k_lens=[record.exact_k_len for record in batch_records],
            batch_tokens=single_batch_token_ids,
            vocab_size=self.vocab_size,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            mask_id=self.mask_id,
            batch_mask_start_pos=batch_tgt_start_idx,
            batch_tgt_mask_pos=[record.tgt_mask_pos for record in batch_records])

        flatten_batch_tgt_start_idx = [j for i in batch_tgt_start_idx for j in i]
        batch["single_attention_mask"] = self._gen_self_attn_mask(single_batch_token_ids, batch_tgt_start_idx=flatten_batch_tgt_start_idx)

        given_len=self.max_tgt_len
        batch["tgt_label"] = self._pad_batch_data_to_len(tgt_label, pad_id=self.bos_id, given_len=given_len)
        batch["tgt_idx"] = self._pad_batch_data_to_len_for_topk(tgt_idx, pad_id=self.pad_id, given_len=given_len)

        batch["single_token_ids"] = pad_batch_data(single_batch_token_ids, pad_id=self.pad_id)
        batch["single_type_ids"] = pad_batch_data(self._get_batch_single_item(batch_records, "type_ids"), pad_id=0)
        batch["single_pos_ids"] = pad_batch_data(self._get_batch_single_item(batch_records, "pos_ids"), pad_id=0)
        if self.use_role:
            batch["single_role_ids"] = pad_batch_data(self._get_batch_single_item(batch_records, "role_ids"), pad_id=0)

        max_len = to_optimized_size(max(map(len, single_batch_token_ids)))
        batch["tgt_label"] = batch["tgt_label"].reshape([-1, self.max_knowledge_num, given_len, 1])
        batch["single_attention_mask"] = batch["single_attention_mask"].reshape([-1, self.max_knowledge_num, max_len, max_len])
        batch["single_token_ids"] = batch["single_token_ids"].reshape([-1, self.max_knowledge_num, max_len, 1])
        batch["single_type_ids"] = batch["single_type_ids"].reshape([-1, self.max_knowledge_num, max_len, 1])
        batch["single_pos_ids"] = batch["single_pos_ids"].reshape([-1, self.max_knowledge_num, max_len, 1])
        if self.use_role:
            batch["single_role_ids"] = batch["single_role_ids"].reshape([-1, self.max_knowledge_num, max_len, 1])

        # for dual
        # token_ids, [n, len, 1]
        batch["dual_src_token_ids"] = pad_batch_data([record.dual_src_token_ids for record in batch_records], pad_id=self.pad_id)
        # [n * k, len, 1]
        batch["dual_knowledge_token_ids"] = pad_batch_data(self._get_batch_knowledge_ids(batch_records, "token_ids"), pad_id=self.pad_id)
        batch["dual_src_type_ids"] = pad_batch_data([record.dual_src_type_ids for record in batch_records], pad_id=0)
        batch["dual_knowledge_type_ids"] = pad_batch_data(self._get_batch_knowledge_ids(batch_records, "type_ids"), pad_id=0)
        batch["dual_src_pos_ids"] = pad_batch_data([record.dual_src_pos_ids for record in batch_records], pad_id=0)
        batch["dual_knowledge_pos_ids"] = pad_batch_data(self._get_batch_knowledge_ids(batch_records, "pos_ids"), pad_id=0)
        if self.use_role:
            batch["dual_src_role_ids"] = pad_batch_data([record.dual_src_role_ids for record in batch_records], pad_id=0)
            batch["dual_knowledge_role_ids"] = pad_batch_data(self._get_batch_knowledge_ids(batch_records, "role_ids"), pad_id=0)
        batch["dual_src_attention_mask"] = self._gen_self_attn_mask(dual_src_batch_token_ids, is_unidirectional=False)
        batch["dual_knowledge_attention_mask"] = self._gen_self_attn_mask(dual_knowledge_batch_token_ids, is_unidirectional=False)

        return batch

    def _pad_batch_records(self, batch_records, is_infer, phase=None):
        """Pad batch records and mask."""
        if self.do_kag_training:
            batch = self._pad_batch_records_for_training(batch_records)
        else:
            batch = super(KAGReader, self)._pad_batch_records(batch_records, is_infer, phase)
        return batch
