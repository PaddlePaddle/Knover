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
"""Dialogue Reader."""

from collections import namedtuple
import os

import numpy as np 
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet

from knover.utils import mask, open_file, pad_batch_data, rindex, str2bool, to_optimized_size
import knover.utils.tokenization as tokenization


class DialogReader(object):
    """The implement of DialogReader."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = parser.add_argument_group("Reader")
        group.add_argument("--max_src_len", type=int, default=128,
                           help="The maximum length of source sequence (context in dialogue generation task).")
        group.add_argument("--max_tgt_len", type=int, default=128,
                           help="The maximum length of target sequence (response in dialogue generation task).")
        group.add_argument("--max_seq_len", type=int, default=256,
                           help="The maximum length of sequence.")
        group.add_argument("--max_knowledge_len", type=int, default=0,
                           help="The maximum length of knowledge sequence.")
        group.add_argument("--knowledge_position", type=str, default="post_src",
                           choices=["post_src", "pre_src"],
                           help="How to concatenate source sequence and knowledge sequence. "
                           "`post_src` means concatenate knowledge sequence after source sequence, "
                           "and `pre_src` means concatenate knowledge sequence before source sequence.")
        group.add_argument("--knowledge_style", type=str, default="original",
                           choices=["original", "reversed"],
                           help="How to concatenate multipe knowledges. `original` concatenate knowledges in "
                           "original order, and `reversed` means concatenate knowledges in reversed order.")
        group.add_argument("--truncate_first_turn", type=str2bool, default=False,
                           help="Whether truncate the first turn.")
        group.add_argument("--file_format", type=str, default="file",
                           choices=["file", "filelist"],
                           help="The input file format.")
        group.add_argument("--data_format", type=str, default="raw",
                           choices=["raw", "tokenized", "numerical"],
                           help="The data format of each file.")
        group.add_argument("--in_tokens", type=str2bool, default=False,
                           help="Whether batchify examples by the number of tokens.")
        group.add_argument("--batch_size", type=int, default=16,
                           help="The size of batches. If `in_tokens` is false, then batchify every X examples."
                           "If `in_tokens` is true, then batchify examples which contains nearly X tokens.")
        group.add_argument("--position_style", type=str, default="continuous",
                           choices=["continuous", "relative"],
                           help="The position encoding style.")
        group.add_argument("--random_seed", type=int, default=11,
                           help="The seed to control the data generation.")
        group.add_argument("--shuffle_pool_size", type=int, default=0,
                           help="The size of shuffle pool."
                           "If it is positive, we will shuffle each X examples and then batchify them.")
        group.add_argument("--sort_pool_size", type=int, default=2 ** 16,
                           help="The size of sorting pool. If it is positive, we will generate batches from sorted "
                           "example pool (containing X examples).")

        tokenizer_group = parser.add_argument_group("Tokenizer")
        tokenizer_group.add_argument("--tokenizer", type=str, default="SentencePieceTokenizer")
        args, _ = parser.parse_known_args()
        tokenizer_cls = getattr(tokenization, args.tokenizer)
        tokenizer_cls.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        tokenizer_cls = getattr(tokenization, args.tokenizer)
        self.tokenizer = tokenizer_cls(args)
        self.vocab = self.tokenizer.vocab
        self.pad_id = args.pad_id = self.tokenizer.pad_id
        self.bos_id = args.bos_id = self.tokenizer.bos_id
        self.eos_id = args.eos_id = self.tokenizer.eos_id
        self.unk_id = args.unk_id = self.tokenizer.unk_id
        self.mask_id = args.mask_id = self.tokenizer.mask_id
        self.vocab_size = args.get("vocab_size", self.tokenizer.vocab_size)
        self.max_src_len = args.max_src_len
        self.max_tgt_len = args.max_tgt_len
        self.max_knowledge_len = args.max_knowledge_len
        self.knowledge_position = args.knowledge_position
        self.knowledge_style = args.knowledge_style
        self.truncate_first_turn = args.truncate_first_turn
        self.file_format = args.file_format
        self.data_format = args.data_format
        self.in_tokens = args.in_tokens
        self.batch_size = args.batch_size
        self.position_style = args.position_style
        self.sort_pool_size = args.sort_pool_size
        self.shuffle_pool_size = args.shuffle_pool_size

        self.reserve_example = args.get("reserve_example", False)

        if self.shuffle_pool_size > 0 and self.sort_pool_size > 0:
            raise ValueError(f"Cannot set `shuffle_pool_size = {self.shuffle_pool_size}`"
                             f" and `sort_pool_size = ${self.sort_pool_size}` both positive.")

        assert args.max_knowledge_len + args.max_src_len + args.max_tgt_len <= args.max_seq_len, \
            "args.max_knowledge_len + args.max_src_len + args.max_tgt_len > args.max_seq_len"

        # random_seed must be set for data slicing when using multi-gpu
        self.global_rng = np.random.RandomState(args.random_seed)

        # training progress
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0
        self.data_id = 0

        # model related
        self.use_role = args.use_role

        self.fields = ["token_ids", "type_ids", "pos_ids"]
        if args.use_role:
            self.fields.append("role_ids")
        self.num_numerical_fields = len(self.fields)
        self.fields += ["tgt_start_idx", "data_id"]

        self.Record = namedtuple("Record", self.fields, defaults=(None,) * len(self.fields))

        if self.reserve_example:
            self.features = {}
        return

    def sort_key(self, record):
        """The key of record.

        We will apply sorting before batching. It can decrease the number of padding and
        speedup training.
        """
        return [to_optimized_size(len(record.token_ids))]

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_epoch, self.current_file_index, self.total_file

    def _parse_src(self, src):
        """Parse source sequence and return corresponding fields."""
        # process src
        src_token_ids = []
        src_pos_ids = []
        if self.use_role:
            src_role_ids = []
            role_id_list = []

        # tokenize src
        s_token_ids_list = []
        for s in src.split("[SEP]"):
            s = s.strip()
            if self.use_role and "\1" in s:
                s, role_id = s.split("\1")
                role_id = int(role_id)
                role_id_list.append(role_id)

            if self.data_format == "tokenized":
                s_tokens = s.split(" ")
            else:
                s_tokens = self.tokenizer.tokenize(s)

            s_token_ids = self.tokenizer.convert_tokens_to_ids(s_tokens) + [self.eos_id]
            s_token_ids_list.append(s_token_ids)

        # trim src
        idx = len(s_token_ids_list) - 1
        total_token_num = 1
        while idx >= 0:
            total_token_num += len(s_token_ids_list[idx])
            if total_token_num > self.max_src_len:
                if self.truncate_first_turn and idx == 0:
                    truncated_ids = s_token_ids_list[idx][:self.max_src_len - total_token_num]
                    if len(truncated_ids) > 1:
                        s_token_ids_list[idx] = truncated_ids[:-1] + [self.eos_id]
                        idx -= 1
                break
            idx -= 1

        if self.use_role and len(role_id_list) == 0:
            for i in range(len(s_token_ids_list)):
                role_id_list.append((len(s_token_ids_list) - i) % 2)
        for i, s_token_ids in enumerate(s_token_ids_list[idx + 1:], idx + 1):
            src_token_ids += s_token_ids
            src_pos_ids += list(range(1, len(s_token_ids) + 1))
            if self.use_role:
                src_role_ids += [role_id_list[i]] * len(s_token_ids)

        field_values = {
            "token_ids": [self.bos_id] + src_token_ids,
            "type_ids": [0] * (len(src_token_ids) + 1),
            "pos_ids": [0] + src_pos_ids
        }
        if self.use_role:
            field_values["role_ids"] = [0] + src_role_ids

        for k in field_values:
            assert len(field_values[k]) == len(field_values["token_ids"]), \
                f"len(field_values[{k}]) != len(field_values['token_ids'])"
        return field_values

    def _parse_knowledge(self, knowledge):
        """Parse knowledge sequence and return corresponding fields."""
        ks_token_ids = [self.bos_id]
        ks_pos_ids = [0]
        if self.knowledge_style == "original":
            ks = knowledge.split("[SEP]")
        elif self.knowledge_style == "reversed":
            ks = reversed(knowledge.split("[SEP]"))
        for k in ks:
            k = k.strip()
            if self.data_format == "tokenized":
                k_tokens = k.split(" ")
            else:
                k_tokens = self.tokenizer.tokenize(k)

            k_token_ids = self.tokenizer.convert_tokens_to_ids(k_tokens) + [self.eos_id]
            ks_token_ids += k_token_ids
            ks_pos_ids += list(range(1, len(k_token_ids) + 1))

        if len(ks_token_ids) > self.max_knowledge_len:
            if self.knowledge_style == "original":
                ks_token_ids = ks_token_ids[:self.max_knowledge_len]
                ks_pos_ids = ks_pos_ids[:self.max_knowledge_len]
            else:
                ks_token_ids = ks_token_ids[-self.max_knowledge_len:]
                ks_pos_ids = ks_pos_ids[-self.max_knowledge_len:]

        field_values = {
            "token_ids": ks_token_ids,
            "type_ids": [2] * len(ks_token_ids),
            "pos_ids": ks_pos_ids
        }
        if self.use_role:
            field_values["role_ids"] = [0] * len(ks_token_ids)

        return field_values

    def _parse_tgt(self, tgt):
        """Parse target sequence and return corresponding fields."""
        # process tgt
        tgt = tgt.strip()
        if self.use_role:
            if "\1" in tgt:
                tgt, role_id = tgt.split("\1")
                role_id = int(role_id)
            else:
                role_id = 0
        if self.data_format == "tokenized":
            tgt_tokens = tgt.split(" ")
        else:
            tgt_tokens = self.tokenizer.tokenize(tgt)

        tgt_token_ids = self.tokenizer.convert_tokens_to_ids(tgt_tokens)
        tgt_token_ids.append(self.eos_id)

        # trim tgt
        tgt_token_ids = tgt_token_ids[:self.max_tgt_len - 1]

        tgt_token_ids = [self.bos_id] + tgt_token_ids

        field_values = {
            "token_ids": tgt_token_ids,
            "type_ids": [1] * len(tgt_token_ids),
            "pos_ids": list(range(len(tgt_token_ids)))
        }
        if self.use_role:
            field_values["role_ids"] = [role_id] * len(tgt_token_ids)

        return field_values

    def _merge_field_values(self, field_values1, field_values2):
        if field_values1 is None:
            return field_values2
        if field_values2 is None:
            return field_values1
        return {
            k: field_values1[k] + field_values2[k]
            for k in field_values1
        }

    def _convert_example_to_record(self, example, is_infer):
        """Convert an example to a record which can be used as the model's input."""
        if "src" in example._fields:
            field_values = self._parse_src(example.src)
            if len(field_values["token_ids"]) == 1:
                raise ValueError(f"Invalid example: context too long / no context - {example}")
        else:
            field_values = None

        if self.max_knowledge_len > 0:
            # add knowledge
            knowledge_field_values = self._parse_knowledge(example.knowledge)
            if self.knowledge_position == "pre_src":
                field_values = self._merge_field_values(knowledge_field_values, field_values)
            else:
                field_values = self._merge_field_values(field_values, knowledge_field_values)

        tgt_start_idx = len(field_values["token_ids"]) if field_values is not None else 0

        if self.position_style == "relative" and field_values is not None:
            ctx_len = len(field_values["token_ids"])
            field_values["pos_ids"] = [
                self.max_tgt_len + ctx_len - i - 1
                for i in range(ctx_len)
            ]

        if not is_infer and hasattr(example, "tgt"):
            tgt_field_values = self._parse_tgt(example.tgt)
            field_values = self._merge_field_values(field_values, tgt_field_values)

        if self.position_style == "continuous":
            field_values["pos_ids"] = list(range(len(field_values["token_ids"])))

        field_values["tgt_start_idx"] = tgt_start_idx
        field_values["data_id"] = example.data_id

        record = self.Record(**field_values)
        return record

    def _read_tsv(self, fp, phase, is_infer, delimiter="\t", quotechar=None):
        """Read a tab separated value file and yield records."""
        headers = next(fp).rstrip("\n").split(delimiter)
        headers.append("data_id")
        Example = namedtuple("Example", headers)

        def __wrapper__():
            for i, line in enumerate(fp):
                line = line.rstrip("\n").split(delimiter)
                example = Example(*line, data_id=self.data_id)
                self.data_id += 1
                if self.reserve_example and (is_infer or phase.endswith("test")):
                    self.features[i] = example
                yield example
        return __wrapper__

    def _read_numerical_file(self, fp, phase, is_infer, delimiter=";"):
        """Read a file which contains numerical data and yield records."""
        for i, line in enumerate(fp):
            cols = line.strip().split(delimiter)
            cols = list(map(lambda x: list(map(int, x.split(" "))), cols))
            if len(cols) > self.num_numerical_fields:
                cols = cols[:self.num_numerical_fields]
            if is_infer:
                tgt_start_idx = len(cols[0])
            else:
                # get the start position of target sequence
                # if you change the numerical data format, you must to make sure the last part of
                # numerical sequence is the target sequence
                tgt_start_idx = rindex(cols[0], self.bos_id)
            record = self.Record(*cols, tgt_start_idx=tgt_start_idx, data_id=self.data_id)
            self.data_id += 1
            yield record

    def _read_file(self, input_file, phase, is_infer):
        """Read a data file and yield records."""
        def __wrapper__():
            with open_file(input_file) as fp:
                if self.data_format == "numerical":
                    yield from self._read_numerical_file(fp, phase, is_infer)
                else:
                    gen_examples = self._read_tsv(fp, phase, is_infer)
                    for example in gen_examples():
                        try:
                            yield self._convert_example_to_record(example, is_infer)
                        except ValueError as e:
                            if "Invalid example" in str(e):
                                print(f"[WARN] {e}")
                            else:
                                raise e

        return __wrapper__

    def _read_files(self, filelist, phase, is_infer, shuffle_files):
        """Read multiply files and yield records."""
        input_files = open(filelist).readlines()
        def __wrapper__():
            if shuffle_files:
                self.global_rng.shuffle(input_files)

            if phase == "train":
                self.total_file = len(input_files)
            for file_index, input_file in enumerate(input_files, 1):
                if phase == "train":
                    self.current_file_index = file_index
                    self.current_file = input_file
                file_reader = self._read_file(input_file.strip(), phase, is_infer)
                for record in file_reader():
                    yield record

        return __wrapper__

    def _shuffle_reader(self, reader, shuffle_pool_size):
        """Shuffle examples."""
        def get_batch(pool):
            self.global_rng.shuffle(pool)
            for record in pool:
                yield record

        def __wrapper__():
            pool = []
            for record in reader():
                pool.append(record)
                if len(pool) == shuffle_pool_size:
                    yield from get_batch(pool)
                    pool = []
            if len(pool) > 0:
                yield from get_batch(pool)

        return __wrapper__

    def _update_max_lens(self, max_lens, record):
        """Update max_lens."""
        if max_lens is None:
            return self.sort_key(record)
        else:
            return [max(max_len, l) for max_len, l in zip(max_lens, self.sort_key(record))]

    def _get_batch(self, reader):
        """Yield batches from record reader."""
        def __wrapper__():
            batch, max_lens = [], None
            for record in reader():
                if record is None:
                    yield batch
                    batch, max_lens = [], None
                    continue

                self.current_example += 1
                max_lens = self._update_max_lens(max_lens, record)
                if self.in_tokens:
                    to_append = (len(batch) + 1) * sum(max_lens) <= self.batch_size
                else:
                    to_append = len(batch) < self.batch_size
                if to_append:
                    batch.append(record)
                else:
                    yield batch
                    batch, max_lens = [record], self.sort_key(record)

            if len(batch) > 0:
                yield batch
        return __wrapper__

    def _get_sorted_batch(self, reader):
        """Yield sorted batches from record reader."""
        def _get_sorted_batch_from_pool(pool):
            """Generate sorted batches from pool."""
            pool = sorted(pool, key=self.sort_key)
            batches = []
            batch, max_lens = [], None
            for record in pool:
                self.current_example += 1
                max_lens = self._update_max_lens(max_lens, record)
                if self.in_tokens:
                    to_append = (len(batch) + 1) * sum(max_lens) <= self.batch_size
                else:
                    to_append = len(batch) < self.batch_size
                if to_append:
                    batch.append(record)
                else:
                    batches.append(batch)
                    batch, max_lens = [record], self.sort_key(record)

            if len(batch) > 0:
                batches.append(batch)
            self.global_rng.shuffle(batches)

            for batch in batches:
                yield batch

        def __wrapper__():
            pool = []
            for record in reader():
                pool.append(record)
                if len(pool) == self.sort_pool_size:
                    yield from _get_sorted_batch_from_pool(pool)
                    pool = []
            if len(pool) > 0:
                yield from _get_sorted_batch_from_pool(pool)

        return __wrapper__

    def _batch_reader(self, reader, phase=None, is_infer=False):
        """Construct a batch reader from a record reader."""
        if self.sort_pool_size > 0 and phase == "train":
            return self._get_sorted_batch(reader)
        else:
            return self._get_batch(reader)

    def _distributed_batch_reader(self, batch_reader, num_part, part_id, is_test=False):
        """Distributed batch reader.

        Slice dataset and feed batches to different devices.

        Args:
            batch_reader: A batch reader.
            num_part: The number of devices.
            part_id: The id of current device.
            is_test: Whether slice dataset in testing phase. When it sets false, we will drop the last batches
                if the number of remainder batches is less than the number of devices.

        Returns:
            reader: A distributed Reader.
        """
        def __wrapper__():
            batches = []
            for batch in batch_reader():
                batches.append(batch)
                if len(batches) == num_part:
                    yield batches[part_id]
                    batches = []
            if is_test and 0 <= part_id < len(batches):
                yield batches[part_id]
            return

        return __wrapper__

    def data_generator(self,
                       input_file=None,
                       reader=None,
                       num_epochs=1,
                       num_part=1,
                       part_id=0,
                       phase=None,
                       is_infer=False):
        """Yield batches from a file or a record reader.

        If `reader` is set, it will yield batches from the record reader, otherwise yield batches from the given
        `input_file`.

        Args:
            input_file: The path of input file. The format of this file is controlled by Reader.file_format and
                Reader.data_format.
            reader: The record reader.
            num_epochs: The number of times the learning algorithm will work through the entire training dataset.
            num_part: The number of devices.
            part_id: The id of current device.
            phase: The type of dataset, which can be one of 'train' / 'valid' / 'test'.
            is_infer: Whether to run inference on this dataset.
        """
        def __wrapper__():
            nonlocal reader
            if reader is None:
                self.data_id = 0
                if self.file_format == "filelist":
                    reader = self._read_files(input_file, phase, is_infer, not phase.endswith("test"))
                else:
                    if phase == "train":
                        self.total_file = 1
                        self.current_file_index = 1
                        self.current_file = input_file
                    reader = self._read_file(input_file, phase, is_infer)

            if self.shuffle_pool_size > 0:
                reader = self._shuffle_reader(reader, self.shuffle_pool_size)

            batch_reader = self._batch_reader(reader, phase, is_infer)
            if phase == "train":
                batch_reader = self._distributed_batch_reader(batch_reader, num_part, part_id)
            elif phase.startswith("distributed"):
                batch_reader = self._distributed_batch_reader(batch_reader, num_part, part_id, is_test=True)

            for epoch_index in range(num_epochs):
                try:
                    if phase == "train":
                        self.current_example = 0
                        self.current_epoch = epoch_index + 1
                    for batch in batch_reader():
                        yield self._pad_batch_records(batch, is_infer, phase=phase)
                except:
                    import traceback
                    traceback.print_exc()
                    raise

        return __wrapper__

    def _gen_self_attn_mask(self, batch_token_ids, batch_tgt_start_idx=None, is_unidirectional=True, num_aux_token=0):
        """Generate self attention masking matrix.

        This is a helpful function to generate different types of attention masking matrix.
        1. Bi-directional: all tokens can attent to all other tokens.
        2. Uni-directional: all tokens can only attent to their former tokens.
        3. Seq2seq: tokens in source sequence can attent each other, tokens in target sequence can only attent the
            tokens in source sequence and the former token in target sequence.

        Args:
            batch_token_ids: A batch of token ids.
            batch_tgt_start_idx: A batch of indices which represent the starting index of target sequence.
            is_unidirectional: Whether generate uni-directional masking matrix. When `batch_tgt_start_idx` is not
                `None` and `is_unidirectional` is True, then it will generate seq2seq masking matrix (source sequence
                is bi-directional attention and target sequence is uni-directional attention).
            num_aux_token: The number of auxiliary tokens. The auxiliary tokens will concatenate to the begin of
                sequence. They are considered as a part of source sequence.
        """
        max_len = to_optimized_size(max(map(len, batch_token_ids)))
        input_mask_data = np.zeros((len(batch_token_ids), max_len + num_aux_token, max_len + num_aux_token))
        if is_unidirectional:
            for index, mask_data in enumerate(input_mask_data):
                start = 0 if batch_tgt_start_idx is None else batch_tgt_start_idx[index]
                end = len(batch_token_ids[index])
                mask_data[:end + num_aux_token, :start + num_aux_token] = 1.0
                # Generate the lower triangular matrix using the slice of matrix
                b = np.tril(np.ones([end - start, end - start]), 0)
                mask_data[start + num_aux_token:end + num_aux_token, start + num_aux_token:end + num_aux_token] = b
        else:
            for index, token_ids in enumerate(batch_token_ids):
                input_mask_data[index, :len(token_ids) + num_aux_token, :len(token_ids) + num_aux_token] = 1.0
        return input_mask_data.astype("float32")

    def _pad_batch_records(self, batch_records, is_infer, phase=None):
        """Padding a batch of records and construct model's inputs.

        This function can be override by its subclass if necessary.
        """
        batch_size = len(batch_records)
        batch = {}
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_type_ids = [record.type_ids for record in batch_records]
        batch_pos_ids = [record.pos_ids for record in batch_records]
        if self.use_role:
            batch_role_ids = [record.role_ids for record in batch_records]
        batch["token_ids"] = pad_batch_data(batch_token_ids, pad_id=self.pad_id)
        batch["type_ids"] = pad_batch_data(batch_type_ids, pad_id=0)
        batch["pos_ids"] = pad_batch_data(batch_pos_ids, pad_id=0)
        if self.use_role:
            batch["role_ids"] = pad_batch_data(batch_role_ids, pad_id=0)

        batch_tgt_start_idx = [record.tgt_start_idx for record in batch_records]
        batch["generation_mask"] = self._gen_self_attn_mask(
            batch_token_ids,
            batch_tgt_start_idx=batch_tgt_start_idx)

        if is_infer:
            bsz = len(batch_token_ids)
            batch["tgt_ids"] = np.full([bsz, 1], self.bos_id, dtype="int64")
            if self.position_style == "continuous":
                batch["tgt_pos"] = np.array(batch_tgt_start_idx, dtype="int64").reshape(-1, 1)
            else:
                batch["tgt_pos"] = np.zeros_like(batch_tgt_start_idx, dtype="int64").reshape(-1, 1)
            batch["tgt_generation_mask"] = batch["generation_mask"][:, :1, :]
            batch_data_id = [record.data_id for record in batch_records]
            batch["data_id"] = np.array(batch_data_id).astype("int64")
        else:
            batch["tgt_label"], batch["tgt_idx"] = mask(
                batch_tokens=batch_token_ids,
                vocab_size=self.vocab_size,
                tgt_starts=batch_tgt_start_idx,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                mask_id=self.mask_id,
                is_unidirectional=True)

        return batch
