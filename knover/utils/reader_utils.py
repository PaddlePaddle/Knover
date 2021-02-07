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
"""Reader utility."""

import copy

import numpy as np


def mask(batch_tokens,
         vocab_size,
         bos_id=1,
         eos_id=2,
         mask_id=3,
         tgt_starts=None,
         labels=None,
         is_unidirectional=False,
         use_latent=False,
         use_bow=False):
    """Add masking and return target's labels and indices.

    Add mask for batch_tokens, return out, mask_label, mask_idx;
    Note: mask_idx (masking index) corresponding to the indices of masking token in batch_tokens after padding.
    """
    batch_tokens = copy.deepcopy(batch_tokens)
    max_len = max(map(len, batch_tokens))
    mask_label = []
    mask_idx = []
    if labels is not None:
        label_idx = []

    if is_unidirectional:
        # unidirectional language model
        if use_latent:
            max_len += 1
            num_aux_token = 1
        else:
            num_aux_token = 0
        for sent_index, sent in enumerate(batch_tokens):
            sent_b_index = tgt_starts[sent_index] if tgt_starts is not None else 0
            need_cal = True
            if labels is not None:
                label_idx.extend([sent_index, len(sent) - 1 + num_aux_token])
                if labels[sent_index] == 0:
                    need_cal = False
            mask_label.extend(sent[sent_b_index + 1:])
            mask_idx.extend([
                [sent_index, i + num_aux_token]
                for i in range(sent_b_index, len(sent) - 1)
            ])
        mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
        mask_idx = np.array(mask_idx).astype("int64").reshape([-1, 2])
        return_list = [mask_label, mask_idx]

        # latent related (bow label and pos)
        if use_latent and use_bow:
            bow_label = []
            bow_idx = []
            for sent_index, sent in enumerate(batch_tokens):
                sent_b_index = tgt_starts[sent_index] if tgt_starts is not None else 0
                def __filter__(tok_id):
                    # TODO: exclude [EOS] from bow loss
                    return True
                bow_idx.extend([
                    sent_index
                    for i in range(sent_b_index + 1, len(sent))
                    if __filter__(sent[i])
                ])
                bow_label.extend([
                    sent[i]
                    for i in range(sent_b_index + 1, len(sent))
                    if __filter__(sent[i])
                ])
            bow_label = np.array(bow_label).astype("int64").reshape([-1, 1])
            bow_idx = np.array(bow_idx).astype("int64").reshape([-1, 1])
            return_list += [bow_label, bow_idx]
    else:
        # bidirectional mask language model
        total_token_num = sum(map(len, batch_tokens))
        prob_mask = np.random.rand(total_token_num)
        # TODO: fix replace_ids, include [UNK]
        replace_ids = np.random.randint(3, high=vocab_size, size=total_token_num)
        prob_index = 0
        for sent_index, sent in enumerate(batch_tokens):
            # add pair label position
            if labels is not None:
                label_idx.append(sent_index * max_len)

            # add mask label and position
            for token_index, token in enumerate(sent):
                if token == eos_id or token == bos_id:
                    continue
                prob = prob_mask[prob_index + token_index]
                if prob > 0.15:
                    continue
                elif 0.03 < prob <= 0.15:
                    # mask
                    mask_label.append(sent[token_index])
                    sent[token_index] = mask_id
                    mask_idx.append([sent_index, token_index])
                elif 0.015 < prob <= 0.03:
                    # random replace
                    mask_label.append(sent[token_index])
                    sent[token_index] = replace_ids[prob_index + token_index]
                    mask_idx.append([sent_index, token_index])
                else:
                    # keep the original token
                    mask_label.append(sent[token_index])
                    mask_idx.append([sent_index, token_index])

            prob_index += len(sent)

        mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
        mask_idx = np.array(mask_idx).astype("int64").reshape([-1, 2])
        return_list = [batch_tokens, mask_label, mask_idx]

    if labels is not None:
        label_idx = np.array(label_idx).astype("int64").reshape([-1, 1])
        assert len(labels) == len(label_idx)
        return_list.append(label_idx)
    return return_list
