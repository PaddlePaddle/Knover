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
"""Unified Transformer model."""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from knover.models import register_model
from knover.core.model import Model
from knover.modules.generator import Generator
from knover.utils import gather, str2bool


@register_model("UnifiedTransformer")
class UnifiedTransformer(Model):
    """Unified Transformer"""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = Model.add_cmdline_args(parser)
        group.add_argument("--weight_sharing", type=str2bool, default=True,
                           help="Whether to share the token embedding with the output FC.")
        group.add_argument("--use_role", type=str2bool, default=False,
                           help="Whether use role embeddings.")

        Generator.add_cmdline_args(parser)
        return group

    def _build_model(self, args):
        self.max_seq_len = args.max_seq_len

        # initializer
        initializer = paddle.nn.initializer.TruncatedNormal(std=args.initializer_range)
        param_attr = paddle.ParamAttr(initializer=initializer)

        self.emb_size = args.get("emb_size", args.hidden_size)
        self.hidden_size = args.hidden_size
        self.dropout = args.hidden_dropout_prob

        self.n_layer = args.num_hidden_layers
        self.n_head = args.num_attention_heads
        self.d_key = args.get("key_size", self.hidden_size // self.n_head)
        self.d_value = args.get("value_size", self.hidden_size // self.n_head)
        self.inner_hidden_size = args.get("inner_hidden_size", self.hidden_size * 4)

        # embeddings
        self.vocab_size = args.vocab_size
        self.type_size = args.type_vocab_size
        self.pos_size = args.max_position_embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_size, weight_attr=param_attr)
        self.type_embedding = nn.Embedding(self.type_size, self.emb_size, weight_attr=param_attr)
        self.pos_embedding = nn.Embedding(self.pos_size, self.emb_size, weight_attr=param_attr)

        # role embeddings
        self.use_role = args.use_role
        if self.use_role:
            self.role_embedding = nn.Embedding(args.role_type_size, self.emb_size, weight_attr=param_attr)

        # embeding mapping
        if self.hidden_size != self.emb_size:
            self.emb_mapping_in = True
        else:
            self.emb_mapping_in = args.get("emb_mapping_in", False)
        if self.emb_mapping_in:
            self.emb_mapping_fc = nn.Linear(self.emb_size, self.hidden_size, weight_attr=param_attr)

        # transformer encoder
        self.normalize_before = args.get("normalize_before", True)
        self.hidden_act = args.hidden_act
        encoder_layer = nn.TransformerEncoderLayer(
            self.hidden_size,
            self.n_head,
            self.inner_hidden_size,
            dropout=self.dropout,
            activation=self.hidden_act,
            act_dropout=0,
            normalize_before=self.normalize_before,
            weight_attr=param_attr)
        if self.normalize_before:
            output_norm = nn.LayerNorm(self.hidden_size)
        else:
            output_norm = None
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_layer, output_norm)

        # lm head
        self.lm_trans_fc = nn.Linear(self.hidden_size, self.hidden_size, weight_attr=param_attr)
        self.activation = getattr(F, self.hidden_act)
        self.lm_trans_norm = nn.LayerNorm(self.hidden_size)
        self.weight_sharing = args.weight_sharing
        if self.weight_sharing:
            self.lm_logits_bias = paddle.create_parameter(
                [self.vocab_size], "float32", name="lm_out_fc.b_0", is_bias=True)
        else:
            self.lm_out_fc = nn.Linear(self.emb_size, self.emb_size, weight_attr=param_attr)

        # task-related
        from knover.modules.generator import GENERATOR_REGISTRY
        generator_cls = GENERATOR_REGISTRY[args.decoding_strategy]
        self.generator = generator_cls(args)
        self.do_generation = args.get("do_generation", False)

    def _gen_input(self,
                   token_ids,
                   type_ids,
                   pos_ids,
                   role_ids,
                   input_mask,
                   aux_emb=None):
        """Generate input embeddings of Transformer

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            pos_ids: represents the position of each token, shape is [batch_size, max_seq_len, 1]
            input_mask: represents the attention masking mastrix in each Transformer blocks,
                shape is [batch_size, max_seq_len, max_seq_len]
            aux_emb: represents the auxiliary input embeddings of Transformer.

        Returns:
            A Tuple contains the input embeddings and the attention masking matrix of Transformer.
        """
        token_emb_out = self.token_embedding(token_ids)
        type_emb_out = self.type_embedding(type_ids)
        pos_emb_out = self.pos_embedding(pos_ids)
        emb_out = token_emb_out + type_emb_out + pos_emb_out

        if self.use_role:
            role_emb_out = self.role_embedding(role_ids)
            emb_out = emb_out + role_emb_out

        # concat auxiliary memory embeddings
        if aux_emb is not None:
            emb_out = paddle.concat([aux_emb, emb_out], axis=1)

        if self.emb_mapping_in:
            emb_out = self.emb_mapping_fc(emb_out)

        # generate n-head self-attention mask
        self_attn_mask = paddle.scale(x=input_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.unsqueeze(self_attn_mask, [1])

        return emb_out, n_head_self_attn_mask

    def _generation_network(self,
                            token_ids,
                            type_ids,
                            pos_ids,
                            role_ids,
                            generation_mask,
                            aux_emb=None):
        """Run Transformer generation network.

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            pos_ids: represents the position of each token, shape is [batch_size, max_seq_len, 1]
            input_mask: represents the attention masking mastrix in each Transformer blocks,
                shape is [batch_size, max_seq_len, max_seq_len]
            aux_emb: represents the auxiliary input embeddings of Transformer.

        Returns:
            The output embeddings of Transformer.
        """
        emb_input, n_head_self_attn_mask = self._gen_input(
            token_ids, type_ids, pos_ids, role_ids, generation_mask, aux_emb=aux_emb)
        if self._generation_caches is None:
            enc_out =  self._encode(emb_input, n_head_self_attn_mask)
        else:
            enc_out, self._generation_caches = self._encode(
                emb_input, n_head_self_attn_mask, self._generation_caches)
        return enc_out

    def _generation_step(self, state):
        # gather caches
        if "parent_idx" in state:
            self._generation_caches = gather(self._generation_caches, state["parent_idx"])
        enc_out = self._generation_network(
            state["token_ids"],
            state["type_ids"],
            state["pos_ids"],
            state.get("role_ids", None),
            state["tgt_generation_mask"],
        )
        logits = self._calc_logits(enc_out)
        return logits

    def _encode(self, emb_input, n_head_self_attn_mask, caches=None):
        """Run Transformer encode pass.

        Args:
            emb_input: represents the input embeddings fo Transformer, shape is [batch_size, max_seq_len, hidden_dim]
            n_head_self_attn_mask: represents the attention masking matrix,
                shape is [batch_size, num_heads, max_seq_len, max_seq_len]

        Returns:
            The output embeddings of Transformer.
        """
        return self.encoder(emb_input, n_head_self_attn_mask, caches)

    def _calc_logits(self, enc_out, tgt_idx=None):
        """Get the logits of generation task.

        The network may share weight with token embeddings.

        Args:
            enc_out: the output embeddings of Transformer, shape is [batch_size, max_seq_len, hidden_dim]
            tgt_idx (optional): the indices of prediction tokens, shape is [num_predictions, 2].

        Returns:
            logits: the logits of prediction task, shape is [num_predictions, vocab_size].
        """
        if tgt_idx is None:
            seq_feat = paddle.reshape(x=enc_out, shape=[-1, self.hidden_size])
        elif len(tgt_idx.shape) == 2 and tgt_idx.shape[1] == 2:
            seq_feat = paddle.gather_nd(enc_out, tgt_idx)
        else:
            raise ValueError(f"Invalid indices shape {tgt_idx.shape} is used")

        seq_trans_feat = self.lm_trans_fc(seq_feat)
        seq_trans_feat = self.activation(seq_trans_feat)
        seq_trans_feat = self.lm_trans_norm(seq_trans_feat)

        if self.weight_sharing:
            logits = paddle.matmul(
                seq_trans_feat, self.token_embedding.weight, transpose_y=True)
            logits += self.lm_logits_bias
        else:
            logits = self.lm_out_fc(seq_trans_feat)
        return logits


    def forward(self, inputs, is_infer=False):
        """Run model main forward."""
        outputs = {}
        if is_infer:
            self._generation_caches = self.encoder.gen_cache(inputs["token_ids"])
        else:
            self._generation_caches = None

        outputs["enc_out"] = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            pos_ids=inputs["pos_ids"],
            role_ids=inputs.get("role_ids", None),
            generation_mask=inputs["generation_mask"]
        )
        return outputs

    def get_metrics(self, inputs, outputs):
        """Get metrics."""
        metrics = {}

        tgt_logits = self._calc_logits(outputs["enc_out"], inputs["tgt_idx"])
        mean_tgt_lm_loss = F.cross_entropy(tgt_logits, inputs["tgt_label"])
        metrics["token_lm_loss"] = mean_tgt_lm_loss

        loss = mean_tgt_lm_loss
        metrics["loss"] = loss
        return metrics

    def get_statistics(self, inputs, outputs):
        """Get statistics."""
        statistics = {}
        if "tgt_label" in inputs:
            statistics["tokens_num"] = inputs["tgt_label"].shape[0]
        statistics["batch_size"] = inputs["token_ids"].shape[0]
        return statistics

    def infer(self, inputs, outputs):
        """Run model inference.

        Only support generation now.
        """
        if self.do_generation:
            outputs = self.generator(self, inputs, outputs)
            data_id_list = outputs["data_id"].numpy()
            token_ids_list = outputs["token_ids"].numpy()
            seq_ids_list = outputs["finished_ids"].numpy()
            score_list = outputs["finished_score"].numpy()
            predictions = []
            for data_id, token_ids, seq_ids, score in zip(data_id_list, token_ids_list, seq_ids_list, score_list):
                if len(seq_ids.shape) == 1:
                    pred = {}
                    pred["data_id"] = int(data_id)
                    pred["decode_score"] = float(score)
                    pred["context_token_ids"] = token_ids
                    pred["response_token_ids"] = seq_ids
                    predictions.append(pred)
                else:
                    for candidate_seq_ids, candidate_score in zip(seq_ids, score):
                        pred = {}
                        pred["data_id"] = int(data_id)
                        pred["decode_score"] = float(candidate_score)
                        pred["context_token_ids"] = token_ids
                        pred["response_token_ids"] = candidate_seq_ids
                        predictions.append(pred)
            return predictions
        else:
            raise NotImplementedError
