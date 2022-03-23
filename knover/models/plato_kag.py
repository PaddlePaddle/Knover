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
"""PlatoKAG model."""

import paddle
import paddle.fluid as fluid

import paddle.fluid.layers as layers
import paddle.nn.functional as F

from knover.core.model import Model
from knover.models import register_model
from knover.models.unified_transformer import UnifiedTransformer
from knover.utils import str2bool


@register_model("PlatoKAG")
class PlatoKAG(UnifiedTransformer):
    """Plato-KAG model."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = UnifiedTransformer.add_cmdline_args(parser)
        group.add_argument("--select_top_k", type=int, default=32)
        group.add_argument("--max_knowledge_num", type=int, default=32)
        return group

    def __init__(self, args, place):
        self.top_k = args.select_top_k
        self.max_mask_len = args.max_tgt_len
        self.batch_size = args.batch_size
        self.max_knowledge_num = args.max_knowledge_num

        self.do_dense_emb = args.get("do_dense_emb", False)
        self.do_kag_training = args.get("do_kag_training", False)

        super(PlatoKAG, self).__init__(args, place)

    def _get_feed_dict_for_kag_training(self):
        """Get the feed list of the model for KAG training."""
        feed_dict = {}
        # dual src, [batch_size, max_seq_len, 1]
        feed_dict["dual_src_token_ids"] = layers.data(
            name="dual_src_token_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["dual_src_type_ids"] = layers.data(
            name="dual_src_type_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["dual_src_pos_ids"] = layers.data(
            name="dual_src_pos_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        # dual knowledge, [batch_size * max_knowledge_num, max_seq_len, 1]
        feed_dict["dual_knowledge_token_ids"] = layers.data(
            name="dual_knowledge_token_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["dual_knowledge_type_ids"] = layers.data(
            name="dual_knowledge_type_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["dual_knowledge_pos_ids"] = layers.data(
            name="dual_knowledge_pos_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        # generation, [batch_size, max_knowledge_num, max_seq_len, 1]
        feed_dict["single_token_ids"] = layers.data(
            name="single_token_ids", shape=[-1, self.max_knowledge_num, self.max_seq_len, 1], dtype="int64")
        feed_dict["single_type_ids"] = layers.data(
            name="single_type_ids", shape=[-1, self.max_knowledge_num, self.max_seq_len, 1], dtype="int64")
        feed_dict["single_pos_ids"] = layers.data(
            name="single_pos_ids", shape=[-1, self.max_knowledge_num, self.max_seq_len, 1], dtype="int64")

        if self.use_role:
            feed_dict["dual_src_role_ids"] = layers.data(
                name="dual_src_role_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
            feed_dict["dual_knowledge_role_ids"] = layers.data(
                name="dual_knowledge_role_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
            feed_dict["single_role_ids"] = layers.data(
                name="single_role_ids", shape=[-1, self.max_knowledge_num, self.max_seq_len, 1], dtype="int64")
        if self.use_turn:
            feed_dict["dual_src_turn_ids"] = layers.data(
                name="dual_src_turn_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
            feed_dict["dual_knowledge_turn_ids"] = layers.data(
                name="dual_knowledge_turn_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
            feed_dict["single_turn_ids"] = layers.data(
                name="single_turn_ids", shape=[-1, self.max_knowledge_num, self.max_seq_len, 1], dtype="int64")

        feed_dict["single_attention_mask"] = layers.data(
            name="single_attention_mask",
            shape=[-1, self.max_knowledge_num, self.max_seq_len, self.max_seq_len],
            dtype=self.dtype)

        feed_dict["dual_src_attention_mask"] = layers.data(
            name="dual_src_attention_mask",
            shape=[-1, self.max_seq_len, self.max_seq_len],
            dtype=self.dtype)

        feed_dict["dual_knowledge_attention_mask"] = layers.data(
            name="dual_knowledge_attention_mask",
            shape=[-1, self.max_seq_len, self.max_seq_len],
            dtype=self.dtype)

        # [batch_size, max_knowledge_num, max_seq_len, 1]
        feed_dict["tgt_label"] = layers.data(
            name="tgt_label", shape=[-1, self.max_knowledge_num, self.max_mask_len, 1], dtype="int64")
        # [batch_size, max_knowledge_num, max_seq_len, 3]
        feed_dict["tgt_idx"] = layers.data(
            name="tgt_idx", shape=[-1, self.max_knowledge_num, self.max_mask_len, 3], dtype="int64")

        feed_dict["data_id"] = layers.data(name="data_id", shape=[-1, 1], dtype="int64")

        return feed_dict

    def _get_feed_dict_for_dense_emb(self):
        """Get the feed list of the model for dense embedding."""
        feed_dict = {}
        feed_dict["token_ids"] = layers.data(
            name="token_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["type_ids"] = layers.data(
            name="type_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["pos_ids"] = layers.data(
            name="pos_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")

        if self.use_role:
            feed_dict["role_ids"] = layers.data(
                name="role_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        if self.use_turn:
            feed_dict["turn_ids"] = layers.data(
                name="turn_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")

        feed_dict["attention_mask"] = layers.data(
            name="attention_mask", shape=[-1, self.max_seq_len, self.max_seq_len], dtype=self.dtype)

        feed_dict["data_id"] = layers.data(name="data_id", shape=[-1, 1], dtype="int64")
        return feed_dict

    def _get_feed_dict(self, is_infer=False):
        """
        Get the feed list of the model.

        Args:
            is_infer(bool): True if running inference.

        Returns:
            dict(str, Variable): The feed dict.
        """
        if self.do_kag_training:
            feed_dict = self._get_feed_dict_for_kag_training()
        elif self.do_dense_emb:
            feed_dict = self._get_feed_dict_for_dense_emb()
        else:
            feed_dict = super(PlatoKAG, self)._get_feed_dict(is_infer)
        return feed_dict

    def _get_topk_gather_idx(self, input_token_idx, topk_idx):
        """Get idx for the topk gathering."""
        K = self.top_k
        # B = batch_size
        # shape: [B]
        idx_0 = layers.cumsum(
            layers.fill_constant_batch_size_like(
                input_token_idx, value=1.0, shape=[-1], dtype="int64"), exclusive=True)
        idx_0 = layers.cast(idx_0, dtype="int64")
        # shape: [B, K, 1]
        idx_0 = layers.unsqueeze(idx_0, [1, 2])
        idx_0 = layers.expand(idx_0, [1, K, 1])
        # shape: [B, K, 1]
        idx_1 = layers.unsqueeze(topk_idx, [2])
        # shape: [B, K, 2]
        idx = layers.concat([idx_0, idx_1], axis=2)
        return idx

    def _get_tgt_idx(self, input_token_idx, selected_tgt_idx):
        """Get idx for the target."""
        K = self.top_k
        T = self.max_mask_len

        # B = batch_size
        # [B, K]
        idx_1 = layers.cumsum(
            layers.fill_constant_batch_size_like(
                input_token_idx, value=1.0, shape=[-1, K], dtype="int64"), exclusive=True, axis=1)
        idx_1 = layers.unsqueeze(idx_1, [2, 3])
        idx_1 = layers.expand(idx_1, [1, 1, T, 1])

        # [B, K, T, 3]
        tgt_idx = layers.concat(
            [
                selected_tgt_idx[:, :, :, :1],
                idx_1,
                selected_tgt_idx[:, :, :, 2:]
            ],
            axis=3)

        # [B * K, T, 3]
        tgt_idx = layers.reshape(x=tgt_idx, shape=[-1, T, 3])
        idx_0 = tgt_idx[:, :, 0] * K + tgt_idx[:, :, 1]
        idx_1 = tgt_idx[:, :, 2]

        # [B * K, T, 2]
        tgt_idx = layers.stack(
            [idx_0, idx_1],
            axis=-1)
        return tgt_idx

    def _forward_for_kag_training(self, inputs):
        """Run forward for KAG training."""
        outputs = {}
        self.generation_caches = None

        outputs["dual_knowledge_enc_out"], k_cps = self._generation_network(
            token_ids=inputs["dual_knowledge_token_ids"],
            type_ids=inputs["dual_knowledge_type_ids"],
            pos_ids=inputs["dual_knowledge_pos_ids"],
            role_ids=inputs.get("dual_knowledge_role_ids", None),
            turn_ids=inputs.get("dual_knowledge_turn_ids", None),
            generation_mask=inputs["dual_knowledge_attention_mask"],
            name="dual_encoder"
        )

        outputs["dual_src_enc_out"], src_cps = self._generation_network(
            token_ids=inputs["dual_src_token_ids"],
            type_ids=inputs["dual_src_type_ids"],
            pos_ids=inputs["dual_src_pos_ids"],
            role_ids=inputs.get("dual_src_role_ids", None),
            turn_ids=inputs.get("dual_src_turn_ids", None),
            generation_mask=inputs["dual_src_attention_mask"],
            name="dual_encoder"
        )

        # B = batch_size
        # S = sequence_len
        # d = hidden_state_dimention
        N = self.max_knowledge_num
        K = self.top_k
        T = self.max_mask_len

        # [B, d]
        src_feat = self._get_pooled_output(outputs["dual_src_enc_out"], name="dual_pool")
        # [B, 1, d]
        src_feat = layers.unsqueeze(src_feat, axes=[1])
        # [B * N, d]
        k_feat = self._get_pooled_output(outputs["dual_knowledge_enc_out"], name="dual_pool")
        # [B, N, d]
        k_feat = layers.reshape(x=k_feat, shape=[-1, N, self.hidden_size])
        # [B, 1, N]
        inner_product = layers.matmul(x=src_feat, y=k_feat, transpose_y=True)
        # [B, N]
        inner_product = layers.squeeze(inner_product, axes=[1])

        # [B, N]
        # Some knowledge which is padded to get a batch, and they should be masked from calculating innner product.
        k_mask = layers.reduce_sum(inputs["dual_knowledge_token_ids"], dim=1)
        k_mask  = k_mask > 0
        k_mask = layers.cast(k_mask, "float32")
        k_mask = layers.reshape(x=k_mask, shape=[-1, N])
        inner_product = inner_product * k_mask - 1e6 * (1 - k_mask)

        top_k_values, topk_idx = layers.topk(input=inner_product, k=K)
        # [B, K, 2]
        idx = self._get_topk_gather_idx(inputs["dual_src_token_ids"], topk_idx)
        # [B * K, 2], for gather nd on 4d data
        nd_idx = layers.reshape(x=idx, shape=[-1, 2])

        # [B, K]
        top_k_values = layers.gather_nd(inner_product, idx)
        top_k_softmax = layers.softmax(top_k_values)
        outputs["log_dual_softmax"] = F.log_softmax(top_k_values)

        # shape: [B, N, T, 1] -> [B * K, T, 1]
        outputs["tgt_label"] = layers.gather_nd(inputs["tgt_label"], nd_idx)
        # shape: [B, N, S, 1] -> [B * K, S, 1]
        selected_token_ids = layers.gather_nd(inputs["single_token_ids"], nd_idx)
        selected_type_ids = layers.gather_nd(inputs["single_type_ids"], nd_idx)
        selected_pos_ids = layers.gather_nd(inputs["single_pos_ids"], nd_idx)

        selected_role_ids = None
        if inputs.get("single_role_ids", None):
            selected_role_ids = layers.gather_nd(inputs["single_role_ids"], nd_idx)
        selected_turn_ids = None
        if inputs.get("single_turn_ids", None):
            selected_turn_ids = layers.gather_nd(inputs["single_turn_ids"], nd_idx)

        selected_attention_mask = layers.gather_nd(inputs["single_attention_mask"], nd_idx)

        # Inputs: quries, keys and values should all be 3-D tensors.
        outputs["single_enc_out"], single_cps = self._generation_network(
            token_ids=selected_token_ids,
            type_ids=selected_type_ids,
            pos_ids=selected_pos_ids,
            role_ids=selected_role_ids,
            turn_ids=selected_turn_ids,
            generation_mask=selected_attention_mask
        )

        # [B, N, T, 3] -> [B, K, T, 3]
        selected_tgt_idx = layers.gather_nd(inputs["tgt_idx"], idx)
        # [B * K, T, 2]
        tgt_idx = self._get_tgt_idx(inputs["dual_src_token_ids"], selected_tgt_idx)
        # [B * K * T, 2]
        outputs["tgt_idx"] = layers.reshape(x=tgt_idx, shape=[-1, 2])

        outputs["checkpoints"] = src_cps + k_cps + single_cps

        return outputs

    def _forward_for_dense_emb(self, inputs):
        """Run forward for dense embedding."""
        outputs = {}
        self.generation_caches = None

        outputs["outputs"], outputs["checkpoints"] = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            pos_ids=inputs["pos_ids"],
            role_ids=inputs.get("role_ids", None),
            turn_ids=inputs.get("turn_ids", None),
            generation_mask=inputs["attention_mask"],
            name="dual_encoder"
        )

        return outputs

    def forward(self, inputs, is_infer=False):
        """Run forward."""
        if self.do_kag_training:
            outputs = self._forward_for_kag_training(inputs)
        elif self.do_dense_emb:
            outputs = self._forward_for_dense_emb(inputs)
        else:
            outputs = super(PlatoKAG, self).forward(inputs, is_infer)

        return outputs

    def _init_build_strategy(self):
        """Initialize the build strategy for Paddle."""
        build_strategy = fluid.BuildStrategy()
        build_strategy.fuse_bn_add_act_ops = False
        build_strategy.fuse_broadcast_ops = False
        build_strategy.enable_inplace = False
        self.dist_strategy.build_strategy = build_strategy

    def _get_statistics_for_kag_training(self, inputs, outputs):
        """Get statistics for KAG training."""
        statistics = {}
        statistics["batch_size"] = inputs["dual_src_token_ids"].shape()[0]
        return statistics

    def get_statistics(self, inputs, outputs):
        """Get statistics for the graph."""
        if self.do_kag_training:
            statistics = self._get_statistics_for_kag_training(inputs, outputs)
        else:
            statistics = super(PlatoKAG, self).get_statistics(inputs, outputs)
        return statistics

    def get_metrics_for_kag_training(self, inputs, outputs):
        """Get metrics for KAG training."""
        metrics = {}
        # outputs["single_enc_out"] -> [B * K, S, D]
        # outputs["tgt_idx"] -> [B * K * T, 2]
        tgt_logits = self._calc_logits(outputs["single_enc_out"], outputs["tgt_idx"])
        # [B * K, T, V]
        tgt_logits = layers.reshape(
            x=tgt_logits, shape=[-1, self.max_mask_len, self.vocab_size])
        # [B * K, T, 1]
        lm_loss = layers.softmax_with_cross_entropy(logits=tgt_logits, label=outputs["tgt_label"])
        need_cal = layers.not_equal(
            outputs["tgt_label"], layers.fill_constant(shape=[1], dtype="int64", value=1)
        )
        need_cal = layers.cast(need_cal, self.dtype)

        lm_loss_on_each_k = layers.reduce_sum(lm_loss * need_cal, dim=1) / (layers.reduce_sum(need_cal, dim=1) + 1e-10)

        # [B * K, T]
        lm_loss = layers.squeeze(input=lm_loss, axes=[-1])
        # [B, K, T]
        lm_loss = layers.reshape(
            x=lm_loss, shape=[-1, self.top_k, self.max_mask_len])
        # [B * K, T, 1] -> [B, K, T]
        need_cal = layers.reshape(x=need_cal, shape=[-1, self.top_k, self.max_mask_len])
        lm_loss = lm_loss * need_cal
        # [B, K, T] -> [B, K]
        lm_loss = layers.reduce_sum(lm_loss, dim=-1) / (layers.reduce_sum(need_cal, dim=-1) + 1e-10)

        # [B, K]
        log_sum = -lm_loss + outputs["log_dual_softmax"]

        # [B]
        loss = -paddle.logsumexp(log_sum, axis=-1)

        metrics["loss"] = layers.mean(loss)
        metrics["mean_mlm_ce"] = layers.mean(lm_loss_on_each_k)
        return metrics

    def get_metrics(self, inputs, outputs):
        """Get metrics for the graph."""
        if self.do_kag_training:
            metrics = self.get_metrics_for_kag_training(inputs, outputs)
        elif self.do_dense_emb:
            metrics = {}
        else:
            metrics = super(PlatoKAG, self).get_metrics(inputs, outputs)
        return metrics

    def infer(self, inputs, outputs):
        """Run inference."""
        if self.do_dense_emb:
            feat = self._get_pooled_output(outputs["outputs"], name="dual_pool")
            predictions = {"emb": feat, "data_id": inputs["data_id"]}
        else:
            predictions = super(PlatoKAG, self).infer(inputs, outputs)
        return predictions

    def infer_step(self, inputs):
        """Run inference for one step."""
        if self.do_dense_emb:
            return Model.infer_step(self, inputs)
        else:
            return super(PlatoKAG, self).infer_step(inputs)
