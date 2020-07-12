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
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from . import register_model
from .model_base import Model
from .transformer_block import encoder, pre_process_layer
from utils.args import str2bool
from utils import repeat_array_or_tensor, slice_array_or_tensor
from .generator import Generator


@register_model("UnifiedTransformer")
class UnifiedTransformer(Model):
    """Unified Transformer"""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = Model.add_cmdline_args(parser)
        group.add_argument("--max_seq_len", type=int, default=256)
        group.add_argument("--weight_sharing", type=str2bool, default=True)
        group.add_argument("--mem_efficient", type=str2bool, default=False)

        Generator.add_cmdline_args(parser)
        return group

    def __init__(self, args, place):
        self.max_seq_len = args.max_seq_len

        self.emb_size = args.emb_size or args.hidden_size
        self.hidden_size = args.hidden_size

        self.n_layer = args.num_hidden_layers
        self.n_head = args.num_attention_heads
        self.d_key = args.get("key_size", self.hidden_size // self.n_head)
        self.d_value = args.get("value_size", self.hidden_size // self.n_head)
        self.inner_hidden_size = args.get("inner_hidden_size", self.hidden_size * 4)

        self.vocab_size = args.vocab_size
        self.max_position_seq_len = args.max_position_embeddings
        self.type_size = args.type_vocab_size
        self.token_emb_name = "word_embedding"
        self.type_emb_name = "sent_embedding"
        self.pos_emb_name = "pos_embedding"

        self.epsilon = args.epsilon or 1e-5
        self.n_layer_per_block = args.n_layer_per_block or 1
        self.pre_encoder_cmd = args.get("pre_encoder_cmd", "nd")
        self.preprocess_cmd = args.get("preprocess_cmd", "")
        self.postprocess_cmd = args.get("postprocess_cmd", "dan")
        self.post_cls_cmd = args.get("post_cls_cmd", "n")
        self.cls_bias = args.get("cls_bias", True)
        if self.hidden_size != self.emb_size:
            self.emb_mapping_in = True
        else:
            self.emb_mapping_in = args.get("emb_mapping_in", False)

        self.hidden_act = args.hidden_act
        self.prepostprocess_dropout = args.hidden_dropout_prob
        self.attention_dropout = args.attention_probs_dropout_prob
        self.weight_sharing = args.weight_sharing

        self.mem_efficient = args.mem_efficient

        self.dtype = "float32"

        # role embeddings
        self.use_role = args.use_role
        if self.use_role:
            self.role_type_size = args.role_type_size
            self.role_emb_name = "role_embedding"

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self.param_initializer = fluid.initializer.TruncatedNormal(
            scale=args.initializer_range)

        # task-related
        self.generator = Generator(args)
        self.do_generation = args.do_generation

        super(UnifiedTransformer, self).__init__(args, place)

    def _gen_input(self,
                   token_ids,
                   type_ids,
                   pos_ids,
                   role_ids,
                   input_mask,
                   aux_emb=None):
        token_emb_out = layers.embedding(
            input=token_ids,
            size=[self.vocab_size, self.emb_size],
            dtype=self.dtype,
            param_attr=fluid.ParamAttr(
                name=self.token_emb_name, initializer=self.param_initializer))
        type_emb_out = layers.embedding(
            input=type_ids,
            size=[self.type_size, self.emb_size],
            dtype=self.dtype,
            param_attr=fluid.ParamAttr(
                name=self.type_emb_name, initializer=self.param_initializer))
        pos_emb_out = layers.embedding(
            input=pos_ids,
            size=[self.max_position_seq_len, self.emb_size],
            dtype=self.dtype,
            param_attr=fluid.ParamAttr(
                name=self.pos_emb_name, initializer=self.param_initializer))
        emb_out = token_emb_out + type_emb_out + pos_emb_out

        if self.use_role:
            role_emb_out = layers.embedding(
                input=role_ids,
                size=[self.role_type_size, self.emb_size],
                dtype=self.dtype,
                param_attr=fluid.ParamAttr(
                    name=self.role_emb_name, initializer=self.param_initializer))
            emb_out = emb_out + role_emb_out

        # auxiliary memory embeddings
        if aux_emb is not None:
            emb_out = layers.concat([aux_emb, emb_out], axis=1)

        # post process of embedding
        emb_out = pre_process_layer(
            emb_out,
            self.pre_encoder_cmd,
            self.prepostprocess_dropout,
            name="pre_encoder",
            epsilon=self.epsilon)
        if self.emb_mapping_in:
            emb_out = layers.fc(
                input=emb_out,
                num_flatten_dims=2,
                size=self.hidden_size,
                param_attr=fluid.ParamAttr(
                    name="emb_hidden_mapping",
                    initializer=self.param_initializer),
                bias_attr="emb_hidden_mapping_bias")

        # generate n-head self-attention mask
        self_attn_mask = input_mask
        self_attn_mask = layers.scale(
            x=self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = layers.stack(
            x=[self_attn_mask] * self.n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        return emb_out, n_head_self_attn_mask

    def _get_pooled_output(self, enc_out, pos):
        enc_out = layers.reshape(
            x=enc_out, shape=[-1, self.hidden_size])
        pos = layers.cast(x=pos, dtype="int32")
        feat = layers.gather(input=enc_out, index=pos)

        pooled_out = layers.fc(
            input=feat,
            size=self.hidden_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self.param_initializer),
            bias_attr="pooled_fc.b_0")
        return pooled_out

    def _generation_network(self,
                            token_ids,
                            type_ids,
                            pos_ids,
                            role_ids,
                            generation_mask,
                            aux_emb=None,
                            gather_idx=None):
        emb_out, n_head_self_attn_mask = self._gen_input(
            token_ids, type_ids, pos_ids, role_ids, generation_mask, aux_emb=aux_emb)
        generation_out = self._encode(
            emb_out, n_head_self_attn_mask, self.generation_caches,
            gather_idx=gather_idx)
        return generation_out

    def _encode(self, emb_out, n_head_self_attn_mask, caches=None, gather_idx=None):
        return encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self.n_layer,
            n_head=self.n_head,
            d_key=self.d_key,
            d_value=self.d_value,
            d_model=self.hidden_size,
            d_inner_hid=self.inner_hidden_size,
            prepostprocess_dropout=self.prepostprocess_dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=0,
            hidden_act=self.hidden_act,
            preprocess_cmd=self.preprocess_cmd,
            postprocess_cmd=self.postprocess_cmd,
            param_initializer=self.param_initializer,
            epsilon=self.epsilon,
            n_layer_per_block=self.n_layer_per_block,
            name="encoder",
            caches=caches,
            gather_idx=gather_idx,
            store=caches is not None
        )

    def _gumbel_softmax(self, logits, tau=0.67, eps=1e-10):
        u = layers.uniform_random_batch_size_like(
            logits, shape=[-1, self.latent_type_size], min=0.0, max=1.0)
        u.stop_gradient = True
        gumbel = 0.0 - layers.log(eps - layers.log(u + eps))
        y = logits + gumbel
        return layers.softmax(y / tau)

    def _get_feed_dict(self, is_infer=False):
        """
        Get the feed list of the model.

        Args:
            is_infer(bool): True if running inference.

        Returns:
            list(Variable): The feed list.
            list(str): The name of each Variable in feed list.
        """
        feed_dict = {}
        feed_dict["token_ids"] = layers.data(name="token_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["type_ids"] = layers.data(name="type_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["pos_ids"] = layers.data(name="pos_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")

        if self.use_role:
            feed_dict["role_ids"] = layers.data(name="role_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")

        feed_dict["generation_mask"] = layers.data(
            name="generation_mask",
            shape=[-1, self.max_seq_len, self.max_seq_len],
            dtype=self.dtype)

        if is_infer:
            feed_dict["tgt_ids"] = layers.data(
                name="tgt_ids",
                shape=[-1, self.max_seq_len, 1],
                dtype="int64",
                lod_level=2)
            feed_dict["tgt_pos"] = layers.data(
                name="tgt_pos",
                shape=[-1, self.max_seq_len, 1],
                dtype="int64",
                lod_level=2)
            feed_dict["init_score"] = layers.data(
                name="init_score",
                shape=[-1, 1],
                dtype="float32",
                lod_level=1)
            feed_dict["parent_idx"] = layers.data(
                name="parent_idx",
                shape=[-1],
                dtype="int64")

            feed_dict["tgt_generation_mask"] = layers.data(
                name="tgt_generation_mask", shape=[-1, 1, self.max_seq_len], dtype="float32")
        else:
            feed_dict["tgt_label"] = layers.data(name="tgt_label", shape=[-1, 1], dtype="int64")
            feed_dict["tgt_pos"] = layers.data(name="tgt_pos", shape=[-1, 1], dtype="int64")

        feed_dict["data_id"] = layers.data(name="data_id", shape=[-1, 1], dtype="int64")
        return feed_dict

    def forward(self, inputs, is_infer=False):
        """
        Run model main forward.
        """
        outputs = {}
        if is_infer:
            self.generation_caches = [{
                "k":
                layers.fill_constant_batch_size_like(
                    input=inputs["token_ids"],
                    shape=[-1, 0, self.d_key * self.n_head],
                    dtype=self.dtype,
                    value=0),
                "v":
                layers.fill_constant_batch_size_like(
                    input=inputs["token_ids"],
                    shape=[-1, 0, self.d_value * self.n_head],
                    dtype=self.dtype,
                    value=0),
            } for i in range(self.n_layer)]
        else:
            self.generation_caches = None

        outputs["enc_out"], generation_checkpoints = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            pos_ids=inputs["pos_ids"],
            role_ids=inputs.get("role_ids", None),
            generation_mask=inputs["generation_mask"],
            gather_idx=inputs.get("parent_idx", None)
        )

        if not is_infer:
            outputs["checkpoints"] = generation_checkpoints
        return outputs

    def _calc_bow_logits(self, enc_out, bow_pos):
        """Get the logits of generation."""
        bow_feat = layers.slice(
            input=enc_out, axes=[1], starts=[0], ends=[1])
        bow_feat = layers.reshape(
            x=bow_feat, shape=[-1, self.hidden_size])
        bow_pos = layers.cast(x=bow_pos, dtype="int32")
        bow_feat = layers.gather(input=bow_feat, index=bow_pos)

        bow_trans_feat = layers.fc(
            input=bow_feat,
            size=self.emb_size,
            act=self.hidden_act,
            param_attr=fluid.ParamAttr(
                name="bow_trans_fc.w_0",
                initializer=self.param_initializer),
            bias_attr=fluid.ParamAttr(name="bow_trans_fc.b_0"))

        bow_trans_feat = pre_process_layer(
            bow_trans_feat, self.post_cls_cmd, name="bow_trans")

        if self.weight_sharing:
            fc_out = layers.matmul(
                x=bow_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self.token_emb_name),
                transpose_y=True)
            if self.cls_bias:
                fc_out += layers.create_parameter(
                    shape=[self.vocab_size],
                    dtype=self.dtype,
                    attr=fluid.ParamAttr(name="bow_out_fc.b_0"),
                    is_bias=True)
        else:
            bow_out_bias_attr = fluid.ParamAttr(name="bow_out_fc.b_0") if self.cls_bias else False
            fc_out = layers.fc(input=bow_trans_feat,
                               size=self.vocab_size,
                               param_attr=fluid.ParamAttr(
                                   name="bow_out_fc.w_0",
                                   initializer=self.param_initializer),
                               bias_attr=bow_out_bias_attr)
        return fc_out

    def _calc_logits(self, enc_out, seq_pos=None):
        """Get the logits of generation."""
        enc_out = layers.reshape(
            x=enc_out, shape=[-1, self.hidden_size])
        if seq_pos is not None:
            seq_pos = layers.cast(x=seq_pos, dtype="int32")
            seq_feat = layers.gather(input=enc_out, index=seq_pos)
        else:
            seq_feat = enc_out

        seq_trans_feat = layers.fc(
            input=seq_feat,
            size=self.emb_size,
            act=self.hidden_act,
            param_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.w_0",
                initializer=self.param_initializer),
            bias_attr=fluid.ParamAttr(name="mask_lm_trans_fc.b_0"))

        seq_trans_feat = pre_process_layer(
            seq_trans_feat, self.post_cls_cmd, name="mask_lm_trans")

        if self.weight_sharing:
            fc_out = layers.matmul(
                x=seq_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self.token_emb_name),
                transpose_y=True)
            if self.cls_bias:
                fc_out += layers.create_parameter(
                    shape=[self.vocab_size],
                    dtype=self.dtype,
                    attr=fluid.ParamAttr(name="mask_lm_out_fc.b_0"),
                    is_bias=True)
        else:
            seq_out_bias_attr = fluid.ParamAttr(name="mask_lm_out_fc.b_0") if self.cls_bias else False
            fc_out = layers.fc(input=seq_trans_feat,
                               size=self.vocab_size,
                               param_attr=fluid.ParamAttr(
                                   name="mask_lm_out_fc.w_0",
                                   initializer=self.param_initializer),
                               bias_attr=seq_out_bias_attr)
        return fc_out

    def _get_metrics(self, inputs, outputs):
        metrics = {}

        tgt_pos = inputs["tgt_pos"]
        tgt_label = inputs["tgt_label"]

        fc_out = self._calc_logits(outputs["enc_out"], tgt_pos)

        tgt_lm_loss = layers.softmax_with_cross_entropy(
            logits=fc_out, label=tgt_label)
        mean_tgt_lm_loss = layers.mean(tgt_lm_loss)
        loss = mean_tgt_lm_loss
        metrics["token_lm_loss"] = mean_tgt_lm_loss

        metrics["loss"] = loss
        return metrics

    def _get_statistics(self, inputs, outputs):
        statistics = {}
        if "tgt_label" in inputs:
            statistics["tokens_num"] = layers.reduce_sum(
                layers.fill_constant_batch_size_like(
                    input=inputs["tgt_label"], value=1.0, shape=[-1], dtype="int64"))
        statistics["batch_size"] = layers.reduce_sum(
            layers.fill_constant_batch_size_like(
                input=inputs["token_ids"], value=1.0, shape=[-1], dtype="int64"))
        return statistics

    def get_metrics_and_statistics(self, inputs, outputs):
        """
        Get metrics and statistics.
        """
        metrics = self._get_metrics(inputs, outputs)
        statistics = self._get_statistics(inputs, outputs)
        return metrics, statistics

    def infer(self, inputs, outputs):
        """
        Run model inference.
        """
        if self.do_generation:
            return self.generator.inference(self, inputs, outputs)
        else:
            raise NotImplementedError

    def _run_generation(self, inputs):
        """
        Run generation.
        """
        batch_size = len(inputs["data_id"])
        inputs["parent_idx"] = np.array(range(batch_size), dtype="int64")
        outputs = self._execute(
            self.infer_program,
            self._get_feed(inputs, is_infer=True),
            self.infer_fetch_dict,
            return_numpy=False)

        predictions = []
        data_id_list = np.array(outputs["data_id"]).reshape(-1).tolist()
        token_ids_list = np.array(outputs["token_ids"]).squeeze(2).tolist()
        seq_ids = outputs["finished_ids"]
        seq_ids_np  = np.array(outputs["finished_ids"])
        seq_scores_np = np.array(outputs["finished_scores"])
        for i, (data_id, token_ids) in enumerate(zip(data_id_list, token_ids_list)):
            start = seq_ids.lod()[0][i]
            end = seq_ids.lod()[0][i + 1]
            for j in range(start, end):
                sub_start = seq_ids.lod()[1][j]
                sub_end = seq_ids.lod()[1][j + 1]
                info = {}
                info["data_id"] = data_id
                info["decode_score"] = float(seq_scores_np[sub_end - 1])
                info["context_token_ids"] = token_ids
                info["response_token_ids"] = seq_ids_np[sub_start:sub_end].tolist()
                predictions.append(info)
        return predictions

    def infer_step(self, inputs):
        """
        Run one inference step.
        """
        if self.do_generation:
            if self.generator.num_samples:
                inputs = {
                    name: repeat_array_or_tensor(array_or_tensor, self.place, self.generator.num_samples)
                    for name, array_or_tensor in inputs.items()
                }

            if self.mem_efficient:
                predictions = []
                for idx in range(0, len(inputs["data_id"]), self.batch_size):
                    part_inputs = {
                        name: slice_array_or_tensor(array_or_tensor, self.place, idx, idx + self.batch_size)
                        for name, array_or_tensor in inputs.items()
                    }
                    part_outputs = self._run_generation(part_inputs)
                    predictions.extend(part_outputs)
                return predictions
            else:
                return self._run_generation(inputs)
        else:
            return self._execute(
                self.infer_program,
                self._get_feed(inputs, is_infer=True),
                self.infer_fetch_dict)
