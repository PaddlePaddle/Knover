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
"""Plato model."""

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers

from . import register_model
from .model_base import Model
from .unified_transformer import UnifiedTransformer
from .transformer_block import encoder, pre_process_layer
from utils import repeat_array_or_tensor
from utils.args import str2bool
from .generator import Generator


@register_model("Plato")
class Plato(UnifiedTransformer):
    """Plato model."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = UnifiedTransformer.add_cmdline_args(parser)
        group.add_argument("--use_bow", type=str2bool, default=True)
        group.add_argument("--use_entropy", type=str2bool, default=False)
        return group

    def __init__(self, args, place):
        # latent related
        self.mask_id = args.mask_id
        self.latent_type_size = args.latent_type_size
        self.latent_emb_name = "latent_embedding"
        self.use_bow = args.use_bow
        self.use_entropy = args.use_entropy

        super(Plato, self).__init__(args, place)

    def _get_feed_dict(self, is_infer=False):
        feed_dict = {}
        feed_dict["token_ids"] = layers.data(name="token_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["type_ids"] = layers.data(name="type_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")
        feed_dict["pos_ids"] = layers.data(name="pos_ids", shape=[-1, self.max_seq_len, 1], dtype="int64")

        if not is_infer:
            feed_dict["recognition_mask"] = layers.data(
                name="recognition_mask",
                shape=[-1, self.max_seq_len + 1, self.max_seq_len + 1],
                dtype=self.dtype)
        feed_dict["generation_mask"] = layers.data(
            name="generation_mask",
            shape=[-1, self.max_seq_len + 1, self.max_seq_len + 1],
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
                name="tgt_generation_mask", shape=[-1, 1, self.max_seq_len + 1], dtype="float32")
            feed_dict["latent_id"] = layers.data(name="latent_id", shape=[-1, 1], dtype="int64")
        else:
            feed_dict["tgt_label"] = layers.data(name="tgt_label", shape=[-1, 1], dtype="int64")
            feed_dict["tgt_pos"] = layers.data(name="tgt_pos", shape=[-1, 1], dtype="int64")

            if self.use_bow:
                feed_dict["bow_label"] = layers.data(name="bow_label", shape=[-1, 1], dtype="int64")
                feed_dict["bow_pos"] = layers.data(name="bow_pos", shape=[-1, 1], dtype="int64")

        feed_dict["data_id"] = layers.data(name="data_id", shape=[-1, 1], dtype="int64")
        return feed_dict

    def _recognition_network(self,
                             token_ids,
                             type_ids,
                             pos_ids,
                             recognition_mask):
        mask_id = layers.fill_constant_batch_size_like(
            input=token_ids, shape=[-1, 1, 1], value=self.mask_id, dtype="int64")
        mask_emb = layers.embedding(
            input=mask_id,
            size=[self.vocab_size, self.emb_size],
            dtype=self.dtype,
            param_attr=fluid.ParamAttr(
                name=self.token_emb_name, initializer=self.param_initializer))
        emb_out, n_head_self_attn_mask = self._gen_input(
            token_ids, type_ids, pos_ids, recognition_mask, aux_emb=mask_emb)

        recognition_out, checkpoints = self._encode(emb_out, n_head_self_attn_mask)

        recognition_feat = layers.slice(
            input=recognition_out, axes=[1], starts=[0], ends=[1])
        recognition_feat = layers.fc(
            input=recognition_feat,
            size=self.hidden_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="recognition_fc.w_0", initializer=self.param_initializer),
            bias_attr="recognition_fc.b_0")
        logits = layers.fc(
            input=recognition_feat,
            size=self.latent_type_size,
            param_attr=fluid.ParamAttr(
                name=self.latent_emb_name, initializer=self.param_initializer),
            bias_attr="recognition_bias")
        return logits, checkpoints

    def _gumbel_softmax(self, logits, tau=0.67, eps=1e-10):
        u = layers.uniform_random_batch_size_like(
            logits, shape=[-1, self.latent_type_size], min=0.0, max=1.0)
        u.stop_gradient = True
        gumbel = 0.0 - layers.log(eps - layers.log(u + eps))
        y = logits + gumbel
        return layers.softmax(y / tau)

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

        latent_embeddings = layers.create_parameter(
            shape=[self.emb_size, self.latent_type_size],
            dtype=self.dtype,
            attr=fluid.ParamAttr(
                name=self.latent_emb_name, initializer=self.param_initializer))

        if is_infer:
            latent_id = inputs["latent_id"]
            weights = layers.one_hot(latent_id, self.latent_type_size)
        else:
            logits, recognition_checkpoints = self._recognition_network(
                token_ids=inputs["token_ids"],
                type_ids=inputs["type_ids"],
                pos_ids=inputs["pos_ids"],
                recognition_mask=inputs["recognition_mask"],
            )
            outputs["post_probs"] = layers.softmax(logits)
            weights = self._gumbel_softmax(logits)
            outputs["checkpoints"] = recognition_checkpoints

        latent_emb = layers.matmul(x=weights, y=latent_embeddings, transpose_y=True)
        outputs["enc_out"], generation_checkpoints = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            pos_ids=inputs["pos_ids"],
            generation_mask=inputs["generation_mask"],
            aux_emb=layers.unsqueeze(latent_emb, axes=[1]),
            gather_idx=inputs.get("parent_idx", None),
        )

        if not is_infer:
            outputs["checkpoints"].extend(generation_checkpoints)
        return outputs

    def _calc_bow_logits(self, enc_out, checkpoints, bow_pos):
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

        checkpoints.append(bow_trans_feat)

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

    def _get_metrics(self, inputs, outputs):
        metrics = super(Plato, self)._get_metrics(inputs, outputs)

        if self.use_bow:
            fc_out = self._calc_bow_logits(outputs["enc_out"], outputs["checkpoints"], inputs["bow_pos"])
            bow_loss = layers.softmax_with_cross_entropy(
                logits=fc_out, label=inputs["bow_label"])
            mean_bow_loss = layers.mean(bow_loss)
            metrics["token_bow_loss"] = mean_bow_loss
            metrics["loss"] = metrics["loss"] + mean_bow_loss

        entropy_loss = layers.reduce_sum(outputs["post_probs"] * layers.log(outputs["post_probs"]), dim=1)
        mean_entropy_loss = layers.mean(entropy_loss)
        metrics["entropy_loss"] = mean_entropy_loss
        if self.use_entropy:
            metrics["loss"] = metrics["loss"] + mean_entropy_loss
        return metrics

    def infer_step(self, inputs):
        """
        Run one inference step.
        """
        if self.do_generation:
            batch_size = len(inputs["data_id"])
            new_bsz = batch_size * self.latent_type_size
            inputs = {
                name: repeat_array_or_tensor(array_or_tensor, self.place, self.latent_type_size)
                for name, array_or_tensor in inputs.items()
            }
            # Add latent_id
            inputs["latent_id"] = np.array(
                [i for i in range(self.latent_type_size) for _ in range(batch_size)],
                dtype="int64"
            ).reshape([-1, 1])

            return super(Plato, self).infer_step(inputs)
        else:
            return self._execute(
                self.infer_program,
                self._get_feed(inputs, is_infer=True),
                self.infer_fetch_dict)
