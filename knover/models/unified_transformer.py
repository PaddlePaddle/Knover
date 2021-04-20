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

from knover.models import register_model
from knover.core.model import Model
from knover.modules.transformer_block import encoder, pre_process_layer
from knover.modules.generator import Generator
from knover.utils import str2bool, repeat_array_or_tensor, slice_array_or_tensor


@register_model("UnifiedTransformer")
class UnifiedTransformer(Model):
    """Unified Transformer"""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = Model.add_cmdline_args(parser)
        group.add_argument("--weight_sharing", type=str2bool, default=True,
                           help="Whether to share the token embedding with the output FC.")
        group.add_argument("--mem_efficient", type=str2bool, default=False,
                           help="Whether to run in memory efficient mode.")
        group.add_argument("--use_role", type=str2bool, default=False,
                           help="Whether use role embeddings.")

        Generator.add_cmdline_args(parser)
        return group

    def __init__(self, args, place):
        self.max_seq_len = args.max_seq_len

        self.emb_size = args.get("emb_size", args.hidden_size)
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

        self.epsilon = args.get("epsilon", 1e-5)
        self.n_layer_per_block = args.get("n_layer_per_block", 1)
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
        self.do_generation = args.get("do_generation", False)

        super(UnifiedTransformer, self).__init__(args, place)

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

        # concat auxiliary memory embeddings
        if aux_emb is not None:
            emb_out = layers.concat([aux_emb, emb_out], axis=1)

        # pre process of input embedding
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
        self_attn_mask = layers.scale(x=input_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = layers.unsqueeze(self_attn_mask, [1])
        n_head_self_attn_mask.stop_gradient = True

        return emb_out, n_head_self_attn_mask

    def _get_pooled_output(self, enc_out, idx=None, name="pooled"):
        """Get pooled output of the last output embedding in Transformer.

        Args:
            enc_out: the output embeddings of Transformer, shape is [batch_size, max_seq_len, hidden_dim]
            idx (optional): the selected indices in pooling operator, shape is [batch_size, 1] or [batch_size, 2].
            name: a string, the name of the pooling layer.

        Returns:
            pooled_out: the pooled output embedding, shape is [batch_size, hidden_dim].
        """
        if idx is None:
            feat = layers.slice(input=enc_out, axes=[1], starts=[0], ends=[1])
        elif len(idx.shape) == 2 and idx.shape[1] == 1:
            enc_out = layers.reshape(x=enc_out, shape=[-1, self.hidden_size])
            feat = layers.gather(input=enc_out, index=idx)
        elif len(idx.shape) == 2 and idx.shape[1] == 2:
            feat = layers.gather_nd(input=enc_out, index=idx)
        else:
            raise ValueError(f"Invalid indices shape {idx.shape} is used")

        pooled_out = layers.fc(
            input=feat,
            size=self.hidden_size,
            act="tanh",
            param_attr=fluid.ParamAttr(name=f"{name}_fc.w_0", initializer=self.param_initializer),
            bias_attr=f"{name}_fc.b_0")
        return pooled_out

    def _get_classifier_output(self, pooled_out, num_classes=2, name="cls"):
        """Get the output logits of the classifier network.

        Args:
            pooled_out: represents the input embedding of classifier network, shape is [batch_size, hidden_dim]
            num_classes: an int, the number of classes in classification task.
            name: a string, the name of classifier network.

        Returns:
            cls_logits: the classification logits, shape is [batch_size, num_classes]
        """
        cls_logits = layers.fc(
            input=pooled_out,
            size=num_classes,
            param_attr=fluid.ParamAttr(name=f"{name}_fc.w_0", initializer=self.param_initializer),
            bias_attr=f"{name}_fc.b_0")
        return cls_logits

    def _generation_network(self,
                            token_ids,
                            type_ids,
                            pos_ids,
                            role_ids,
                            generation_mask,
                            aux_emb=None,
                            gather_idx=None):
        """Run Transformer generation network.

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            pos_ids: represents the position of each token, shape is [batch_size, max_seq_len, 1]
            input_mask: represents the attention masking mastrix in each Transformer blocks,
                shape is [batch_size, max_seq_len, max_seq_len]
            aux_emb: represents the auxiliary input embeddings of Transformer.
            gather_idx: the gather index of saved embedding in Transformer.

        Returns:
            A tuple contains the output embeddings of Transformer and the checkpoints of Transformer in this pass.
        """
        emb_out, n_head_self_attn_mask = self._gen_input(
            token_ids, type_ids, pos_ids, role_ids, generation_mask, aux_emb=aux_emb)
        # return self._encode(
        #     emb_out, n_head_self_attn_mask, None,
        #     gather_idx=gather_idx)
        return self._encode(
           emb_out, n_head_self_attn_mask, self.generation_caches,
           gather_idx=gather_idx)

    def _encode(self, emb_input, n_head_self_attn_mask, caches=None, gather_idx=None):
        """Run Transformer encode pass.

        Args:
            emb_input: represents the input embeddings fo Transformer, shape is [batch_size, max_seq_len, hidden_dim]
            n_head_self_attn_mask: represents the attention masking matrix,
                shape is [batch_size, num_heads, max_seq_len, max_seq_len]

        Returns:
            A tuple contains the output embeddings of Transformer and the checkpoints of Transformer in this pass.
        """
        return encoder(
            enc_input=emb_input,
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
            enc_out = layers.reshape(
                x=enc_out, shape=[-1, self.hidden_size])
            seq_feat = enc_out
        elif len(tgt_idx.shape) == 2 and tgt_idx.shape[1] == 2:
            seq_feat = layers.gather_nd(input=enc_out, index=tgt_idx)
        else:
            raise ValueError(f"Invalid indices shape {tgt_idx.shape} is used")

        seq_trans_feat = layers.fc(
            input=seq_feat,
            size=self.emb_size,
            act=self.hidden_act,
            param_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.w_0",
                initializer=self.param_initializer),
            bias_attr="mask_lm_trans_fc.b_0")

        seq_trans_feat = pre_process_layer(
            seq_trans_feat, self.post_cls_cmd, name="mask_lm_trans")

        if self.weight_sharing:
            logits = layers.matmul(
                x=seq_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self.token_emb_name),
                transpose_y=True)
            if self.cls_bias:
                logits += layers.create_parameter(
                    shape=[self.vocab_size],
                    dtype=self.dtype,
                    attr=fluid.ParamAttr(name="mask_lm_out_fc.b_0"),
                    is_bias=True)
        else:
            seq_out_bias_attr = "mask_lm_out_fc.b_0" if self.cls_bias else False
            logits = layers.fc(
                input=seq_trans_feat,
                size=self.vocab_size,
                param_attr=fluid.ParamAttr(
                    name="mask_lm_out_fc.w_0",
                    initializer=self.param_initializer),
                bias_attr=seq_out_bias_attr)
        return logits

    def _get_feed_dict(self, is_infer=False):
        """Get model's input feed dict.

        Args:
            is_infer: If true, get inference input feed dict, otherwise get training / evaluation input feed dict.

        Returns:
            feed_dict: A feed dict mapping keys to feed input variable.
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
                name="tgt_ids", shape=[-1, self.max_seq_len, 1], dtype="int64", lod_level=2)
            feed_dict["tgt_pos"] = layers.data(
                name="tgt_pos", shape=[-1, self.max_seq_len, 1], dtype="int64", lod_level=2)
            feed_dict["init_score"] = layers.data(name="init_score", shape=[-1, 1], dtype="float32", lod_level=1)
            feed_dict["parent_idx"] = layers.data(name="parent_idx", shape=[-1], dtype="int64")

            feed_dict["tgt_generation_mask"] = layers.data(
                name="tgt_generation_mask", shape=[-1, 1, self.max_seq_len], dtype="float32")

            feed_dict["data_id"] = layers.data(name="data_id", shape=[-1, 1], dtype="int64")
            # feed_dict["tgt_label"] = layers.data(name="tgt_label", shape=[-1, 1], dtype="int64")
            # feed_dict["tgt_idx"] = layers.data(name="tgt_idx", shape=[-1, 2], dtype="int64")
        else:
            feed_dict["tgt_label"] = layers.data(name="tgt_label", shape=[-1, 1], dtype="int64")
            feed_dict["tgt_idx"] = layers.data(name="tgt_idx", shape=[-1, 2], dtype="int64")

        return feed_dict

    def forward(self, inputs, is_infer=False):
        """Run model main forward."""
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
            # print("====================")
            # print("k: ", self.d_key * self.n_head, "v: ", self.d_value * self.n_head)
            # print("====================")
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

    def get_metrics(self, inputs, outputs):
        """Get metrics."""
        metrics = {}

        tgt_logits = self._calc_logits(outputs["enc_out"], inputs["tgt_idx"])
        tgt_lm_loss = layers.softmax_with_cross_entropy(
            logits=tgt_logits, label=inputs["tgt_label"])
        mean_tgt_lm_loss = layers.mean(tgt_lm_loss)
        metrics["token_lm_loss"] = mean_tgt_lm_loss

        loss = mean_tgt_lm_loss
        metrics["loss"] = loss
        return metrics

    def get_statistics(self, inputs, outputs):
        """Get statistics."""
        statistics = {}
        if "tgt_label" in inputs:
            statistics["tokens_num"] = inputs["tgt_label"].shape()[0]
        statistics["batch_size"] = inputs["token_ids"].shape()[0]
        return statistics

    def infer(self, inputs, outputs):
        """Run model inference.

        Only support generation now.
        """
        if self.do_generation:
            return self.generator.inference(self, inputs, outputs)
        else:
            raise NotImplementedError

    def _get_batch_size(self, inputs):
        """Get the batch size of inputs."""
        if "data_id" not in inputs:
            raise ValueError("Cannot find `data_id` in inputs.")
        elif isinstance(inputs["data_id"], np.ndarray):
            return len(inputs["data_id"])
        elif isinstance(inputs["data_id"], fluid.LoDTensor):
            return inputs["data_id"].shape()[0]
        else:
            raise ValueError(f"Invalid type of `data_id`: {type(inputs['data'])}")

    def _run_generation(self, inputs):
        """Run generation."""
        batch_size = self._get_batch_size(inputs)
        inputs["parent_idx"] = np.array(range(batch_size), dtype="int64")
        with open("run_generation_outputprogram_before.txt", "w") as f:
            f.write(str(self.infer_program))
            print("gen_output_program_____________________")
            
        outputs = self._execute(
            self.infer_program,
            inputs,
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
                pred = {}
                pred["data_id"] = data_id
                pred["decode_score"] = float(seq_scores_np[sub_end - 1])
                pred["context_token_ids"] = token_ids
                pred["response_token_ids"] = seq_ids_np[sub_start:sub_end].tolist()
                predictions.append(pred)
        return predictions

    def infer_step(self, inputs):
        """Run one inference step."""
        # handle DataLoader input type in distributed mode.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if self.do_generation:
            batch_size = self._get_batch_size(inputs)
            if self.generator.num_samples:
                inputs = {
                    name: repeat_array_or_tensor(array_or_tensor, self.place, self.generator.num_samples)
                    for name, array_or_tensor in inputs.items()
                }

            if self.mem_efficient:
                predictions = []
                for idx in range(0, batch_size, self.batch_size):
                    part_inputs = {
                        name: slice_array_or_tensor(array_or_tensor, self.place, idx, idx + self.batch_size)
                        for name, array_or_tensor in inputs.items()
                    }
                    part_outputs = self._run_generation(part_inputs)
                    predictions.extend(part_outputs)
            else:
                predictions = self._run_generation(inputs)
            return predictions
        else:
            return self._execute(
                self.infer_program,
                self._get_feed(inputs, is_infer=True),
                self.infer_fetch_dict)
