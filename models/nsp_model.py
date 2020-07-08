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
"""NSP model."""

import paddle.fluid as fluid
import paddle.fluid.layers as layers

from . import register_model
from .model_base import Model
from .unified_transformer import UnifiedTransformer


@register_model("NSPModel")
class NSPModel(UnifiedTransformer):
    """NSP model."""

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

        feed_dict["attention_mask"] = layers.data(
            name="attention_mask",
            shape=[-1, self.max_seq_len, self.max_seq_len],
            dtype=self.dtype)
        feed_dict["label_pos"] = layers.data(name="label_pos", shape=[-1, 1], dtype="int64")

        if not is_infer:
            feed_dict["label"] = layers.data(name="label", shape=[-1, 1], dtype="int64")
            feed_dict["tgt_label"] = layers.data(name="tgt_ids", shape=[-1, 1], dtype="int64")
            feed_dict["tgt_pos"] = layers.data(name="tgt_pos", shape=[-1, 1], dtype="int64")

        feed_dict["data_id"] = layers.data(name="data_id", shape=[-1, 1], dtype="int64")
        return feed_dict

    def _get_feed(self, inputs, is_infer=False):
        return Model._get_feed(self, inputs, is_infer)

    def forward(self, inputs, is_infer=False):
        outputs = {}
        self.generation_caches = None
        outputs["enc_out"], self.checkpoints = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            pos_ids=inputs["pos_ids"],
            role_ids=inputs.get("role_ids", None),
            generation_mask=inputs["attention_mask"]
        )
        return outputs

    def _get_metrics(self, inputs, outputs):
        metrics = {}
        fc_out = self._calc_logits(outputs["enc_out"], inputs["tgt_pos"])
        lm_loss = layers.softmax_with_cross_entropy(logits=fc_out, label=inputs["tgt_pos"])
        need_cal = layers.not_equal(
            inputs["tgt_label"], layers.fill_constant(shape=[1], dtype="int64", value=1)
        )
        need_cal = layers.cast(need_cal, self.dtype)
        mean_lm_loss = layers.reduce_sum(lm_loss * need_cal) / (layers.reduce_sum(need_cal) + 1e-10)

        pooled_out = self._get_pooled_output(outputs["enc_out"], inputs["label_pos"])
        nsp_fc_out = layers.fc(
            input=pooled_out,
            size=2,
            param_attr=fluid.ParamAttr(
                name="next_sent_fc.w_0", initializer=self.param_initializer),
            bias_attr="next_sent_fc.b_0")
        nsp_loss, nsp_softmax = layers.softmax_with_cross_entropy(
            logits=nsp_fc_out, label=inputs["label"], return_softmax=True)

        nsp_acc = layers.accuracy(nsp_softmax, inputs["label"])
        mean_nsp_loss = layers.mean(nsp_loss)

        metrics["loss"] = mean_lm_loss + mean_nsp_loss
        metrics["lm_loss"] = mean_lm_loss
        metrics["nsp_loss"] = mean_nsp_loss
        metrics["nsp_acc"] = nsp_acc
        return metrics

    def infer(self, inputs, outputs):
        pooled_out = self._get_pooled_output(outputs["enc_out"], inputs["label_pos"])
        nsp_fc_out = layers.fc(
            input=pooled_out,
            size=2,
            param_attr=fluid.ParamAttr(
                name="next_sent_fc.w_0", initializer=self.param_initializer),
            bias_attr="next_sent_fc.b_0")
        scores = layers.softmax(nsp_fc_out)
        predictions = {"scores": scores, "data_id": inputs["data_id"]}
        return predictions

    def infer_step(self, inputs):
        return Model.infer_step(self, inputs)
