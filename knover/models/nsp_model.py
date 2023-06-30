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

from knover.core.model import Model
from knover.models import register_model
from knover.models.unified_transformer import UnifiedTransformer
from knover.utils import str2bool


@register_model("NSPModel")
class NSPModel(UnifiedTransformer):
    """NSP model."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = UnifiedTransformer.add_cmdline_args(parser)
        group.add_argument("--use_mlm", type=str2bool, default=True,
                           help="Whether to train model with MLM loss.")
        return group

    def __init__(self, args, place):
        self.use_mlm = args.use_mlm
        super(NSPModel, self).__init__(args, place)

    def _get_feed_dict(self, is_infer=False):
        """Get model's input feed dict.

        Args:
            is_infer: If true, get inference input feed dict, otherwise get training / evaluation input feed dict.

        Returns:
            feed_dict: A feed dict mapping keys to feed input variable.
        """
        feed_dict = {}
        feed_dict["token_ids"] = layers.data(name="token_ids", shape=[-1, -1, 1], dtype="int64")
        feed_dict["type_ids"] = layers.data(name="type_ids", shape=[-1, -1, 1], dtype="int64")
        feed_dict["pos_ids"] = layers.data(name="pos_ids", shape=[-1, -1, 1], dtype="int64")

        if self.use_role:
            feed_dict["role_ids"] = layers.data(name="role_ids", shape=[-1, -1, 1], dtype="int64")

        feed_dict["attention_mask"] = layers.data(name="attention_mask", shape=[-1, -1, -1], dtype=self.dtype)
        feed_dict["label_idx"] = layers.data(name="label_idx", shape=[-1, 2], dtype="int64")

        if not is_infer:
            feed_dict["label"] = layers.data(name="label", shape=[-1, 1], dtype="int64")
            feed_dict["tgt_label"] = layers.data(name="tgt_label", shape=[-1, 1], dtype="int64")
            feed_dict["tgt_idx"] = layers.data(name="tgt_idx", shape=[-1, 2], dtype="int64")
        else:
            feed_dict["data_id"] = layers.data(name="data_id", shape=[-1], dtype="int64")
        return feed_dict

    def forward(self, inputs, is_infer=False):
        """Run model main forward."""
        outputs = {}
        self.generation_caches = None
        outputs["enc_out"], outputs["checkpoints"] = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            pos_ids=inputs["pos_ids"],
            role_ids=inputs.get("role_ids", None),
            generation_mask=inputs["attention_mask"]
        )
        return outputs

    def get_metrics(self, inputs, outputs):
        """Get metrics."""
        metrics = {}
        tgt_logits = self._calc_logits(outputs["enc_out"], inputs["tgt_idx"])
        lm_loss = layers.softmax_with_cross_entropy(logits=tgt_logits, label=inputs["tgt_label"])
        need_cal = layers.not_equal(
            inputs["tgt_label"], layers.fill_constant(shape=[1], dtype="int64", value=1)
        )
        need_cal = layers.cast(need_cal, self.dtype)
        mean_lm_loss = layers.reduce_sum(lm_loss * need_cal) / (layers.reduce_sum(need_cal) + 1e-10)

        pooled_out = self._get_pooled_output(outputs["enc_out"], inputs["label_idx"])
        nsp_logits = self._get_classifier_output(pooled_out, name="next_sent")
        nsp_loss, nsp_softmax = layers.softmax_with_cross_entropy(
            logits=nsp_logits, label=inputs["label"], return_softmax=True)

        nsp_acc = layers.accuracy(nsp_softmax, inputs["label"])
        mean_nsp_loss = layers.mean(nsp_loss)

        loss = mean_nsp_loss
        if self.use_mlm:
            loss = loss + mean_lm_loss
            metrics["token_lm_loss"] = mean_lm_loss
        metrics["loss"] = loss
        metrics["nsp_loss"] = mean_nsp_loss
        metrics["nsp_acc"] = nsp_acc
        return metrics

    def infer(self, inputs, outputs):
        """Run model inference."""
        pooled_out = self._get_pooled_output(outputs["enc_out"], inputs["label_idx"])
        nsp_logits = self._get_classifier_output(pooled_out, name="next_sent")
        scores = layers.softmax(nsp_logits)
        predictions = {"scores": scores, "data_id": inputs["data_id"]}
        return predictions

    def infer_step(self, inputs):
        """Run one inference step."""
        return Model.infer_step(self, inputs)
