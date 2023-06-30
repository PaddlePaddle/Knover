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
"""Classifier."""

import paddle.fluid as fluid
import paddle.fluid.layers as layers

from knover.core.model import Model
from knover.models import register_model
from knover.models.unified_transformer import UnifiedTransformer


@register_model("Classifier")
class Classifier(UnifiedTransformer):
    """Classifier."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = UnifiedTransformer.add_cmdline_args(parser)
        return group

    def __init__(self, args, place):
        self.num_classes = args.num_classes
        super(Classifier, self).__init__(args, place)

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

        if not is_infer:
            feed_dict["label"] = layers.data(name="label", shape=[-1, 1], dtype="int64")
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
        pooled_out = self._get_pooled_output(outputs["enc_out"])
        cls_logits = self._get_classifier_output(pooled_out, num_classes=self.num_classes, name="cls")
        cls_loss, cls_softmax = layers.softmax_with_cross_entropy(
            logits=cls_logits, label=inputs["label"], return_softmax=True)

        cls_acc = layers.accuracy(cls_softmax, inputs["label"])
        mean_cls_loss = layers.mean(cls_loss)

        metrics["loss"] = mean_cls_loss
        metrics["cls_loss"] = mean_cls_loss
        metrics["cls_acc"] = cls_acc

        # statistics for recall & precision & f1
        if self.num_classes == 2:
            pred = layers.argmax(cls_softmax, axis=1)
            label = layers.squeeze(inputs["label"], axes=[1])
            metrics["stat_tp"] = layers.reduce_sum(
                layers.logical_and(pred == 1, label == 1).astype("float32")
            )
            metrics["stat_fp"] = layers.reduce_sum(
                layers.logical_and(pred == 1, label == 0).astype("float32")
            )
            metrics["stat_tn"] = layers.reduce_sum(
                layers.logical_and(pred == 0, label == 0).astype("float32")
            )
            metrics["stat_fn"] = layers.reduce_sum(
                layers.logical_and(pred == 0, label == 1).astype("float32")
            )
        return metrics

    def infer(self, inputs, outputs):
        """Run model inference."""
        pooled_out = self._get_pooled_output(outputs["enc_out"])
        cls_logits = self._get_classifier_output(pooled_out, num_classes=self.num_classes, name="cls")
        scores = layers.softmax(cls_logits)
        predictions = {"scores": scores, "data_id": inputs["data_id"]}
        return predictions

    def infer_step(self, inputs):
        """Run one inference step."""
        return Model.infer_step(self, inputs)
