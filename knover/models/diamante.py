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
"""Diamante model."""

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.nn.functional as F

from knover.core.model import Model
from knover.models import register_model
from knover.models.unified_transformer import UnifiedTransformer


@register_model("Diamante")
class Diamante(UnifiedTransformer):
    """diamante model"""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = UnifiedTransformer.add_cmdline_args(parser)
        return group

    def __init__(self, args, place):
        self.batch_size = args.batch_size
        super(Diamante, self).__init__(args, place)

    def _get_feed_dict(self, is_infer=False):
        """Get model's input feed dict.

        Args:
            is_infer: If true, get inference input feed dict, otherwise get training / evaluation input feed dict.

        Returns:
            feed_dict: A feed dict mapping keys to feed input variable.
        """
        feed_dict = super(Diamante, self)._get_feed_dict(is_infer)
        if not is_infer or not self.do_generation:
            feed_dict["label_idx"] = layers.data(name="label_idx", shape=[-1, 2], dtype="int64")
            if is_infer:
                feed_dict["data_id"] = layers.data(name="data_id", shape=[-1], dtype="int64")
            else:
                feed_dict["label"] = layers.data(name="label", shape=[-1, 1], dtype="int64")
        return feed_dict

    def _get_similarity_score(self, feat):
        """Get similarity score."""
        similarity_score = layers.fc(
            input=feat,
            size=1,
            act=None,
            param_attr=fluid.ParamAttr(
                name="similarity_fc.w_0",
                initializer=self.param_initializer),
            bias_attr="similarity_fc.b_0")

        return similarity_score

    def get_metrics(self, inputs, outputs):
        """Get metrics."""
        # NLL loss
        metrics = super(Diamante, self).get_metrics(inputs, outputs)

        # Preference Estimation loss
        feat = layers.gather_nd(outputs["enc_out"], inputs["label_idx"])
        similarity_score = self._get_similarity_score(feat)

        pos_sim = layers.concat([similarity_score[0::3], similarity_score[0::3], similarity_score[1::3]], axis=0)
        neg_sim = layers.concat([similarity_score[1::3], similarity_score[2::3], similarity_score[2::3]], axis=0)

        labels = layers.fill_constant_batch_size_like(pos_sim, [-1, 1], dtype=self.dtype, value=1)

        logits = F.sigmoid(pos_sim - neg_sim)
        ranking_loss = F.log_loss(logits, labels)
        mean_ranking_loss = layers.mean(ranking_loss)
        metrics["ranking_loss"] = mean_ranking_loss

        # joint loss
        loss = metrics["ranking_loss"] + metrics["token_lm_loss"]

        metrics["loss"] = loss
        metrics["score_gap"] = layers.mean(logits)
        metrics["acc"] = layers.mean(layers.cast(pos_sim > neg_sim, self.dtype))

        metrics["human_bot_gap"] = layers.mean(logits[0::3])
        metrics["human_random_gap"] = layers.mean(logits[1::3])
        metrics["bot_random_gap"] = layers.mean(logits[2::3])

        return metrics

    def infer(self, inputs, outputs):
        """Run model inference.

        Only support generation now.
        """
        predictions = super(Diamante, self).infer(inputs, outputs)

        if self.do_generation:
            model_input = {}
            model_input["pos_ids"] = predictions["pos_ids"]
            model_input["token_ids"] = layers.fill_constant_batch_size_like(
                model_input["pos_ids"], [-1, 1, 1], "int64", self.generator.eos_id)
            model_input["type_ids"] = layers.fill_constant_batch_size_like(
                model_input["pos_ids"], [-1, 1, 1], "int64", 1)
            if self.use_role:
                model_input["role_ids"] = layers.fill_constant_batch_size_like(
                    model_input["pos_ids"], [-1, 1, 1], "int64", 0)

            generation_mask = predictions["generation_mask"]
            append_mask = layers.fill_constant_batch_size_like(generation_mask, [-1, 1], "float32", 1)
            append_mask = layers.unsqueeze(append_mask, [2])
            generation_mask = layers.concat([generation_mask, append_mask], axis=2)
            model_input["generation_mask"] = generation_mask

            enc_out, _ = self._generation_network(**model_input)
            feat = enc_out[:, 0, :]
            ranking_score = self._get_similarity_score(feat)
            ranking_score = F.sigmoid(ranking_score)
            predictions["ranking_score"] = ranking_score
        else:
            # ranking score
            feat = layers.gather_nd(outputs["enc_out"], inputs["label_idx"])
            ranking_score = self._get_similarity_score(feat)
            predictions["ranking_score"] = F.sigmoid(ranking_score)

        return predictions

    def _run_generation(self, inputs):
        """Run generation."""
        batch_size = self._get_batch_size(inputs)
        inputs["parent_idx"] = np.array(range(batch_size), dtype="int64")
        outputs = self._execute(
            self.infer_program,
            inputs,
            self.infer_fetch_dict,
            return_numpy=False)

        predictions = []
        data_id_list = np.array(outputs["data_id"]).tolist()
        token_ids_list = np.array(outputs["token_ids"]).squeeze(2).tolist()
        ranking_score = np.array(outputs["ranking_score"]).tolist()

        seq_ids = outputs["finished_ids"]
        seq_ids_np  = np.array(outputs["finished_ids"])
        seq_scores_np = np.array(outputs["finished_scores"])
        for i, (data_id, token_ids, score) in enumerate(zip(data_id_list, token_ids_list, ranking_score)):
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
                pred["ranking_score"] = score[0]
                predictions.append(pred)

        return predictions
