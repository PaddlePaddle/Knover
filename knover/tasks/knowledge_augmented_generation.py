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
"""Knowledge-Augmented Generation task."""

import math

from knover.data.kag_reader import KAGReader
from knover.tasks import register_task
from knover.tasks.dialog_generation import DialogGeneration
from knover.utils.args import str2bool


@register_task("KnowledgeAugmentedGeneration")
class KnowledgeAugmentedGeneration(DialogGeneration):
    """Define knowledge-augmented generation task."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = parser.add_argument_group("Task")
        group.add_argument("--do_generation", type=str2bool, default=True,
                           help="Whether to run generation on inference phase. "
                           "Dialogue generation support two type of inference: generation and scoring.")

        group.add_argument("--filter_cross_repetition", type=str2bool, default=True,
                           help="Whether to filter cross turn repetion or not.")
        group.add_argument("--ranking_score", type=str, default="decode_score",
                           help="Which score will be used to rerank.")
        group.add_argument("--do_kag_training", type=str2bool, default=False)
        group.add_argument("--multi_eval", type=str2bool, default=False)

        KAGReader.add_cmdline_args(parser)
        return group

    def __init__(self, args):
        super(KnowledgeAugmentedGeneration, self).__init__(args)
        self.reader = KAGReader(args)
        self.do_kag_training = args.do_kag_training
        self.multi_eval = args.multi_eval
        return

    def merge_metrics_and_statistics(self, outputs, part_outputs):
        """Merge two evaulation output.

        Args:
            outputs: Original outputs which contains metrics and statistics.
            part_outputs: New outputs which contains metrics and statistics.

        Returns:
            Return merged output which contains metrics and statistics.
        """
        if not self.do_kag_training:
            return super(KnowledgeAugmentedGeneration, self).merge_metrics_and_statistics(outputs, part_outputs)
        if outputs is None:
            return part_outputs

        if part_outputs is None:
            return outputs

        batch_size = outputs.pop("batch_size")
        part_batch_size = part_outputs.pop("batch_size")

        new_outputs = {
            "batch_size": batch_size + part_batch_size,
        }
        for k in outputs:
            if k.startswith("token_"):
                pass
            else:
                new_outputs[k] = (
                    outputs[k] * batch_size + part_outputs[k] * part_batch_size
                ) / new_outputs["batch_size"]
        return new_outputs

    def get_metrics(self, outputs):
        """Get metrics."""
        if not self.do_kag_training:
            return super(KnowledgeAugmentedGeneration, self).get_metrics(outputs)

        if outputs is None:
            raise ValueError("metrics is None")
        outputs = dict(outputs)
        metrics = {}
        batch_size = outputs.pop("batch_size", None)
        for k in outputs:
            if k.startswith("token_"):
                metrics[k[6:]] = outputs[k]
            else:
                metrics[k] = outputs[k]
            if k == "token_lm_loss":
                metrics["ppl"] = math.exp(outputs[k])
        return metrics

    def _post_process_scoring_output(self, predictions):
        return [
            {
                "data_id": data_id,
                "lm_loss": float(lm_loss),
                "ppl": math.exp(lm_loss / tokens_num),
                "tokens_num": int(tokens_num),
                "token_lm_loss": float(lm_loss / tokens_num),
                "gt_response": self.reader.features[data_id].tgt.split("\x01")[0]
            }
            for data_id, lm_loss, tokens_num in zip(
                predictions["data_id"].tolist(), predictions["lm_loss"], predictions["tokens_num"]
            )
        ]

