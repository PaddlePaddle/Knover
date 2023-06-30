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
"""Classification task."""

from knover.core.task import Task
from knover.data.classification_reader import ClassificationReader
from knover.tasks import register_task


@register_task("Classification")
class Classification(Task):
    """Define classification task."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = ClassificationReader.add_cmdline_args(parser)
        group.add_argument("--num_classes", type=int, default=2,
                           help="The num classes in classification task.")
        return group

    def __init__(self, args):
        super(Classification, self).__init__(args)
        self.reader = ClassificationReader(args)
        self.num_classes = args.num_classes
        return

    def merge_metrics_and_statistics(self, outputs, part_outputs):
        """Merge two evaulation output.

        Args:
            outputs: Original outputs which contains metrics and statistics.
            part_outputs: New outputs which contains metrics and statistics.

        Returns:
            Return merged output which contains metrics and statistics.
        """
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
            if k.startswith("stat_"):
                new_outputs[k] = outputs[k] + part_outputs[k]
            else:
                new_outputs[k] = (
                    outputs[k] * batch_size + part_outputs[k] * part_batch_size
                ) / new_outputs["batch_size"]
        return new_outputs

    def get_metrics(self, outputs):
        """Get metrics."""
        if outputs is None:
            raise ValueError("metrics is None")
        outputs = dict(outputs)
        # pop statistics
        outputs.pop("batch_size", None)
        if self.num_classes == 2:
            tp = outputs.pop("stat_tp")
            fp = outputs.pop("stat_fp")
            tn = outputs.pop("stat_tn")
            fn = outputs.pop("stat_fn")

            outputs["precision"] = tp / (tp + fp + 1e-10)
            outputs["recall"] = tp / (tp + fn + 1e-10)
            outputs["f1"] = (2 * outputs["precision"] * outputs["recall"]) \
                / (outputs["precision"] + outputs["recall"] + 1e-10)
        return outputs

    def _post_process_infer_output(self, predictions):
        """Post-process inference output."""
        predictions = [{"data_id": data_id.tolist(), "score": score.tolist()[1]}
                       for data_id, score in zip(predictions["data_id"], predictions["scores"])]
        return predictions
