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
"""Task base."""

from abc import (
    abstractmethod,
    ABC
)

from models.model_base import Model


class Task(ABC):
    """
    Basic task.
    """

    def __init__(self, args):
        return

    def train_step(self, model: Model, inputs):
        """Run one training step."""
        outputs = model.train_step(inputs)
        outputs = {k: v.tolist()[0] for k, v in outputs.items()}
        return outputs

    def eval_step(self, model: Model, inputs):
        """Run one evaluation step"""
        outputs = model.eval_step(inputs)
        outputs = {k: v.tolist()[0] for k, v in outputs.items()}
        return outputs

    def infer_step(self, model: Model, inputs):
        """Run one inference step."""
        predictions = model.infer_step(inputs)
        outputs = self._post_process_infer_output(predictions)
        return outputs

    def _post_process_infer_output(self, predictions):
        """
        Post-process inference output.
        """
        return predictions

    @abstractmethod
    def merge_mertrics_and_statistics(self, outputs, part_outputs):
        """
        Merge metrics and statistics.
        """
        pass

    @abstractmethod
    def show_metrics(self, outupts):
        """
        Show metrics.
        """
        pass
