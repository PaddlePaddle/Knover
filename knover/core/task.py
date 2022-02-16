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

from abc import abstractmethod, ABC

import numpy as np

from knover.core.model import ModelInterface


class Task(ABC):
    """Basic task."""

    def __init__(self, args):
        self.debug_mode = False
        return

    def train_step(self, model: ModelInterface, inputs):
        """Run one training step."""
        outputs = model.train_step(inputs)
        outputs = {k: v.tolist()[0] if isinstance(v, np.ndarray) else v
                   for k, v in outputs.items()}
        return outputs

    def eval_step(self, model: ModelInterface, inputs):
        """Run one evaluation step"""
        outputs = model.eval_step(inputs)
        outputs = {k: v.tolist()[0] if isinstance(v, np.ndarray) else v
                   for k, v in outputs.items()}
        return outputs

    def infer_step(self, model: ModelInterface, inputs):
        """Run one inference step."""
        predictions = model.infer_step(inputs)
        outputs = self._post_process_infer_output(predictions)
        return outputs

    def _post_process_infer_output(self, predictions):
        """Post-process inference output."""
        return predictions

    def merge_metrics_and_statistics(self, outputs, part_outputs):
        """Merge metrics and statistics."""
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
            new_outputs[k] = (
                outputs[k] * batch_size + part_outputs[k] * part_batch_size
            ) / new_outputs["batch_size"]
        return new_outputs

    def merge_distributed_metrics_and_statistics(self, outputs):
        """Merge metrics and statistics in distributed mode."""
        import paddle
        batch_size = outputs.pop("batch_size")
        bsz_tensor = paddle.to_tensor(np.array([batch_size]).astype(np.int))
        paddle.distributed.all_reduce(bsz_tensor)
        new_outputs = {"batch_size": bsz_tensor.numpy()[0]}
        for k in outputs:
            tensor = paddle.to_tensor(np.array([outputs[k] * batch_size]).astype(np.float))
            paddle.distributed.all_reduce(tensor)
            new_outputs[k] = tensor.numpy()[0] / new_outputs["batch_size"]
        return new_outputs

    def get_metrics(self, outputs):
        """Get metrics."""
        if outputs is None:
            raise ValueError("metrics is None")
        outputs = dict(outputs)
        # pop statistics
        outputs.pop("batch_size", None)
        return outputs

    def get_data_loader(self, model: ModelInterface, *args, is_infer=False, **kwargs):
        """Get the model's DataLoader.

        Args:
            model: the trained model.
            is_infer: whether to run model in inference mode.
            args: the arguments of Reader.data_generator.
            kwargs: the arguments of Reader.data_generator.

        Returns:
            loader: DataLoader.
        """
        generator = self.reader.data_generator(*args, is_infer=is_infer, **kwargs)
        return model.get_data_loader(generator, is_infer)

    def debug(self, debug_mode=True):
        """Switch debug mode."""
        self.debug_mode = debug_mode
        return
