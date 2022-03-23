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
"""Dense embedding task."""

import math

from knover.core.task import Task
from knover.data.dense_embedding_reader import DenseEmbeddingReader
from knover.tasks import register_task
from knover.utils.args import str2bool


@register_task("DenseEmbedding")
class DenseEmbedding(Task):
    """Generate Dense Embeddings."""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline argurments."""
        group = DenseEmbeddingReader.add_cmdline_args(parser)
        group.add_argument("--do_dense_emb", type=str2bool, default=True)
        return group

    def __init__(self, args):
        super(DenseEmbedding, self).__init__(args)
        self.reader = DenseEmbeddingReader(args)
        return

    def _post_process_infer_output(self, predictions):
        """Post-process inference output."""
        predictions = [{"data_id": data_id.tolist()[0], "emb": emb.tolist()}
                    for data_id, emb in zip(predictions["data_id"], predictions["emb"])]
        return predictions
