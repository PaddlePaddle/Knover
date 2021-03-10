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
"""LR schedulers."""

import math

from paddle.optimizer.lr import LRScheduler


class CosineDecay(LRScheduler):

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 end_lr=0.0001,
                 last_epoch=-1,
                 verbose=False):
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        assert learning_rate > end_lr
        super(CosineDecay, self).__init__(learning_rate, last_epoch,
                                          verbose)

    def get_lr(self):
        if self.last_epoch < self.decay_steps:
            return self.end_lr + (self.base_lr - self.end_lr) * 0.5 * (
                math.cos((self.last_epoch * math.pi / self.decay_steps)) + 1)
        else:
            return self.end_lr

