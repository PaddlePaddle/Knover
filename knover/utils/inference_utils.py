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
"""Inference utils."""

import os

import paddle.fluid as fluid


def create_predictor(inference_model_path, place=None, return_numpy=True):
    """Create predictor."""
    if place is None:
        gpu_id = int(os.getenv("FLAGS_selected_gpus", 0))
        place = fluid.CUDAPlace(gpu_id)
    exe = fluid.Executor(place)

    scope = fluid.Scope()
    with fluid.scope_guard(scope):
        inference_prog, feed_target_names, fetch_targets = fluid.io.load_inference_model(
            inference_model_path, exe)

    def __predict__(inputs):
        with fluid.scope_guard(scope):
            outputs = exe.run(
                inference_prog,
                feed=inputs,
                fetch_list=fetch_targets,
                return_numpy=return_numpy)
            return outputs
    return __predict__
