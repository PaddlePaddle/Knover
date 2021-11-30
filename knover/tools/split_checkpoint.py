#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Split parameters for model parallelism."""

import argparse
import pickle

import numpy as np
import paddle
import paddle.fluid as fluid


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--split_type", type=str, default="mp",
                        choices=["mp"])
    parser.add_argument("--num_partitions", type=int, default=2)

    return parser.parse_args()


def mp_convert_fn(prog_state, num_partitions):
    mp_prog_state = {}
    for k in prog_state:
        weight = prog_state[k]
        if "multi_head_att" in k:
            if "w_0" in k:
                if len(weight.shape) == 1:
                    # beta*_pow_acc
                    splited_weights = [weight] * num_partitions
                elif "output_fc" in k:
                    splited_weights = np.split(weight, num_partitions, axis=0)
                else:
                    splited_weights = np.split(weight, num_partitions, axis=1)

                for i, w in enumerate(splited_weights):
                    mp_prog_state[k.replace("fc", f"fc_{i}")] = w
            else:
                assert "b_0" in k
                if weight.shape[0] == 1:
                    # beta*_pow_acc
                    splited_weights = [weight] * num_partitions
                elif "output_fc" in k:
                    splited_weights = [weight] * num_partitions
                else:
                    splited_weights = np.split(weight, num_partitions, axis=0)

                for i, w in enumerate(splited_weights):
                    if "output_fc" in k:
                        mp_prog_state[k] = w
                    else:
                        mp_prog_state[k.replace("fc", f"fc_{i}")] = w
        elif "ffn_fc" in k:
            if "w_0" in k:
                if "fc_0" in k:
                    # fc_0.w_0
                    if len(weight.shape) == 1:
                        for i in range(num_partitions):
                            mp_prog_state[k.replace("fc_0", f"fc_0_{i}")] = weight
                    else:
                        for i, w in enumerate(np.split(weight, num_partitions, axis=1)):
                            mp_prog_state[k.replace("fc_0", f"fc_0_{i}")] = w
                else:
                    # fc_1.w_0
                    if len(weight.shape) == 1:
                        for i in range(num_partitions):
                            mp_prog_state[k.replace("fc_1", f"fc_1_{i}")] = weight
                    else:
                        for i, w in enumerate(np.split(weight, num_partitions, axis=0)):
                            mp_prog_state[k.replace("fc_1", f"fc_1_{i}")] = w
            else:
                assert "b_0" in k
                if "fc_0" in k:
                    # fc_0.b_0
                    if weight.shape[0] == 1:
                        for i in range(num_partitions):
                            mp_prog_state[k.replace("fc_0", f"fc_0_{i}")] = weight
                    else:
                        for i, w in enumerate(np.split(weight, num_partitions, axis=0)):
                            mp_prog_state[k.replace("fc_0", f"fc_0_{i}")] = w
                else:
                    # fc_1.b_0
                    for i in range(num_partitions):
                        mp_prog_state[k] = weight
        else:
            mp_prog_state[k] = weight

    return mp_prog_state


def main(args):
    paddle.enable_static()
    prog_state = fluid.io.load_program_state(args.param_path)

    if args.split_type == "mp":
        prog_state = mp_convert_fn(prog_state, args.num_partitions)
    else:
        raise ValueError(f"split type: {args.split_type} is not supported now.")

    program = fluid.Program()
    for k in prog_state:
        weight = prog_state[k]
        param = program.global_block().create_parameter(
            shape=weight.shape,
            dtype=weight.dtype,
            name=k)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(program)

    for k in prog_state:
        param_tensor = fluid.global_scope().find_var(k).get_tensor()
        param_tensor.set(prog_state[k], exe.place)

    fluid.io.save_params(exe, args.save_path, main_program=program)
    return


if __name__ == "__main__":
    args = setup_args()
    main(args)
