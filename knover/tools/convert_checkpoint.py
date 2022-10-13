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
"""Convert checkpoint between different format."""

import argparse
import pickle
import regex

import numpy as np
import paddle
import paddle.fluid as fluid


def setup_args():
    """Setup convert checkpoints arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_path", type=str, required=True,
                        help="The path of parameters")
    parser.add_argument("--save_path", type=str, required=True,
                        help="The path of converted parameters.")
    parser.add_argument("--convert_type", type=str, default="static2dygraph",
                        choices=["static2dygraph", "dygraph2static", "paddle2pkl", "fp16"],
                        help="The argument determine how to convert checkpoint.")

    return parser.parse_args()


def basic_convert_fn(state_dict, src_tgt_mp, transpose=False):
    """Basic convert function."""
    new_state_dict = {}
    for k in sorted(state_dict):
        if hasattr(state_dict[k], "numpy"):
            state_dict[k] = state_dict[k].numpy()
        if not isinstance(state_dict[k], np.ndarray):
            continue
        flag = False
        for src, tgt in src_tgt_mp.items():
            pat = regex.compile(src)
            match_k = k.replace(".gamma", ".weight").replace(".beta", ".bias")
            res = pat.match(match_k)
            if res:
                if tgt is not None:
                    new_k = tgt.format(*res.groups())
                    if transpose and new_k.endswith(".w_0"):
                        new_state_dict[new_k] = np.transpose(state_dict[k])
                    else:
                        new_state_dict[new_k] = state_dict[k]
                    print(f"Convert {k} -> {new_k}: {new_state_dict[new_k].shape}")
                else:
                    print("Skip", k)
                flag = True
                break
        if not flag:
            raise ValueError(f"Cannot convert {k}!")

    print("Convert checkpoint sucessfully.")
    return new_state_dict


def load_program_state(param_path):
    """Loading parameters from both.

    Support both static and dygraph mode.
    """
    paddle.enable_static()
    prog_state = fluid.io.load_program_state(param_path)
    return prog_state


def save_static(state_dict, save_path):
    """Save model in static mode(paddle 1.x).

    This function will save all model parameters in a directory.
    """
    paddle.enable_static()
    program = fluid.Program()
    for k in state_dict:
        weight = state_dict[k]
        param = program.global_block().create_parameter(
            shape=weight.shape,
            dtype=weight.dtype,
            name=k)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(program)

    for k in state_dict:
        param_tensor = fluid.global_scope().find_var(k).get_tensor()
        param_tensor.set(state_dict[k], exe.place)

    fluid.io.save_params(exe, save_path, main_program=program)
    print(f"Save parameters into {save_path} in static mode(paddle 1.x) sucessfully.")


def save_dygraph(state_dict, save_path):
    """Save model in dygraph mode.

    This function will save all model parameters in a file.
    """
    paddle.disable_static()
    paddle.save(state_dict, save_path)
    print(f"Save parameters into {save_path} in dygraph mode sucessfully.")


def static2dygraph(prog_state):
    """Convert parameters from static mode(paddle 1.x) to dygraph mode."""
    return basic_convert_fn(prog_state, {
        ".*moment.*": None,
        ".*pow_acc.*": None,
        "@LR_DECAY_COUNTER@": None,
        "num_.*_steps.*": None,
        "loss_scaling_0": None,
        "encoder_layer_(\\d+)_multi_head_att_query_fc.b_0": "encoder.layers.{}.self_attn.q_proj.bias",
        "encoder_layer_(\\d+)_multi_head_att_key_fc.b_0": "encoder.layers.{}.self_attn.k_proj.bias",
        "encoder_layer_(\\d+)_multi_head_att_value_fc.b_0": "encoder.layers.{}.self_attn.v_proj.bias",
        "encoder_layer_(\\d+)_multi_head_att_output_fc.b_0": "encoder.layers.{}.self_attn.out_proj.bias",
        "encoder_layer_(\\d+)_multi_head_att_query_fc.w_0": "encoder.layers.{}.self_attn.q_proj.weight",
        "encoder_layer_(\\d+)_multi_head_att_key_fc.w_0": "encoder.layers.{}.self_attn.k_proj.weight",
        "encoder_layer_(\\d+)_multi_head_att_value_fc.w_0": "encoder.layers.{}.self_attn.v_proj.weight",
        "encoder_layer_(\\d+)_multi_head_att_output_fc.w_0": "encoder.layers.{}.self_attn.out_proj.weight",
        "encoder_layer_(\\d+)_ffn_fc_0.b_0": "encoder.layers.{}.linear1.bias",
        "encoder_layer_(\\d+)_ffn_fc_1.b_0": "encoder.layers.{}.linear2.bias",
        "encoder_layer_(\\d+)_ffn_fc_0.w_0": "encoder.layers.{}.linear1.weight",
        "encoder_layer_(\\d+)_ffn_fc_1.w_0": "encoder.layers.{}.linear2.weight",
        "encoder_layer_(\\d+)_pre_ffn_layer_norm_scale": "encoder.layers.{}.norm2.weight",
        "encoder_layer_(\\d+)_pre_ffn_layer_norm_bias": "encoder.layers.{}.norm2.bias",
        "encoder_layer_(\\d+)_pre_att_layer_norm_scale": "encoder.layers.{}.norm1.weight",
        "encoder_layer_(\\d+)_pre_att_layer_norm_bias": "encoder.layers.{}.norm1.bias",
        "word_embedding": "token_embedding.weight",
        "pos_embedding": "pos_embedding.weight",
        "sent_embedding": "type_embedding.weight",
        "mask_lm_out_fc.b_0": "lm_logits_bias",
        "mask_lm_trans_fc.b_0": "lm_trans_fc.bias",
        "mask_lm_trans_fc.w_0": "lm_trans_fc.weight",
        "post_encoder_layer_norm_bias": "encoder.norm.bias",
        "post_encoder_layer_norm_scale": "encoder.norm.weight",
        "mask_lm_trans_layer_norm_bias": "lm_trans_norm.bias",
        "mask_lm_trans_layer_norm_scale": "lm_trans_norm.weight",
    })


def dygraph2static(state_dict):
    """Convert parameters from dygraph mode to static mode(paddle 1.x)."""
    return basic_convert_fn(state_dict, {
      "encoder.layers.(\\d+).self_attn.q_proj.bias": "encoder_layer_{}_multi_head_att_query_fc.b_0",
      "encoder.layers.(\\d+).self_attn.k_proj.bias": "encoder_layer_{}_multi_head_att_key_fc.b_0",
      "encoder.layers.(\\d+).self_attn.v_proj.bias": "encoder_layer_{}_multi_head_att_value_fc.b_0",
      "encoder.layers.(\\d+).self_attn.out_proj.bias": "encoder_layer_{}_multi_head_att_output_fc.b_0",
      "encoder.layers.(\\d+).self_attn.q_proj.weight": "encoder_layer_{}_multi_head_att_query_fc.w_0",
      "encoder.layers.(\\d+).self_attn.k_proj.weight": "encoder_layer_{}_multi_head_att_key_fc.w_0",
      "encoder.layers.(\\d+).self_attn.v_proj.weight": "encoder_layer_{}_multi_head_att_value_fc.w_0",
      "encoder.layers.(\\d+).self_attn.out_proj.weight": "encoder_layer_{}_multi_head_att_output_fc.w_0",
      "encoder.layers.(\\d+).linear1.bias": "encoder_layer_{}_ffn_fc_0.b_0",
      "encoder.layers.(\\d+).linear2.bias": "encoder_layer_{}_ffn_fc_1.b_0",
      "encoder.layers.(\\d+).linear1.weight": "encoder_layer_{}_ffn_fc_0.w_0",
      "encoder.layers.(\\d+).linear2.weight": "encoder_layer_{}_ffn_fc_1.w_0",
      "encoder.layers.(\\d+).norm2.weight": "encoder_layer_{}_pre_ffn_layer_norm_scale",
      "encoder.layers.(\\d+).norm2.bias": "encoder_layer_{}_pre_ffn_layer_norm_bias",
      "encoder.layers.(\\d+).norm1.weight": "encoder_layer_{}_pre_att_layer_norm_scale",
      "encoder.layers.(\\d+).norm1.bias": "encoder_layer_{}_pre_att_layer_norm_bias",
      "token_embedding.weight": "word_embedding",
      "pos_embedding.weight": "pos_embedding",
      "type_embedding.weight": "sent_embedding",
      "lm_logits_bias": "mask_lm_out_fc.b_0",
      "lm_trans_fc.bias": "mask_lm_trans_fc.b_0",
      "lm_trans_fc.weight": "mask_lm_trans_fc.w_0",
      "encoder.norm.bias": "post_encoder_layer_norm_bias",
      "encoder.norm.weight": "post_encoder_layer_norm_scale",
      "lm_trans_norm.bias": "mask_lm_trans_layer_norm_bias",
      "lm_trans_norm.weight": "mask_lm_trans_layer_norm_scale"
    })


def to_fp16(state_dict):
    """Convert parameters from fp32 to fp16."""
    new_state_dict = {}
    for k in state_dict:
        # [NOTE] Do not convert parameters in LayerNorm.
        if "layer_norm" in k:
            new_state_dict[k] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k].astype("float16")
    return new_state_dict


def convert_checkpoint(args):
    """Main function of converting checkpoint."""
    if args.convert_type == "paddle2pkl":
        state_dict = load_program_state(args.param_path)
        pickle.dump(state_dict, open(args.save_path, "wb"))
    elif args.convert_type == "static2dygraph":
        state_dict = load_program_state(args.param_path)
        state_dict = static2dygraph(state_dict)
        save_dygraph(state_dict, args.save_path)
    elif args.convert_type == "dygraph2static":
        state_dict = paddle.load(args.param_path)
        state_dict = dygraph2static(state_dict)
        save_static(state_dict, args.save_path)
    elif args.convert_type == "fp16":
        state_dict = load_program_state(args.param_path)
        state_dict = to_fp16(state_dict)
        save_static(state_dict, args.save_path)
    else:
        raise ValueError(f"convert_type: {args.convert_type} is not supported now.")

    return


if __name__ == "__main__":
    args = setup_args()
    convert_checkpoint(args)
