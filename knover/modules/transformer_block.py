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
"""Transformer block."""

from functools import partial

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers


def _build_linear_column_parallel(x, n_in, n_out, name, initializer, num_partitions, part_id):
    return paddle.distributed.split(
        x,
        size=(n_in, n_out),
        operation="linear",
        axis=1,
        gather_out=False,
        num_partitions=num_partitions,
        weight_attr=fluid.ParamAttr(name=name + f"_{part_id}.w_0", initializer=initializer),
        bias_attr=name + f"_{part_id}.b_0")


def _build_linear_row_parallel(x, n_in, n_out, name, initializer, num_partitions, part_id):
    return paddle.distributed.split(
        x,
        size=(n_in, n_out),
        operation="linear",
        axis=0,
        gather_out=True,
        num_partitions=num_partitions,
        weight_attr=fluid.ParamAttr(name=name + f"_{part_id}.w_0", initializer=initializer),
        bias_attr=name + ".b_0")


def gen_cache(x, d_key, d_value, n_head, topo=None):
    num_splits = topo.mp_info.size if topo is not None else 1
    k = layers.fill_constant_batch_size_like(
        input=x,
        shape=[-1, n_head // num_splits, 0, d_key],
        dtype="float32",
        value=0)
    v = layers.fill_constant_batch_size_like(
        input=x,
        shape=[-1, n_head // num_splits, 0, d_value],
        dtype="float32",
        value=0)
    return k, v


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         gather_idx=None,
                         store=False,
                         param_initializer=None,
                         name="multi_head_att",
                         topo=None):
    """Multi-Head Attention.

    Note that attn_bias is added to the logit before computing softmax activiation to
    mask certain selected positions so that they will not be considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError("Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """Add linear projection to queries, keys, and values."""
        if topo is None or topo.mp_info.size == 1:
            q = layers.fc(input=queries,
                          size=d_key * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=name + "_query_fc.w_0",
                              initializer=param_initializer),
                          bias_attr=name + "_query_fc.b_0")
            k = layers.fc(input=keys,
                          size=d_key * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=name + "_key_fc.w_0",
                              initializer=param_initializer),
                          bias_attr=name + "_key_fc.b_0")
            v = layers.fc(input=values,
                          size=d_value * n_head,
                          num_flatten_dims=2,
                          param_attr=fluid.ParamAttr(
                              name=name + "_value_fc.w_0",
                              initializer=param_initializer),
                          bias_attr=name + "_value_fc.b_0")
        else:
            q = _build_linear_column_parallel(
                queries,
                d_model,
                d_key * n_head,
                f"{name}_query_fc",
                param_initializer,
                topo.mp_info.size,
                topo.mp_info.rank)
            k = _build_linear_column_parallel(
                keys,
                d_model,
                d_key * n_head,
                f"{name}_key_fc",
                param_initializer,
                topo.mp_info.size,
                topo.mp_info.rank)
            v = _build_linear_column_parallel(
                values,
                d_model,
                d_value * n_head,
                f"{name}_value_fc",
                param_initializer,
                topo.mp_info.size,
                topo.mp_info.rank)
        return q, k, v

    def __split_heads(x, n_head):
        """Split input embeddings into multiply chunks.

        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [batch_size, max_seq_len, hidden_size] then output a tensor
        with shape [batch_size, num_heads, max_seq_len, hidden_size // num_heads].
        """
        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """Merge multiply chunks into output embeddings.

        Transpose and then reshape the last two dimensions of input tensor x
        into one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def __scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """Scaled Dot-Product Attention"""
        product = layers.matmul(x=q, y=k, transpose_y=True, alpha=d_key ** -0.5)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product, use_cudnn=True)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)
    if topo is not None and topo.mp_info.size > 1:
        n_head = n_head // topo.mp_info.size

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        cache_k, cache_v = cache["k"], cache["v"]
        if gather_idx is not None:
            select_k = layers.gather(cache_k, index=gather_idx)
            select_v = layers.gather(cache_v, index=gather_idx)
        else:
            select_k, select_v = cache_k, cache_v

        if store:
            k = layers.concat([select_k, k], axis=2)
            v = layers.concat([select_v, v], axis=2)
            layers.assign(k, cache["k"])
            layers.assign(v, cache["v"])
        else:
            # Cannot support static cache now.
            assert False
            layers.assign(select_k, cache["k"])
            layers.assign(select_v, cache["v"])
            k = layers.concat([select_k, k], axis=2)
            v = layers.concat([select_v, v], axis=2)

    ctx_multiheads = __scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    if topo is None or topo.mp_info.size == 1:
        proj_out = layers.fc(input=out,
                             size=d_model,
                             num_flatten_dims=2,
                             param_attr=fluid.ParamAttr(
                                 name=name + "_output_fc.w_0",
                                 initializer=param_initializer),
                             bias_attr=name + "_output_fc.b_0")
    else:
        n_head = n_head * topo.mp_info.size
        proj_out = _build_linear_row_parallel(
            out,
            d_value * n_head,
            d_model,
            f"{name}_output_fc",
            param_initializer,
            topo.mp_info.size,
            topo.mp_info.rank)
    return proj_out


def positionwise_feed_forward(x,
                              d_inner_hid,
                              d_hid,
                              dropout_rate,
                              hidden_act,
                              param_initializer=None,
                              name="ffn",
                              topo=None):
    """Position-wise Feed-Forward Networks.

    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    if topo is None or topo.mp_info.size == 1:
        hidden = layers.fc(input=x,
                           size=d_inner_hid,
                           num_flatten_dims=2,
                           act=hidden_act,
                           param_attr=fluid.ParamAttr(
                               name=name + "_fc_0.w_0",
                               initializer=param_initializer),
                           bias_attr=name + "_fc_0.b_0")
    else:
        hidden = _build_linear_column_parallel(
            x,
            d_hid,
            d_inner_hid,
            f"{name}_fc_0",
            param_initializer,
            topo.mp_info.size,
            topo.mp_info.rank)
        hidden = getattr(layers, hidden_act)(hidden)
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)

    if topo is None or topo.mp_info.size == 1:
        out = layers.fc(input=hidden,
                        size=d_hid,
                        num_flatten_dims=2,
                        param_attr=fluid.ParamAttr(
                            name=name + "_fc_1.w_0", initializer=param_initializer),
                        bias_attr=name + "_fc_1.b_0")
    else:
        out = _build_linear_row_parallel(
            hidden,
            d_inner_hid,
            d_hid,
            f"{name}_fc_1",
            param_initializer,
            topo.mp_info.size,
            topo.mp_info.rank)
    return out


def pre_post_process_layer(prev_out,
                           out,
                           process_cmd,
                           dropout_rate=0.,
                           epsilon=1e-5,
                           name=""):
    """Add a pre-process or post-process between sub layers.

    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out_dtype = out.dtype
            if out_dtype == paddle.float16:
                out = layers.cast(out, "float32")
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.ParamAttr(
                    name=name + "_layer_norm_scale",
                    initializer=fluid.initializer.Constant(1.)),
                bias_attr=fluid.ParamAttr(
                    name=name + "_layer_norm_bias",
                    initializer=fluid.initializer.Constant(0.)),
                epsilon=epsilon)
            if out_dtype == paddle.float16:
                out = layers.cast(out, "float16")
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    dropout_implementation="upscale_in_train",
                    is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name="",
                  epsilon=1e-5,
                  cache=None,
                  gather_idx=None,
                  store=False,
                  topo=None):
    """A Transformer encoder block.

    The encoder layers that can be stacked to form a deep encoder.
    This module consists of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components are companied
    with the pre_process_layer / post_process_layer to add residual connection,
    layer normalization and dropout.
    """
    attn_output = multi_head_attention(
        pre_process_layer(
            enc_input,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + "_pre_att"),
        None,
        None,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + "_multi_head_att",
        cache=cache,
        gather_idx=gather_idx,
        store=store,
        topo=topo)
    attn_output = post_process_layer(
        enc_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        epsilon=epsilon,
        name=name + "_post_att")
    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            epsilon=epsilon,
            name=name + "_pre_ffn"),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + "_ffn",
        topo=topo)
    ffd_output = post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        epsilon=epsilon,
        name=name + "_post_ffn")
    return ffd_output, [ffd_output]


def encoder(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            pre_encoder_cmd="nd",
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            epsilon=1e-5,
            n_layer_per_block=1,
            param_share="normal",
            caches=None,
            gather_idx=None,
            store=False,
            topo=None,
            name="encoder"):
    """A Transformer Encoder.

    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    checkpoints = []
    names = []
    if param_share == "inner_share":
        for _ in range(n_layer // n_layer_per_block):
            for i in range(n_layer_per_block):
                names.append(name + "_layer_" + str(i))
    else:
        for i in range(n_layer // n_layer_per_block):
            for _ in range(n_layer_per_block):
                names.append(name + "_layer_" + str(i))

    enc_input = pre_process_layer(
        enc_input,
        pre_encoder_cmd,
        prepostprocess_dropout,
        epsilon=epsilon,
        name=f"pre_{name}")
    for i in range(n_layer):
        enc_output, cps = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            epsilon=epsilon,
            name=names[i],
            cache=caches[i] if caches is not None else None,
            gather_idx=gather_idx,
            store=store,
            topo=topo)
        checkpoints.extend(cps)
        enc_input = enc_output
    enc_output = pre_process_layer(
        enc_output,
        preprocess_cmd,
        prepostprocess_dropout,
        epsilon=epsilon,
        name=f"post_{name}")

    return enc_output, checkpoints
