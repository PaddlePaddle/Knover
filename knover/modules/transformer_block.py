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
"""Transformer block."""

import collections

import paddle
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear, RowParallelLinear
from paddle.distributed.fleet.utils import recompute
from paddle.fluid import layers
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.nn.layer.transformer import _convert_param_attr_to_list, _convert_attention_mask


def parallel_matmul(lm_output, logit_weights, parallel_output):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


class MultiHeadAttention(nn.Layer):
    """Multi-Head Attention.

    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None,
                 fuse_qkv=False,
                 num_partitions=1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout, mode="upscale_in_train")
        self.need_weights = need_weights
        self.fuse_qkv = fuse_qkv

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert self.num_heads % num_partitions == 0
        self.num_heads = self.num_heads // num_partitions

        if self.fuse_qkv:
            assert self.kdim == embed_dim, "embed_dim should be equal to kdim"
            assert self.vdim == embed_dim, "embed_dim should be equal to vidm"

            self.qkv_proj = ColumnParallelLinear(
                embed_dim,
                3 * embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False)
        else:
            self.q_proj = ColumnParallelLinear(
                embed_dim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False)

            self.k_proj = ColumnParallelLinear(
                self.kdim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False)

            self.v_proj = ColumnParallelLinear(
                self.vdim,
                embed_dim,
                weight_attr=weight_attr,
                has_bias=True,
                gather_output=False)

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            weight_attr=weight_attr,
            has_bias=True,
            input_is_parallel=True)

    def _fuse_prepare_qkv(self, x):
        """Prapares linear projected queries, keys and values in fused style."""
        mix_layer = self.qkv_proj(x)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, self.num_heads, 3 * self.head_dim])
        mix_layer = paddle.transpose(mix_layer, [0, 2, 1, 3])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)
        return q, k, v

    def _prepare_qkv(self, query, key, value, cache=None):
        """Prapares linear projected queries, keys and values."""
        q = self.q_proj(query)
        q = tensor.reshape(q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)

        if cache is not None:
            return q, k, v, self.Cache(k, v)
        else:
            return q, k, v

    def compute_kv(self, key, value):
        """Prapares linear projected  keys and values.

        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.

        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(k, perm=[0, 2, 1, 3])
        v = tensor.reshape(v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """Generates cache for faster decoding step by step.

        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self,
                query,
                key,
                value,
                attn_bias=None,
                cache=None):
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is not None:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)
        elif self.fuse_qkv:
            q, k, v = self._fuse_prepare_qkv(query)
        else:
            q, k, v = self._prepare_qkv(query, key, value)
        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

        if isinstance(attn_bias, str) and attn_bias == "upper_triangle":
            weights = incubate.softmax_mask_fuse_upper_triangle(product)
        elif attn_bias is not None:
            weights = F.softmax(product + attn_bias)
        else:
            weights = F.softmax(product)

        weights = self.dropout(weights)

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerEncoderLayer(nn.Layer):
    """Transformer encoder layer.

    It contains Multi-head Attention and Position-wise Feed-forward Network.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=True,
                 fuse_qkv=False,
                 weight_attr=None,
                 bias_attr=None,
                 num_partitions=1):
        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            num_partitions=num_partitions,
            fuse_qkv=fuse_qkv)

        self.linear1 = ColumnParallelLinear(
            d_model,
            dim_feedforward,
            weight_attr=weight_attrs[2],
            gather_output=False,
            has_bias=True)

        self.linear2 = RowParallelLinear(
            dim_feedforward,
            d_model,
            weight_attr=weight_attrs[2],
            input_is_parallel=True,
            has_bias=True)

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self,
                src,
                src_mask=None,
                cache=None):
        residual = src

        if self.normalize_before:
            src = self.norm1(src)

        if cache is not None:
            src, incremental_cache = self.self_attn(src, src, src, src_mask, cache)
        else:
            src = self.self_attn(src, src, src, src_mask)

        src = residual + self.dropout1(src)

        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)

        # src = self.linear2(F.gelu(self.linear1(src), approximate=True))
        src = self.activation(self.linear1(src))
        src = self.dropout2(src)
        src = self.linear2(src)

        src = residual + self.dropout1(src)

        if not self.normalize_before:
            src = self.norm2(src)

        if cache is not None:
            return src, incremental_cache
        else:
            return src

    def gen_cache(self, memory):
        """Generates cache for faster decoding step by step.

        The generated cache is Cache or StaticCache produced by `MultiHeadAttention.gen_cache`.
        See `MultiHeadAttention.gen_cache` for more details.
        """
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        return incremental_cache


class TransformerEncoder(nn.Layer):
    """TransformerEncoder is a stack of N encoder layers."""

    def __init__(self, encoder_layers, norm=None, use_recompute=False):
        super(TransformerEncoder, self).__init__()
        # TODO: use LayerList (https://github.com/PaddlePaddle/Paddle/blob/bed652d6ece3791c6a68d0a61f0f1007fc044a91/python/paddle/nn/layer/transformer.py#L652)
        self.layers = encoder_layers
        self.norm = norm
        self.use_recompute = use_recompute

    def forward(self,
                src,
                src_mask=None,
                caches=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        if caches is not None:
            new_caches = []
        if self.use_recompute and self.training:
            self.checkpoints = []

        for i, mod in enumerate(self.layers):
            if caches is not None:
                output, new_cache = mod(output, src_mask, caches[i])
                new_caches.append(new_cache)
            elif self.use_recompute and self.training:
                output = recompute(mod, output, src_mask)
            else:
                output = mod(output, src_mask)
            if self.use_recompute and self.training:
                self.checkpoints.append(output.name)

        if self.norm is not None:
            output = self.norm(output)
        if caches is not None:
            return output, new_caches
        else:
            return output

    def gen_cache(self, memory, do_zip=False):
        """Generates cache for faster decoding step by step.

        The generated cache is a list, and each element in it is Cache or StaticCache
        produced by `TransformerLayer.gen_cache`. See `TransformerLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
       """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache
