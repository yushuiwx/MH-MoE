# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    from torch.nn import LayerNorm
    from torchscale.component.rms_norm import RMSNorm

from .multiway_network import MultiwayWrapper
from .xpos_relative_position import XPOS

if torch.cuda.get_device_capability()[0] > 7:
    from flash_attn.flash_attn_interface import flash_attn_func
    Use_Xformers = False
else:
    try:
        from xformers.ops import memory_efficient_attention, LowerTriangularMask, MemoryEfficientAttentionCutlassOp
    except ModuleNotFoundError:
        print("No Xformers Detected")
    Use_Xformers = True

from .utils_quant import QuantizeLinear, QuantizeEmbedding, act_quant_fn, AlphaInit, QuantConfig, LearnableBias, AttnQuantizerUnsigned
from .model_parallel import ModelParallelLinear
from .embedding import RotaryEmbedding, apply_rotary_pos_emb


# def qkv_init_method(tensor, **kwargs):
#     nn.init.xavier_uniform_(tensor, gain=1 / math.sqrt(2))
#
# def out_init_method(tensor, **kwargs):
#     nn.init.xavier_uniform_(tensor)

def ffn_init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.args = args
        self.quant_before_rope = getattr(args, "quant_before_rope", False)
        self.binary_query = getattr(args, "binary_query", False)
        self.binary_attn = getattr(args, "binary_attn", False)
        self.attn_bits = getattr(args, "attn_bits", 32)
        self.query_bits = getattr(args, "query_bits", 32)
        self.attn_quant_method = getattr(args, "attn_quant_method", "attn_absmax_per_token")
        self.attn_input_absmean_scale = getattr(args, "attn_input_absmean_scale", 1.0)
        self.attn_quant_symmetric = getattr(args, "attn_quant_symmetric", False)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_kv_heads = args.n_kv_heads
        assert self.num_heads % self.n_kv_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.flash_attention = args.flash_attention
        assert self.self_attention ^ self.encoder_decoder_attention
        if self.flash_attention:
            assert self.self_attention and dropout == 0

        has_bias = not args.no_bias

        config = QuantConfig(args)
        weight_bits = 32 if args.quant_ffn_only else config.weight_bits
        input_bits = 32 if args.quant_ffn_only else config.input_bits
        if self.args.model_parallel_size > 1:
            self.k_proj = MultiwayWrapper(
                args,
                ModelParallelLinear(
                    args,
                    embed_dim,
                    self.n_kv_heads * self.head_dim,
                    bias=has_bias,
                    parallel_mode='column',
                    init_method=ffn_init_method,
                    clip_val=config.clip_init_val,
                    weight_bits=weight_bits,
                    input_bits=input_bits,
                    weight_layerwise=config.weight_layerwise,
                    input_layerwise=config.input_layerwise,
                    weight_quant_method=config.weight_quant_method,
                    input_quant_method=config.input_quant_method,
                    learnable=config.learnable_scaling,
                    symmetric=config.sym_quant_qkvo,
                    hadamard_group=config.hadamard_group,
                    blockwise_quant=config.blockwise_quant,
                    weight_blocksize=config.weight_blocksize,
                    grad_act=config.grad_act,
                    weight_blockscale=config.weight_blockscale,
                    smoothquant=config.smoothquant,
                    smoothquant_alpha=config.smoothquant_alpha,
                    absmean_alpha=config.absmean_alpha,
                    sparse_blocksize=config.sparse_blocksize,
                    sparse_ratio=config.sparse_ratio,  
                    sparse_alpha=config.sparse_alpha,
                )
            )
            self.v_proj = MultiwayWrapper(
                args,
                ModelParallelLinear(
                    args,
                    embed_dim,
                    self.n_kv_heads * self.head_dim,
                    bias=has_bias,
                    parallel_mode='column',
                    init_method=ffn_init_method,
                    clip_val=config.clip_init_val,
                    weight_bits=weight_bits,
                    input_bits=input_bits,
                    weight_layerwise=config.weight_layerwise,
                    input_layerwise=config.input_layerwise,
                    weight_quant_method=config.weight_quant_method,
                    input_quant_method=config.input_quant_method,
                    learnable=config.learnable_scaling,
                    symmetric=config.sym_quant_qkvo,
                    hadamard_group=config.hadamard_group,
                    blockwise_quant=config.blockwise_quant,
                    weight_blocksize=config.weight_blocksize,
                    grad_act=config.grad_act,
                    weight_blockscale=config.weight_blockscale,
                    smoothquant=config.smoothquant,
                    smoothquant_alpha=config.smoothquant_alpha,
                    absmean_alpha=config.absmean_alpha,
                    sparse_blocksize=config.sparse_blocksize,
                    sparse_ratio=config.sparse_ratio,  
                    sparse_alpha=config.sparse_alpha,
                )
            )
            self.q_proj = MultiwayWrapper(
                args,
                ModelParallelLinear(
                    args,
                    embed_dim,
                    embed_dim,
                    bias=has_bias,
                    parallel_mode='column',
                    init_method=ffn_init_method,
                    clip_val=config.clip_init_val,
                    weight_bits=weight_bits,
                    input_bits=input_bits,
                    weight_layerwise=config.weight_layerwise,
                    input_layerwise=config.input_layerwise,
                    weight_quant_method=config.weight_quant_method,
                    input_quant_method=config.input_quant_method,
                    learnable=config.learnable_scaling,
                    symmetric=config.sym_quant_qkvo,
                    hadamard_group=config.hadamard_group,
                    blockwise_quant=config.blockwise_quant,
                    weight_blocksize=config.weight_blocksize,
                    grad_act=config.grad_act,
                    weight_blockscale=config.weight_blockscale,
                    smoothquant=config.smoothquant,
                    smoothquant_alpha=config.smoothquant_alpha,
                    absmean_alpha=config.absmean_alpha,
                    sparse_blocksize=config.sparse_blocksize,
                    sparse_ratio=config.sparse_ratio,  
                    sparse_alpha=config.sparse_alpha,
                )
            )
            self.out_proj = MultiwayWrapper(
                args,
                ModelParallelLinear(
                    args,
                    embed_dim,
                    embed_dim,
                    bias=has_bias,
                    parallel_mode='row',
                    init_method=ffn_init_method,
                    clip_val=config.clip_init_val,
                    weight_bits=weight_bits,
                    input_bits=input_bits,
                    weight_layerwise=config.weight_layerwise,
                    input_layerwise=config.input_layerwise,
                    weight_quant_method=config.weight_quant_method,
                    input_quant_method=config.input_quant_method,
                    learnable=config.learnable_scaling,
                    symmetric=config.sym_quant_qkvo,
                    hadamard_group=config.hadamard_group,
                    blockwise_quant=config.blockwise_quant,
                    weight_blocksize=config.weight_blocksize,
                    grad_act=config.grad_act,
                    weight_blockscale=config.weight_blockscale,
                    smoothquant=config.smoothquant,
                    smoothquant_alpha=config.smoothquant_alpha,
                    absmean_alpha=config.absmean_alpha,
                    sparse_blocksize=config.sparse_blocksize,
                    sparse_ratio=config.sparse_ratio,  
                    sparse_alpha=config.sparse_alpha,
                )
            )
        else:
            self.k_proj = MultiwayWrapper(
                args,
                QuantizeLinear(
                    embed_dim,
                    self.n_kv_heads * self.head_dim,
                    bias=has_bias,
                    clip_val=config.clip_init_val,
                    weight_bits=weight_bits,
                    input_bits=input_bits,
                    weight_layerwise=config.weight_layerwise,
                    input_layerwise=config.input_layerwise,
                    weight_quant_method=config.weight_quant_method,
                    input_quant_method=config.input_quant_method,
                    learnable=config.learnable_scaling,
                    symmetric=config.sym_quant_qkvo,
                    hadamard_group=config.hadamard_group,
                    blockwise_quant=config.blockwise_quant,
                    weight_blocksize=config.weight_blocksize,
                    grad_act=config.grad_act,
                    weight_blockscale=config.weight_blockscale,
                    smoothquant=config.smoothquant,
                    smoothquant_alpha=config.smoothquant_alpha,
                    absmean_alpha=config.absmean_alpha,
                    input_absmean_alpha=config.input_absmean_alpha,
                    sparse_blocksize=config.sparse_blocksize,
                    sparse_ratio=config.sparse_ratio,  
                    sparse_alpha=config.sparse_alpha,
                )
            )
            self.v_proj = MultiwayWrapper(
                args,
                QuantizeLinear(
                    embed_dim,
                    self.n_kv_heads * self.head_dim,
                    bias=has_bias,
                    clip_val=config.clip_init_val,
                    weight_bits=weight_bits,
                    input_bits=input_bits,
                    weight_layerwise=config.weight_layerwise,
                    input_layerwise=config.input_layerwise,
                    weight_quant_method=config.weight_quant_method,
                    input_quant_method=config.input_quant_method,
                    learnable=config.learnable_scaling,
                    symmetric=config.sym_quant_qkvo,
                    hadamard_group=config.hadamard_group,
                    blockwise_quant=config.blockwise_quant,
                    weight_blocksize=config.weight_blocksize,
                    grad_act=config.grad_act,
                    weight_blockscale=config.weight_blockscale,
                    smoothquant=config.smoothquant,
                    smoothquant_alpha=config.smoothquant_alpha,
                    absmean_alpha=config.absmean_alpha,
                    input_absmean_alpha=config.input_absmean_alpha,
                    sparse_blocksize=config.sparse_blocksize,
                    sparse_ratio=config.sparse_ratio,  
                    sparse_alpha=config.sparse_alpha,
                )
            )
            self.q_proj = MultiwayWrapper(
                args,
                QuantizeLinear(
                    embed_dim,
                    embed_dim,
                    bias=has_bias,
                    clip_val=config.clip_init_val,
                    weight_bits=weight_bits,
                    input_bits=input_bits,
                    weight_layerwise=config.weight_layerwise,
                    input_layerwise=config.input_layerwise,
                    weight_quant_method=config.weight_quant_method,
                    input_quant_method=config.input_quant_method,
                    learnable=config.learnable_scaling,
                    symmetric=config.sym_quant_qkvo,
                    hadamard_group=config.hadamard_group,
                    blockwise_quant=config.blockwise_quant,
                    weight_blocksize=config.weight_blocksize,
                    grad_act=config.grad_act,
                    weight_blockscale=config.weight_blockscale,
                    smoothquant=config.smoothquant,
                    smoothquant_alpha=config.smoothquant_alpha,
                    absmean_alpha=config.absmean_alpha,
                    input_absmean_alpha=config.input_absmean_alpha,
                    sparse_blocksize=config.sparse_blocksize,
                    sparse_ratio=config.sparse_ratio,  
                    sparse_alpha=config.sparse_alpha,
                )
            )
            self.out_proj = MultiwayWrapper(
                args,
                QuantizeLinear(
                    embed_dim,
                    embed_dim,
                    bias=has_bias,
                    clip_val=config.clip_init_val,
                    weight_bits=weight_bits,
                    input_bits=input_bits,
                    weight_layerwise=config.weight_layerwise,
                    input_layerwise=config.input_layerwise,
                    weight_quant_method=config.weight_quant_method,
                    input_quant_method=config.input_quant_method,
                    learnable=config.learnable_scaling,
                    symmetric=config.sym_quant_qkvo,
                    hadamard_group=config.hadamard_group,
                    blockwise_quant=config.blockwise_quant,
                    weight_blocksize=config.weight_blocksize,
                    grad_act=config.grad_act,
                    weight_blockscale=config.weight_blockscale,
                    smoothquant=config.smoothquant,
                    smoothquant_alpha=config.smoothquant_alpha,
                    absmean_alpha=config.absmean_alpha,
                    input_absmean_alpha=config.input_absmean_alpha,
                    sparse_blocksize=config.sparse_blocksize,
                    sparse_ratio=config.sparse_ratio,  
                    sparse_alpha=config.sparse_alpha,
                )
            )

        # self.inner_attn_ln = (
        #     MultiwayWrapper(args, LayerNorm(self.embed_dim, eps=args.layernorm_eps))
        #     if subln and self.self_attention
        #     else None
        # )
        # self.ln_embed_dim = self.embed_dim // args.group_norm_size
        # assert args.model_parallel_size == 1
        # assert (self.embed_dim // args.model_parallel_size) % self.ln_embed_dim == 0
        if args.rms_norm:
            Layernorm = RMSNorm
        else:
            Layernorm = LayerNorm

        self.ln_embed_dim = self.embed_dim // args.group_norm_size
        assert self.embed_dim % args.model_parallel_size == 0
        assert (self.embed_dim // args.model_parallel_size) % self.ln_embed_dim == 0
        elementwise = True if self.args.model_parallel_size <= 1 else False
        self.inner_attn_ln = (
            MultiwayWrapper(args, Layernorm(self.ln_embed_dim, eps=args.layernorm_eps, elementwise_affine=elementwise))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )
        self.partial_rotary_factor = args.partial_rotary_factor
        self.rotary_emb = (
            RotaryEmbedding(
                int(args.partial_rotary_factor * self.head_dim), 
                args.max_target_positions
            )
            if args.rotary_embed and self.self_attention
            else None
        )
        self.num_heads = self.num_heads // args.model_parallel_size
        self.n_kv_heads = self.n_kv_heads // args.model_parallel_size
        self.kv_quant_group = getattr(args, "kv_quant_group", 1)

        # if args.bmt:
        #     self.qkv_layernorm = MultiwayWrapper(args, Layernorm(self.embed_dim, eps=args.layernorm_eps))
        #     self.out_layernorm = MultiwayWrapper(args, Layernorm(self.embed_dim, eps=args.layernorm_eps))

        # if args.model_parallel_size <= 1:
        #     self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -1)
        # nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        # nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        # nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        # nn.init.xavier_uniform_(self.out_proj.weight)
        # nn.init.constant_(self.out_proj.bias, 0.0)
        # if self.args.bmt:
        #     self.qkv_layernorm.reset_parameters()
        #     self.out_layernorm.reset_parameters()

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_causal=False,
        position_ids=None, # pass position_ids for rotary embedding
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.n_kv_heads != self.num_heads:
            k = repeat(k, 'b l d -> b l (n d)', n = self.num_heads // self.n_kv_heads)
            v = repeat(v, 'b l d -> b l (n d)', n = self.num_heads // self.n_kv_heads)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.quant_before_rope:
            if self.binary_query:
                q = act_quant_fn(
                    q,
                    clip_val=None,
                    num_bits=self.query_bits,
                    symmetric=self.attn_quant_symmetric,
                    quant_method=self.attn_quant_method,
                    layerwise=False,
                    input_parallel=False,
                    grad_act=False,
                    scale=self.attn_input_absmean_scale,
                )

            if self.kv_quant_group != 1:
                k = k.view(bsz, self.num_heads, src_len, self.kv_quant_group, self.head_dim // self.kv_quant_group)
                v = v.view(bsz, self.num_heads, src_len, self.kv_quant_group, self.head_dim // self.kv_quant_group)
            if self.binary_attn:
                k = act_quant_fn(
                    k,
                    clip_val=None,
                    num_bits=self.attn_bits,
                    symmetric=self.attn_quant_symmetric,
                    quant_method=self.attn_quant_method,
                    layerwise=False,
                    input_parallel=False,
                    grad_act=False,
                    scale=self.attn_input_absmean_scale,
                )
                v = act_quant_fn(
                    v,
                    clip_val=None,
                    num_bits=self.attn_bits,
                    symmetric=self.attn_quant_symmetric,
                    quant_method=self.attn_quant_method,
                    layerwise=False,
                    input_parallel=False,
                    grad_act=False,
                    scale=self.attn_input_absmean_scale,
                )
            if self.kv_quant_group != 1:
                k = k.view(bsz, self.num_heads, src_len, self.head_dim)
                v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        if self.rotary_emb is not None:
            kv_seq_len = k.shape[-2]
            if incremental_state is not None and "prev_key" in incremental_state:
                kv_seq_len += incremental_state["prev_key"][0].shape[-2]
            cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            if self.partial_rotary_factor == 1.0:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids) # q,k apply rotary embedding
                # [bsz, nh, t, hd]
                if not self.quant_before_rope:
                    if self.kv_quant_group != 1:
                        k = k.view(bsz, self.num_heads, src_len, self.kv_quant_group, self.head_dim // self.kv_quant_group)
                        v = v.view(bsz, self.num_heads, src_len, self.kv_quant_group, self.head_dim // self.kv_quant_group)
                    if self.binary_attn:
                        k = act_quant_fn(
                            k,
                            clip_val=None,
                            num_bits=self.attn_bits,
                            symmetric=self.attn_quant_symmetric,
                            quant_method=self.attn_quant_method,
                            layerwise=False,
                            input_parallel=False,
                            grad_act=False,
                            scale=self.attn_input_absmean_scale,
                        )
                        v = act_quant_fn(
                            v,
                            clip_val=None,
                            num_bits=self.attn_bits,
                            symmetric=self.attn_quant_symmetric,
                            quant_method=self.attn_quant_method,
                            layerwise=False,
                            input_parallel=False,
                            grad_act=False,
                            scale=self.attn_input_absmean_scale,
                        )
                    if self.kv_quant_group != 1:
                        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
                        v = v.view(bsz, self.num_heads, src_len, self.head_dim)
            else:
                # Partial rotary embedding
                assert self.kv_quant_group == 1
                query_rot, query_pass = (
                    q[..., : self.rotary_emb.dim],
                    q[..., self.rotary_emb.dim :],
                )
                key_rot, key_pass = (
                    k[..., : self.rotary_emb.dim],
                    k[..., self.rotary_emb.dim :],
                )
                # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
                query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
                if not self.quant_before_rope:
                    if self.binary_attn:
                        key_pass = act_quant_fn(
                            key_pass,
                            clip_val=None,
                            num_bits=self.attn_bits,
                            symmetric=self.attn_quant_symmetric,
                            quant_method=self.attn_quant_method,
                            layerwise=False,
                            input_parallel=False,
                            grad_act=False,
                            scale=self.attn_input_absmean_scale,
                        )
                        v = act_quant_fn(
                            v,
                            clip_val=None,
                            num_bits=self.attn_bits,
                            symmetric=self.attn_quant_symmetric,
                            quant_method=self.attn_quant_method,
                            layerwise=False,
                            input_parallel=False,
                            grad_act=False,
                            scale=self.attn_input_absmean_scale,
                        )
                # [batch_size, seq_length, num_heads, head_dim]
                q = torch.cat((query_rot, query_pass), dim=-1)
                k = torch.cat((key_rot, key_pass), dim=-1)

        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if self.xpos is not None:
            if incremental_state is not None:
                offset = src_len - 1
            else:
                offset = 0
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)
        
        # if not self.quant_before_rope:
        #     if self.binary_query:
        #         q = act_quant_fn(
        #             q,
        #             clip_val=None,
        #             num_bits=self.query_bits,
        #             symmetric=self.attn_quant_symmetric,
        #             quant_method=self.attn_quant_method,
        #             layerwise=False,
        #             input_parallel=False,
        #             grad_act=False,
        #             scale=self.attn_input_absmean_scale,
        #         )

        #     if self.binary_attn:
        #         k = act_quant_fn(
        #             k,
        #             clip_val=None,
        #             num_bits=self.attn_bits,
        #             symmetric=self.attn_quant_symmetric,
        #             quant_method=self.attn_quant_method,
        #             layerwise=False,
        #             input_parallel=False,
        #             grad_act=False,
        #             scale=self.attn_input_absmean_scale,
        #         )
        #         v = act_quant_fn(
        #             v,
        #             clip_val=None,
        #             num_bits=self.attn_bits,
        #             symmetric=self.attn_quant_symmetric,
        #             quant_method=self.attn_quant_method,
        #             layerwise=False,
        #             input_parallel=False,
        #             grad_act=False,
        #             scale=self.attn_input_absmean_scale,
        #         )
        
        if self.flash_attention:
            assert is_causal
            if Use_Xformers:
                attn_bias = LowerTriangularMask()
                attn = memory_efficient_attention(q, k, v, attn_bias, op=MemoryEfficientAttentionCutlassOp)
            else:
                q = rearrange(q, '(b h) l d -> b l h d', b=bsz)
                k = rearrange(k, '(b h) l d -> b l h d', b=bsz)
                v = rearrange(v, '(b h) l d -> b l h d', b=bsz)
                attn = flash_attn_func(q, k, v, causal=is_causal)
                attn = rearrange(attn, 'b l h d -> (b h) l d')
            attn_weights = None
        else:
            q *= self.scaling
            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                attn_weights = torch.nan_to_num(attn_weights)
                attn_mask = attn_mask.unsqueeze(0)
                attn_weights += attn_mask

            if key_padding_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if rel_pos is not None:
                rel_pos = rel_pos.view(attn_weights.size())
                attn_weights = attn_weights + rel_pos

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
            attn_probs = self.dropout_module(attn_weights)
            attn = torch.bmm(attn_probs, v)

        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, self.head_dim * self.num_heads).transpose(0, 1)

        if self.inner_attn_ln is not None:
            # if self.args.model_parallel_size > 1:
            #     attn = attn.contiguous()
            #     attn = gather_from_model_parallel_region(attn)
            if (self.embed_dim // self.args.model_parallel_size) > self.ln_embed_dim:
                attn = rearrange(attn, 'b l (n d) -> b l n d', d=self.ln_embed_dim)
            attn = self.inner_attn_ln(attn)
            if (self.embed_dim // self.args.model_parallel_size) > self.ln_embed_dim:
                attn = rearrange(attn, 'b l n d -> b l (n d)')
            # if self.args.model_parallel_size > 1:
            #     attn = scatter_to_model_parallel_region(attn)

        # if self.args.bmt:
        #     attn = self.out_layernorm(self.out_proj(attn)) + attn
        # else:
        #     attn = self.out_proj(attn)
        attn = self.out_proj(attn)
        if attn_weights is not None:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)

        return attn, attn_weights
