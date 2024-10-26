# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    from torch.nn import LayerNorm
    from torchscale.component.rms_norm import RMSNorm
from .utils_quant import QuantizeLinear, QuantizeEmbedding, act_quant_fn, AlphaInit, QuantConfig
from einops import rearrange
from .model_parallel import ModelParallelLinear

from .xmoe.global_groups import get_moe_group


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = self.get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_rng_state(self):
        state = {"torch_rng_state": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
        return state

    def set_rng_state(self, state):
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng_state"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.set_rng_state(self.rng_state)


def make_experts(args, embed_dim, expert_ffn_dim, is_moe_layer):
    world_size = (
        1
        if not torch.distributed.is_initialized()
        else torch.distributed.get_world_size()
    )
    expert_list = []
    ddp_rank = args.ddp_rank
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert (
            args.moe_expert_count % world_size == 0
        ), f"{args.moe_expert_count}, {world_size}"
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(
                    FeedForwardNetwork(
                        args,
                        embed_dim,
                        expert_ffn_dim,
                        args.activation_fn,
                        args.dropout,
                        args.activation_dropout,
                        args.layernorm_eps,
                        args.subln,
                        is_moe_layer,
                    )
                )
    else:
        assert (
            world_size % args.moe_expert_count == 0
        ), f"{world_size}, {args.moe_expert_count}"

        # with set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
        moe_idx, _ = get_moe_group(args.moe_expert_count)

        with set_torch_seed(start_seed + moe_idx):
            expert_list.append(
                FeedForwardNetwork(
                    args,
                    embed_dim,
                    expert_ffn_dim,
                    args.activation_fn,
                    args.dropout,
                    args.activation_dropout,
                    args.layernorm_eps,
                    args.subln,
                    is_moe_layer,
                )
            )
    experts = nn.ModuleList(expert_list)
    return experts


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'silu':
        return F.silu
    elif activation == 'leaky_relu':
        return F.leaky_relu
    else:
        raise NotImplementedError


def ffn_init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
        is_moe_layer=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.use_quant_for_activation = getattr(args, "use_quant_for_activation", False)
        self.activation_fn = get_activation_fn(activation=str(activation_fn)) if not self.use_quant_for_activation else None
        self.negative_slope = getattr(args, "negative_slope", -1.0)
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.relu_squared = getattr(args, "relu_squared", False)
        self.glu = getattr(args, "glu", False)
        ratio = self.embed_dim // args.moe_lora_rank if args.moe_lora_rank != -1 and is_moe_layer else 1

        if args.rms_norm:
            Layernorm = RMSNorm
        else:
            Layernorm = LayerNorm

        # self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        # self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        has_bias = not args.no_bias
        config = QuantConfig(args)
        ffn_bits = config.ffn_bits if config.ffn_bits != -1 else config.input_bits
        fc2_bits = config.fc2_bits if config.fc2_bits != -1 else ffn_bits
        ffn_quant_method = config.ffn_quant_method if config.ffn_quant_method != "" else config.input_quant_method
        fc2_quant_method = config.fc2_quant_method if config.fc2_quant_method != "" else ffn_quant_method
        fc2_input_absmean_scale = config.fc2_input_absmean_scale if config.fc2_input_absmean_scale != -1.0 else config.input_absmean_alpha
        fc2_sparse_blocksize = config.fc2_sparse_blocksize if config.fc2_sparse_blocksize != -1 else config.sparse_blocksize
        fc2_sparse_ratio = config.fc2_sparse_ratio if config.fc2_sparse_ratio != -1.0 else config.sparse_ratio
        if self.args.model_parallel_size > 1:
            self.fc1 = ModelParallelLinear(
                args,
                self.embed_dim // ratio, 
                ffn_dim * ratio,
                bias=has_bias,
                parallel_mode='column',
                init_method=ffn_init_method,
                clip_val=config.clip_init_val,
                weight_bits=config.weight_bits,
                input_bits=ffn_bits,
                weight_layerwise=config.weight_layerwise,
                input_layerwise=config.input_layerwise,
                weight_quant_method=config.weight_quant_method,
                input_quant_method=ffn_quant_method,
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
            self.fc2 = ModelParallelLinear(
                args,
                ffn_dim * ratio, 
                self.embed_dim // ratio,
                bias=has_bias,
                parallel_mode='row',
                init_method=ffn_init_method,
                clip_val=config.clip_init_val,
                weight_bits=config.weight_bits,
                input_bits=fc2_bits,
                weight_layerwise=config.weight_layerwise,
                input_layerwise=config.input_layerwise,
                weight_quant_method=config.weight_quant_method,
                input_quant_method=fc2_quant_method,
                learnable=config.learnable_scaling,
                symmetric=config.sym_quant_ffn_attn,
                hadamard_group=config.hadamard_group,
                blockwise_quant=config.blockwise_quant,
                weight_blocksize=config.weight_blocksize,
                grad_act=config.grad_act,
                weight_blockscale=config.weight_blockscale,
                smoothquant=config.smoothquant,
                smoothquant_alpha=config.smoothquant_alpha,
                absmean_alpha=config.absmean_alpha,
                input_absmean_alpha=fc2_input_absmean_scale,
                sparse_blocksize=fc2_sparse_blocksize,
                sparse_ratio=fc2_sparse_ratio,  
                sparse_alpha=config.sparse_alpha,
            )
            self.fc3 = ModelParallelLinear(
                args,
                self.embed_dim // ratio,
                ffn_dim * ratio,
                bias=has_bias,
                parallel_mode='column',
                init_method=ffn_init_method,
                clip_val=config.clip_init_val,
                weight_bits=config.weight_bits,
                input_bits=ffn_bits,
                weight_layerwise=config.weight_layerwise,
                input_layerwise=config.input_layerwise,
                weight_quant_method=config.weight_quant_method,
                input_quant_method=ffn_quant_method,
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
            ) if self.glu or (activation_fn == 'silu' or activation_fn == 'leaky_relu' or (activation_fn == 'relu' and not self.relu_squared)) else None
        else:
            self.fc1 = QuantizeLinear(
                self.embed_dim // ratio, 
                ffn_dim * ratio,
                bias=has_bias,
                clip_val=config.clip_init_val,
                weight_bits=config.weight_bits,
                input_bits=ffn_bits,
                weight_layerwise=config.weight_layerwise,
                input_layerwise=config.input_layerwise,
                weight_quant_method=config.weight_quant_method,
                input_quant_method=ffn_quant_method,
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
            self.fc2 = QuantizeLinear(
                ffn_dim * ratio,
                self.embed_dim // ratio,
                bias=has_bias,
                clip_val=config.clip_init_val,
                weight_bits=config.weight_bits,
                input_bits=fc2_bits,
                weight_layerwise=config.weight_layerwise,
                input_layerwise=config.input_layerwise,
                weight_quant_method=config.weight_quant_method,
                input_quant_method=fc2_quant_method,
                learnable=config.learnable_scaling,
                symmetric=config.sym_quant_ffn_attn,
                hadamard_group=config.hadamard_group,
                blockwise_quant=config.blockwise_quant,
                weight_blocksize=config.weight_blocksize,
                grad_act=config.grad_act,
                weight_blockscale=config.weight_blockscale,
                smoothquant=config.smoothquant,
                smoothquant_alpha=config.smoothquant_alpha,
                absmean_alpha=config.absmean_alpha,
                input_absmean_alpha=fc2_input_absmean_scale,
                sparse_blocksize=fc2_sparse_blocksize,
                sparse_ratio=fc2_sparse_ratio,
                sparse_alpha=config.sparse_alpha,
            )
            self.fc3 = QuantizeLinear(
                self.embed_dim // ratio,
                ffn_dim * ratio,
                bias=has_bias,
                clip_val=config.clip_init_val,
                weight_bits=config.weight_bits,
                input_bits=ffn_bits,
                weight_layerwise=config.weight_layerwise,
                input_layerwise=config.input_layerwise,
                weight_quant_method=config.weight_quant_method,
                input_quant_method=ffn_quant_method,
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
            ) if self.glu or (activation_fn == 'silu' or activation_fn == 'leaky_relu' or (activation_fn == 'relu' and not self.relu_squared)) else None

        self.ffn_dim = ffn_dim * ratio
        self.ln_embed_dim = ffn_dim * ratio // args.group_norm_size
        assert ffn_dim * ratio % args.model_parallel_size == 0
        assert (ffn_dim * ratio // args.model_parallel_size) % self.ln_embed_dim == 0
        elementwise = True if self.args.model_parallel_size <= 1 else False
        self.ffn_layernorm = Layernorm(self.ln_embed_dim, eps=layernorm_eps, elementwise_affine=elementwise) if subln else None
        self.quant_ffn_output = getattr(args, "quant_ffn_output", False)
        self.output_bits = ffn_bits
        self.ffn_quant_method = ffn_quant_method
        self.nozero_rmsnorm = getattr(args, "nozero_rmsnorm", False)

        if args.model_parallel_size <= 1:
            self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()
        if self.fc3 is not None:
            self.fc3.reset_parameters()

    def forward(self, x):
        if self.fc3 is not None:
            # x_shape = x.shape
            # x = x.reshape(-1, x.size(-1))
            if self.activation_fn is not None:
                gate = self.activation_fn(self.fc3(x), negative_slope=self.negative_slope) if self.negative_slope != -1.0 else self.activation_fn(self.fc3(x))
                if self.relu_squared:
                    gate = gate * gate.abs()
            else:
                gate = self.fc3(x)
            x = self.fc1(x) * gate
            if self.ffn_layernorm is not None:
                if (self.ffn_dim // self.args.model_parallel_size) > self.ln_embed_dim:
                    x = rearrange(x, 'b (n d) -> b n d', d=self.ln_embed_dim)
                x = self.ffn_layernorm(x)
                if self.nozero_rmsnorm:
                    dtype = x.dtype
                    nozero = (x != 0.0).sum(dim=-1, keepdim=True).sqrt()
                    x = x * nozero / math.sqrt(self.ln_embed_dim)
                    x = x.type(dtype)
                if (self.ffn_dim // self.args.model_parallel_size) > self.ln_embed_dim:
                    x = rearrange(x, 'b n d -> b (n d)')
            x= self.fc2(
                    x, 
                    # vis_input=True, 
                    # vis_output=True,
                )
            # x = x.view(x_shape)
            x = self.dropout_module(x)
        else:
            # x_shape = x.shape
            # x = x.reshape(-1, x.size(-1))
            x = self.fc1(x)
            if self.activation_fn is not None:
                x = self.activation_fn(x.float()).type_as(x)
            if self.relu_squared:
                x = x * x.abs()
            x = self.activation_dropout_module(x)
            if self.ffn_layernorm is not None:
                if (self.ffn_dim // self.args.model_parallel_size) > self.ln_embed_dim:
                    x = rearrange(x, 'b (n d) -> b n d', d=self.ln_embed_dim)
                x = self.ffn_layernorm(x)
                if self.nozero_rmsnorm:
                    dtype = x.dtype
                    nozero = (x != 0.0).sum(dim=-1, keepdim=True).sqrt()
                    x = x * nozero / math.sqrt(self.ln_embed_dim)
                    x = x.type(dtype)
                if (self.ffn_dim // self.args.model_parallel_size) > self.ln_embed_dim:
                    x = rearrange(x, 'b n d -> b (n d)')
            x = self.fc2(x)
            # x = x.view(x_shape)
            x = self.dropout_module(x)
            
        if self.quant_ffn_output:
            x = act_quant_fn(
                x,
                clip_val=None,
                num_bits=self.output_bits,
                symmetric=True,
                quant_method=self.ffn_quant_method,
                layerwise=True,
                input_parallel=False,
                grad_act=False,
                scale=1.0,
            )
        return x
