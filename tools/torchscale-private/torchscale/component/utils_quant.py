# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import logging
import math
import os
import torch.distributed as dist
from fairseq.distributed.utils import get_model_parallel_group, get_model_parallel_world_size, all_reduce
from torch.distributed import ReduceOp


def get_max_num(directory_path, file_pattern):
    import re
    max_number = 0
    # directory_path = '/mnt/data/xxx'
    # file_pattern = r'fc2_activation_(\d+)\.pdf'

    # Loop through each file in the directory
    for file_name in os.listdir(directory_path):
        match = re.match(file_pattern, file_name)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number

    return max_number


def visualize_weight_distribution_from_tensor(weight, pic_path):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.hist(weight.ravel(), bins=100, color='blue', alpha=0.7, density=True)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    if pic_path is not None:
        plt.savefig(pic_path, dpi=600)

def visualize_input_distribution_from_tensor(weight, pic_path):
    # weight = weight[0]
    weight = weight[:, 100, : ]
    weight = weight.reshape(1, -1).cpu().numpy()[0]
    import matplotlib.pyplot as plt
    plt.clf()
    # plt.hist(weight.ravel(), bins=100, color='blue', alpha=0.7, density=True)
    plt.plot(list(range(len(weight))), weight)
    plt.xlabel('Values')
    plt.ylabel('Dims')
    plt.grid(True)
    if pic_path is not None:
        plt.savefig(pic_path, dpi=600)


class QuantConfig(object):
    def __init__(self, args, **kwargs):
        self.weight_bits = getattr(args, 'weight_bits', 1)
        self.input_bits = getattr(args, 'input_bits', 8)
        self.clip_init_val = 2.5
        self.clip_lr = 1e-4
        self.clip_wd = 0.0
        self.embed_layerwise = False
        self.weight_layerwise = not getattr(args, 'weight_featurewise', False)
        self.input_layerwise = True
        self.sym_quant_ffn_attn = False
        self.sym_quant_qkvo = True
        self.learnable_scaling = True
        self.weight_quant_method =  getattr(args, "weight_quant_method", "bwn")
        self.input_quant_method = getattr(args, "input_quant_method", "elastic")
        self.hadamard_group = getattr(args, "hadamard_group", 32)
        self.blockwise_quant = getattr(args, 'blockwise_quant', False)
        weight_blocksize = getattr(args, 'weight_blocksize', "-1,-1").split(',')
        self.weight_blocksize = (int(weight_blocksize[0]), int(weight_blocksize[1]))
        self.grad_act = getattr(args, 'grad_act', False)
        self.weight_blockscale = getattr(args, 'weight_blockscale', 'none')
        self.smoothquant = getattr(args, 'smoothquant', False)
        self.smoothquant_alpha = getattr(args, 'smoothquant_alpha', 0.5)
        self.ffn_bits = getattr(args, 'ffn_bits', -1)
        self.ffn_quant_method = getattr(args, "ffn_quant_method", "")
        self.absmean_alpha = getattr(args, "absmean_alpha", 1.0)
        self.fc2_bits = getattr(args, 'fc2_bits', -1)
        self.input_absmean_alpha = getattr(args, "input_absmean_alpha", 1.0)
        self.fc2_quant_method = getattr(args, "fc2_quant_method", "")
        self.fc2_input_absmean_scale = getattr(args, "fc2_input_absmean_scale", -1.0)
        self.sparse_blocksize = getattr(args, "sparse_blocksize", 16)
        self.fc2_sparse_blocksize = getattr(args, "fc2_sparse_blocksize", -1)
        self.sparse_ratio = getattr(args, "sparse_ratio", 0.4)
        self.fc2_sparse_ratio = getattr(args, "fc2_sparse_ratio", -1.0)
        self.sparse_alpha = getattr(args, "sparse_alpha", 1.0)
        self.sparse_before_quant = getattr(args, "sparse_before_quant", False)


class AttnQuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        # Qn = 0
        # Qp = 2 ** num_bits - 1
        # if num_bits == 1:
        #     min_val = 0
        #     input_ = input
        # else:
        #     min_val = input.min().item()
        #     input_ = input - min_val

        # s = Qp / input_.max(dim=-1, keepdim=True)[0]
        # return (input_ * s).round().clamp(Qn, Qp) + min_val
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        # ctx.save_for_backward(input)
        # dtype = input.dtype
        # input = input.float()
        e = input.mean()
        result = (input-e).sign()
        return result
        # if layerwise:
        #     s = input.size()
        #     m = input.norm(p=1).div(input.nelement())
        #     e = input.mean()
        #     result = (input-e).sign().mul(m.expand(s))
        # else:
        #     n = input[0].nelement()  # W of size axb, return a vector of  ax1
        #     s = input.size()
        #     m = input.norm(1, 1, keepdim=True).div(n)
        #     e = input.mean()
        #     result = (input-e).sign().mul(m.expand(s))

        # return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class AbsmaxPerTokenINT4QuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None

class AbsmaxPerTokenINT4bos8QuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        # dtype = input.dtype
        # input = input.float()
        # Qn = -2 ** (num_bits - 1)
        # Qp = 2 ** (num_bits - 1) - 1
        # s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        # result = (input * s).round().clamp(Qn, Qp) / s
        # return result.type(dtype)
        dtype = input.dtype
        input = input.float()
        x1 = input[:, :1, :]
        x2 = input[:, 1:, :]
        num_bits = 8
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s1 = Qp / x1.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result1 = (x1 * s1).round().clamp(Qn, Qp) / s1

        num_bits = 4
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s2 = Qp / x2.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result2 = (x2 * s2).round().clamp(Qn, Qp) / s2

        result = torch.concat([result1, result2], dim=1)
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class AbsmaxPerTokenClipINT4QuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        num_bits = ctx.num_bits
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None


class AbsmaxPerTokenQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class AbsmaxPerTokenQuantizerSparseSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, sparse_blocksize):
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * s).round().clamp(Qn, Qp)
        mask = result.abs() > sparse_blocksize
        result *= mask
        result /= s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None

class AbsmaxPerTokenQuantizerClipSparseSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, sparse_blocksize):
        dtype = input.dtype
        input = input.float()
        ctx.sparse_blocksize = sparse_blocksize
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * s).round().clamp(Qn, Qp)
        ctx.save_for_backward(input, s)
        mask = result.abs() > sparse_blocksize
        result *= mask
        result /= s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        sparse_blocksize = ctx.sparse_blocksize
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        input, s = ctx.saved_tensors
        result = (input * s).round().clamp(Qn, Qp)
        indicate_small = (result.abs() <= sparse_blocksize) * (result.abs() > 0)
        indicate_middle = 1.0 - indicate_small.float() # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        # grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class AbsmaxPerTokenQuantizerTopKClipSparseSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, sparse_ratio):
        dtype = input.dtype
        dim = input.shape[-1]
        input = input.float()
        ctx.sparse_ratio = sparse_ratio
        ctx.num_bits = num_bits
        ctx.dim = dim
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        input = (input * s).round().clamp(Qn, Qp)
        k = int((1 - sparse_ratio) * dim)
        values, indices = torch.topk(input.abs(), k, dim=-1)
        mask = torch.zeros_like(input, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
        input *= mask
        result = input / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        sparse_ratio = ctx.sparse_ratio
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        input, s = ctx.saved_tensors
        k = int((1 - sparse_ratio) * dim)
        values, indices = torch.topk(input.abs(), k, dim=-1)
        mask = torch.ones_like(input)
        mask.scatter_(dim=-1, index=indices, src=0.5 * torch.ones_like(indices))
        grad_input = mask * grad_output
        # grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class AbsmaxPerTokenQuantizerMaxTopKBlockSparseSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, block_size, sparse_ratio):
        dtype = input.dtype
        bsz, seq, hidden = input.shape[0], input.shape[1], input.shape[2]
        sparsed_block = int(sparse_ratio * hidden // block_size)
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        input = input.view(bsz, seq, hidden // block_size, block_size)
        s = input.abs().max(dim=-1).values
        threhold = s.kthvalue(k=sparsed_block, dim=-1, keepdim=True).values
        mask = (s > threhold).unsqueeze(dim=-1)
        input *= mask
        input = input.view(bsz, seq, hidden)

        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None

class AbsmaxPerTokenQuantizerTopKBlockSparseSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, block_size, sparse_ratio):
        dtype = input.dtype
        bsz, seq, hidden = input.shape[0], input.shape[1], input.shape[2]
        sparsed_block = int(sparse_ratio * hidden // block_size)
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        input = input.view(bsz, seq, hidden // block_size, block_size)
        s = input.abs().mean(dim=-1)
        threhold = s.kthvalue(k=sparsed_block, dim=-1, keepdim=True).values
        mask = (s > threhold).unsqueeze(dim=-1)
        input *= mask
        input = input.view(bsz, seq, hidden)

        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None


class AbsmaxPerTokenQuantizerTopKSparseSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, sparse_ratio):
        dtype = input.dtype
        dim = input.shape[-1]
        input = input.float()
        # ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        input = (input * s).round().clamp(Qn, Qp)
        k = int((1 - sparse_ratio) * dim)
        values, indices = torch.topk(input.abs(), k, dim=-1)
        mask = torch.zeros_like(input, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
        # kth_values = values[:, :, -1].unsqueeze(-1)
        # mask = input.abs() >= kth_values
        input *= mask
        # ctx.save_for_backward(input, s)
        result = input / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # num_bits = ctx.num_bits
        # if num_bits == 1:
        #     Qn = -1
        #     Qp = 1
        # else:
        #     Qn = -2 ** (num_bits - 1)
        #     Qp = 2 ** (num_bits - 1) - 1

        # input, s = ctx.saved_tensors
        # q_w = input * s
        # indicate_small = (q_w < Qn).float()
        # indicate_big = (q_w > Qp).float()
        # indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        # grad_input = indicate_middle * grad_output
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class AbsmaxPerTokenQuantizerMeanSparseSigned(torch.autograd.Function):
    @staticmethod
    # @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, sparse_alpha):
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        threhold = input.abs().mean(dim=-1, keepdim=True) * sparse_alpha
        mask = input.abs() > threhold
        input *= mask
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


# class AbsmaxPerTokenQuantizerReLUSigned(torch.autograd.Function):
#     @staticmethod
#     @torch.compile
#     def forward(ctx, input, clip_val, num_bits, layerwise):
#         dtype = input.dtype
#         input = input.float()
#         if num_bits == 1:
#             Qn = -1
#             Qp = 1
#         else:
#             Qn = -2 ** (num_bits - 1)
#             Qp = 2 ** (num_bits - 1) - 1
#         s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
#         result = (input * s).round().clamp(Qn, Qp) / s
#         return result.type(dtype)

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         return grad_input, None, None, None


class AttnAbsmaxPerTokenQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        # ctx.num_bits = num_bits
        # ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        # num_bits = ctx.num_bits
        # if num_bits == 1:
        #     Qn = -1
        #     Qp = 1
        # else:
        #     Qn = -2 ** (num_bits - 1)
        #     Qp = 2 ** (num_bits - 1) - 1

        # input, s = ctx.saved_tensors
        # q_w = input * s
        # indicate_small = (q_w < Qn).float()
        # indicate_big = (q_w > Qp).float()
        # indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        # grad_input = indicate_middle * grad_output
        return grad_input, None, None, None


class AttnAbsmaxPerTokenQuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        Qn = 0
        Qp = 2 ** num_bits - 1
        if num_bits == 1:
            min_val = 0
            input_ = input
        else:
            min_val = input.min(dim=-1, keepdim=True).values
            input_ = input - min_val

        s = Qp / input_.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input_ * s).round().clamp(Qn, Qp) / s + min_val
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class AttnAbsmaxPerTokenINT4QuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        # ctx.num_bits = num_bits
        # ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        # num_bits = ctx.num_bits
        # if num_bits == 1:
        #     Qn = -1
        #     Qp = 1
        # else:
        #     Qn = -2 ** (num_bits - 1)
        #     Qp = 2 ** (num_bits - 1) - 1

        # input, s = ctx.saved_tensors
        # q_w = input * s
        # indicate_small = (q_w < Qn).float()
        # indicate_big = (q_w > Qp).float()
        # indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        # grad_input = indicate_middle * grad_output
        return grad_input, None, None, None


class AttnAbsmaxPerTokenINT4QuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        Qn = 0
        Qp = 2 ** num_bits - 1
        if num_bits == 1:
            min_val = 0
            input_ = input
        else:
            min_val = input.min(dim=-1, keepdim=True).values
            input_ = input - min_val

        s = Qp / input_.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input_ * s).round().clamp(Qn, Qp) / s + min_val
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class AbsmeanQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, absmean_alpha):
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = absmean_alpha * math.sqrt(Qp) / input.abs().mean().clamp(min=1e-5)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class AbsmeanClipQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = math.sqrt(Qp) / input.abs().mean().clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None


class AbsmeanPerTokenClipbos8QuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        x1 = input[:, :1, :]
        x2 = input[:, 1:, :]
        num_bits = 8
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s1 = Qp / x1.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result1 = (x1 * s1).round().clamp(Qn, Qp) / s1

        num_bits = 2
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1
        s2 = math.sqrt(Qp) / scale / x2.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(x2, s2)
        result2 = (x2 * s2).round().clamp(Qn, Qp) / s2

        result = torch.concat([result1, result2], dim=1)
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_output[:, :1, :]
        num_bits = 2
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1

        x2, s2 = ctx.saved_tensors
        q_w = x2 * s2
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input2 = indicate_middle * grad_output[:, 1:, :]

        grad_input = torch.concat([grad_input1, grad_input2], dim=1)
        return grad_input, None, None, None, None
    

class AbsmeanPerTokenQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = math.sqrt(Qp) / scale / input.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None
    

class AbsmeanPerTokenClipQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = math.sqrt(Qp) / scale / input.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None


class AbsmeanPerTokenClipQuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min(dim=-1, keepdim=True).item()
            input_ = input - min_val

        s = math.sqrt(Qp) / scale / input_.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        if num_bits != 1:
            result = result + min_val
        return result.type(dtype)

    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        num_bits = ctx.num_bits
        Qn = 0
        Qp = 2 ** (num_bits) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None


class TwnPerTokenClipQuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        Qn = 0
        Qp = 2
        min_val = input.min(dim=-1, keepdim=True).values
        input_ = input - min_val

        s = math.sqrt(Qp) / scale / input_.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        result = result + min_val
        return result.type(dtype)

    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        num_bits = ctx.num_bits
        Qn = 0
        Qp = 2

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None
    

class AbsmeanPerTokenClipINT2QuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = math.sqrt(Qp) / scale / input.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None


class AbsmeanPerTokenClipINT2Scale1_2QuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = math.sqrt(Qp) / scale / input.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None


class AbsmeanPerTokenClipINT2Scale2QuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = math.sqrt(Qp) / scale / input.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None


class AbsmeanPerTokenClipINT2QuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, scale):
        ctx.num_bits = num_bits
        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        s = math.sqrt(Qp) / scale / input_.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        if num_bits != 1:
            result = result + min_val
        return result

    @staticmethod
    def backward(ctx, grad_output):
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None
    

class AbsmeanPerTokenClipBWNQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, scale=1.0):
        dtype = input.dtype
        input = input.float()
        ctx.num_bits = num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        s = math.sqrt(Qp) / scale / input.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_input = grad_output.clone()
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None


class AbsmeanPerTokenClipBWNQuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, scale):
        ctx.num_bits = num_bits
        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        s = math.sqrt(Qp) / scale / input_.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, s)
        result = (input * s).round().clamp(Qn, Qp) / s
        if num_bits != 1:
            result = result + min_val
        return result

    @staticmethod
    def backward(ctx, grad_output):
        num_bits = ctx.num_bits
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        input, s = ctx.saved_tensors
        q_w = input * s
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None, None
    

class AbsmaxPerTokenQuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        dtype = input.dtype
        input = input.float()
        Qn = 0
        Qp = 2 ** num_bits - 1
        if num_bits == 1:
            min_val = 0
            input_ = input
        else:
            min_val = input.min(dim=-1, keepdim=True).values
            input_ = input - min_val

        s = Qp / input_.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input_ * s).round().clamp(Qn, Qp) / s + min_val
        # result = (input_ * s).round().clamp(Qn, Qp) / s
        return result.type(dtype) 


    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class ElasticQuantBinarizerSigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    @torch.compile
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel()) if not Qp else 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        if num_bits == 1:
            input_ = input
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        if num_bits != 1:
            w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class AlphaInit(nn.Parameter):
    def __init__(self, tensor):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)


class AbsmaxQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, alpha, num_bits, layerwise, input_parallel=False):
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        s = Qp / input.abs().max().clamp(min=1e-5)
        # if input_parallel:
        #     dist.all_reduce(s, op=ReduceOp.MIN, group=get_model_parallel_group())

        result = (input * s).round().clamp(Qn, Qp) / s
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


def AbsmaxQuantizerActSigned(input, alpha, num_bits, layerwise, input_parallel=False):
    if num_bits == 1:
        Qn = -1
        Qp = 1
    else:
        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1

    s = Qp / input.abs().max()
    if input_parallel:
        dist.all_reduce(s, op=ReduceOp.MIN, group=get_model_parallel_group())

    input = torch.tanh(input)
    quantized = (input * s).round().clamp(Qn, Qp) / s

    return input - input.detach() + quantized

    # @staticmethod
    # def backward(ctx, grad_output):
    #     grad_input = grad_output.clone()
    #     return grad_input, None, None, None, None


class AbsmaxBlockQuantizerSigned(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, input, alpha, num_bits, layerwise, group):
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1

        shape = input.shape
        assert input.shape[-1] % group == 0
        input = input.reshape(-1, input.shape[-1] // group, group)
        s = Qp / input.abs().max(dim=-1, keepdim=True).values
        input= (input * s).round().clamp(Qn, Qp) / s

        return input.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class AbsmaxQuantizerUnsigned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, input_parallel=False):
        dtype = input.dtype
        input = input.float()
        Qn = 0
        Qp = 2 ** num_bits - 1
        if num_bits == 1:
            min_val = 0
            input_ = input
        else:
            min_val = input.min().item()
            # if input_parallel:
            #     dist.all_reduce(min_val, op=ReduceOp.MIN, group=get_model_parallel_group())
            input_ = input - min_val

        s = Qp / input_.abs().max().clamp(min=1e-5)
        # if input_parallel:
        #     dist.all_reduce(s, op=ReduceOp.MIN, group=get_model_parallel_group())

        result = (input_ * s).round().clamp(Qn, Qp) / s + min_val
        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def AbsmaxQuantizerActUnsigned(input, alpha, num_bits, layerwise, input_parallel=False):
    Qn = 0
    Qp = 2 ** num_bits - 1
    if num_bits == 1:
        min_val = 0
        input_ = input
    else:
        min_val = input.min().item()
        if input_parallel:
            dist.all_reduce(min_val, op=ReduceOp.MIN, group=get_model_parallel_group())
        input_ = input - min_val

    s = Qp / input_.abs().max()
    if input_parallel:
        dist.all_reduce(s, op=ReduceOp.MIN, group=get_model_parallel_group())

    input = torch.tanh(input)
    quantized = (input_ * s).round().clamp(Qn, Qp) / s + min_val

    return input - input.detach() + quantized

    # @staticmethod
    # def backward(ctx, grad_output):
    #     grad_input = grad_output.clone()
    #     return grad_input, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        dtype = input.dtype
        input = input.float()
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:  # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None
        

class BwnQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, input_parallel=False):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        dtype = input.dtype
        input = input.float()
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            if input_parallel:
                dist.all_reduce(m, op=ReduceOp.SUM, group=get_model_parallel_group())
                dist.all_reduce(e, op=ReduceOp.SUM, group=get_model_parallel_group())
                m /= get_model_parallel_world_size()
                e /= get_model_parallel_world_size()
            result = (input-e).sign().mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            if input_parallel:
                dist.all_reduce(m, op=ReduceOp.SUM, group=get_model_parallel_group())
                dist.all_reduce(e, op=ReduceOp.SUM, group=get_model_parallel_group())
                m /= get_model_parallel_world_size()
                e /= get_model_parallel_world_size()
            result = (input-e).sign().mul(m.expand(s))

        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class Bwn01Quantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise, input_parallel=False):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        dtype = input.dtype
        input = input.float()
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            if input_parallel:
                dist.all_reduce(m, op=ReduceOp.SUM, group=get_model_parallel_group())
                dist.all_reduce(e, op=ReduceOp.SUM, group=get_model_parallel_group())
                m /= get_model_parallel_world_size()
                e /= get_model_parallel_world_size()
            result = (input-e).sign()
            result = (torch.ones_like(result) + result) // 2
            result = result.mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            if input_parallel:
                dist.all_reduce(m, op=ReduceOp.SUM, group=get_model_parallel_group())
                dist.all_reduce(e, op=ReduceOp.SUM, group=get_model_parallel_group())
                m /= get_model_parallel_world_size()
                e /= get_model_parallel_world_size()
            result = (input-e).sign()
            result = (torch.ones_like(result) + result) // 2
            result = result.mul(m.expand(s))

        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None

class BwnRandomQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, input_parallel=False):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        dtype = input.dtype
        input = input.float()
        fan = torch.nn.init._calculate_correct_fan(input, mode='fan_in')
        bound = 0.1 * math.sqrt(6.0 / fan)
        
        s = input.size()
        m = input.norm(p=1).div(input.nelement())
        e = input.mean()
        input = input - e
        mask = (input >= - bound) & (input <= bound)
        random_signs = torch.where(torch.rand_like(input) < 0.5, 1, -1)
        input = torch.where(mask, random_signs, input.sign())
        result = input.mul(m.expand(s))

        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class WeightAbsmaxQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, input_parallel=False):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        # n = input[0].nelement()  # W of size out_features x in_features, return a vector of  out_features x 1
        # s = input.size()
        # m = input.norm(1, 1, keepdim=True).div(n)
        # scale = Qp / m.abs().max().clamp_(min=1e-5)
        # m = (m * scale).round().clamp(Qn, Qp) / scale

        n = input[1].nelement()  # W of size out_features x in_features, return a vector of  1 x in_features
        s = input.size()
        m = input.norm(dim=0, p=1, keepdim=True).div(n)
        scale = Qp / m.abs().max().clamp(min=1e-5)
        m = (m * scale).round().clamp(Qn, Qp) / scale
        
        e = input.mean()
        result = (input-e).sign().mul(m.expand(s))

        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class WeightAbsmaxPerInchannelQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, input_parallel=False):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        dtype = input.dtype
        input = input.float()
        if num_bits == 1:
            Qn = -1
            Qp = 1
        else:
            Qn = -2 ** (num_bits - 1)
            Qp = 2 ** (num_bits - 1) - 1
        # n = input[0].nelement()  # W of size out_features x in_features, return a vector of  out_features x 1
        # s = input.size()
        # m = input.norm(1, 1, keepdim=True).div(n)
        # scale = Qp / m.abs().max().clamp_(min=1e-5)
        # m = (m * scale).round().clamp(Qn, Qp) / scale

        n = input[1].nelement()  # W of size out_features x in_features, return a vector of  1 x in_features
        s = input.size()
        m = input.norm(dim=0, p=1, keepdim=True).div(n)
        scale = Qp / m.abs().max().clamp(min=1e-5)
        m = (m * scale).round().clamp(Qn, Qp) / scale
        
        e = input.mean()
        result = (input-e).sign().mul(m.expand(s))

        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class BwnPerTokenClipSignedQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        dtype = input.dtype
        input = input.float()

        alpha = input.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, alpha)
        result = input.sign() * alpha

        return result.type(dtype)

    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        Qn, Qp = -1, 1
        input, alpha = ctx.saved_tensors
        q_w = input / alpha

        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)

        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None
    

class BwnPerTokenClipUnsignedQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        dtype = input.dtype
        input = input.float()

        Qn, Qp = 0, 1
        alpha = input.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        ctx.save_for_backward(input, alpha)
        result = (input / alpha).round().clamp(Qn, Qp) * alpha

        return result.type(dtype)

    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        Qn, Qp = 0, 1
        input, alpha = ctx.saved_tensors
        q_w = input / alpha

        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)

        grad_input = indicate_middle * grad_output
        return grad_input, None, None, None


class BwnPerTokenQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    @torch.compile
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        dtype = input.dtype
        input = input.float()
        s = input.size()
        n = input.size(-1)
        m = input.norm(dim=-1, p=1, keepdim=True).div(n)
        e = input.mean(dim=-1, keepdim=True)
        result = (input-e).sign().mul(m.expand(s))

        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class BwnGradQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, input_parallel=False):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        dtype = input.dtype
        input = input.float()
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            if input_parallel:
                dist.all_reduce(m, op=ReduceOp.SUM, group=get_model_parallel_group())
                dist.all_reduce(e, op=ReduceOp.SUM, group=get_model_parallel_group())
                m /= get_model_parallel_world_size()
                e /= get_model_parallel_world_size()
            result = (input-e).sign().mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            if input_parallel:
                dist.all_reduce(m, op=ReduceOp.SUM, group=get_model_parallel_group())
                dist.all_reduce(e, op=ReduceOp.SUM, group=get_model_parallel_group())
                m /= get_model_parallel_world_size()
                e /= get_model_parallel_world_size()
            result = (input-e).sign().mul(m.expand(s))

        return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        input = ctx.saved_tensors[0]
        grad_input *= (1 + torch.tanh(input)**2)
        return grad_input, None, None, None, None
    

class BwnBlockQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, blocksize=(-1, -1), use_blockscale='none'):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        dtype = input.dtype
        h, w = input.shape[0], input.shape[1]
        bh, bw = blocksize[0], blocksize[1]

        if use_blockscale == 'init' or use_blockscale == 'random':
            result = input.sign()
        else:
            input = input.float()
            input = input.unfold(0, bh, bh).unfold(1, bw, bw)

            m = input.norm(p=1, dim=(-2, -1)) / bh / bw
            m = m.unsqueeze(-1).unsqueeze(-1).expand_as(input)

            result = m * input.sign()
            result = result.permute(0, 2, 1, 3).reshape(h, w)
        
        if use_blockscale == 'random4':
            result *= math.sqrt(w)

        if use_blockscale == 'random2':
            # result *= math.sqrt(w) 
            result *= math.pow(w, 0.25)

        return result.type(dtype)
        # """
        # :param input: tensor to be binarized
        # :return: quantized tensor
        # """
        # ctx.save_for_backward(input)
        # dtype = input.dtype
        # input = input.float()
        # if layerwise:
        #     s = input.size()
        #     m = input.norm(p=1).div(input.nelement())
        #     e = input.mean()
        #     result = (input-e).sign().mul(m.expand(s))
        # else:
        #     n = input[0].nelement()  # W of size axb, return a vector of  ax1
        #     s = input.size()
        #     m = input.norm(1, 1, keepdim=True).div(n)
        #     e = input.mean()
        #     result = (input-e).sign().mul(m.expand(s))

        # return result.type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None


class BwnQuantizerWithScale(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        dtype = input.dtype
        input_fp16 = input
        input = input.float()
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            e = input.mean()
            result = (input-e).sign().mul(m.expand(s))

        result = result.type(dtype)
        ctx.save_for_backward(input_fp16, result)
        ctx.grad_scale = 1.0 / math.sqrt(input.numel())
        return result * alpha

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        input, result = ctx.saved_tensors
        grad_alpha = (grad_output * result).sum() * ctx.grad_scale
        return grad_input, grad_alpha, None, None


class BwnQuantizerInputAware(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, weight, clip_val, num_bits, layerwise, input):
        """
        :param input: tensor to be binarized
        :return: quantized tensor
        """
        e = weight.mean()
        wb = (weight - e).sign()
        out_dim = weight.shape[0]
        t1 = nn.functional.linear(input, weight).view(-1, out_dim).float()
        t2 = nn.functional.linear(input, wb).view(-1, out_dim).float()
        m = ((t1 * t2).mean(dim=0)) / (t2.square().mean(dim=0))
        return (m.unsqueeze(1) * wb).type(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class HadamardMultiplier(nn.Module):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, group, dim, learnable=False, reverse=False):
        super(HadamardMultiplier, self).__init__()
        self.group = group
        self.dim = dim
        H_group = self.constructH(group)
        H_group = H_group.t() if reverse else H_group
        self.H = nn.Parameter(H_group.repeat(dim // group, 1, 1), requires_grad=learnable)

    def constructH(self, group):
        H = torch.ones(1, 1).cuda()

        for i in range(int(math.log2(group))):
            H = torch.cat((
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1)), dim=0
            ) / math.sqrt(2)
        assert H.shape[0] == group
        return H

    def forward(self, x):
        x_shape2 = x.shape
        x = x.reshape(-1, x.shape[-1])

        x = x.reshape(-1, self.dim // self.group, self.group).transpose(0, 1)
        x = torch.bmm(x, self.H).transpose(0, 1)
        # H = torch.block_diag(*self.H)
        # x = torch.mm(x, H)

        x = x.reshape(x_shape2)
        return x


def act_quant_fn(
        input, 
        clip_val, 
        num_bits, 
        symmetric, 
        quant_method, 
        layerwise, 
        input_parallel=False, 
        grad_act=False,
        scale=1.0,
        sparse_blocksize=16,
        sparse_ratio=0.4,
        sparse_alpha=1.0,
    ):
    if num_bits == 32:
        return input
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmean_per_token_unsigned_clipped":
        quant_fn = AbsmeanPerTokenClipQuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "twn_per_token_unsigned_clipped" and num_bits == 1:
        quant_fn = TwnPerTokenClipQuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "elastic" and num_bits >= 1 and symmetric:
        quant_fn = ElasticQuantBinarizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "elastic" and num_bits >= 1 and not symmetric:
        quant_fn = ElasticQuantBinarizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax" and num_bits >= 1 and symmetric:
        quant_fn = AbsmaxQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax" and num_bits >= 1 and not symmetric:
        quant_fn = AbsmaxQuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_sparse" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_blocksize)
    elif quant_method == "absmax_per_token_sparse" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_blocksize)
    elif quant_method == "absmax_per_token_sparse_clipped" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerClipSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_blocksize)
    elif quant_method == "absmax_per_token_sparse_clipped" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerClipSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_blocksize)
    elif quant_method == "absmax_per_token_sparse_topk" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerTopKSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_ratio)
    elif quant_method == "absmax_per_token_sparse_topk" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerTopKSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_ratio)
    elif quant_method == "absmax_per_token_blocksparse_topk" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerTopKBlockSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_blocksize, sparse_ratio)
    elif quant_method == "absmax_per_token_blocksparse_topk" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerTopKBlockSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_blocksize, sparse_ratio)
    elif quant_method == "absmax_per_token_blocksparse_maxtopk" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerMaxTopKBlockSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_blocksize, sparse_ratio)
    elif quant_method == "absmax_per_token_blocksparse_maxtopk" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerMaxTopKBlockSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_blocksize, sparse_ratio)
    elif quant_method == "absmax_per_token_sparse_topk_clip" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerTopKClipSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_ratio)
    elif quant_method == "absmax_per_token_sparse_topk_clip" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerTopKClipSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_ratio)
    elif quant_method == "absmax_per_token_sparse_mean" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerMeanSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_alpha)
    elif quant_method == "absmax_per_token_sparse_mean" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerMeanSparseSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, sparse_alpha)
    elif quant_method == "absmax_per_token_sparse_me" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_sparse_me" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_relu" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_relu" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_relu2" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_relu2" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_sp" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_sp" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token" and num_bits == 8 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token" and num_bits == 8 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token" and num_bits == 4 and symmetric:
        quant_fn = AbsmaxPerTokenINT4QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token" and num_bits == 4 and not symmetric:
        quant_fn = AbsmaxPerTokenINT4QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_bos8" and num_bits == 4 and symmetric:
        quant_fn = AbsmaxPerTokenINT4bos8QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_bos8" and num_bits == 4 and not symmetric:
        quant_fn = AbsmaxPerTokenINT4bos8QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_clipped" and num_bits == 4 and symmetric:
        quant_fn = AbsmaxPerTokenClipINT4QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_clipped" and num_bits == 4 and not symmetric:
        quant_fn = AbsmaxPerTokenClipINT4QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "attn_absmax_per_token" and num_bits == 8 and symmetric:
        quant_fn = AttnAbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "attn_absmax_per_token" and num_bits == 8 and not symmetric:
        quant_fn = AttnAbsmaxPerTokenQuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "attn_absmax_per_token" and num_bits == 4 and symmetric:
        quant_fn = AttnAbsmaxPerTokenINT4QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "attn_absmax_per_token" and num_bits == 4 and not symmetric:
        quant_fn = AttnAbsmaxPerTokenINT4QuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "attn_absmax_per_token" and num_bits == 3 and symmetric:
        quant_fn = AttnAbsmaxPerTokenINT4QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "attn_absmax_per_token" and num_bits == 3 and not symmetric:
        quant_fn = AttnAbsmaxPerTokenINT4QuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 4 and symmetric:
        quant_fn = AbsmeanPerTokenClipQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 4 and not symmetric:
        quant_fn = AbsmeanPerTokenClipQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped_bos8" and num_bits == 2 and symmetric:
        quant_fn = AbsmeanPerTokenClipbos8QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped_bos8" and num_bits == 2 and not symmetric:
        quant_fn = AbsmeanPerTokenClipbos8QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 2 and scale == 1.0 and symmetric:
        quant_fn = AbsmeanPerTokenClipINT2QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 2 and scale == 1.0 and not symmetric:
        quant_fn = AbsmeanPerTokenClipINT2QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 2 and scale == 2.0 and symmetric:
        quant_fn = AbsmeanPerTokenClipINT2Scale2QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 2 and scale == 2.0 and not symmetric:
        quant_fn = AbsmeanPerTokenClipINT2Scale2QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 2 and scale == 0.5 and symmetric:
        quant_fn = AbsmeanPerTokenClipINT2Scale1_2QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 2 and scale == 0.5 and not symmetric:
        quant_fn = AbsmeanPerTokenClipINT2Scale1_2QuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 1 and symmetric:
        quant_fn = AbsmeanPerTokenClipBWNQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token_clipped" and num_bits == 1 and not symmetric:
        quant_fn = AbsmeanPerTokenClipBWNQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token" and num_bits >= 1 and symmetric:
        quant_fn = AbsmeanPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmean_per_token" and num_bits >= 1 and not symmetric:
        quant_fn = AbsmeanPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise, scale)
    elif quant_method == "absmax_per_token_mixsigned" and num_bits >= 1 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerSigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_mixsigned" and num_bits >= 1 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_unsigned" and num_bits >= 1 and symmetric:
        quant_fn = AbsmaxPerTokenQuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "absmax_per_token_unsigned" and num_bits >= 1 and not symmetric:
        quant_fn = AbsmaxPerTokenQuantizerUnsigned
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "bwn_per_token" and num_bits == 1:
        quant_fn = BwnPerTokenQuantizer
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "bwn_per_token_clipped" and num_bits == 1:
        quant_fn = BwnPerTokenClipSignedQuantizer
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "bwn_per_token_clipped_mixsigned" and num_bits >= 1 and symmetric:
        quant_fn = BwnPerTokenClipSignedQuantizer
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    elif quant_method == "bwn_per_token_clipped_mixsigned" and num_bits >= 1 and not symmetric:
        quant_fn = BwnPerTokenClipUnsignedQuantizer
        input = quant_fn.apply(input, clip_val, num_bits, layerwise)
    else:
        raise ValueError("Unknownquant_method")

    # if input_parallel:
    #     input = quant_fn.apply(input, clip_val, num_bits, layerwise, input_parallel)
    # else:
    #     input = quant_fn.apply(input, clip_val, num_bits, layerwise)

    return input


def weight_quant_fn(
        weight,  
        clip_val,  
        num_bits,  
        symmetric, 
        quant_method, 
        layerwise, 
        input_parallel=False, 
        weight_blocksize=(-1, -1),
        use_blockscale='none',
        absmean_alpha=1.0,
    ):
    if num_bits == 32:
        return weight
    elif quant_method == "bwn" and num_bits == 1 and (weight_blocksize[0] == -1) and (weight_blocksize[1] == -1):
        quant_fn = BwnQuantizer
        weight = quant_fn.apply(weight, clip_val, num_bits, layerwise, input_parallel)
    elif quant_method == "bwn01" and num_bits == 1 and (weight_blocksize[0] == -1) and (weight_blocksize[1] == -1):
        quant_fn = Bwn01Quantizer
        weight = quant_fn.apply(weight, clip_val, num_bits, layerwise, input_parallel)
    # elif quant_method == "bwn_per_token" and num_bits == 1 and (weight_blocksize[0] == -1) and (weight_blocksize[1] == -1):
    #     quant_fn = BwnPerTokenQuantizer
    #     weight = quant_fn.apply(weight, clip_val, num_bits, layerwise)
    elif quant_method == "random_bwn" and num_bits == 1 and (weight_blocksize[0] == -1) and (weight_blocksize[1] == -1):
        quant_fn = BwnRandomQuantizer
        weight = quant_fn.apply(weight, clip_val, num_bits, layerwise, input_parallel)
    elif quant_method == "bwn" and num_bits == 1 and (weight_blocksize[0] != -1) and (weight_blocksize[1] != -1):
        quant_fn = BwnBlockQuantizer
        weight = quant_fn.apply(weight, clip_val, num_bits, layerwise, weight_blocksize, use_blockscale)
    elif quant_method == "absmax":
        quant_fn = AbsmaxQuantizerSigned
        weight = quant_fn.apply(weight, clip_val, num_bits, layerwise, input_parallel)
    elif quant_method == "absmean":
        quant_fn = AbsmeanQuantizerSigned
        weight = quant_fn.apply(weight, clip_val, num_bits, layerwise, absmean_alpha)
    elif quant_method == "twn":
        quant_fn = TwnQuantizer
        weight = quant_fn.apply(weight, clip_val, num_bits, layerwise)
    # elif quant_method == "absmean_clip":
    #     quant_fn = AbsmeanClipQuantizerSigned
    #     weight = quant_fn.apply(weight, clip_val, num_bits, layerwise)
    # elif quant_method == "weight_absmax":
    #     quant_fn = WeightAbsmaxQuantizer
    #     weight = quant_fn.apply(weight, clip_val, num_bits, layerwise, input_parallel)
    else:
        raise ValueError("Unknown quant_method")

    
    return weight


class BinaryLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        e = weight.mean()
        wb = (weight - e).sign()
        shape = input.shape
        input = input.view(-1, shape[-1])
        t1 = torch.mm(input, weight.t()).float()
        t2 = torch.mm(input, wb.t()).float()
        m = ((t1 * t2).mean(dim=0)) / (t2.square().mean(dim=0))
        weight_ = (m.unsqueeze(1) * wb).type(weight.dtype)
        return torch.mm(input, weight_.t()).view(*shape[:-1], -1).clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        out_dim = weight.shape[1]
        e = weight.mean()
        wb = (weight - e).sign()
        shape = grad_output.shape
        grad_output = grad_output.view(-1, shape[-1])
        t1 = torch.mm(grad_output, weight).view(-1, out_dim).float()
        t2 = torch.mm(grad_output, wb).view(-1, out_dim).float()
        m = ((t1 * t2).mean(dim=0)) / (t2.square().mean(dim=0))
        weight_ = (m.unsqueeze(0) * wb).type(weight.dtype)
        grad_input = torch.mm(grad_output, weight_).view(*shape[:-1], -1).clone()
        grad_weight = torch.mm(grad_output.t(), input.view(-1, input.shape[-1]))
        return grad_input, grad_weight


class QuantizeLinear(nn.Linear):

    def __init__(self,
            *kargs,
            clip_val=2.5,
            weight_bits=8,
            input_bits=8,
            learnable=False,
            symmetric=True,
            weight_layerwise=True,
            input_layerwise=True,
            weight_quant_method="bwn",
            input_quant_method="uniform",
            learnable_step=False,
            hadamard_group=32,
            blockwise_quant=False,
            weight_blocksize=(-1, -1),
            grad_act=False,
            weight_blockscale='none',
            smoothquant=False,
            smoothquant_alpha=0.5,
            absmean_alpha=1.0,
            input_absmean_alpha=1.0,
            sparse_blocksize=16,
            sparse_ratio=0.4,
            sparse_alpha=1.0,
            **kwargs
        ):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self._build_weight_clip_val(weight_quant_method, learnable, init_val=clip_val)
        self._build_input_clip_val(input_quant_method, learnable, init_val=clip_val)
        self.blockwise_quant = blockwise_quant
        self.weight_blocksize = weight_blocksize
        self.grad_act = grad_act
        self.weight_blockscale_init = weight_blockscale
        self.absmean_alpha = absmean_alpha
        self.input_absmean_alpha = input_absmean_alpha
        self.smoothquant = smoothquant
        self.smoothquant_alpha = smoothquant_alpha
        self.weight_blockscale = None
        self.sparse_blocksize = sparse_blocksize
        self.sparse_ratio = sparse_ratio
        self.sparse_alpha = sparse_alpha
        # if weight_blockscale == 'init':
        #     bh, bw = self.weight_blocksize[0], self.weight_blocksize[1]
        #     blockwise_norm = self.weight.unfold(0, bh, bh).unfold(1, bw, bw)
        #     blockwise_norm = blockwise_norm.norm(p=1, dim=(-2, -1)) / (bh * bw)
        #     blockwise_norm = blockwise_norm.unsqueeze(-1).unsqueeze(-1)
        #     self.weight_blockscale = nn.Parameter(blockwise_norm, requires_grad=True).to(self.weight.device)
        # elif weight_blockscale == 'random':
        #     self.weight_blockscale = nn.Parameter(
        #         torch.ones(self.out_features // weight_blocksize[0], self.in_features // weight_blocksize[1], 1, 1),
        #         requires_grad=True,
        #     ).to(self.weight.device)
        #     nn.init.kaiming_uniform_(self.weight_blockscale, a=math.sqrt(5))
        #     self.weight_blockscale.data /= math.sqrt(weight_blocksize[1])
        # elif weight_blockscale == 'random2':
        #     self.weight_blockscale = nn.Parameter(
        #         torch.ones(self.out_features // weight_blocksize[0], self.in_features // weight_blocksize[1], 1, 1),
        #         requires_grad=True,
        #     ).to(self.weight.device)
        #     # self.weight_blockscale.data /= math.sqrt(self.in_features)
        #     self.weight_blockscale.data /= math.pow(self.in_features, 0.25)
        # elif weight_blockscale == 'random3':
        #     self.weight_blockscale = nn.Parameter(
        #         torch.ones(self.out_features // weight_blocksize[0], self.in_features // weight_blocksize[1], 1, 1),
        #         requires_grad=True,
        #     ).to(self.weight.device)
        # elif weight_blockscale == 'random4':
        #     self.weight_blockscale = nn.Parameter(
        #         torch.ones(self.out_features // weight_blocksize[0], self.in_features // weight_blocksize[1], 1, 1),
        #         requires_grad=True,
        #     ).to(self.weight.device)
        #     nn.init.kaiming_uniform_(self.weight_blockscale, a=math.sqrt(5))
        #     self.weight_blockscale.data /= math.sqrt(weight_blocksize[1])
        # elif weight_blockscale == 'sort':
        #     def reorder_weights(weights):
        #         sorted_rows = [torch.sort(torch.abs(row), descending=True)[1] for row in weights]
        #         return torch.stack([row[indices] for row, indices in zip(weights, sorted_rows)])
        #     self.weight.data = reorder_weights(self.weight.data)
        #     self.weight_blockscale = None
        # elif weight_blockscale == 'sort2':
        #     def reorder_weights(weights):
        #         sorted_rows = [torch.sort(torch.abs(row), descending=True)[1] for row in weights]
        #         return torch.stack([row[indices] for row, indices in zip(weights, sorted_rows)])
        #     self.weight.data = reorder_weights(self.weight.data)
        #     self.weight_blockscale = nn.Parameter(
        #         torch.ones(self.out_features // weight_blocksize[0], self.in_features // weight_blocksize[1], 1, 1),
        #         requires_grad=True,
        #     ).to(self.weight.device)
        # elif weight_blockscale == 'hadmard':
        #     assert weight_blocksize[1] == weight_blocksize[0]
        #     self.H = HadamardMultiplier(group=weight_blocksize[0], dim=self.out_features, learnable=False)
        #     self.Ht = HadamardMultiplier(group=weight_blocksize[0], dim=self.out_features, learnable=False, reverse=True)
        #     self.weight_blockscale = None
        # else:
        #     self.weight_blockscale = None

    def _build_weight_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        elif quant_method == 'elastic':
            assert learnable, 'Elastic method must use leranable step size!'
            self.weight_clip_val = AlphaInit(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        elif quant_method == 'bwn_with_scale':
            self.weight_clip_val = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('weight_clip_val', None)

    def _build_input_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('input_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.input_clip_val = nn.Parameter(self.input_clip_val)
        elif quant_method == 'elastic' or quant_method == 'bwn':
            assert learnable, 'Elastic method must use leranable step size!'
            self.input_clip_val = AlphaInit(torch.tensor(1.0))  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('input_clip_val', None)

    def forward(self, input, vis_input=False, vis_output=False):
        # quantize input
        # if self.smoothquant:
        #     hidden_dim = input.shape[-1]
        #     activation_scale = torch.max(input.reshape(-1, hidden_dim).abs().detach(), dim=0).values
        #     activation_scale = activation_scale.pow(self.smoothquant_alpha).clamp(min=1e-5)
        #     weight = weight * activation_scale
        #     input = input / activation_scale

        # if vis_input:
        #     #1-4 fc2 8bit; 5-10, fc2 4bit, absmax; 11- fc2 4bit absmean; #
        #     print(input.abs().max() / input.abs().mean())
        #     max_num = get_max_num("/home/shumma/torchscale-examples/bitnet15", file_pattern=r'fc2_activation_(\d+)\.pdf')
        #     visualize_weight_distribution_from_tensor(input.cpu(), pic_path=f"/home/shumma/torchscale-examples/bitnet15/fc2_activation_{max_num + 1}.pdf")
        # if self.input_quant_method == "absmax_per_token_relu" and self.symmetric:
        #     input = nn.functional.relu(input)
        if self.input_quant_method == "absmax_per_token_sp" and self.symmetric:
            if self.input_bits == 1:
                Qp = 1
            else:
                Qp = 2 ** (self.input_bits - 1) - 1
            s = input.abs().max(dim=-1, keepdim=True).values / Qp * self.sparse_blocksize
            mask = input.abs() > s
            input *= mask
        elif self.input_quant_method == "absmax_per_token_relu2" and self.symmetric:
            input = nn.functional.relu(input)
            input = input * input
        elif self.input_quant_method == "absmax_per_token_relu" and self.symmetric:
            input = nn.functional.relu(input)
        elif self.input_quant_method == "absmax_per_token_sparse_me" and self.symmetric:
            threhold = input.abs().mean(dim=-1, keepdim=True) * self.sparse_alpha
            mask = input.abs() > threhold
            input *= mask

        weight = self.weight
        input = act_quant_fn(
            input, 
            self.input_clip_val,
            num_bits=self.input_bits,
            symmetric=self.symmetric,
            quant_method=self.input_quant_method,
            layerwise=self.input_layerwise,
            input_parallel=False,
            grad_act=self.grad_act,
            scale=self.input_absmean_alpha,
            sparse_blocksize=self.sparse_blocksize,
            sparse_ratio=self.sparse_ratio,
            sparse_alpha=self.sparse_alpha,
        )
        weight = weight_quant_fn(
            weight, 
            self.weight_clip_val,
            num_bits=self.weight_bits,
            symmetric=self.symmetric,
            quant_method=self.weight_quant_method,
            layerwise=self.weight_layerwise,
            input_parallel=False,
            weight_blocksize=self.weight_blocksize,
            use_blockscale=self.weight_blockscale_init,
            absmean_alpha=self.absmean_alpha,
        )
        # if vis_output:
        #     max_num = get_max_num("/home/shumma/torchscale-examples/bitnet15", file_pattern=r'fc2_activation_output_(\d+)\.pdf')
        #     visualize_weight_distribution_from_tensor(input.cpu(), pic_path=f"/home/shumma/torchscale-examples/bitnet15/fc2_activation_output_{max_num + 1}.pdf")
        # if self.weight_blockscale_init == "hadmard":
        #     weight = weight_quant_fn(
        #         self.H(self.weight.t()), 
        #         self.weight_clip_val,
        #         num_bits=self.weight_bits,
        #         symmetric=self.symmetric,
        #         quant_method=self.weight_quant_method,
        #         layerwise=self.weight_layerwise,
        #         input_parallel=False,
        #         weight_blocksize=self.weight_blocksize,
        #         use_blockscale=self.weight_blockscale_init,
        #     )
        #     out = input @ weight
        #     out = self.Ht(out)
        # else:
        #     weight = weight_quant_fn(
        #         weight, 
        #         self.weight_clip_val,
        #         num_bits=self.weight_bits,
        #         symmetric=self.symmetric,
        #         quant_method=self.weight_quant_method,
        #         layerwise=self.weight_layerwise,
        #         input_parallel=False,
        #         weight_blocksize=self.weight_blocksize,
        #         use_blockscale=self.weight_blockscale_init,
        #         absmean_alpha=self.absmean_alpha,
        #     )

        #     if self.weight_blockscale is not None:
        #         h, w = weight.shape[0], weight.shape[1]
        #         bh, bw = self.weight_blocksize[0], self.weight_blocksize[1]
        #         weight = weight.unfold(0, bh, bh).unfold(1, bw, bw)
        #         if self.weight_blockscale_init == 'init' or self.weight_blockscale_init == 'random' \
        #             or self.weight_blockscale_init == 'random3' or self.weight_blockscale_init == 'sort2' \
        #             or self.weight_blockscale_init == 'random2' or self.weight_blockscale_init == 'random4':
        #             weight = weight * self.weight_blockscale.expand_as(weight)
        #         else:
        #             raise NotImplementedError
        #         weight = weight.permute(0, 2, 1, 3).reshape(h, w)

        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, learnable=False, symmetric=True,
                 embed_layerwise=False, weight_quant_method="twn", **kwargs):
        super(QuantizeEmbedding, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.embed_layerwise = embed_layerwise
        self.weight_quant_method = weight_quant_method
        self._build_embed_clip_val(weight_quant_method, learnable, init_val=clip_val)

    def _build_embed_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('embed_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.embed_clip_val = nn.Parameter(self.embed_clip_val)
        elif quant_method == 'elastic':
            assert learnable, 'Elastic method must use leranable step size!'
            self.embed_clip_val = AlphaInit(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('embed_clip_val', None)

    def forward(self, input):
        weight = weight_quant_fn(
            self.weight, self.embed_clip_val,
            num_bits=self.weight_bits,
            symmetric=self.symmetric,
            quant_method=self.weight_quant_method,
            layerwise=self.embed_layerwise,
            input_parallel=False,
        )

        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

        return out
