# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce, pack, unpack

from .tutel_moe_layer import fused_cumsum_sub_one, has_tutel
# from tutel import moe as tutel_moe
# from tutel_moe.jit_kernels.gating import fast_cumsum_sub_one
# import tutel_moe.fast_cumsum_sub_one as fast_cumsum_sub_one
import torch
from torch.distributions.normal import Normal
# use a fixed temperature to compute balance loss
TEMPERATURE_FOR_L_UAX = 0.07

# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 0.25

# logging
SAMPLE_FRACTION = 0.2


def create_combinations(num_experts):
        """
        This function creates a tensor with all possible combinations of 0s and 1s for the given number of experts.
        Each row in the tensor represents a unique combination of experts' presence (1) or absence (0).
        """
        # Generate all possible combinations from 0 to 2^num_experts - 1
        binary_combinations = [bin(i)[2:].zfill(int(math.log2(num_experts))) for i in range(num_experts)]
        # Convert to tensor
        combinations_tensor = torch.tensor([[int(bit) for bit in combination] for combination in binary_combinations], dtype=torch.float16)
        return combinations_tensor


def compute_combinations(tensor, combination_indices):
        """
        This function computes the result of applying all combinations of experts to the tensor.
        """
        seq, _ = tensor.shape
        num_experts = combination_indices.shape[0]

        # Calculate probabilities and their complements
        tensor_expanded = tensor.unsqueeze(1).expand(-1, num_experts, -1)
        complement_expanded = (1 - tensor).unsqueeze(1).expand(-1, num_experts, -1)

        # Use the combination indices to select between the tensor and its complement
        combination_selector = combination_indices.unsqueeze(0).expand(seq, -1, -1)
        combined_tensor = torch.where(combination_selector == 1, tensor_expanded, complement_expanded)

        # Multiply across the experts' dimension to get the final combination values
        result_tensor = combined_tensor.prod(dim=2)
        return result_tensor


def compute_sorted_location(x, importance_scores):
    sorted_x = x[importance_scores.argsort(dim=0)]
    sorted_cumsum = fused_cumsum_sub_one(sorted_x) * sorted_x
    return sorted_cumsum[importance_scores.argsort(dim=0).argsort(dim=0)]

gumbel_map: Dict[torch.device, Callable] = {}


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def one_hot(indices: torch.Tensor, num_classes: int, unsqueeze_indices=False) -> Tensor:
    if unsqueeze_indices:
        indices = indices.unsqueeze(-1)
    assert indices.shape[-1] == 1, "last dimension of indices must be have size 1"
    output = torch.zeros(
        indices.shape[:-1] + (num_classes,), device=indices.device, dtype=indices.dtype
    )
    output.scatter_(len(output.shape) - 1, indices, 1)
    return output


def entropy(probs):
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)

def _one_hot_with_dtype(data, num_classes, dtype, hot_value=1):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), hot_value)
    return result


def gshard_loss(scores_w_noise, top_ids):
    num_samples, num_global_experts = int(scores_w_noise.size(0)), int(scores_w_noise.size(1))
    mask = _one_hot_with_dtype(top_ids[:, 0], num_global_experts, dtype=scores_w_noise.dtype,
        hot_value=num_global_experts / num_samples)
    me = torch.sum(scores_w_noise, dim=0)
    ce = torch.sum(mask, dim=0)
    l_aux = torch.sum(me * ce) / num_samples
    return l_aux


def topkgating(
    top_k,
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    second_expert_policy="sampling",
    normalize_gate_prob_before_dropping=False,
    eval_mode=False,
    moe_eval_capacity_token_fraction=0.25,
    batch_prioritized_routing=False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()
    gates = F.softmax(logits, dim=1)
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
    #     capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    # else:
    #     # capacity = 2S/E
    #     capacity = 2 * math.ceil(num_tokens / num_experts)

    # # Create a mask for 1st's expert per token
    # indices1_s = torch.argmax(gates, dim=1, keepdim=True)
    # mask1 = one_hot(indices1_s, num_experts)
    assert top_k <= num_experts, print("top-k should be smaller or equal to expert number.")
    topk_indices = torch.topk(gates, top_k, dim=1).indices
    indices_s = [x.view(-1) for x in topk_indices.chunk(top_k, dim=1)]

    masks_se = [_one_hot_with_dtype(x, num_classes=num_experts, dtype=x.dtype) for x in indices_s]
    gates_s = [(gates * x).sum(dim=1) for x in masks_se]


    # if second_expert_policy == "sampling":
    #     # Create a mask for 2nd's expert per token using Gumbel-max trick
    #     # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    #     logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # else:
    #     logits_w_noise = logits
    # # Replace top-expert with min value
    # logits_except1 = logits_w_noise.masked_fill(mask1.bool(), float("-inf"))
    # indices2_s = torch.argmax(logits_except1, dim=1, keepdim=True)
    # mask2 = one_hot(indices2_s, num_experts)
    # gates1_s = (gates * mask1).sum(dim=1)
    # gates2_s = (gates * mask2).sum(dim=1)

    l_aux = gshard_loss(gates, topk_indices)

    if batch_prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        compute_location = lambda x: compute_sorted_location(x, importance_scores)
    else:
        compute_location = fused_cumsum_sub_one

    locations1 = compute_location(masks_se[0])

    locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]

    if top_k > 1:
        acc_base = None
        for k in range(1, top_k):
            acc_base = torch.sum(masks_se[k - 1], dim=0, keepdim=True) if acc_base is None else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True)
            locations2 = compute_location(masks_se[k])
            locations2 += acc_base
            locations_s.append(torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32))

        if normalize_gate_prob_before_dropping:
            denom_s = torch.clamp(sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps)
            gates_s = [x / denom_s for x in gates_s]
    else:
        locations2 = locations1
    locations2 = locations2[-1] + 1
    indices_s = [x.to(torch.int32) for x in indices_s]


    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = 2S/E
        capacity = top_k * math.ceil(num_tokens / num_experts)
    
    return l_aux, metadata, capacity, num_experts, indices_s, locations_s, gates_s

    # if normalize_gate_prob_before_dropping:
    #     # Normalize gate probabilities
    #     denom_s = gates1_s + gates2_s
    #     # Avoid divide-by-zero
    #     denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    #     gates1_s = gates1_s / denom_s
    #     gates2_s = gates2_s / denom_s

    # if second_expert_policy == "random":
    #     sampled = (2 * gates2_s) > torch.rand_like(gates2_s)
    #     mask2 = mask2 * sampled.repeat(num_experts, 1).transpose(1, 0)

    # # Compute locations in capacity buffer
    # if input_mask is not None and input_mask.any():
    #     nonpadding = ~input_mask
    #     mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)
    #     mask2 = mask2 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    # if batch_prioritized_routing:
    #     # if batch_prioritized_routing:
    #     importance_scores = -1 * gates.max(dim=1)[0]
    #     sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
    #     sorted_cumsum1 = fused_cumsum_sub_one(sorted_mask1) * sorted_mask1
    #     importance_sorted_locations1 = sorted_cumsum1[
    #         importance_scores.argsort(dim=0).argsort(dim=0)
    #     ]

    #     sorted_mask2 = mask2[importance_scores.argsort(dim=0)]
    #     sorted_cumsum2 = fused_cumsum_sub_one(sorted_mask2) * sorted_mask2
    #     importance_sorted_locations2 = sorted_cumsum2[
    #         importance_scores.argsort(dim=0).argsort(dim=0)
    #     ]

    #     importance_sorted_locations2 += torch.sum(mask1, dim=0, keepdim=True)

    #     locations1, locations2 = (
    #         importance_sorted_locations1,
    #         importance_sorted_locations2,
    #     )
    # else:
    #     locations1 = fused_cumsum_sub_one(mask1)
    #     locations2 = fused_cumsum_sub_one(mask2)
    #     # Update 2nd's location by accounting for locations of 1st
    #     locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # # Compute l_aux
    # me = torch.mean(gates, dim=0)
    # ce = torch.mean(mask1.to(gates.dtype), dim=0)
    # l_aux = torch.mean(me * ce)
    # l_aux = l_aux * num_experts * num_experts

    # # for logging purposes
    # metadata["overflow_expert1"] = (
    #     100 * torch.sum(mask1 * torch.ge(locations1, capacity)) / torch.sum(mask1)
    # )
    # metadata["overflow_expert2"] = (
    #     100 * torch.sum(mask2 * torch.ge(locations2, capacity)) / torch.sum(mask2)
    # )

    # # Remove locations outside capacity from mask
    # mask1_, mask2_ = mask1, mask2
    # mask1 = mask1 * torch.lt(locations1, capacity)
    # mask2 = mask2 * torch.lt(locations2, capacity)

    # # for logging (percent of tokens routed to each expert)
    # expert1_hist = (
    #     100
    #     * torch.histc(
    #         (indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
    #     )
    #     / num_tokens
    # )
    # metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    # expert1_hist = (
    #     torch.sort(expert1_hist, dim=0, descending=True).values
    #     + torch.finfo(torch.float32).tiny
    # )

    # expert2_hist = (
    #     100
    #     * torch.histc(
    #         (indices2_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts
    #     )
    #     / num_tokens
    # )
    # metadata["unused_expert2_count"] = (expert2_hist == 0).sum()
    # expert2_hist = (
    #     torch.sort(expert2_hist, dim=0, descending=True).values
    #     + torch.finfo(torch.float32).tiny
    # )

    # sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    # metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    # metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

    # metadata["expert2_balance_top"] = expert2_hist[:sample_count].sum()
    # metadata["expert2_balance_bottom"] = expert2_hist[-sample_count:].sum()

    # if not normalize_gate_prob_before_dropping:
    #     # Normalize gate probabilities
    #     gates1_s = (gates * mask1).sum(dim=1)
    #     gates2_s = (gates * mask2).sum(dim=1)
    #     denom_s = gates1_s + gates2_s
    #     # Avoid divide-by-zero
    #     denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    #     gates1_s /= denom_s
    #     gates2_s /= denom_s

    # locations1_s = torch.sum(locations1 * mask1_, dim=1)
    # locations2_s = torch.sum(locations2 * mask2_, dim=1)

    # # Store the capacity location for each token
    # locations1_s = torch.sum(locations1 * mask1, dim=1)
    # locations2_s = torch.sum(locations2 * mask2, dim=1)

    # # Calculate combine_weights and dispatch_mask
    # gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    # gates2 = gates2_s.unsqueeze(-1) * mask2.to(gates2_s.dtype)  # einsum("s,se->se")
    # locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    # locations2_sc = one_hot(locations2_s, num_classes=capacity, unsqueeze_indices=True)
    # combine1_sec = torch.bmm(
    #     # einsum("se,sc->sec")
    #     gates1.unsqueeze(-1),
    #     locations1_sc.to(gates1.dtype).unsqueeze(1),
    # )
    # combine2_sec = torch.bmm(
    #     # einsum("se,sc->sec")
    #     gates2.unsqueeze(-1),
    #     locations2_sc.to(gates2.dtype).unsqueeze(1),
    # )
    # combine_weights = combine1_sec + combine2_sec
    # dispatch_mask = combine_weights.bool()
    # if use_fp32:
    #     return l_aux, combine_weights.to(orig_dtype), dispatch_mask, metadata
    # else:
    #     return l_aux, combine_weights, dispatch_mask, metadata


class TopkGate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        args,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        second_expert_policy="sampling",
        normalize_gate_prob_before_dropping=False,
        moe_eval_capacity_token_fraction=0.25,
        batch_prioritized_routing=False,
        use_xmoe=False,
    ) -> None:
        super().__init__()
        if use_xmoe:
            self.wg_reduction = torch.nn.Linear(model_dim, 16, bias=False)
            wg = torch.empty(num_experts, 16)
            torch.nn.init.orthogonal_(wg, gain=0.32)
            self.register_parameter("wg", torch.nn.Parameter(wg))
        else:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        
        self.use_fp32 = use_fp32
        self.second_expert_policy = second_expert_policy
        self.normalize_gate_prob_before_dropping = normalize_gate_prob_before_dropping
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.batch_prioritized_routing = batch_prioritized_routing
        self.use_xmoe = use_xmoe
        self.top_k = args.moe_top_k
        print("TopkGate:", self.top_k, "has_tutel", has_tutel)
        self.args = args

    def forward(self, input, mask=None):  # type: ignore
        if self.use_xmoe:
            input = self.wg_reduction(input)
            with torch.no_grad():
                wg_norm = self.wg.norm(p=2.0, dim=1, keepdim=True)
                self.wg.mul_(1.5 / wg_norm)
            logits = self._cosine(input, self.wg)
            logits = self._make_finite(logits)
        else:
            logits = self.wg(input)
        return topkgating(
            self.top_k,
            logits,
            mask,
            use_fp32=self.use_fp32,
            second_expert_policy=self.second_expert_policy,
            normalize_gate_prob_before_dropping=self.normalize_gate_prob_before_dropping,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            batch_prioritized_routing=self.batch_prioritized_routing,
        )

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
