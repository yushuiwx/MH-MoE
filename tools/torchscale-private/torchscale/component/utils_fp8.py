# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import warnings
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import math
import torch
from torch import nn
import transformer_engine.pytorch as te
import transformer_engine_extensions as tex
from torchscale.component.utils_quant import weight_quant_fn, act_quant_fn, AlphaInit

from transformer_engine.pytorch.utils import (
    cast_if_needed,
)
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.module.linear import _Linear
from transformer_engine.pytorch.float8_tensor import _FromFloat8Func, _ToFloat8Func


__all__ = ["Linear"]


class QuantizeFP8Linear(te.Linear):
    def __init__(
        self,
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
        **kwargs
    ) -> None:
        super(QuantizeFP8Linear, self).__init__(*kargs, **kwargs)
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

    @no_torch_dynamo
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """
        inp = act_quant_fn(
            inp, self.input_clip_val,
            num_bits=self.input_bits,
            symmetric=self.symmetric,
            quant_method=self.input_quant_method,
            layerwise=self.input_layerwise,
            input_parallel=False,
            grad_act=self.grad_act,
            scale=self.input_absmean_alpha,
        )
        weight = weight_quant_fn(
            self.weight, 
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
        with self.prepare_forward(inp, is_first_microbatch) as inp:
            assert self.fp8 or not self.primary_weights_in_fp8, \
                   "Need to run inside fp8_autocast region when weights are stored in FP8."
            bias_tensor = (
                self.bias if self.parameters_split is None
                else self.bias_tensor if not torch.is_grad_enabled()
                else self.noop_cat("bias_tensor", self.bias_names,
                    self.updated_parameters_split)
            )
            weight_tensor = (
                weight if self.parameters_split is None
                else self.weight_tensor if not torch.is_grad_enabled()
                else self.noop_cat("weight_tensor", self.weight_names,
                    self.updated_parameters_split)
            )

            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_t_fp8 = self.get_fp8_weights_scratchpad(
                is_first_microbatch
            )

            if torch.is_grad_enabled():
                linear_fn = _Linear.apply
                args = []
            else:
                linear_fn = _Linear.forward
                args = [None]
            args += (
                weight_tensor,
                weight1_fp8,
                weight1_t_fp8,
                inp,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                torch.is_grad_enabled(),
                self.primary_weights_in_fp8,
                self.ub_split_rs,
                self.ub_split_ag,
                self.ub_atomic_gemm_rs,
                self.ub_atomic_gemm_ag,
            )
            out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out