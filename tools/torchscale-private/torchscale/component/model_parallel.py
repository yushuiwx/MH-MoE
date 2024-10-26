import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from fairseq.model_parallel.megatron.mpu import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from fairseq.model_parallel.megatron.mpu.mappings import (
        scatter_to_model_parallel_region,
        reduce_from_model_parallel_region,
        copy_to_model_parallel_region,
        gather_from_model_parallel_region
    )
    from fairseq.model_parallel.megatron.mpu.initialize import get_model_parallel_world_size, get_model_parallel_rank
    from fairseq.model_parallel.megatron.mpu.utils import divide
    from .utils_quant import AlphaInit, LearnableBias, act_quant_fn, weight_quant_fn, QuantizeLinear

    class RowParallelQuantizeLinear(RowParallelLinear):
        def __init__(
            self,
            *args,
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
            super().__init__(*args, **kwargs)
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
            self.move = None
            # self.move = LearnableBias(self.input_size_per_partition) if weight_bits != 32 else None
            # assert self.weight_quant_method == "bwn"
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

        def _build_weight_clip_val(self, quant_method, learnable, init_val):
            if quant_method == 'uniform':
                # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
                self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
                if learnable:
                    self.weight_clip_val = nn.Parameter(self.weight_clip_val)
            elif quant_method == 'elastic':
                assert learnable, 'Elastic method must use leranable step size!'
                self.weight_clip_val = AlphaInit(
                    torch.tensor(1.0))  # stepsize will be initialized in the first quantization
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

        def forward(self, input_):
            if self.move is not None:
                input_ = self.move(input_)

            # Set up backprop all-reduce.
            if self.input_is_parallel:
                input_parallel = input_
            else:
                input_parallel = scatter_to_model_parallel_region(input_)

            input_parallel = act_quant_fn(
                input_parallel, 
                self.input_clip_val,
                num_bits=self.input_bits,
                symmetric=self.symmetric,
                quant_method=self.input_quant_method,
                layerwise=self.input_layerwise,
                input_parallel=False,
                # input_parallel=self.input_is_parallel,
                grad_act=self.grad_act,
                scale=self.input_absmean_alpha,
                sparse_blocksize=self.sparse_blocksize,
                sparse_ratio=self.sparse_ratio,
                sparse_alpha=self.sparse_alpha,
            )
            weight = weight_quant_fn(
                self.weight, 
                self.weight_clip_val,
                num_bits=self.weight_bits,
                symmetric=self.symmetric,
                quant_method=self.weight_quant_method,
                layerwise=self.weight_layerwise,
                input_parallel=False,
                # input_parallel=self.input_is_parallel,
                weight_blocksize=self.weight_blocksize,
                use_blockscale=self.weight_blockscale_init,
                absmean_alpha=self.absmean_alpha,
            )

            # Matrix multiply.
            output_parallel = F.linear(input_parallel, weight)
            # All-reduce across all the partitions.
            output_ = reduce_from_model_parallel_region(output_parallel)
            if self.bias is not None:
                output = output_ + self.bias
            else:
                output = output_
            return output


    class ColumnParallelQuantizeLinear(ColumnParallelLinear):
        def __init__(
            self,
            *args,
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
            super().__init__(*args, **kwargs)
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
            # self.move = LearnableBias(self.input_size) if weight_bits != 32 else None
            self.move = None
            # assert self.weight_quant_method == "bwn"
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

        def _build_weight_clip_val(self, quant_method, learnable, init_val):
            if quant_method == 'uniform':
                # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
                self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
                if learnable:
                    self.weight_clip_val = nn.Parameter(self.weight_clip_val)
            elif quant_method == 'elastic':
                assert learnable, 'Elastic method must use leranable step size!'
                self.weight_clip_val = AlphaInit(
                    torch.tensor(1.0))  # stepsize will be initialized in the first quantization
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

        def forward(self, input_):
            if self.move is not None:
                input_ = self.move(input_)

            # Set up backprop all-reduce.
            input_parallel = copy_to_model_parallel_region(input_)

            input_parallel = act_quant_fn(
                input_parallel, 
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

            # Matrix multiply.
            output_parallel = F.linear(input_parallel, weight, self.bias)
            if self.gather_output:
                # All-gather across the partitions.
                output = gather_from_model_parallel_region(output_parallel)
            else:
                output = output_parallel
            return output
except ModuleNotFoundError:
    print("No Megatron-LM")


def ModelParallelLinear(
        args,
        input_dim,
        output_dim,
        bias=True,
        parallel_mode='row',
        init_method=nn.init.xavier_uniform_,
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
        sparse_alpha=1.0
    ):
    if args.model_parallel_size <= 1:
        return QuantizeLinear(
            input_dim,
            output_dim,
            bias=bias,
            clip_val=clip_val,
            weight_bits=weight_bits,
            input_bits=input_bits,
            learnable=learnable,
            symmetric=symmetric,
            weight_layerwise=weight_layerwise,
            input_layerwise=input_layerwise,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
            learnable_step=learnable_step,
            hadamard_group=hadamard_group,
            blockwise_quant=blockwise_quant,
            weight_blocksize=weight_blocksize,
            grad_act=grad_act,
            weight_blockscale=weight_blockscale,
            smoothquant=smoothquant,
            smoothquant_alpha=smoothquant_alpha,
            absmean_alpha=absmean_alpha,
            input_absmean_alpha=input_absmean_alpha,
            sparse_blocksize=sparse_blocksize,
            sparse_ratio=sparse_ratio,
            sparse_alpha=sparse_alpha,
        )
    elif parallel_mode == 'row':
        if weight_bits < 32 or input_bits < 32:
            return RowParallelQuantizeLinear(
                input_dim,
                output_dim,
                bias=bias,
                input_is_parallel=True,
                init_method=init_method,
                clip_val=clip_val,
                weight_bits=weight_bits,
                input_bits=input_bits,
                learnable=learnable,
                symmetric=symmetric,
                weight_layerwise=weight_layerwise,
                input_layerwise=input_layerwise,
                weight_quant_method=weight_quant_method,
                input_quant_method=input_quant_method,
                learnable_step=learnable_step,
                hadamard_group=hadamard_group,
                blockwise_quant=blockwise_quant,
                weight_blocksize=weight_blocksize,
                grad_act=grad_act,
                weight_blockscale=weight_blockscale,
                smoothquant=smoothquant,
                smoothquant_alpha=smoothquant_alpha,
                absmean_alpha=absmean_alpha,
                input_absmean_alpha=input_absmean_alpha,
                sparse_blocksize=sparse_blocksize,
                sparse_ratio=sparse_ratio,
                sparse_alpha=sparse_alpha,
            )
        return RowParallelLinear(input_dim, output_dim, bias, input_is_parallel=True, init_method=init_method)
    else:
        assert parallel_mode == 'column'
        if weight_bits < 32 or input_bits < 32:
            return ColumnParallelQuantizeLinear(
                input_dim,
                output_dim,
                bias=bias,
                gather_output=False,
                init_method=init_method,
                clip_val=clip_val,
                weight_bits=weight_bits,
                input_bits=input_bits,
                learnable=learnable,
                symmetric=symmetric,
                weight_layerwise=weight_layerwise,
                input_layerwise=input_layerwise,
                weight_quant_method=weight_quant_method,
                input_quant_method=input_quant_method,
                learnable_step=learnable_step,
                hadamard_group=hadamard_group,
                blockwise_quant=blockwise_quant,
                weight_blocksize=weight_blocksize,
                grad_act=grad_act,
                weight_blockscale=weight_blockscale,
                smoothquant=smoothquant,
                smoothquant_alpha=smoothquant_alpha,
                absmean_alpha=absmean_alpha,
                input_absmean_alpha=input_absmean_alpha,
                sparse_blocksize=sparse_blocksize,
                sparse_ratio=sparse_ratio,
                sparse_alpha=sparse_alpha,
            )
        return ColumnParallelLinear(input_dim, output_dim, bias, gather_output=False, init_method=init_method)