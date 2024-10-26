# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)

            return self.weight * hidden_states
        else:
            return hidden_states