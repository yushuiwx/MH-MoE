import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List
from dataclasses import dataclass, field
from omegaconf import II, DictConfig
import torch.distributed as dist
from typing import Tuple, Optional, Callable

from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.dataclass import FairseqDataclass
from collections.abc import Collection

@dataclass
class FairseqLionConfig(FairseqDataclass):
    lion_betas: str = field(
        default="(0.9, 0.99)", metadata={"help": ""}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "weight decay"}
    )
    use_triton: bool = field(
        default=False, metadata={"help": ""}
    )
    # TODO common vars below in parent
    # tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")
    # block_wise: bool = field(default=False, metadata={"help": "Enables block-wise optimization for 8-bit Adam"})


@register_optimizer("lion", dataclass=FairseqLionConfig)
class FairseqLion(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: DictConfig, params):
        super().__init__(cfg)
        self._optimizer = Lion(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0] if isinstance(self.cfg.lr, Collection) else self.cfg.lr,
            "betas": eval(self.cfg.lion_betas),
            "weight_decay": self.cfg.weight_decay,
            "use_triton": self.cfg.use_triton,
        }

    def average_params(self):
        """average Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            raise NotImplementedError
            # from lion_pytorch.triton import update_fn as triton_update_fn
            # self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss


def exists(val):
    return val is not None


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)