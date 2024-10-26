import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from dataclasses import dataclass, field
from omegaconf import II, DictConfig
import torch.distributed as dist

from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.dataclass import FairseqDataclass
from collections.abc import Collection

@dataclass
class FairseqSophiaConfig(FairseqDataclass):
    sophia_betas: str = field(
        default="(0.965, 0.99)", metadata={"help": "betas for Adam optimizer"}
    )
    sophia_rho: float = field(
        default=0.04, metadata={"help": ""}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "weight decay"}
    )
    maximize: bool = field(
        default=False, metadata={"help": ""}
    )
    capturable: bool = field(
        default=False, metadata={"help": ""}
    )
    # TODO common vars below in parent
    # tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")
    # block_wise: bool = field(default=False, metadata={"help": "Enables block-wise optimization for 8-bit Adam"})


@register_optimizer("sophia", dataclass=FairseqSophiaConfig)
class FairseqSophia(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: DictConfig, params):
        super().__init__(cfg)
        self._optimizer = SophiaG(params, **self.optimizer_config)

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
            "betas": eval(self.cfg.sophia_betas),
            "rho": self.cfg.sophia_rho,
            "weight_decay": self.cfg.weight_decay,
            "maximize": self.cfg.maximize,
            "capturable": self.cfg.capturable,
        }

    def average_params(self):
        """average Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["hessian"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["hessian"], op=dist.ReduceOp.SUM)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return False


class SophiaG(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1, *,
        maximize: bool = False,
        capturable: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(params_with_grad,
                    grads,
                    exp_avgs,
                    hessian,
                    state_steps,
                    bs=bs,
                    beta1=beta1,
                    beta2=beta2,
                    rho=group['rho'],
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    maximize=group['maximize'],
                    capturable=group['capturable'])

        return loss


def sophiag(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        hessian: List[Tensor],
        state_steps: List[Tensor],
        capturable: bool = False,
        *,
        bs: int,
        beta1: float,
        beta2: float,
        rho: float,
        lr: float,
        weight_decay: float,
        maximize: bool,
    ):
        if not all(isinstance(t, torch.Tensor) for t in state_steps):
            raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

        func = _single_tensor_sophiag

        func(params,
             grads,
             exp_avgs,
             hessian,
             state_steps,
             bs=bs,
             beta1=beta1,
             beta2=beta2,
             rho=rho,
             lr=lr,
             weight_decay=weight_decay,
             maximize=maximize,
             capturable=capturable)


def _single_tensor_sophiag(
        params: List[Tensor],
       grads: List[Tensor],
       exp_avgs: List[Tensor],
       hessian: List[Tensor],
       state_steps: List[Tensor],
       *,
       bs: int,
       beta1: float,
       beta2: float,
       rho: float,
       lr: float,
       weight_decay: float,
       maximize: bool,
       capturable: bool,
    ):
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            hess = hessian[i]
            step_t = state_steps[i]

            if capturable:
                assert param.is_cuda and step_t.is_cuda and bs.is_cuda

            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                exp_avg = torch.view_as_real(exp_avg)
                hess = torch.view_as_real(hess)
                param = torch.view_as_real(param)

            # update step
            step_t += 1

            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            if capturable:
                step = step_t
                step_size = lr
                step_size_neg = step_size.neg()

                ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
                param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
            else:
                step = step_t.item()
                step_size_neg = - lr

                ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
                param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)