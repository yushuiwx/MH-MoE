# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch.nn import MSELoss
from omegaconf import II


@dataclass
class DistillCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    gt_alpha: float = field(
        default=1.0,
    )
    kd_alpha: float = field(
        default=0.0,
    )
    mse_alpha: float = field(
        default=0.0,
    )


@register_criterion("distill", dataclass=DistillCriterionConfig)
class DistillCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, gt_alpha, kd_alpha, mse_alpha):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.gt_alpha = gt_alpha
        self.kd_alpha = kd_alpha
        self.mse_alpha = mse_alpha

    def forward(self, model, sample, teacher_probs, teacher_inner_states, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        padding_mask = sample['net_input']['src_tokens'] != self.padding_idx
        gt_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)

        if teacher_probs is not None:
            kd_loss, _ = self.compute_kd_loss(model, net_output, teacher_probs, padding_mask, reduce=reduce)
        else:
            kd_loss = torch.zeros_like(gt_loss)
            
        if teacher_inner_states is not None:
            mse_loss, _ = self.compute_mse_loss(net_output[1]['inner_states'], teacher_inner_states)
        else:
            mse_loss = torch.zeros_like(gt_loss)

        loss = gt_loss * self.gt_alpha + kd_loss * self.kd_alpha + mse_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "gt_loss": gt_loss.data,
            "kd_loss": kd_loss.data,
            "mse_loss": mse_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    def compute_kd_loss(self, model, net_output, teacher_probs, padding_mask, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        teacher_probs = teacher_probs.view(-1, lprobs.size(-1))
        # loss = F.kl_div(
        #     lprobs,
        #     teacher_probs,
        #     reduction="none",
        # )
        loss = - teacher_probs * lprobs
        padding_mask = padding_mask.view(-1).unsqueeze(-1)
        loss *= padding_mask.float()

        loss = torch.sum(loss) if reduce else loss

        return loss, loss

    def compute_mse_loss(self, student_inner_states, teacher_inner_states):
        assert len(student_inner_states) == len(teacher_inner_states)
        loss = 0.
        for stu, tea in zip(student_inner_states, teacher_inner_states):
            loss += F.mse_loss(
                stu / stu.norm(2, keepdim=True, dim=-1), 
                tea / tea.norm(2, keepdim=True, dim=-1), 
                reduction="sum",
            ) / len(student_inner_states) * self.mse_alpha
        return loss, loss
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        gt_loss_sum = sum(log.get("gt_loss", 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get("kd_loss", 0) for log in logging_outputs)
        mse_loss_sum = sum(log.get("mse_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "gt_loss", gt_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "kd_loss", kd_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mse_loss", mse_loss_sum / sample_size, sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", gt_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["gt_loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
