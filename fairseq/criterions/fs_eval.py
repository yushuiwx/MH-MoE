# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class FewshotEvalConfig(FairseqDataclass):
    fewshot_type: int = field(
        default=0,
        metadata={"help":"0: <s> x1 y1 <eos> x2 y2 <eos> x3 [y3],  1: <s> x1 y1 x2 y2 x3 [y3]"}
    )


@register_criterion("fs_eval", dataclass=FewshotEvalConfig)
class FewshotEvalCriterion(FairseqCriterion):
    def __init__(self, cfg: FewshotEvalConfig, task):
        super().__init__(task)
        self.fewshot_type = cfg.fewshot_type
        # context examples
        self.context_output = None
        self.context_tokens = None
        self.option_set = None

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens[tokens!=1]
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        # eos_mask = tokens == self.task.source_dictionary.eos()
        # doc_mask = eos_mask[1:] & eos_mask[:-1]
        # sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = self.task.tokenizer.decode(self.task.dictionary.string(tokens))
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model.eval()
        feature_only = False
        if hasattr(model, "gpt_model"):
            feature_only = True
        
        net_input = {
            "src_tokens": sample["net_input"]["src_tokens"].clone(),
            "src_lengths": sample["net_input"]["src_lengths"].clone(),
        }
        net_output, extra = model(
            # **sample["net_input"],
            **net_input,
            features_only=feature_only
        )
            
        net_output = net_output[:, :-1, :]
        net_output = (net_output, extra)
        targets = sample["net_input"]["src_tokens"][:, 1:].unsqueeze(-1)
        loss_mask = sample["net_input"]["gpt_loss_mask"][:, 1:]

        gpt_model = model
        if hasattr(model, "gpt_model"):
            gpt_model = model.gpt_model

        lprobs = gpt_model.get_normalized_probs(net_output, log_probs=True)
        loss = torch.gather(lprobs, -1, targets).squeeze(-1) * (loss_mask != False).int()
        loss = loss.sum(-1) / loss_mask.int().sum(-1)

        true_pred = torch.argmax(lprobs, -1)
        # print(f"targets is {self.decode(targets.squeeze(-1)[0][loss_mask[0]])}")

        option_num = self.task.fewshot_task.class_num
        fewshot_labels = sample["targets"].view(-1)
        
        assert sample["targets"].size(0) % option_num == 0
        sample_size = sample["targets"].size(0) // option_num

        pred_label = torch.argmax(loss.view(-1, option_num), dim=1)
        target_label = fewshot_labels.view(-1, option_num)[:,0]

        logging_output = {}

        logging_output.update(
            {
                "loss": loss.sum().data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
                "ncorrect": (pred_label == target_label).sum(),
                "npos": (target_label == 0).sum(),
                "nneg": (target_label == 1).sum(),
            }
        )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            npos = sum(log.get("npos", 0) for log in logging_outputs)
            nneg = sum(log.get("nneg", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
            metrics.log_scalar(
                "pos_proportion", 100.0 * npos / nsentences, nsentences, round=1
            )
            metrics.log_scalar(
                "neg_proportion", 100.0 * nneg / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True