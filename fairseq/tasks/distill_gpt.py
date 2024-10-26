import os
import json
from argparse import Namespace
from typing import Optional
import torch

from fairseq import utils
# from fairseq.data import Dictionary
from .data.dictionary import Dictionary
from fairseq.tasks import FairseqTask, register_task
# from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from .gpt import GPTPretrainingTask, GPTLanguageModelingConfig
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from dataclasses import dataclass, field
import sentencepiece
import tiktoken
from omegaconf import II, MISSING

from .data.llama_lm_loader import TiktokenLmLoader as LMLoader
from .data.utils import EOL_SYMBOL

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"

import logging
logger = logging.getLogger(__name__)


@register_task('distill_gpt', dataclass=GPTLanguageModelingConfig)
class DistillGPTPretrainingTask(GPTPretrainingTask):
    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        if model.decoder.teacher is not None:
            model.decoder.teacher.eval()
            with torch.no_grad():  # 确保不计算梯度
                teacher_net_output = model.decoder.teacher(**sample['gpt']["net_input"])
                teacher_probs = model.decoder.teacher.get_normalized_probs(teacher_net_output, log_probs=False)
                teacher_inner_states = teacher_net_output[1]["inner_states"]
        else:
            teacher_probs, teacher_inner_states = None, None
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample['gpt'], teacher_probs, teacher_inner_states)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output