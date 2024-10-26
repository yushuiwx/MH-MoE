import os
import json
from argparse import Namespace
import torch

from fairseq import utils
from .data.dictionary import Dictionary
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from dataclasses import dataclass, field
import sentencepiece
import tiktoken
from omegaconf import II, MISSING

# from .data.llama_lm_loader import TiktokenLmLoader
from .data.llama_lm_loader import TiktokenLmLoader as LlamaLoader
from .data.tiktoken_lm_loader_v32 import TiktokenLmLoader
from .data.utils import EOL_SYMBOL

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"

import logging
logger = logging.getLogger(__name__)


@dataclass
class GPTLanguageModelingConfig(LanguageModelingConfig):
    spm_model: str = field(
        default="",
        metadata={
            "help": "sentencepice model to tokenize the data"
        },
    )
    tiktoken_model: str = field(
        default="",
        metadata={
            "help": "tiktoken model to tokenize the data"
        },
    )
    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON, metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE, metadata={"help": "path to vocab.bpe"}
    )
    dict_path: str = field(
        default="",
        metadata={
            "help": "sentencepice model to tokenize the data"
        },
    )
    batch_read_ahead: int = field(
        default=10000,
        metadata={"help": "batch read ahead size for infinibatch"},
    )
    pad_to_max_len: bool = field(
        default=False,
        metadata={"help": "pad each sentence to max length"},
    )
    absolute_path: bool = field(
        default=False,
        metadata={"help": "use absolute path in data config"},
    )
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")


@register_task('gpt', dataclass=GPTLanguageModelingConfig)
class GPTPretrainingTask(LanguageModelingTask):
    def __init__(self, args, dictionary, tokenizer, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary=output_dictionary, targets=targets)
        self.cfg = args
        self.tokenizer = tokenizer
    
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        if len(cfg.spm_model) > 0 and len(cfg.dict_path) == 0:
            logger.info("load dictionary directly from spm_model")
            dictionary = Dictionary.load(cfg.spm_model, from_SPM=True)
        elif len(cfg.dict_path) > 0:
            dictionary = Dictionary.load(cfg.dict_path)
        else:
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        dictionary.add_symbol(EOL_SYMBOL)
        dictionary.pad_to_multiple_(cfg.required_batch_size_multiple)

        output_dictionary = dictionary

        args = cfg
        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        if cfg.spm_model != "":
            tokenizer = sentencepiece.SentencePieceProcessor(model_file=cfg.spm_model)
        elif cfg.tiktoken_model != "":
            cl100k_base = tiktoken.get_encoding(cfg.tiktoken_model)
            tokenizer = tiktoken.Encoding(
                # If you're changing the set of special tokens, make sure to use a different name
                # It should be clear from the name what behaviour to expect.
                name="cl100k_im",
                pat_str=cl100k_base._pat_str,
                mergeable_ranks=cl100k_base._mergeable_ranks,
                special_tokens={
                    **cl100k_base._special_tokens,
                    "<fim_prefix>": 100264,
                    "<fim_middle>": 100265,
                    "<fim_suffix>": 100266,
                    "<fim_pad>": 100267,
                    "<reponame>": 100268,
                    "<filename>": 100269,
                    "<gh_stars>": 100270,
                    "<issue_start>": 100271,
                    "<issue_comment>": 100272,
                    "<issue_closed>": 100273,
                    "<jupyter_start>": 100274,
                    "<jupyter_text>": 100275,
                    "<jupyter_code>": 100276,
                    "<jupyter_output>": 100277,
                    "<empty_output>": 100278,
                    "<commit_before>": 100279,
                    "<commit_msg>": 100280,
                    "<commit_after>": 100281,
                }
            )
        else:
            tokenizer = GPT2BPE(Namespace(
                gpt2_vocab_bpe=cfg.gpt2_vocab_bpe,
                gpt2_encoder_json=cfg.gpt2_encoder_json))

        return cls(cfg, dictionary, tokenizer, output_dictionary, targets=targets)
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if "tnlg" in self.cfg.data:
            self.datasets[split] = {
                'data': json.load(open(f'{self.cfg.data}/json/{split}-nogithub-noarvix-nopubmed-mtnlg.json')) if split == 'train' else json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data_dir': self.cfg.data,
                'shuffle': True if split == 'train' else False,
            }
        else:
            self.datasets[split] = {
                'data': json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data_dir': self.cfg.data,
                'shuffle': True if split == 'train' else False,
            }
        self.datasets[split] = Namespace(**self.datasets[split])
    
    def dataset(self, split):
        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        
        return self.datasets[split]
    
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False
    ):
        # LMLoader = TiktokenLmLoader
        if self.cfg.tiktoken_model != "":
            LMLoader = TiktokenLmLoader
        else:
            LMLoader = LlamaLoader

        return LMLoader(
                self.cfg,
                dataset,
                self.dictionary,
                self.tokenizer,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

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
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample['gpt'])
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

