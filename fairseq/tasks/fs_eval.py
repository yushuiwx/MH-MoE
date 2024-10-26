import os
import json
import torch
import numpy as np
from argparse import Namespace
import logging
from .data.dictionary import Dictionary
from fairseq.data import (
    data_utils,
    # Dictionary,
    encoders,
    BaseWrapperDataset,
    IdDataset,
    NumSamplesDataset,
    NumelDataset,
    NestedDictionaryDataset,
    SortDataset,
    NumelDataset,
    RightPadDataset,
    RawLabelDataset,
    FairseqDataset,
)

from fairseq.tasks import register_task, FairseqDataclass, FairseqTask
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from dataclasses import dataclass, field
from omegaconf import II, MISSING
from typing import Optional
from fairseq import utils
# from .data.spm_lm_loader import SpmLmLoader as LMLoader, EOL_SYMBOL
from .data.llama_lm_loader import TiktokenLmLoader as LMLoader
from .data.utils import EOL_SYMBOL

from .fewshot_task import CB, BoolQ, COPA, MultiRC, HellaSwag, StoryCloze, Winogrande, Winograd, PIQA, Lambada

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"

logger = logging.getLogger(__name__)

task_map = {
    "cb": CB,
    "boolq": BoolQ,
    "copa": COPA,
    "multirc": MultiRC,
    "hellaswag": HellaSwag,
    "storycloze": StoryCloze,
    "winogrande": Winogrande,
    "winograd": Winograd,
    "piqa": PIQA,
    "lambada": Lambada,
}


@dataclass
class FewshotEvalConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    seed: int = II("common.seed")
    eval_data: str = field(default="", metadata={"help": "dataset name"})
    test_split: str = field(default="test", metadata={"help": "test data split"})
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")

    tokens_per_sample: int = field(
        default=2048,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )

    gpt2_encoder_json: str = field(
        default=DEFAULT_ENCODER_JSON, metadata={"help": "path to encoder.json"}
    )
    gpt2_vocab_bpe: str = field(
        default=DEFAULT_VOCAB_BPE, metadata={"help": "path to vocab.bpe"}
    )

    gpt_dict: str = field(
        default="", metadata={"help": "gpt dict file"}
    )

    shuffle: bool = field(
        default=False,
        metadata={"help": "shuffle the in-context examples."},
    )

    k: int = field(
        default=4,
        metadata={"help": "k shot"},
    )
    temp_index: int = field(
        default=0,
        metadata={"help": "temp num"},
    )
    train_num: int = field(
        default=5000,
        metadata={"help": "k shot"},
    )
    valid_num: int = field(
        default=1000,
        metadata={"help": "k shot"},
    )

    all_gpt_emb: bool = field(
        default=False,
        metadata={
            "help": "whether to add connector"
        },
    )
    spm_model: str = field(
        default="",
        metadata={
            "help": "sentencepice model to tokenize the data"
        },
    )
    dict_path: str = field(
        default="",
        metadata={
            "help": "sentencepice model to tokenize the data"
        },
    )
    pad_to_max_length: int = field(
        default=0,
        metadata={
           "help": "pad to max length for each batch in moe mode"
        },
    )       


@register_task('fs_eval', dataclass=FewshotEvalConfig)
class FewshotEval(FairseqTask):

    def __init__(self, cfg, dictionary, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = dictionary
        self.seed = cfg.seed
        self.tokenizer = tokenizer

        # context examples
        self.mlm_tokens = None
        self.mlm_mask = None
        self.gpt_tokens = None 
        self.gpt_mask = None
        self.option_set = None
        self.fewshot_task = task_map[self.cfg.eval_data](tokenizer=self.tokenizer, dictionary=self.dictionary, k=cfg.k, temp_index=cfg.temp_index, train_num=cfg.train_num, valid_num=cfg.valid_num, seed=cfg.seed)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        if len(cfg.spm_model) > 0 and len(cfg.dict_path) == 0:
            logger.info("load dictionary directly from spm_model")
            dictionary = Dictionary.load(cfg.spm_model, from_SPM=True)
            dictionary.add_symbol(EOL_SYMBOL)
        elif len(cfg.dict_path) > 0:
            dictionary = Dictionary.load(cfg.dict_path)
            dictionary.add_symbol(EOL_SYMBOL)
        else:
            dictionary = Dictionary.load(cfg.gpt_dict)

        dictionary.pad_to_multiple_(cfg.required_batch_size_multiple)
        logger.info("dictionary: {} types".format(len(dictionary)))

        if len(cfg.spm_model) > 0:
            tokenizer = SentencepieceBPE(Namespace(
            sentencepiece_model=cfg.spm_model,
            sentencepiece_enable_sampling=False,
            sentencepiece_alpha=None))
        else:
            tokenizer = GPT2BPE(Namespace(
                gpt2_vocab_bpe=cfg.gpt2_vocab_bpe,
                gpt2_encoder_json=cfg.gpt2_encoder_json))

        return cls(cfg, dictionary, tokenizer)

    def load_dataset(self, split, combine=False, **kwargs):
        pad_length = None
        if self.cfg.pad_to_max_length > 0:
            pad_length = self.cfg.pad_to_max_length

        src_tokens, mlm_src_tokens, gpt_input_mask, mlm_mask, gpt_loss_mask, labels = self.fewshot_task.get_data_for_fewshot(cut_long_sequece=pad_length)

        src_tokens = RawArrayDataset(src_tokens)
        mlm_src_tokens = MLMContextDataset(RawArrayDataset(mlm_src_tokens), self.dictionary.pad(), datatype="token")
        mlm_mask = MLMContextDataset(RawArrayDataset(mlm_mask), self.dictionary.pad(), datatype="mask")
        gpt_input_mask = RawArrayDataset(gpt_input_mask, datatype="mask")
        gpt_loss_mask = RawArrayDataset(gpt_loss_mask, datatype="mask")
        label_ids = RawLabelDataset(labels)

        '''
            Input format: src_tokens + option_tokens
        '''
        data_dict = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.dictionary.pad(),
                    pad_length=pad_length,
                ),  
                'gpt_input_mask': RightPadDataset(
                    gpt_input_mask,
                    pad_idx=False,
                ),  
                'gpt_loss_mask': RightPadDataset(
                    gpt_loss_mask,
                    pad_idx=False,
                    pad_length=pad_length,
                ),
                'mlm_mask': mlm_mask,
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'targets': label_ids,
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        if not self.cfg.all_gpt_emb:
            data_dict["net_input"]["mlm_src_tokens"] = mlm_src_tokens

        dataset = NestedDictionaryDataset(
            data_dict,
            sizes=[src_tokens.sizes],
        )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class RawArrayDataset(FairseqDataset):

    def __init__(self, dataset, datatype="token"):
        super().__init__()
        self.dataset = dataset
        self.datatype = datatype
        if hasattr(dataset, 'sizes'):
            self._sizes = dataset.sizes
        else:
            try:
                self._sizes = np.array([len(x) for x in self.dataset])
            except:
                self._sizes =  np.array([1 for x in self.dataset])

    def __getitem__(self, index):
        if type(self.dataset[index][0]) != list:
            if self.datatype == "token":
                return torch.Tensor(self.dataset[index]).long()
            else:
                return torch.Tensor(self.dataset[index]).bool()
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

class MLMContextDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, datatype):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.datatype = datatype
        assert datatype == "token" or datatype == "mask"
        if hasattr(dataset, 'sizes'):
            self._sizes = dataset.sizes
        else:
            try:
                self._sizes = np.array([len(x) for x in self.dataset])
            except:
                self._sizes =  np.array([1 for x in self.dataset])

    def __getitem__(self, index):
        item = self.dataset[index]
        return item 

    @property
    def sizes(self):
        return self._sizes
    
    def collater(self, samples):
        temp_samples = []
        for sample in samples:
            for item in sample:
                if self.datatype == "token":
                    try:
                        temp_samples.append(torch.Tensor(item).long())
                    except:
                        print(0)
                else:
                    try:
                        temp_samples.append(torch.Tensor(item).bool())
                    except:
                        print(f"item is {item}")
        
        if len(temp_samples) == 0:
            if self.datatype == "token":
                temp_samples.append(torch.Tensor([0]).long())
            else:
                temp_samples.append(torch.Tensor([False]).bool())

        return collate_tokens(temp_samples, 
                    pad_idx=self.pad_idx if self.datatype == "token" else False, 
                    left_pad=False)