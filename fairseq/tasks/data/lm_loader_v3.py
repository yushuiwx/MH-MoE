import glob
import os
import torch
import numpy as np
import time
import json
import random
import itertools
import hydra
import copy
from omegaconf import DictConfig, OmegaConf

from infinibatch import iterators
from .basic_loader import BaseBatchGen
from .utils import NativeCheckpointableIterator, WeightIterator, EOL_SYMBOL, FixedBlockwiseShuffleIterator


class LMLoader(BaseBatchGen):

    def __init__(
            self,
            args,
            dataset,
            dictionary,
            tokenizer,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            epoch=1,
            num_shards=1,
            shard_id=0,
    ):
        super().__init__()
        self.args = args
        self.data = dataset.data
        self.data_dir = dataset.data_dir
        self.shuffle = dataset.shuffle
        self.dictionary = dictionary
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.max_positions = max_positions
        self.tokens_per_sample = args.tokens_per_sample
        self.mlm_cut_length = getattr(args, "mlm_cut_length", 0)
        self.mlm_tokens_proportion = getattr(args, "mlm_tokens_proportion", 0)
        self.pad_to_max_len = getattr(args, "pad_to_max_len", False)
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = str(seed)
        self.epoch = epoch
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead
        self.sharded_checkpoint = True

        self._build_iter()
    
    def _build_iter(self):
        tokenized_lines = self._tokenize()
        self.padded_batches = self._batchify(tokenized_lines)
        
        if self.shuffle:
            prefetch_batches = iterators.PrefetchIterator(
                self.padded_batches, 
                buffer_size=10000, 
                buffer_in_main_process=True, 
                log_empty_buffer_warning=True and self.shard_id == 0,
            )
        else:
            prefetch_batches = self.padded_batches

        prefetch_batches = iterators.MapIterator(
            prefetch_batches, self._move_to_tensor
        )

        self._iter = prefetch_batches

    def _tokenize(self):
        '''
        data:
        {
            'source': list[Path],
        }
        '''
        dataset = list(zip(self.data['source']))

        if self.shuffle:
            chunk_files = \
                iterators.InfinitePermutationSourceIterator(
                    dataset,
                    seed=self.seed, 
                    shuffle=self.shuffle, 
                    num_instances=self.num_shards, 
                    instance_rank=self.shard_id,
                )
        else:
            chunk_files = \
                iterators.ChunkedSourceIterator(
                    dataset,
                    num_instances=self.num_shards, 
                    instance_rank=self.shard_id,
                )
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.SamplingRandomMapIterator(tokenized_lines, self._prepare, self.seed)
        
        return tokenized_lines

    def getstate(self):
        state = super().getstate()
        state["epoch"] = self.epoch
        state["iterations_in_epoch"] = None
        return state

    def _batchify(self, lines):
        
        if self.max_sentences is not None:
            if self.batch_read_ahead > 0:
                lines = FixedBlockwiseShuffleIterator(lines, self.batch_read_ahead, self.seed)
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:
            # -
            def dynamic_batch_size(sample):
                lengths = [len(x) for x in sample]
                batch_size = self.max_tokens // max(lengths) // self.required_batch_size_multiple * self.required_batch_size_multiple
                return max(1, batch_size)
            
            batches = iterators.BucketedReadaheadBatchIterator(
                    lines,
                    read_ahead=self.batch_read_ahead, 
                    key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None, 
                    batch_size=dynamic_batch_size, 
                    shuffle=self.shuffle,
                    seed=self.seed,
            )

        def collate(batch):
            batch_size = len(batch)
            mlm_batch_size = sum([len(x[2]) for x in batch]) 

            gpt_max_length = max([len(x[0]) for x in batch])
            if self.pad_to_max_len:
                gpt_max_length = self.tokens_per_sample + 1

            mlm_max_length = 0
            mlm_ntokens = 0
            for x in batch:
                for y in x[2]:
                    mlm_max_length = max(mlm_max_length, len(y))
                    mlm_ntokens += len(y)

            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            gpt_target_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            mlm_source_ids = np.full(shape=(mlm_batch_size, mlm_max_length), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)
            mlm_mask_all = np.full(shape=(mlm_batch_size, mlm_max_length), dtype=np.int32, fill_value=0)

            mlm_index = 0
            for i, (gpt_ids, gpt_input_mask, mlm_ids_list, mlm_mask_list, gpt_loss_mask) in enumerate(batch):
                gpt_source_ids[i, :len(gpt_ids)-1] = gpt_ids[:-1]
                gpt_target_ids[i, :len(gpt_ids)-1] = gpt_ids[1:]
                gpt_input_mask_all[i, :len(gpt_ids)-1] = gpt_input_mask[:-1]
                gpt_loss_mask_all[i, :len(gpt_ids)-1] = gpt_loss_mask[1:]
                
                for j, (mlm_ids, mlm_mask) in enumerate(zip(mlm_ids_list, mlm_mask_list)):
                    mlm_source_ids[mlm_index, :len(mlm_ids)] = mlm_ids
                    mlm_mask_all[mlm_index, :len(mlm_mask)] = mlm_mask
                    mlm_index += 1
            
            ret_batch = {
                'text':{
                    'net_input': {
                        'src_tokens': gpt_source_ids.astype(np.int64),
                        'mlm_src_tokens': mlm_source_ids.astype(np.int64) if mlm_batch_size !=0 else None,
                        'gpt_input_mask': gpt_input_mask_all.astype(np.bool_),
                        'gpt_loss_mask': gpt_loss_mask_all.astype(np.bool_),
                        'mlm_mask': mlm_mask_all.astype(np.bool_) if mlm_batch_size !=0 else None
                    },
                    'target': gpt_target_ids.astype(np.int64),
                    'nsentences': batch_size,
                    'ntokens': sum([len(x[0]) for x in batch]),
                    'mlm_ntokens': mlm_ntokens
                }
            }

            return ret_batch

        def collate_for_gpt(batch):
            batch_size = len(batch)
            gpt_max_length = max([len(x[0]) for x in batch])
            if self.pad_to_max_len:
                gpt_max_length = self.tokens_per_sample + 1

            gpt_source_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                 fill_value=self.dictionary.pad())
            gpt_target_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
                                     fill_value=self.dictionary.pad())
            gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
            gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)

            for i, (gpt_ids, gpt_input_mask, mlm_ids_list, mlm_mask_list, gpt_loss_mask) in enumerate(batch):
                gpt_source_ids[i, :len(gpt_ids)-1] = gpt_ids[:-1]
                gpt_target_ids[i, :len(gpt_ids)-1] = gpt_ids[1:]
                gpt_input_mask_all[i, :len(gpt_ids)-1] = gpt_input_mask[:-1]
                gpt_loss_mask_all[i, :len(gpt_ids)-1] = gpt_loss_mask[1:]
            
            ret_batch = {
                'gpt':{
                    'net_input': {
                        'src_tokens': gpt_source_ids.astype(np.int64),
                    },
                    'target': gpt_target_ids.astype(np.int64),
                    'nsentences': batch_size,
                    'ntokens': sum([len(x[0]) for x in batch]),
                    'mlm_ntokens': 0
                }
            }

            return ret_batch

        if self.mlm_tokens_proportion == 0:
            padded_batches = iterators.MapIterator(
                batches, collate_for_gpt
            )
        else:
            padded_batches = iterators.MapIterator(
                batches, collate
            )

        return padded_batches

    def _prepare(self, doc):
        mlm_tokens = []
        mlm_mask = []
        gpt_input_mask = [0] * len(doc)
        gpt_loss_mask = [1] * len(doc)
        full_tokens = doc
        return full_tokens, gpt_input_mask, mlm_tokens, mlm_mask, gpt_loss_mask
    
    def _read_from_files(self, source_file):
        data = []
        file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.read().strip().split('\n')

        gpt_format_text = []
        for line in lines:
            gpt_format_text.extend(list(filter(None, json.loads(line)["text"].split("\n"))))
            gpt_format_text.append('')

        tokenized_lines = [self.tokenizer.encode(line) for line in gpt_format_text]
        tokenized_ids = [self.dictionary.encode_line(line, add_if_not_exist=False) for line in tokenized_lines]

        doc = [self.dictionary.bos()]
        for ids in tokenized_ids:
            if len(ids) > self.tokens_per_sample: # drop too long sentence
                continue

            if len(doc) + len(ids) > self.tokens_per_sample:
                if len(doc) > 5/2*self.mlm_cut_length + 1:
                    data.append(doc)
                doc = [self.dictionary.bos()]
            doc.extend(ids)

        if len(doc) > 1 and len(doc) <= self.tokens_per_sample:
            if len(doc) > 5/2*self.mlm_cut_length + 1:
                data.append(doc)

        return data
