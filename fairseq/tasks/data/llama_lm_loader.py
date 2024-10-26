import json
import os
import multiprocessing
import itertools
import math
import random

from infinibatch import iterators
from functools import partial
from .lm_loader_v3 import LMLoader
from .utils import NativeCheckpointableIterator, WeightNoRandomStateIterator, EOL_SYMBOL, SelectManyNoSkipIterator

MAX_SHARD_SIZE = 10000000 # 10M files

NON_JSON_SET = ['c4/shard', 'cc-100/shard', 'wiki/shard_v1']
TIKTOKEN_LINE_BREAK = set(['<0x0A>'])

class TiktokenLmLoader(LMLoader):
    def _tokenize(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(
                self._tokenize_foreach_lang(data)
            )
            if 'weight' in data:
                weights.append(float(data['weight']))
            else:
                weights.append(int(data['count']))
        
        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightNoRandomStateIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = iterators.MultiplexIterator(control_iterator, multilingual_iters)
        
        return tokenized_lines

    def _tokenize_foreach_lang(self, data):
        # if 'epoch' in data:
        _random = random.Random(self.seed)
        data_source = data['source']
        epoch_num = 50
        temp_list = math.ceil(epoch_num) * data_source
        _random.shuffle(temp_list)
        dataset = list(zip(temp_list))
        # print('data name: ', data['name'], 'len(dataset): ', len(dataset))
        chunk_files = iterators.ChunkedSourceIterator(
            dataset,
            num_instances=self.num_shards, 
            instance_rank=self.shard_id,)
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.MapIterator(tokenized_lines, self._prepare)
        
        return tokenized_lines

    @staticmethod
    def fs_encode_line(fs_dict, words, append_eos=True):
        ids = []
        for i, word in enumerate(words):
            idx = fs_dict.index(word)
            ids.append(idx)
        if append_eos:
            ids.append(fs_dict.eos_index)
        return ids

    @staticmethod
    def _doc_to_ids(text, spm_tokenizer=None, fs_dict=None):
        assert EOL_SYMBOL in fs_dict.indices

        tokenized_ids = [] # list of list of ids

        tokens = spm_tokenizer.encode(text, out_type=str)

        current_list = []
        line_break_flag = False
        for token in tokens:
            if token in TIKTOKEN_LINE_BREAK:
                line_break_flag = True
                current_list.append(fs_dict.index(token))
            else:
                if line_break_flag:
                    tokenized_ids.append(current_list)
                    current_list = []
                    line_break_flag = False
                current_list.append(fs_dict.index(token))
        if len(current_list) > 0:
            tokenized_ids.append(current_list)
        tokenized_ids[-1].append(fs_dict.eos_index)
        return tokenized_ids

    def _read_from_files(self, source_file):
        data = []
        if self.args.absolute_path:
            file_path = source_file
        else:
            file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file

        lines_to_ids = False
        for non_json_key in NON_JSON_SET:
            if non_json_key in file_path:
                lines_to_ids = True
        if lines_to_ids:
            text = "\n".join(lines)
            tokenized_ids = []
            try:
                ret = TiktokenLmLoader._doc_to_ids(text, spm_tokenizer=self.tokenizer, fs_dict=self.dictionary)
                tokenized_ids.extend(ret)
            except BaseException as e:
                print(e)
                print(lines)
        else:
            tokenized_ids = []
            for doc_jsonstr in lines:
                try:
                    json_obj = json.loads(doc_jsonstr)
                    if 'text' in json_obj:
                        text = json_obj['text']
                    elif 'content' in json_obj:
                        text = json_obj['content']
                    elif 'raw_content_lines' in json_obj:
                        text = "\n".join(json_obj['raw_content_lines'])
                    if len(text) == 0:
                        continue
                    ret = TiktokenLmLoader._doc_to_ids(text, spm_tokenizer=self.tokenizer, fs_dict=self.dictionary)
                    tokenized_ids.extend(ret)
                except BaseException as e:
                    print(e)
                    print(doc_jsonstr)
            
        # ###################################################

        doc = [self.dictionary.bos()]
        for ids in tokenized_ids:
            if len(doc) + len(ids) > self.tokens_per_sample + 1:
                doc.extend(ids)
                doc = doc[:self.tokens_per_sample + 1]
                data.append(doc)
                doc = [self.dictionary.bos()]
            else:
                doc.extend(ids)

        if len(doc) > 1 and len(doc) <= self.tokens_per_sample + 1:
            data.append(doc)

        return data
