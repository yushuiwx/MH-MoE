import os
import json
import argparse
import torch
import datasets
from transformers import AutoTokenizer
import random
import glog
import sentencepiece

from quip_sharp.dictionary import Dictionary
from quip_sharp.lib.utils import LMEvalAdaptor
from quip_sharp.lib.utils.unsafe_import import model_from_hf_path
from lm_eval import evaluator, tasks

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')
parser.add_argument('--spm_model', default='', type=str)
parser.add_argument('--ctx_size', default=2048, type=int)


def main(args):
    model, model_str = model_from_hf_path(args.hf_path,
                                          use_cuda_graph=False,
                                          use_flash_attn=not args.no_use_flash_attn)
    dictionary = None
    if args.spm_model == "":
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        glog.info('loaded model!')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.spm_model)
        EOL_SYMBOL = "</line>"
        print("load dictionary directly from spm_model")
        dictionary = Dictionary.load(args.spm_model, from_SPM=True)
        dictionary.add_symbol(EOL_SYMBOL)
        dictionary.pad_to_multiple_(2)

    task_names = args.tasks.split(",")

    lm_eval_model = LMEvalAdaptor(model_str, model, tokenizer, args.batch_size, args.ctx_size, dictionary)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=args.num_fewshot,
    )

    print(evaluator.make_table(results))

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = args.hf_path
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
