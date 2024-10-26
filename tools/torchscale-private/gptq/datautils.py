import numpy as np
import torch
import sentencepiece
# from fairseq.data import Dictionary
from quip_sharp.dictionary import Dictionary

def fs_encode_line(fs_dict, words, append_eos=True):
    ids = []
    for i, word in enumerate(words):
        idx = fs_dict.index(word)
        ids.append(idx)
    if append_eos:
        ids.append(fs_dict.eos_index)
    return ids


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, spm_model="", dict_path=""):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    class TokenizerTrainWrapper:
        def __init__(self, input_ids, pad_index):
            self.input_ids = input_ids
            attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
            attention_mask[input_ids != pad_index] = 1
            self.attention_mask = attention_mask

    if spm_model == "":
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    else:
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=spm_model)
        EOL_SYMBOL = "</line>"
        dictionary = Dictionary.load(dict_path)
        dictionary.add_symbol(EOL_SYMBOL)
        dictionary.pad_to_multiple_(2)

        # trainenc = tokenizer.EncodeAsIds("".join(traindata['text']))
        # trainenc = torch.Tensor([dictionary.bos_index] + trainenc).long().reshape(1, -1)

        # text = "Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall ."

        text = ""
        tokens = tokenizer.encode(text, out_type=str)
        tokenized_tokens = fs_encode_line(dictionary, tokens, append_eos=False)
        trainenc = torch.Tensor([dictionary.bos()] + tokenized_tokens).long().reshape(1, -1)
        trainenc = TokenizerTrainWrapper(trainenc, dictionary.pad_index)

        testenc = tokenizer.EncodeAsIds("".join(testdata['text']))
        testenc = torch.Tensor([dictionary.bos_index] + testenc).long().reshape(1, -1)
        testenc = TokenizerTrainWrapper(testenc, dictionary.pad_index)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, spm_model="", dict_path=""):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, spm_model="", dict_path=""):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    if spm_model == "":
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

        import random
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        import random
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
    else:
        class TokenizerTrainWrapper:
            def __init__(self, input_ids, pad_index):
                self.input_ids = input_ids
                attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
                attention_mask[input_ids != pad_index] = 1
                self.attention_mask = attention_mask
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        tokenizer = sentencepiece.SentencePieceProcessor(model_file=spm_model)
        EOL_SYMBOL = "</line>"
        if dict_path != "":
            dictionary = Dictionary.load(dict_path)
        else:
            print("load dictionary directly from spm_model")
            dictionary = Dictionary.load(spm_model, from_SPM=True)
        dictionary.add_symbol(EOL_SYMBOL)
        dictionary.pad_to_multiple_(2)
        import random
        random.seed(seed)
        trainloader = []
        # text = "Conduct effective product control and valuation control of treasury products and derivatives in banking book. Duties include daily P&L reporting and analysis, balance sheet substantiation, pricing parameter verification and market conformity checking. First point of contact for front office to handle ad hoc queries on position, P&L. University graduate with degree in Financial Engineering, Quantitative Finance, Risk Management, Finance, Economics, Accounting or related disciplines. Further professional qualification such as ACCA, CPA, CFA, FRM would be a plus. At least 2 year of working experience in equity and interest rate related treasury products and/or product control, solid experience in South East Asia is an advantage. Proficiency in financial instruments & market operations."
        # text = "Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as \" Craig \" in the episode \" Teddy 's Story \" of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall ."
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                # trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                # trainenc = tokenizer.EncodeAsIds(traindata[i]['text'])
                tokens = tokenizer.encode(traindata[i]['text'].replace('\n', ''), out_type=str)
                tokenized_tokens = fs_encode_line(dictionary, tokens, append_eos=False)
                trainenc = torch.Tensor([dictionary.bos()] + tokenized_tokens).long().reshape(1, -1)
                trainenc = TokenizerTrainWrapper(trainenc, dictionary.pad())
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        import random
        random.seed(42)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                # tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
                # tmp = tokenizer.EncodeAsIds(valdata[i]['text'])
                tokenized_tokens = [dictionary.bos()]
                for line in valdata[i]['text'].split('\n'):
                    tokens = tokenizer.encode(line, out_type=str)
                    tokenized_tokens += fs_encode_line(dictionary, tokens, append_eos=True)
                tmp = torch.Tensor(tokenized_tokens).long().reshape(1, -1)
                tmp = TokenizerTrainWrapper(tmp, dictionary.pad())
                # break
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            # i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            # j = i + seqlen
            # valenc.append(tmp.input_ids[:, i:j])
            valenc.append(tmp.input_ids[:, :seqlen])
        valenc = torch.hstack(valenc)
        valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model, spm_model="", dict_path=""):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model, spm_model="", dict_path=""):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', spm_model="", dict_path="",
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, spm_model=spm_model, dict_path=dict_path)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model, spm_model=spm_model, dict_path=dict_path)
        return get_ptb(nsamples, seed, seqlen, model, spm_model=spm_model, dict_path=dict_path)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model, spm_model=spm_model, dict_path=dict_path)
        return get_c4(nsamples, seed, seqlen, model, spm_model=spm_model, dict_path=dict_path)
