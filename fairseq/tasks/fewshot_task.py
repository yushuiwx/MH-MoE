import datasets
import random
import torch
import json
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, accuracy_score
# from .data.spm_lm_loader import SpmLmLoader as LMLoader, EOL_SYMBOL
from .data.llama_lm_loader import TiktokenLmLoader as LMLoader
from .data.utils import EOL_SYMBOL

class BaseTask(object):
    def __init__(self, tokenizer, dictionary, seed=1, k=2, temp_index=0, prune_valid_set=False, train_num=500, valid_num=10000, mlm_eos=False, label_bidirect=False, gpt_maxlen_persample=1024, mlm_maxlen=512, gpt_maxlen=2048):
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.tokenizer = tokenizer
        self.dictionary = dictionary

        self.k = k 
        self.temp_index = temp_index
        self.prune_valid_set = prune_valid_set
        self.train_num = train_num
        self.valid_num = valid_num
        self.class_num = 1
        self.gpt_maxlen_persample = gpt_maxlen_persample
        self.mlm_maxlen = mlm_maxlen
        self.gpt_maxlen = gpt_maxlen

        # mlm tokens end with </s>
        self.mlm_eos = mlm_eos
        # put the context labels also in bidirect mlm
        self.label_bidirect = label_bidirect

    def templates_set_without_newline(self):
        raise NotImplementedError("Please provide the templates!")

    def preprocess_example(self):
        raise NotImplementedError("Preprocess single example!")

    def get_dataset_train(self, data):
        return data.shuffle(seed=self.seed)
    
    def get_dataset_valid(self, data):
        if self.prune_valid_set:
            return data.shuffle(seed=self.seed)
        else:
            return data

    def compute_f1(self, golds, preds):
        return f1_score(golds, preds)
    
    def compute_accuracy(self, golds, preds):
        return accuracy_score(golds, preds)

    def compute_metric(self, golds, preds, metric="accuracy"):
        if metric == "accuracy":
            return self.compute_accuracy(golds, preds)
        elif metric == "f1":
            return self.compute_f1(golds, preds)

    def get_data_for_fewshot(self, cut_long_sequece=None):
        src_tokens_train, mlm_src_tokens_train, gpt_input_mask_train, mlm_mask_train, gpt_loss_mask_train, _ = self.tokenized_data('train', cut_long_sequece)
        src_tokens_valid, mlm_src_tokens_valid, gpt_input_mask_valid, mlm_mask_valid, gpt_loss_mask_valid, labels = self.tokenized_data('valid', cut_long_sequece)

        for i in range(len(labels) // self.class_num):
            idx_train = np.random.choice(np.arange(min(len(self.dataset_train), self.train_num)), self.k, replace=False)
            context_src_tokens_list = src_tokens_train[idx_train]
            mlm_src_tokens_train_list = mlm_src_tokens_train[idx_train]
            gpt_input_mask_train_list = gpt_input_mask_train[idx_train]
            mlm_mask_train_list = mlm_mask_train[idx_train]

            context_src_tokens = [0]
            context_gpt_input_mask = [False]
            for m in range(self.k):
                if len(context_src_tokens_list[m]) < (self.gpt_maxlen - len(gpt_input_mask_valid[i*self.class_num])):
                    context_src_tokens.extend(context_src_tokens_list[m][1:])
                    context_gpt_input_mask.extend(gpt_input_mask_train_list[m][1:])

            for j in range(i*self.class_num, (i+1)*self.class_num):
                src_tokens_valid[j] = context_src_tokens + src_tokens_valid[j][1:]
                mlm_src_tokens_valid[j] = list(mlm_src_tokens_train_list) + [mlm_src_tokens_valid[j]]
                gpt_input_mask_valid[j] = context_gpt_input_mask + gpt_input_mask_valid[j][1:]
                mlm_mask_valid[j] = list(mlm_mask_train_list) + [mlm_mask_valid[j]]
                gpt_loss_mask_valid[j] = [False]*len(context_src_tokens) + gpt_loss_mask_valid[j][1:]

        return src_tokens_valid, mlm_src_tokens_valid, gpt_input_mask_valid, mlm_mask_valid, gpt_loss_mask_valid, labels


    def tokenized_data(self, split='train', cut_long_sequece=None):
        src_tokens = []
        mlm_src_tokens = []
        gpt_input_mask = []
        mlm_mask = []
        gpt_loss_mask = []
        labels = []
        dataset = self.dataset_train if split == 'train' else self.dataset_valid
        cut_num = 0
        if split == 'train':
            min_num = self.train_num
        else:
            min_num = self.valid_num if self.prune_valid_set else len(dataset)

        def encode(sentence):
            splitlines = list(filter(None, sentence.splitlines()))
            return torch.cat([self.dictionary.encode_line(self.tokenizer.encode(line), add_if_not_exist=False) for line in splitlines]).tolist()

        for i in range(min(len(dataset), min_num)):
            example = dataset[i]
            input_str, label_str, label = self.preprocess_example(example)
            if i < 2:
                print(f"input str is {input_str}")
                print(f"label str is {label_str}")

            if split == 'train':
                input_str, label_str = input_str[label], label_str[label]
                input_token = encode(input_str)[0:-1]
                # append eos token
                label_token = encode(label_str)

                if cut_long_sequece is not None: 
                    temp_token = input_token + label_token
                    if len(temp_token) + 1 > cut_long_sequece: # cut the front part
                        cut_num += 1
                        temp_token = temp_token[-cut_long_sequece+1:]
                    src_tokens.append([0] + temp_token)
                    
                else:
                    src_tokens.append([0] + input_token + label_token)
                if len(input_token) + 1 > self.mlm_maxlen:
                    mlm_src_tokens.append([0] + input_token[:self.mlm_maxlen-1])
                    gpt_input_mask.append([False] + [True]*(self.mlm_maxlen-1) + [False]*(len(label_token)+len(input_token)-self.mlm_maxlen+1))
                    mlm_mask.append([False] + [True]*(self.mlm_maxlen-1))
                else:
                    mlm_src_tokens.append([0] + input_token)
                    gpt_input_mask.append([False] + [True]*len(input_token) + [False]*len(label_token))
                    mlm_mask.append([False] + [True]*len(input_token))
                gpt_loss_mask.append([False]*(len(gpt_input_mask[-1])))
                labels.append(label)
            elif split == 'valid':
                for j in range(len(input_str)):
                    sub_input_str, sub_label_str = input_str[j], label_str[j]
                    input_token = encode(sub_input_str)[0:-1]
                    label_token = encode(sub_label_str)[0:-1]

                    if cut_long_sequece is not None: 
                        temp_token = input_token + label_token
                        if len(temp_token) + 5 > cut_long_sequece: # cut the front part
                            cut_num += 1
                            temp_token = temp_token[-cut_long_sequece+5:]
                        src_tokens.append([0] + temp_token)
                    else:
                        src_tokens.append([0] + input_token + label_token)
                    if len(input_token) + 1 > self.mlm_maxlen:
                        mlm_src_tokens.append([0] + input_token[:self.mlm_maxlen-1])
                        gpt_input_mask.append([False] + [True]*(self.mlm_maxlen-1) + [False]*(len(label_token)+len(input_token)-self.mlm_maxlen+1))
                        mlm_mask.append([False] + [True]*(self.mlm_maxlen-1))
                    else:
                        mlm_src_tokens.append([0] + input_token)
                        gpt_input_mask.append([False] + [True]*len(input_token) + [False]*len(label_token))
                        mlm_mask.append([False] + [True]*len(input_token))
                    gpt_loss_mask_item = [False]*(len(input_token)+1) + [True]*len(label_token)
                    gpt_loss_mask.append(gpt_loss_mask_item[-len(src_tokens[-1]):])
                    labels.append(label)

        if cut_num > 0:
            print(f"cut {cut_num} examples")
            
        return np.array(src_tokens), np.array(mlm_src_tokens), np.array(gpt_input_mask), np.array(mlm_mask), np.array(gpt_loss_mask), np.array(labels)


# super-glue
class CB(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'cb')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 3

    def templates_set_without_newline(self):
        return [
            ("{premise} Question: {hypothesis}. True, False, or Neither? Answer:", " {answer}", ["True", "False", "Neither"]),
            ("{premise} Based on the paragraph above can we conclude that \"{hypothesis}\"? Yes, No, or Maybe?", " Answer: {answer}.", ["Yes", "No", "Maybe"]),
            ("{premise} Can we infer the following? {hypothesis}.", " {answer}", ["Yes", "No", "Maybe"]),
            ("Read the following paragraph and determine if the hypothesis is true: {premise} Hypothesis: {hypothesis}.", " {answer}", ["Yes", "No", "Maybe"]),
            ("Can we draw the following hypothesis from the context? Context: {premise} Hypothesis: {hypothesis}. Answer:", " {answer}", ["Yes", "No", "Maybe"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{premise}", example["premise"]).replace("{hypothesis}", example["hypothesis"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label


class BoolQ(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'boolq')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("Passage: {passage} After reading this passage, I have a question: {question}? True or False?", " {answer}", ["False", "True"]),
            ("Text: {passage} Question: {question}? Answer:", " {answer}", ["No", "Yes"]),
            ("{passage} Based on the above text, what's the best answer to this question: {question}?", " {answer}", ["No", "Yes"]),
            ("Based on the following passage, {question}? {passage} Please answer yes or no.", " {answer}", ["No", "Yes"]),
            ("Exercise: read the text and answer the question by True or False. Text: {passage} Question: {question}?", " {answer}", ["False", "True"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{question}", example["question"]).replace("{passage}", example["passage"])
        input_str = [input_str] * self.class_num
        answer_str = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        label = example["label"]
        return input_str, answer_str, label


class COPA(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'copa')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 2

    def preprocess_example(self, example):
        def copa_answer(string):
            string = ' ' + string[0].lower() + string[1:]
            return string

        text_first = example["premise"]

        if self.temp_index == 0:
            if text_first[-1] == '.':
                text_first = text_first[:-1]
            if example["question"] == "cause":
                text_first = text_first + " because"
            else:
                text_first = text_first + " so"
            input_str = [text_first] * self.class_num
            answer_str = [copa_answer(example["choice1"]), copa_answer(example["choice2"])]
        elif self.temp_index == 2:
            input_str = [text_first + " What is the " + example["question"] + "?"] * self.class_num
            answer_str = [' ' + example["choice1"], ' ' + example["choice2"]]
        elif self.temp_index == 3:
            if text_first[-1] == '.':
                text_first = text_first[:-1]
            if example["question"] == "cause":
                text_first = text_first + " because"
            else:
                text_first = text_first + " so"
            input_str = [text_first] * self.class_num
            answer_str = [' '+example["choice1"], ' '+example["choice2"]]
        elif self.temp_index == 4:
            if text_first[-1] == '.':
                text_first = text_first[:-1]
            if example["question"] == "cause":
                text_first = text_first + " This happened because"
            else:
                text_first = text_first + " As a consequence,"
            input_str = [text_first] * self.class_num
            answer_str = [copa_answer(example["choice1"]), copa_answer(example["choice2"])]
        label = example["label"]
        return input_str, answer_str, label


class MultiRC(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('super_glue', 'multirc')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.templates = self.templates_set_without_newline()
        self.class_num = 2

    def templates_set_without_newline(self):
        return [
            ("{paragraph} Question: \"{question}\" Response: \"{response}\" Does the response correctly answer the question? Answer:", " {answer}", ["No", "Yes"]),
            ("{paragraph} Question: \"{question}\" Response: \"{response}\" Based on the paragraph, is the response to the question is factually correct?", " {answer}", ["No", "Yes"]),
            ("{paragraph} Based on the paragraph, does the response \"{response}\" correctly answer the question \"{question}\"?", " {answer}", ["No", "Yes"]),
            ("{paragraph} According to the above paragraph, the correct answer to the question \"{question}\" is \"{response}\"? Answer:", " {answer}", ["No", "Yes"]),
            ("{paragraph} Question: \"{question}\" Answer: \"{response}\" Is this answer to the question True or False?", " {answer}", ["False", "True"])
        ]

    def preprocess_example(self, example):
        input_temp, output_temp, options = self.templates[self.temp_index]
        input_str = input_temp.replace("{paragraph}", example["paragraph"]).replace("{question}", example["question"]).replace("{response}", example["answer"])
        answer_str = output_temp.replace("{answer}", options[example["label"]])
        options_list = [output_temp.replace("{answer}", options[i]) for i in range(len(options))]
        return input_str, answer_str, options_list

# completion
class HellaSwag(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('hellaswag')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 4

    def preprocess_example(self, example):
        input_str = [example["ctx"]] * self.class_num
        answer_str = []
        for i in range(self.class_num):
            answer_str.append(' ' + example["endings"][i])
        label = int(example["label"])
        return input_str, answer_str, label


class PIQA(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('piqa')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 2

    def preprocess_example(self, example):
        input_str = [example["goal"]] * self.class_num
        answer_str = []
        for i in range(self.class_num):
            if i == 0:
                answer_str.append(' ' + example["sol1"])
            else:
                answer_str.append(' ' + example["sol2"])
                
        label = int(example["label"])
        return input_str, answer_str, label


class Lambada(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_dict = {"text": []}
        with open("/mnt/msranlp/yaru/data/fs_eval/lambada/lambada_test.jsonl", "r") as fin:
            for line in fin:
                data_dict["text"].append(json.loads(line)['text'])
        dataset = Dataset.from_dict(data_dict)
        self.dataset_train = self.get_dataset_train(dataset)
        self.dataset_valid = self.get_dataset_valid(dataset)
        self.class_num = 1

    def preprocess_example(self, example):
        text = example["text"]
        space_index = text.rfind(' ')
        input_str = [text[:space_index]]
        answer_str = [text[space_index:]]
        return input_str, answer_str, 0

# download mannualy
class StoryCloze(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('story_cloze', data_dir="/mnt/msranlp/yaru/data/fs_eval/story_cloze")
        # dataset = load_dataset('story_cloze', data_dir="/mnt/share/data/nlpunilm/yaru/data/fs_eval/story_cloze")
        self.dataset_train = self.get_dataset_train(dataset['validation'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 2

    def preprocess_example(self, example):
        input_str = example["input_sentence_1"] + ' ' + example["input_sentence_2"] + ' ' + example["input_sentence_3"] + ' ' + example["input_sentence_4"] 
        input_str = [input_str] * self.class_num
        answer_str = [' '+example["sentence_quiz1"], ' '+example["sentence_quiz2"]]
        label = int(example["answer_right_ending"]) - 1
        return input_str, answer_str, label

# Winograd tasks
class Winogrande(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('winogrande', 'winogrande_xs')
        self.dataset_train = self.get_dataset_train(dataset['train'])
        self.dataset_valid = self.get_dataset_valid(dataset['validation'])
        self.class_num = 2

    def preprocess_example(self, example):
        cut_index = example["sentence"].index('_')
        text_first = example["sentence"][:cut_index]
        text_second = example["sentence"][cut_index+1:]
        input_str = [text_first+example["option1"], text_first+example["option2"]]
        answer_str = [text_second] * self.class_num
        label = int(example["answer"]) - 1
        return input_str, answer_str, label


class Winograd(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = load_dataset('winograd_wsc', 'wsc285')
        self.dataset_train = self.get_dataset_train(dataset['test'])
        self.dataset_valid = self.get_dataset_valid(dataset['test'])
        self.class_num = 2

    def preprocess_example(self, example):
        text_first = example["text"][:example["pronoun_loc"]]
        text_second = example["text"][example["pronoun_loc"]+len(example["pronoun"]):]
        input_str = []
        for option in example["options"]:
            input_str.append(text_first+option)
        answer_str = [text_second] * self.class_num
        label = example["label"]
        return input_str, answer_str, label