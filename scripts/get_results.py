import os, json
import numpy as np

BASE_DIR="/mnt/msranlp/shaohanh/data/fs_eval/harness/"

def acc(y_true, y_pred): 
    # print(y_true, y_pred)
    # print(len(y_true), len(y_pred))
    assert len(y_true) == len(y_pred)
    correct_predictions = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])  
    total_predictions = len(y_true)  
    accu = correct_predictions / total_predictions  
    return accu
  

def multi_acc(task, _file):
    true_preds = []
    preds = []
    with open(_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'begin validation on "valid" subset' in line:
                preds = []
            if 'target_logits is' in line:
                # import pudb; pu.db
                preds.extend(eval(line.replace('target_logits is', '')))
    if 'anli' in task:
        class_num = 3
    elif 'boolq' in task:
        class_num = 2
    elif 'rte' in task:
        class_num = 2
    elif 'wic' in task:
        class_num = 2
    elif 'winogrande' in task:
        class_num = 2
    elif 'xnli' in task:
        class_num = 3
    else:
        raise "error task"

    for i in range(0, len(preds), class_num):  
        group = preds[i:i+class_num]  
        max_index = group.index(max(group))  
        true_preds.append(max_index)  

    targets = []
    with open(os.path.join(BASE_DIR, task.replace('harness_', '')), 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if 'winogrande' in task:
                targets.append(int(obj['answer']) - 1)
            else:
                targets.append(obj['label'])
    # print("*" * 100)
    # print(len(true_preds), len(targets))
    return acc(true_preds, targets)


def arc_acc(task, _file):
    targets = []
    completion_len = []
    choices_num = []
    with open(os.path.join(BASE_DIR, task.replace('harness_', '')), 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            targets.append(obj['gold'])
            choices_num.append(len(obj['choices']))
            completion_len.extend([float(len(i)) for i in obj["choices"]])
    
    true_preds = []
    preds = []
    preds_norm = []
    with open(_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'begin validation on "valid" subset' in line:
                preds = []
            if 'target_logits is' in line:
                preds.extend(eval(line.replace('target_logits is', '')))
    
    # print(preds, completion_len)
    for i in range(len(preds)):
        preds_norm.append(preds[i] / completion_len[i])

    i = 0
    while i < len(preds):
        group = preds[i:i+choices_num[len(true_preds)]]  
        i += choices_num[len(true_preds)]
        max_index = group.index(max(group))  
        true_preds.append(max_index)  

    acc_score = acc(true_preds, targets)
    # print(true_preds, targets, preds_norm)
    true_preds = []
    i = 0
    while i < len(preds_norm):
        group = preds_norm[i:i+choices_num[len(true_preds)]]  
        i += choices_num[len(true_preds)]
        max_index = group.index(max(group))  
        true_preds.append(max_index)  

    acc_norm_score = acc(true_preds, targets)

    alpha = 1
    if 'xnli' in task or 'hendrycks' in task:
        alpha = 100

    return str(acc_score * alpha) + f'\n{task}_accnorm\t' + str(acc_norm_score * alpha)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def record_eval(task, _file):

    targets = []
    completion_len = []
    choices_num = []
    entities = []
    # import pudb; pu.db
    with open(os.path.join(BASE_DIR, task.replace('harness_', '')), 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            targets.append(obj['answers'])
            choices_num.append(len(obj['entities']))
            entities.append(obj['entities'])
    
    true_preds = []
    preds = []
    with open(_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'begin validation on "valid" subset' in line:
                preds = []
            if 'target_logits is' in line:
                preds.extend(eval(line.replace('target_logits is', '')))
    
    # print(len(preds))
    # print(sum(choices_num))
    i = 0
    while i < len(preds):
        instance_idx = len(true_preds)
        group = preds[i:i+choices_num[instance_idx]]  
        i += choices_num[instance_idx]
        max_index = group.index(max(group))  
        true_preds.append(entities[instance_idx][max_index])  

    import transformers.data.metrics.squad_metrics as squad_metrics
    f1_scores = []
    em_scores = []
    for pred, target in zip(true_preds, targets):
        f1_score = metric_max_over_ground_truths(squad_metrics.compute_f1, pred, target)
        f1_scores.append(f1_score)
        em_score = metric_max_over_ground_truths(squad_metrics.compute_exact, pred, target)
        em_scores.append(em_score)

    return str(sum(f1_scores)/len(f1_scores)) + f'\n{task}_em\t' + str(sum(em_scores)/len(em_scores))


def truthfullqa(task, _file):
    def mc1(lls, _):
            # The gold answers in `mc1_targets` are always first (index = `0`).
            return np.argmax(lls) == 0

    def mc2(lls, labels):
        # Split on the first `0` as everything before it is true (`1`).
        split_idx = list(labels).index(0)
        # Compute the normalized probability mass for the correct answer.
        ll_true, ll_false = lls[:split_idx], lls[split_idx:]
        p_true, p_false = np.exp(np.array(ll_true)), np.exp(np.array(ll_false))
        p_true = p_true / (sum(p_true) + sum(p_false))
        return sum(p_true)

    mc1_targets = []
    mc2_targets = []
    with open(os.path.join(BASE_DIR, 'truthfulqa_mc'), 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            mc1_targets.append(obj['mc1_targets']["labels"])
            mc2_targets.append(obj['mc2_targets']["labels"])

    preds = []
    with open(_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'begin validation on "valid" subset' in line:
                preds = []
            if 'target_logits is' in line:
                preds.extend(eval(line.replace('target_logits is', '')))
    scores = []
    i = 0
    while i < len(preds):
        instance_idx = len(scores)
        if 'mc1' in task:
            choice_len = len(mc1_targets[instance_idx])
        else: 
            choice_len = len(mc2_targets[instance_idx])
        lls = preds[i:i+choice_len]  
        i += choice_len
        if 'mc1' in task:
            scores.append(mc1(lls, ''))
        else:
            scores.append(mc2(lls, mc2_targets[instance_idx]))
    return sum(scores) / len(scores)

def lambada_acc(task, _file, tokenizer_name='llama'):
    targets_length = []

    if tokenizer_name == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path='huggyllama/llama-7b')
    elif tokenizer_name == 'gpt4':
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
    else:
        assert NotImplementedError

    with open(os.path.join(BASE_DIR, task.replace('harness_', '')), 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text = obj['text'].split(' ')[-1].strip()
            if tokenizer_name == 'llama':
                targets_length.append(len(tokenizer.tokenize(text)))
            elif tokenizer_name == 'gpt4':
                targets_length.append(len(tokenizer.encode(text)))
    
    preds = []
    targets = []
    with open(_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'begin validation on "valid" subset' in line:
                preds = []
                targets = []
            if 'target_logits is' in line:
                results = eval(line.replace('target_logits is', '')) # [pred, target]
                preds.extend(results[0])
                targets.extend(results[1])
    assert len(preds) == sum(targets_length)

    n = len(targets_length)
    c = 0
    b =0
    for l in targets_length:
        if preds[b: b+l] == targets[b: b+l]:
            c += 1
        b += l
        
    print(c / n)


eval_map = {
    "harness_anli_r1": multi_acc,
    "harness_anli_r2": multi_acc,
    "harness_anli_r3": multi_acc,
    "harness_arc_challenge": arc_acc, # using harness_eval criteria
    "harness_arc_easy": arc_acc, # using harness_eval criteria
    "harness_boolq": multi_acc,
    "harness_hellaswag": arc_acc,
    "harness_openbookqa": arc_acc,
    "harness_piqa": arc_acc,
    "harness_record": record_eval,    # using harness_eval criteria
    "harness_rte": multi_acc,
    "harness_truthfullqa_mc1": truthfullqa, # using harness_eval criteria
    "harness_truthfullqa_mc2": truthfullqa, # using harness_eval criteria
    "harness_wic": multi_acc,
    "harness_winogrande": multi_acc,
    "harness_arc_challenge_25s": arc_acc, # using harness_eval criteria
    "harness_hellaswag_10s": arc_acc,
    "harness_lambada_openai": lambada_acc,
}

LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]

def construct_tasks():
    tasks = {}
    for lang in LANGS:
        tasks[f"xnli_{lang}"] = multi_acc
    return tasks

xnli_tasks = construct_tasks()
eval_map.update(xnli_tasks)

# mmlu
SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

eval_map.update({f"hendrycksTest-{sub}": arc_acc for sub in SUBJECTS})

def get_acc(task, _file):
    eval_func = eval_map[task]
    return eval_func(task, _file)


def pull_data(folder, tokenizer='llama'):
    tasks = []
    tasks = ["harness_anli_r1", "harness_anli_r2", "harness_anli_r3", "harness_arc_challenge", "harness_arc_easy", "harness_boolq", "harness_hellaswag", "harness_openbookqa", "harness_piqa", "harness_record", "harness_rte", "harness_truthfullqa_mc1", "harness_truthfullqa_mc2", "harness_wic", "harness_winogrande", ]
    # tasks += ["xnli_ar", "xnli_b:qg", "xnli_de", "xnli_el", "xnli_en", "xnli_es", "xnli_fr", "xnli_hi", "xnli_ru", "xnli_sw", "xnli_th", "xnli_tr", "xnli_ur", "xnli_vi", "xnli_zh", ]
    # tasks = [
    #      "harness_arc_challenge", "harness_arc_easy", "harness_boolq", "harness_hellaswag", "harness_openbookqa", "harness_piqa", "harness_record", "harness_winogrande", ]
    # tasks = ["harness_arc_challenge", "harness_arc_easy", "harness_boolq", "harness_hellaswag", "harness_openbookqa", "harness_piqa", "harness_winogrande", ]

    # tasks = ["harness_winogrande"]
    # tasks += [f"hendrycksTest-{sub}" for sub in SUBJECTS]
    shots = [0]

    all_lines = []
    for task in tasks:
        line = task
        for shot in shots:
            if 'hendrycksTest' in task:
                _file = os.path.join(folder, f'{task}_{shot}')
            else:
                _file = os.path.join(folder, f'{task.lower()}_{shot}')

            if not os.path.exists(_file):
                continue
            acc = get_acc(task, _file)
            try:
                if 'lambada' in task:
                    acc = lambada_acc(task, _file, tokenizer_name=tokenizer)
                else:
                    acc = get_acc(task, _file)
            except:
                acc = 0
                eval_func = eval_map[task]
                if eval_func == arc_acc:
                    acc = str(acc) + f'\n{task}_accnorm\t0'
                # print(acc)
            line += '\t' + str(acc)
        print(line)
        all_lines.append(line)
    # print('\n'.join(all_lines))
    return all_lines


    
if __name__ == '__main__':
    import sys
    folder = sys.argv[1]
    pull_data(sys.argv[1])
    # for example: python eval/gpt_zs/eval.py  /mnt/localdata/msranlp_1/shaohanh/exp/unigpt_exp/gpt-medium-cc-xpos/checkpoint_1_150000_nl_zero-shot/
