import sys
import os
import json
import argparse
import csv
import math
import numpy as np
from tabulate import tabulate

parser = argparse.ArgumentParser()

# common args
parser.add_argument('--prefix', type=str)
parser.add_argument('--folders', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--step', type=int, default=300)
parser.add_argument('--model', type=str, default='base')

args = parser.parse_args()

folders = args.folders.split(',')
tasks = ['hellaswag', 'winogrande', 'piqa']

def get_fewshot(f):
    def get_from_line(line, metric):
        ss = line.split('|')
        for s in ss:
            ts = s.strip().split(' ')
            if ts[0] == metric:
                return float(ts[1])
        return -1

    def get_from_file(filename, keyword, metric):
        try:
            if not os.path.isfile(filename):
                print('{} not found'.format(filename))
                return -1
            with open(filename, "r") as input:
                ret = -1
                for line in input.readlines():
                    if keyword in line and 'valid on' in line:
                        cur_res = get_from_line(line, metric)
                        ret = max(cur_res, ret)
                if ret <= 0 or ret != ret:
                    print("{}: accuracy is {}".format(filename, ret))
                    ret = 0
                return ret
        except Exception as e:
            return -1

    keyword = "valid"
    metric = "accuracy"
    result_dict = {}

    for exp in folders:
        result_dict[exp] = {}
        log_dir = os.path.join(args.prefix, exp, f"checkpoint_1_{args.step}000-eval")
        for task in tasks:
            result_dict[exp][task] = get_from_file(os.path.join(log_dir, f"{task}.log"), keyword, metric)
    
    return result_dict


def print_summary_table(r_fewshot):
    tb_list = []
    for exp in folders:
        line = [exp] + [r_fewshot[exp][task] for task in tasks]
        line += [sum(r_fewshot[exp].values()) / len(tasks)]
        tb_list.append(line)
    print(tabulate(tb_list, headers=['Run'] + [task for task in tasks] + ['Avg']))


with open(args.output, 'w') as f:
    r_fewshot = get_fewshot(f)
    print_summary_table(r_fewshot)