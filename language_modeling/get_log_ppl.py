import os

branch = ['0326-gpt-small-tiktoken-fullsentence-rsv1', '0508-gpt-small-tiktoken-rsv1_1', '0508-gpt-small-tiktoken-cc-2249', '0508-gpt-small-tiktoken-cc-baseline', '0326-gpt-small-tiktoken-fullsentence-rsv0-multilingual', '0517-gpt-small-tiktoken-fullsentence-mptv1']
# /mnt/msranlp/shaohanh/exp/lm_main_exp/{branch}/{ckpt}_ppl.out
# Loss: 4.75827169418335, ppl 27.06340980529785

def get_ppl():
    results = {}
    for one_branch in branch:
        print(one_branch)
        for _file in os.listdir(f'/mnt/msranlp/shaohanh/exp/lm_main_exp/{one_branch}'):
            # if _file.endswith('_ppl.out'):
            if not _file.endswith('_ppl.out') and 'ppl.out' in _file:
                with open(f'/mnt/msranlp/shaohanh/exp/lm_main_exp/{one_branch}/{_file}') as f:
                    valid_loss = 0
                    for line in f:
                        if 'Loss: ' in line:
                            valid_loss = float(line.split(',')[0].split(' ')[-1])
                    results[one_branch + '\t' + _file] = valid_loss   
                    print(one_branch + '\t' + _file, valid_loss)                 
    return results

results = get_ppl()
print(results)

