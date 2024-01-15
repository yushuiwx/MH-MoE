import json

"""
ArXiv_ftfy_cleaned_id
BookCorpus2_ftfy_cleaned_id
Books3_ftfy_cleaned_id
CC-2020-50_id_cleaned
CC-2021-04_id_cleaned
Github_ftfy_id
Gutenberg_PG-19_ftfy_cleaned_id_cleaned
NIH_ExPorter_ftfy_id
OpenWebText2_ftfy_cleaned_id
Pile-CC_id_cleaned
PubMed_Abstracts_ftfy_id
rn_dedup_shuf_cleaned_0.7_cleaned
StackExchange_ftfy_id
stories_dedup0.7_shuf_cleaned
Wikipedia_en_ftfy_id
"""
final_json = [
    # {"name": "CC", "weight": 0.5, "source":[]},
    {"name": "Github", "weight": 0.3, "source":[]},
    # {"name": "Wiki", "weight": 3, "source":[]},
    {"name": "Book", "weight": 2, "source":[]},
    {"name": "Arxiv", "weight": 1, "source":[]},
    {"name": "StackExchange", "weight": 1, "source":[]},
    {"name": "Realnews", "weight": 1, "source":[]},
]

map_dict = {
    'BookCorpus2_ftfy_cleaned_id': 'Book',
    'Books3_ftfy_cleaned_id': 'Book',
    'Gutenberg_PG-19_ftfy_cleaned_id_cleaned': 'Book',
    'rn_dedup_shuf_cleaned_0.7_cleaned': 'Realnews',
    'StackExchange_ftfy_id': 'StackExchange',
}

print('weight sum', sum([item['weight'] for item in final_json]))

mtnlg_json = open(r"C:\Users\shaohanh\Downloads\tnlg_config\json\train_1.json", 'r', encoding='utf-8')
mtnlg_json = json.load(mtnlg_json)

for item in mtnlg_json:
    item_name = item['name']
    if item_name in map_dict:
        # print(item_name)
        for final_item in final_json:
            if final_item['name'] == map_dict[item_name]:
                print(final_item['name'], item_name)
                for prev_source_item in item['source']:
                    # convert to abs path
                    final_item['source'].append(prev_source_item.replace('../tnlg', '/mnt/msranlp/shaohanh/data/tnlg'))

for final_item in final_json:
    if final_item['name'] in ['Book', 'Realnews', 'StackExchange']:
        final_item['weight'] = len(final_item['source']) * final_item['weight']

# arxiv data
for final_item in final_json:
    if final_item['name'] == 'Arxiv':
        for i in range(16843):
            final_item['source'].append(f'/mnt/msranlp/tengchao/dataset/arxiv_latex/data/{i}')
        final_item['weight'] = 16843 * final_item['weight']

# arxiv data
for final_item in final_json:
    if final_item['name'] == 'Github':
        for i in range(126237):
            final_item['source'].append('/mnt/msranlp/xingxingzhang/data/github/the-stack-dedup_jsonl_filtered_std/split{0:010d}.jsonl'.format(i))
        final_item['weight'] = 126237 * final_item['weight']

# CC-zh
# for final_item in final_json:
#     if final_item['name'] == 'CC-zh':
#         cczh_source = json.load(open(r"C:\Users\shaohanh\Downloads\data_cc100_bpe.json", "r", encoding='utf-8'))
#         # final_item['source'] = C4_source[0]['source']
#         for item in cczh_source['zh-Hans']:
#             final_item['source'].append(f'/mnt/msranlp/shaohanh/res/cc-100/shard/{item}')
wiki_source = json.load(open(r"C:\Users\shaohanh\Downloads\data_cc100_bpe.json", "r", encoding='utf-8'))
wiki_prob = json.load(open(r"C:\Users\shaohanh\Downloads\lang_prob_max_cc100-ccnet_wiki_alp0.7.json", "r", encoding='utf-8'))
for lang in wiki_source.keys():
    if lang not in wiki_prob:
        continue
    wiki_item = {"name": "CC100-" + lang, "weight": 1, "source":[]}
    for item in wiki_source[lang]:
        wiki_item['source'].append(f'/mnt/msranlp/shaohanh/res/cc-100/shard/{item}')
    wiki_item['weight'] = len(wiki_item['source']) * wiki_item['weight'] * wiki_prob[lang]
    final_json.append(wiki_item)

# Wiki
# for final_item in final_json:
    # if final_item['name'] == 'Wiki':
        # langs = 'enbg, ca, cs, da, de, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk, zh'
wiki_source = json.load(open(r"C:\Users\shaohanh\Downloads\data_wiki_bpe.json", "r", encoding='utf-8'))
wiki_prob = json.load(open(r"C:\Users\shaohanh\Downloads\lang_prob_wiki_alp0.7.json", "r", encoding='utf-8'))
for lang in wiki_source.keys():
    if lang not in wiki_prob:
        continue
    wiki_item = {"name": "Wiki-" + lang, "weight": 3, "source":[]}
    for item in wiki_source[lang]:
        wiki_item['source'].append(f'/mnt/msranlp/shaohanh/res/wiki/shard_v1/{item}')
    wiki_item['weight'] = len(wiki_item['source']) * wiki_item['weight'] * wiki_prob[lang]
    final_json.append(wiki_item)

for final_item in final_json:
    print(final_item['name'], len(final_item['source']))

writer_file = open(r"C:\Users\shaohanh\Downloads\tnlg_config\json\mv1_train.json", 'w', encoding='utf-8')
json.dump(final_json, writer_file, indent=4)