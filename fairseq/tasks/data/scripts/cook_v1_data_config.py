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
    {"name": "CC", "weight": 0.356930634, "source":[]},
    {"name": "C4", "weight": 0.291962128, "source":[]},
    {"name": "Github", "weight": 0.073802638, "source":[]},
    {"name": "Wiki", "weight": 0.019015173, "source":[]},
    {"name": "Book", "weight": 0.088737472, "source":[]},
    {"name": "Arxiv", "weight": 0.03486115, "source":[]},
    {"name": "StackExchange", "weight": 0.01386523, "source":[]},
    {"name": "Openwebtext2", "weight": 0.023768966, "source":[]},
    {"name": "Realnews", "weight": 0.043576437, "source":[]},
    {"name": "CC-zh", "weight": 0.011884483, "source":[]},
    {"name": "Wiki-others", "weight": 0.04159569, "source":[]},
]

map_dict = {
    'BookCorpus2_ftfy_cleaned_id': 'Book',
    'Books3_ftfy_cleaned_id': 'Book',
    'CC-2020-50_id_cleaned': 'CC',
    'CC-2021-04_id_cleaned': 'CC',
    'Gutenberg_PG-19_ftfy_cleaned_id_cleaned': 'Book',
    'OpenWebText2_ftfy_cleaned_id': 'Openwebtext2',
    'Pile-CC_id_cleaned': 'CC',
    'rn_dedup_shuf_cleaned_0.7_cleaned': 'Realnews',
    'StackExchange_ftfy_id': 'StackExchange',
    'stories_dedup0.7_shuf_cleaned': 'CC',
    'Wikipedia_en_ftfy_id': 'Wiki',
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

# arxiv data
for final_item in final_json:
    if final_item['name'] == 'Arxiv':
        for i in range(16843):
            final_item['source'].append(f'/mnt/msranlp/tengchao/dataset/arxiv_latex/data/{i}')

# arxiv data
for final_item in final_json:
    if final_item['name'] == 'Github':
        for i in range(126237):
            final_item['source'].append('/mnt/msranlp/xingxingzhang/data/github/the-stack-dedup_jsonl_filtered_std/split{0:010d}.jsonl'.format(i))

# C4 data
for final_item in final_json:
    if final_item['name'] == 'C4':
        C4_source = json.load(open(r"C:\Users\shaohanh\Downloads\mono-en-c4\json\train.json", "r", encoding='utf-8'))
        final_item['source'] = C4_source[0]['source']

# CC-zh
for final_item in final_json:
    if final_item['name'] == 'CC-zh':
        cczh_source = json.load(open(r"C:\Users\shaohanh\Downloads\data_cc100_bpe.json", "r", encoding='utf-8'))
        # final_item['source'] = C4_source[0]['source']
        for item in cczh_source['zh-Hans']:
            final_item['source'].append(f'/mnt/msranlp/shaohanh/res/cc-100/shard/{item}')

# Wiki-others
for final_item in final_json:
    if final_item['name'] == 'Wiki-others':
        langs = 'bg, ca, cs, da, de, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk, zh'
        wiki_source = json.load(open(r"C:\Users\shaohanh\Downloads\data_wiki_bpe.json", "r", encoding='utf-8'))
        for lang in langs.split(', '):
            for item in wiki_source[lang]:
                final_item['source'].append(f'/mnt/msranlp/shaohanh/res/wiki/shard_v1/{item}')

for final_item in final_json:
    print(final_item['name'], len(final_item['source']))

writer_file = open(r"C:\Users\shaohanh\Downloads\tnlg_config\json\v1_train.json", 'w', encoding='utf-8')
json.dump(final_json, writer_file, indent=4)