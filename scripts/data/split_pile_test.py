import json, tqdm

def split_pile_test():
    _file = r"D:\Downloads\pile_test.jsonl"
    writer_dict = {}
    with open(_file, 'r', encoding='utf-8') as fin:
        for line in tqdm.tqdm(fin):
            obj = json.loads(line)
            writer_name = obj["meta"]["pile_set_name"].split(' ')[0]
            if writer_name not in writer_dict:
                writer_dict[writer_name] = open(_file + '.' + writer_name, 'w', encoding='utf-8')
            writer_dict[writer_name].write(line)
    for writer_name in writer_dict:
        print(writer_name)
        writer_dict[writer_name].close()
    
split_pile_test()