"""Read, split and save the kaggle dataset for our model"""

import tqdm
import json
import sys
from collections import Counter
import numpy as np
from sklearn import preprocessing
from utils.utils import *

def truncate_text(texts, max_len=500, padding_idx=0, unknown_idx=1, unknown='<UNK>'):
    if max_len is None:
        return texts

    texts = np.asarray([list(x[:max_len]) + [padding_idx] * (max_len - len(x)) for x in texts])
    texts[(texts == padding_idx).all(axis=1), 0] = unknown_idx
    return texts
def label_pro(label):
    le = preprocessing.LabelEncoder()
    le.fit(label)
    return le
def get_data(path_dataset, params,label_file=1):
    data=[]
    label=[]
    if label_file==1:
        with open(path_dataset, encoding='utf-8') as fp:
            for line in tqdm.tqdm(fp, desc='Tokenizing'):
                line=line.split(",")
                if line[0]=="id":
                    continue
                data.append(line[1])
                label.append(line[2].strip("\n"))
            with open(params.vocab_path) as f:
                 vocab=json.load(f)
            text = np.asarray([[vocab.get(word, vocab['<UNK>']) for word in line.split(" ")]
                               for line in data])

            data=truncate_text(text,params.sentence_length,params.padding_idx)

            le=label_pro(label)
            label=le.transform(label)
            tags = {i: word for i, word in enumerate(le.classes_)}
            params.tag_path="data/tag.json"
            save_dict_to_json(tags,params.tag_path)
            params.tag_len=len(le.classes_)
            return data,label,params
    else:
        with open(path_dataset, encoding='utf-8') as fp:
            for line in tqdm.tqdm(fp, desc='Tokenizing'):
                line = line.split(",")
                if line[0] == "id":
                    continue
                data.append(line[1])
            with open(params.vocab_path) as f:
                 vocab=json.load(f)
            text = np.asarray([[vocab.get(word, vocab['<UNK>']) for word in line.split(" ")]
                               for line in data])
            data = truncate_text(text, params.sentence_length, params.padding_idx)
            return data
if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/datagrand_2021_train.csv'
    tokenized_path="data/1.txt"

