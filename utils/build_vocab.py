
import tqdm
import os
import sys
from collections import Counter
import numpy as np
from utils.utils import *
def vocab(paths,params, pad='<PAD>', unknown='<UNK>',vocab_path="data/vocab.json"):
    vocab = [pad, unknown]
    data=[]
    for path_dataset in paths:
        with open(path_dataset, encoding='utf-8') as fp:
            for line in tqdm.tqdm(fp, desc='Tokenizing'):
                line = line.split(",")
                if line[0] == "id":
                    continue
                data.append(line[1])
    counter = Counter(token for t in data for token in set(t.split(" ")))
    for word, freq in sorted(counter.items(), reverse=True):
        vocab.append(word)
    #vocab=np.asarray(vocab)
    vocab = {word: i for i, word in enumerate(vocab)}
    params.vocab_size = len(vocab)
    params.vocab_path=vocab_path

    return save_dict_to_json(vocab,vocab_path)