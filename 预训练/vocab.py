
import tqdm
import os
import json
from collections import Counter
import numpy as np
def vocab(paths, pad='[PAD]', unknown='[UNK]',vocab_path="data/vocab.txt"):
    vocab = [pad, unknown]
    data=[]

    for line in open(paths, 'r'):
        line=json.loads(line)

        title=line["title"]
        content = line["content"]

        data.append(title)
        data.append(content)
    counter = Counter(token for t in data for token in set(t.split(" ")))
    for word, freq in sorted(counter.items(), reverse=True):
        vocab.append(word)
    #vocab=np.asarray(vocab)
    vocab = {word: i for i, word in enumerate(vocab)}
    with open(vocab_path,"w",encoding='utf-8') as fp:
         for i in vocab:
             fp.write(i)
             fp.write("\n")


    return 0
if __name__ == "__main__":
    vocab(paths="data/datagrand_2021_unlabeled_data.txt")