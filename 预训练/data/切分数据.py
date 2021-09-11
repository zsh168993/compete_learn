
import json
from collections import Counter
paths=r"D:/datagrand_2021_unlabeled_data.json"
tokenized_path=r"E:\比赛\预训练\data\test_datagrand_2021_unlabeled_data.txt"
count=0
num=0
data=[]
pad='[PAD]'
unknown='[UNK]'
vocab_path="vocab.txt"
vocab = [pad, unknown]
counter = Counter()#token for t in data for token in set(t.split(" "))
with open(paths, encoding='utf-8') as fp, open(tokenized_path, 'w', encoding='utf-8') as fout:
    for line in open(paths, 'r'):
        line = json.loads(line)
        title = line["title"]
        content = line["content"]
        counter.update(set(title.split(" ")))
        counter.update(set(content.split(" ")))
        line=title+content
        x=line.split()
        num=num+len(x)

        print(line, file=fout)
        count=count+1
        if count==20000:#1000000 4.9g
            break


for word, freq in sorted(counter.items(), reverse=True):
    vocab.append(word)
# vocab=np.asarray(vocab)
vocab = {word: i for i, word in enumerate(vocab)}
with open(vocab_path, "w", encoding='utf-8') as fp:
    for i in vocab:
        fp.write(i)
        fp.write("\n")

print(num/count)