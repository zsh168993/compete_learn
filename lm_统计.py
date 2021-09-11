from collections import Counter
import numpy as np


"""语料"""
corpus = '''她的菜很好 她的菜很香 她的他很好 他的菜很香 他的她很好
很香的菜 很好的她 很菜的他 她的好 菜的香 他的菜 她很好 他很菜 菜很好'''.split()


"""语料预处理"""
counter = Counter()  # 词频统计
for sentence in corpus:
    for word in sentence:
        counter[word] += 1
counter = counter.most_common()
lec = len(counter)
word2id = {counter[i][0]: i for i in range(lec)}
id2word = {i: w for w, i in word2id.items()}


"""N-gram建模训练"""
unigram = np.array([i[1] for i in counter]) / sum(i[1] for i in counter)

bigram = np.zeros((lec, lec)) + 1e-8
for sentence in corpus:
    sentence = [word2id[w] for w in sentence]
    for i in range(1, len(sentence)):
        bigram[[sentence[i - 1]], [sentence[i]]] += 1
for i in range(lec):
    bigram[i] /= bigram[i].sum()


"""句子概率"""
def prob(sentence):
    s = [word2id[w] for w in sentence]
    les = len(s)
    if les < 1:
        return 0
    p = unigram[s[0]]
    if les < 2:
        return p
    for i in range(1, les):
        p *= bigram[s[i - 1], s[i]]
    return p

print('很好的菜', prob('很好的菜'))
print('菜很好的', prob('菜很好的'))
print('菜好的很', prob('菜好的很'))


"""排列组合"""
def permutation_and_combination(ls_ori, ls_all=None):
    ls_all = ls_all or [[]]
    le = len(ls_ori)
    if le == 1:
        ls_all[-1].append(ls_ori[0])
        ls_all.append(ls_all[-1][: -2])
        return ls_all
    for i in range(le):
        ls, lsi = ls_ori[:i] + ls_ori[i + 1:], ls_ori[i]
        ls_all[-1].append(lsi)
        ls_all = permutation_and_combination(ls, ls_all)
    if ls_all[-1]:
        ls_all[-1].pop()
    else:
        ls_all.pop()
    return ls_all

print('123排列组合', permutation_and_combination([1, 2, 3]))


"""给定词组，返回最大概率组合的句子"""
def max_prob(words):
    pc = permutation_and_combination(words)  # 生成排列组合
    p, w = max((prob(s), s) for s in pc)
    return p, ''.join(w)

print(*max_prob(list('香很的菜')))
print(*max_prob(list('好很的他菜')))
print(*max_prob(list('好很的的她菜')))
