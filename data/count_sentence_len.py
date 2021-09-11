import tqdm
paths="E:\比赛\data\datagrand_2021_train.csv"
leng=0
num=0
with open(paths, encoding='utf-8') as fp:
    for line in tqdm.tqdm(fp, desc='Tokenizing'):
        x=line.split()
        leng=leng+len(x)
        num=num+1

print(leng/num)