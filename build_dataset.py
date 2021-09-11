"""Read, split and save the kaggle dataset for our model"""

import tqdm
import os
import sys



if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/datagrand_2021_train.csv'
    tokenized_path="data/1.txt"
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    with open(path_dataset, encoding='utf-8') as fp,open(tokenized_path, 'w',encoding='utf-8') as fout:
        for line in tqdm.tqdm(fp, desc='Tokenizing'):
            print(line, file=fout)
    print("- done.")

