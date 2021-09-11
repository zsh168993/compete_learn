from utils.dataprocess import *
from utils.utils import *
from utils.build_vocab import *

from torch.utils.data import DataLoader,TensorDataset,RandomSampler
from  model.lstm import *
from  model.cnn import *
from  model.roberta import *
from  stack.lstm_cnn_stacking import *

from  train_model.train import *

from sklearn.model_selection import train_test_split
import argparse

import os

import torch
import torch.optim as optim
import json
import torch.nn as nn
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='lstm',
                    help="model")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='data',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default="best",
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

path_dataset = 'data/datagrand_2021_train.csv'
json_path="data/params.json"

def main(model):
    # Load the parameters from json file
    args = parser.parse_args()
    args.model=model
    if args.model=="lstm" :
       json_path = os.path.join(args.model_dir, 'lstm/lstm_params.json')
       # Set the logger
       set_logger(os.path.join(args.model_dir, 'lstm/train.log'))
       params = Params(json_path)
       params.model = args.model
    elif args.model=="cnn":
        json_path = os.path.join(args.model_dir, 'cnn/cnn_params.json')
        set_logger(os.path.join(args.model_dir, 'cnn/train.log'))
        params = Params(json_path)
        params.model = args.model
    else:
        json_path= os.path.join(args.model_dir, 'stack/lstm_cnn_params.json')

        # Set the logger
        set_logger(os.path.join(args.model_dir, 'stack/train.log'))
        params= Params(json_path)
        params.model=args.model

    paths=["data/datagrand_2021_train.csv","data/datagrand_2021_test.csv"]
    vocab(paths,params)
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
    path_dataset = 'data/datagrand_2021_train.csv'
    train, train_label,params = get_data(path_dataset,params)
    params.save(json_path)
    train_x, valid_x, train_labels, valid_labels = train_test_split(train, train_label,
                                                                    test_size=0.2,
                                                                    random_state=2021,stratify = train_label)
    path_dataset = "data/datagrand_2021_test.csv"
    testdata = get_data(path_dataset, params, label_file=0)


    train_dataset=TensorDataset(torch.tensor(train), torch.tensor(train_label))
    valid_dataset = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_labels))
    train_loader = DataLoader(train_dataset,batch_size=params.batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4)


    loss_fn = nn.CrossEntropyLoss()
    if args.model == "robert":
        import datasets
        from datasets import load_dataset
        from transformers import AutoTokenizer
        model_path = r"E:\比赛\预训练\chinese_roberta_L-8_H-512"
        train_path_dataset = 'data/datagrand_2021_train.csv'#datagrand_2021_train
        test_path_dataset = "data/datagrand_2021_test.csv"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        train_datasets = load_dataset("csv", data_files=train_path_dataset)
        num=len(train_datasets["train"]["text"])
        splitnum=int(np.ceil(num * 0.8))
        train=train_datasets["train"]["text"][:splitnum]
        dev = train_datasets["train"]["text"][splitnum:]
        train_datasets = tokenizer(
            train,
            padding=True,
            truncation=True,
            max_length=50,

        )
        valid_datasets = tokenizer(
            dev,
            padding=True,
            truncation=True,
            max_length=50,

        )

        train_datasets = TensorDataset(torch.tensor(np.array(train_datasets['input_ids'])),
                             torch.tensor(np.array(train_datasets['token_type_ids'])),
                             torch.tensor(np.array(train_datasets['attention_mask'])), torch.tensor(train_label[:splitnum]))
        valid_datasets = TensorDataset(torch.tensor(np.array(valid_datasets['input_ids'])),
                             torch.tensor(np.array(valid_datasets['token_type_ids'])),
                             torch.tensor(np.array(valid_datasets['attention_mask'])), torch.tensor(train_label[splitnum:]))

        train_loader = DataLoader(train_datasets,batch_size=8,shuffle=False, num_workers=4)
        valid_loader = DataLoader(valid_datasets, batch_size=8, shuffle=False, num_workers=4)

        model = RoBerToModelClassification(model_path,clf_dropout=0.4, n_class=35)
        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(model.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            if space[0] == 'transformer':
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # # bert other module
            # {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            #  "weight_decay": 0.00, 'lr': 2e-5},
            # {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            #  "weight_decay": 0.0, 'lr': 2e-5},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.00, 'lr': 2e-3},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr':2e-3},
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        train_and_evaluate(model, train_loader, valid_loader, optimizer, loss_fn, metrics, params,
                           model_dir=r"E:\比赛\data", restore_file=None)

    elif args.model == "lstm":
        model = Net_lstm(params)
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        train_and_evaluate(model, train_loader, valid_loader, optimizer, loss_fn, metrics, params,
                           model_dir=r"E:\比赛\data", restore_file=None)

        print("finish!")
    elif args.model == "cnn":
        model = Net_cnn(params)
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        train_and_evaluate(model, train_loader, valid_loader, optimizer, loss_fn, metrics, params,
                           model_dir=r"E:\比赛\data", restore_file=None)

        print("finish!")
    elif args.model == "stacking":
        stacking(train_x, valid_x, train_labels, valid_labels, loss_fn, metrics, params,
                           model_dir=r"E:\比赛\data", restore_file=None)




if __name__ == '__main__':
    model = "robert"
    main(model)


    def to_result(model):
        # the model testdata
        import csv

        # 1. 创建文件对象
        f = open('result.csv', 'w', encoding='utf-8', newline="")

        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)

        # 3. 构建列表头
        csv_writer.writerow(["id", "label"])

        args = parser.parse_args()
        if model == "lstm":
            json_path = os.path.join(args.model_dir, 'lstm/lstm_params.json')
            params = Params(json_path)
            # use GPU if available
            params.cuda = torch.cuda.is_available()
            model = Net_lstm(params).cuda() if params.cuda else Net_lstm(params)
            load_checkpoint(os.path.join(args.model_dir + "/lstm", args.restore_file + '.pth.tar'), model)
        elif model == "cnn":
            json_path = os.path.join(args.model_dir, 'cnn/cnn_params.json')
            params = Params(json_path)
            # use GPU if available
            params.cuda = torch.cuda.is_available()
            model = Net_cnn(params).cuda() if params.cuda else Net_cnn(params)
            load_checkpoint(os.path.join(args.model_dir + "/cnn", args.restore_file + '.pth.tar'), model)

        path_dataset = "data/datagrand_2021_test.csv"
        testdata = get_data(path_dataset, params, label_file=0)

        dataset = TensorDataset(torch.tensor(testdata))
        test_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, num_workers=4)
        model.eval()

        # summary for current eval loop
        summ = []

        # compute metrics over the dataset
        num = 0
        with open(params.tag_path) as f:
            tag = json.load(f)
        for i, data_batch in enumerate(test_loader):
            # compute model output
            output_batch = model(data_batch[0])
            predicts = output_batch.max(1)

            for j in predicts[1]:
                csv_writer.writerow([num, tag[str(j.item())]])
                num = num + 1

        # 5. 关闭文件
        f.close()
    #to_result( model = model )

