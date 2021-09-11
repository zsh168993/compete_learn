# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from  train_model.train import *
from train_model.evaluate import evaluate

class BasicModel(nn.Module):
    """Parent class of basic models"""

    def train_k(self, optimizer, loss_fn, train_loader, metrics, params):
        """return a trained model and eval metric o validation data"""
        pass

    def predict(self, model, x_test):
        """return the predicted result"""
        pass

    def get_oof(self,x_train, y_train,x_test,y_test,optimizer,params,loss_fn, metrics,model_dir,n_folds=5):
        """K-fold stacking"""
        num_train, num_test = x_train.shape[0], x_test.shape[0]
        oof_train = np.zeros((num_train,params.tag_len))
        oof_test = np.zeros((num_test,params.tag_len))
        oof_test_all_fold = np.zeros((num_test,n_folds, params.tag_len))

        KF = KFold(n_splits=n_folds, random_state=2021,shuffle=True)
        for i, (train_index, val_index) in enumerate(KF.split(x_train)):
            print('{0} fold, train {1}, val {2}'.format(i, len(train_index), len(val_index)))
            x_tra, y_tra = x_train[train_index], y_train[train_index]
            x_val, y_val = x_train[val_index], y_train[val_index]
            # data
            train_dataset = TensorDataset(torch.tensor(x_tra), torch.tensor(y_tra))
            valid_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
            train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4)
            valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4)
            for epoch in range(1):
                 self.train_k(optimizer, loss_fn, train_loader, metrics, params)
            x= self.predict(loss_fn, valid_loader, metrics, params)
            oof_train[val_index]=x

            test_dataset = TensorDataset(torch.tensor(x_test),torch.tensor(y_test))
            test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4)
            oof_test_all_fold[:,i, :] = self.predict(loss_fn, test_loader, metrics, params,falg=0)
        oof_test = np.mean(oof_test_all_fold, axis=1)

        return oof_train, oof_test


# create two models for first-layer stacking: xgb and lgb
class Net_lstm(BasicModel):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net_lstm, self).__init__()

        # the embedding takes as input the vocab_size and the embedding_dim
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim, batch_first=True,bidirectional=True)

        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Linear(2*params.lstm_hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        self.padding_idx=params.padding_idx
        self.attention = nn.Linear(2*params.lstm_hidden_dim, params.tag_len, bias=False)
    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x seq_len x embedding_dim

        s = s.to(torch.int64)
        emb = self.embedding(s)
        emb = self.dropout(emb)
        lengths, masks = (s != self.padding_idx).sum(dim=-1), s != self.padding_idx
        inputs,lengths,masks=emb[:, :lengths.max()], lengths, masks[:, :lengths.max()]

        # run the LSTM along the sentences of length seq_len
        # dim: batch_size x seq_len x lstm_hidden_dim
        idx = torch.argsort(lengths, descending=True)  # 40
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs[idx], lengths[idx],
                                                          batch_first=True)  # packed_inputs[0](2182,300) packed_inputs[1]112
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            self.lstm(packed_inputs)[0], batch_first=True)  # outputs(40,112,512)
        inputs=self.dropout(outputs[torch.argsort(idx)])#N, L, hidden_size * 2
        #标签注意力
        masks = torch.unsqueeze(masks, 1)  # N, 1, L # self.attention,   hiddensize * 2, numtag
        # attention = self.attention(inputs).transpose(1, 2).masked_fill(1.0 - masks, -np.inf)  # N, labels_num, L
        attention = self.attention(inputs).transpose(1, 2).masked_fill(~masks, -np.inf)  # N, labels_num, L 理解#https://zhuanlan.zhihu.com/p/151783950
        attention = F.softmax(attention, -1)
        s=attention @ inputs  # N, labels_num, 2*hidden_size
        # make the Variable contiguous in memory (a PyTorch artefact)


        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)
        s=torch.squeeze(s, -1)

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags

    def train_k(self, optimizer, loss_fn, train_loader, metrics, params):


        train(self, optimizer, loss_fn, train_loader, metrics, params)

    def predict(self, loss_fn, data_iterator, metrics, params,falg=0):
        print('test with Net_lstm model')

        return  evaluate(self, loss_fn, data_iterator, metrics, params,falg)


class Net_cnn(BasicModel):

    def __init__(self, params):

        super(Net_cnn, self).__init__()

        self.is_training = True
        self.dropout_rate = params.dropout_rate
        self.tag_len = params.tag_len
        self.padding_idx=params.padding_idx
        self.config = params

        self.embedding = nn.Embedding(num_embeddings=params.vocab_size,
                                      embedding_dim=params.embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(

                          nn.Conv1d(in_channels=params.embedding_dim,
                                    out_channels=params.feature_size,
                                    kernel_size=h),
                          nn.BatchNorm1d(num_features=params.feature_size),#num_features： 来自期望输入的特征数，C from an expected input of size (N,C,L) or L from input of size (N,L)
                          nn.ReLU(),#64 * feature_size 50 * params.sentence_length - h + 1
                          nn.MaxPool1d(kernel_size=params.sentence_length - h + 1))
                          #nn.MaxPool1d(kernel_size=2,stride=1))#构建一个卷积核大小为1 x params.sentence_length - h + 1
                                                     #out:64 * 50 * (params.sentence_length - h + 1)+2-1
            for h in params.window_sizes
        ])
        # size=0
        # for h in params.window_sizes:
        #     size=size+params.feature_size*(params.sentence_length - h)
        self.fc = nn.Linear(in_features=params.feature_size*len(params.window_sizes),
                            out_features=params.tag_len)

    def forward(self, s):
        s = s.to(torch.int64)
        embed_x = self.embedding(s)
        out = F.dropout(input=embed_x, p=self.dropout_rate)
        embed_x = embed_x.permute(0, 2, 1)
        # print('embed size 2',embed_x.size())  # 32*256*35
        out = [conv(embed_x) for conv in self.convs]  # out[i]:batch_size x feature_size*1
        # for o in out:
        #    print('o',o.size())  # 32*100*1
        out = torch.cat(out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        # print(out.size(1)) # 32*400*1
        #out = out.view(out.size(0), -1)
        # print(out.size())  # 32*400
        out = out.view(-1, out.size(1))
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out)


        return F.log_softmax(out, dim=1)   # dim: batch_size*seq_len x num_tags

    def train_k(self, optimizer, loss_fn, train_loader, metrics, params):
        train(self, optimizer, loss_fn, train_loader, metrics, params)

    def predict(self, loss_fn, data_iterator, metrics, params,falg=0):
        print('test with Net_cnn model')

        return  evaluate(self, loss_fn, data_iterator, metrics, params,falg)

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(np.sum(mask))

def acc_f1_macro(outputs, labels):
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    f1 = f1_score(labels, outputs, average="macro")

    return f1
# get output of first layer models and construct as input for the second layer
def stacking(train_x, valid_x, train_labels, valid_labels, loss_fn, metrics, params,
                           model_dir=r"E:\比赛\data", restore_file=None):


    # lstm_classifier = Net_lstm(params)
    # optimizer = optim.Adam(lstm_classifier.parameters(), lr=params.learning_rate)
    # lstm_oof_train, lstm_oof_test = lstm_classifier.get_oof(train_x, train_labels,valid_x,valid_labels,optimizer,params,loss_fn, metrics,model_dir)
    # print(lstm_oof_train.shape, lstm_oof_test.shape)
    #
    # print("lstm",acc_f1_macro(lstm_oof_test, valid_labels))

    cnn_classifier = Net_cnn(params)
    optimizer = optim.Adam(cnn_classifier.parameters(), lr=params.learning_rate)
    cnn_oof_train, cnn_oof_test = cnn_classifier.get_oof(train_x, train_labels,valid_x,valid_labels,optimizer,params,loss_fn, metrics,model_dir)


    print(cnn_oof_train.shape, cnn_oof_test.shape)

    print("cnn", acc_f1_macro(cnn_oof_test, valid_labels))
    input_train = [cnn_oof_train, cnn_oof_train]
    input_test = [cnn_oof_test, cnn_oof_test]

    stacked_train = np.concatenate([f for f in input_train], axis=1)
    stacked_test = np.concatenate([f for f in input_test], axis=1)
    print(stacked_train.shape, stacked_test.shape)



    # use LR as the model of the second layer
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

    # split for validation
    n = int(stacked_train.shape[0] * 0.8)
    x_tra, y_tra = stacked_train[:n], train_labels[:n]
    x_val, y_val = stacked_train[n:], train_labels[n:]
    model = LinearRegression()
    model.fit(x_tra, y_tra)
    y_pred = model.predict(x_val)
    y_val = y_val.ravel()
    y_pred = y_pred.astype(np.int32)
    print(f1_score(y_pred, y_val, average="macro"))


    # predict on test data
    final_model = LinearRegression()
    final_model.fit(stacked_train, train_labels)
    test_prediction = final_model.predict(stacked_test)
    valid_labels=valid_labels.ravel()
    test_prediction = test_prediction.astype(np.int32)
    print(f1_score(test_prediction, valid_labels, average="macro"))

