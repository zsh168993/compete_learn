"""Defines the neural network, losss function and metrics"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





class Net_cnn(nn.Module):

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


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


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
metrics = {
    'accuracy': accuracy,
    "acc_f1_macro":acc_f1_macro
    # could add more metrics such as accuracy for each token type
}
