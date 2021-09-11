import torch
import numpy as np
from torch import nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RoBerToModelClassification(nn.Module):
    def __init__(self, model_name, clf_dropout=0.4, n_class=8):
        super(RoBerToModelClassification, self).__init__()
        self.transformer = BertModel.from_pretrained(model_name,
                                                     hidden_dropout_prob=0.1)
        self.bert_config = self.transformer.config
        self.dropout = nn.Dropout(clf_dropout)
        self.linear = nn.Linear(self.bert_config.hidden_size, n_class)
        self._init_weights([self.linear], initializer_range=self.bert_config.initializer_range)

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        output= self.transformer(input_ids, position_ids, token_type_ids)
        hidden_states=output[0]
        # avg_pool = torch.mean(hidden_states, 1)
        # max_pool, _ = torch.max(hidden_states, 1)
        # h_conc = torch.cat((avg_pool, max_pool, hidden_states[:, 0, :]), 1)
        pooled_output = self.dropout(hidden_states)
        avg_output = torch.mean(pooled_output, 1).view(-1, self.bert_config.hidden_size)
        logits = self.linear(self.dropout(avg_output))
        return logits

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
