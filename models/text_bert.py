from nlp_tools.tokenizers import WhitespaceTokenizer
from torch import nn
import logging
from transformers import BertModel
from utils.model_utils import use_cuda, device
import numpy as np
# from cfg import bert_path
import torch.nn.functional as F

# build word encoder
dropout = 0.15


class BertSoftmaxModel(nn.Module):
    def __init__(self, bert_path, label_encoder):
        super(BertSoftmaxModel, self).__init__()
        self.all_parameters = {}
        parameters = []
        self.dropout = nn.Dropout(dropout)

        self.tokenizer = WhitespaceTokenizer(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        bert_parameters = self.get_bert_parameters()

        self.dense = nn.Linear(768, label_encoder.label_size, bias=True)
        parameters.extend(
            list(filter(lambda p: p.requires_grad, self.dense.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        self.all_parameters["bert_parameters"] = bert_parameters
        self.pooled = False
        logging.info('Build Bert encoder with pooled {}.'.format(self.pooled))

    def encode(self, tokens):
        tokens = self.tokenizer.tokenize(tokens)
        return tokens

    def get_bert_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def forward(self, batch_inputs):
        input_ids, token_type_ids = batch_inputs

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids)

        # if self.training:
        #     sequence_output = self.dropout(sequence_output)

        out = self.dense(sequence_output)

        score = out
        # score = F.softmax(out, dim=-1)  # dim=-1： 对最后一维进行softmax
        # print("score:{}".format(score.shape))
        # print(score[0,:,:])
        score = score.view(score.shape[0] * score.shape[1], score.shape[2])
        return score