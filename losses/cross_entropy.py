import torch
from torch import nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, fc_out, label):
        one_hot_lable = torch.FloatTensor(fc_out.shape[0], 3)
        one_hot_lable.zero_()
        one_hot_lable.scatter_(1, torch.reshape(label, (fc_out.shape[0], 1)), 1)
        loss = one_hot_lable * torch.softmax(fc_out, 1)
        loss = -torch.sum(torch.log(torch.sum(loss, 1)))/fc_out.shape[0]

        return loss