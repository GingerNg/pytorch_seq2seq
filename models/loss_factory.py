import torch.nn as nn
from losses.focal_loss import FocalLoss
from losses.contrastive_loss import ContrastiveLoss
from utils.model_utils import use_cuda, device


def cross_entropy_loss():
    criterion = nn.CrossEntropyLoss()  # obj
    return criterion


def focal_loss():
    criterion = FocalLoss(4)  # obj
    return criterion


def binarg_loss():
    # return nn.BCEWithLogitsLoss()
    return nn.BCELoss()


def contrastive_loss():
    return ContrastiveLoss()


def smb_loss(detection_output, correct_output, detection_label, correct_label, coefficient=0.1):
    """[overall objective]

    Args:
        detection_output ([type]): [description]
        correct_output ([type]): [description]
        detection_label ([type]): [description]
        correct_label ([type]): [description]
        coefficient (float, optional): [[0; 1]]. Defaults to 0.5.

    Returns:
        [type]: [description]s
    """
    # if use_cuda:
    #     coefficient = coefficient.to(device)
    # print(detection_output.shape, detection_label.shape)
    detection_criterion = cross_entropy_loss()
    correct_criterion = cross_entropy_loss()
    # x_input = x_input.view(x_input.size(0) * x_input.size(1), x_input.size(2))
    detection_loss = detection_criterion(
        detection_output.view(detection_output.size(0) * detection_output.size(1), detection_output.size(2)),
        detection_label.view(detection_label.size(0) * detection_label.size(1))
    )
    correct_loss = correct_criterion(
        correct_output.view(correct_output.size(0) * correct_output.size(1), correct_output.size(2)),
        correct_label.view(correct_label.size(0) * correct_label.size(1))
        )
    return coefficient*detection_loss + (1-coefficient)*correct_loss
    # return detection_loss
    # return correct_loss
