import os
import sys
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import numpy as np
from ..backend.common import get_session,addindent,get_time_prefix,get_class
from ..backend.pytorch_backend import *



# def accuracy(input, target,axis=1):
#     input_tensor=input.clone().detach()
#     target_tensor=target.clone().detach()
#     if input_tensor.dtype!=torch.int64:
#         input_tensor=argmax(input_tensor,axis).squeeze()
#     if target_tensor.dtype!=torch.int64:
#         target_tensor=argmax(target_tensor,axis).squeeze()
#     if input_tensor.shape!=target_tensor.shape:
#         raise  ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape,target_tensor.shape))
#     else:
#         return input_tensor.eq(target_tensor).float().mean()


# 计算准确度

def accuracy(output, target, topk=1,axis=1):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    input_tensor=output.clone().detach()
    target_tensor=target.clone().detach()
    if input_tensor.dtype!=torch.int64:
        input_tensor=argmax(input_tensor,axis).squeeze()
    if target_tensor.dtype!=torch.int64:
        target_tensor=argmax(target_tensor,axis).squeeze()
    if input_tensor.shape!=target_tensor.shape:
        raise  ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape,target_tensor.shape))

    batch_size = target_tensor.size(0)
    if len(target_tensor.size())>=3 or topk==1:
        return input_tensor.eq(target_tensor).float().mean()
    else:
        _, pred = input_tensor.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target_tensor.view(1, -1).expand_as(pred))
        res = []

        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(1 / batch_size)

