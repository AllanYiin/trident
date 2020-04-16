from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..backend.common import  get_session
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['max_norm','non_neg','unit_norm','min_max_norm','maxnorm','nonneg','unitnorm','minmaxnorm','get_constraints']


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_session=get_session()
_epsilon=_session.epsilon


def l1_reg(model,loss):
    with torch.enable_grad():
        reg =_epsilon
        for name, param in model.named_parameters():
            if 'bias' not in name:
                loss = loss + (reg * torch.sum(torch.abs(param)))

def l2_reg(model,loss):
    with torch.enable_grad():
        reg =_epsilon
        for name, param in model.named_parameters():
            if 'bias' not in name:
                loss = loss + (0.5 * reg * torch.sum(torch.pow(param, 2)))

def orth_reg(model,loss):
    with torch.enable_grad():
        reg =_epsilon
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0])
                loss = loss + (reg * sym.abs().sum())
