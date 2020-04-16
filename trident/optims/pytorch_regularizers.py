from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backend.common import get_session,addindent,get_time_suffix,get_class,format_time,get_terminal_size,snake2camel,camel2snake,get_function

__all__ = ['l1_reg','l2_reg','orth_reg','get_reg']


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_session=get_session()
_epsilon=_session.epsilon


def l1_reg(model,loss,weight=1e-6):
    with torch.enable_grad():
        for name, param in model.named_parameters():
            if 'bias' not in name:
                loss = loss + (weight * torch.sum(torch.abs(param)))

def l2_reg(model,loss,weight=1e-6):
    with torch.enable_grad():
        for name, param in model.named_parameters():
            if 'bias' not in name:
                loss = loss + (0.5 * weight * torch.sum(torch.pow(param, 2)))


def orth_reg(model,loss,weight=1e-6):
    with torch.enable_grad():
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0])
                loss = loss + (weight * sym.abs().sum())



def get_reg(reg_name):
    if reg_name is None:
        return None
    if '_reg' not in reg_name:
        reg_name=reg_name+'_reg'
    reg_modules = ['trident.optimizers.pytorch_regularizers']
    reg_fn = get_function(camel2snake(reg_name), reg_modules)
    return reg_fn