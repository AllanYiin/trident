from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from trident.backend.common import get_session,addindent,get_time_suffix,get_class,format_time,get_terminal_size,snake2camel,camel2snake,get_function
from trident.backend.pytorch_ops import *

_session = get_session()
_epsilon = _session.epsilon

__all__ = ['l1_reg','l2_reg','orth_reg','get_reg','total_variation_norm_reg']


def l1_reg(model:nn.Module,reg_weight=1e-6):
    #with torch.enable_grad():
    loss =to_tensor(0.0,requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name and  not any_abnormal_number(param) and param.requires_grad==True:
            loss = loss + (reg_weight * sum(abs(param)))
    return loss


def l2_reg(model:nn.Module,reg_weight=1e-6):
    loss =to_tensor(0.0,requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name  and  not any_abnormal_number(param) and  param.requires_grad==True:
            loss=loss+ reg_weight *(param**2).sum()
    return loss


def orth_reg(model:nn.Module,reg_weight=1e-6):
    loss =to_tensor(0.0,requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name and param.requires_grad==True:
            param_flat = param.view(param.shape[0], -1)
            sym = torch.mm(param_flat, torch.t(param_flat))
            sym -= torch.eye(param_flat.shape[0])
            loss = loss + (reg_weight * sym.abs().sum())
    return loss

def total_variation_norm_reg(output:torch.Tensor,reg_weight=1):
    b,c,h,w=int_shape(output)
    total_variation_norm_reg.reg_weight=reg_weight
    assert len(output.size())==4
    count_w = output.size()[2] * (output.size()[3] - 1)
    count_h = (output.size()[2] - 1) * output.size()[3]

    diff_i = torch.sum(torch.pow(output[:, :, :, 1:] - output[:, :, :, :-1], 2))
    diff_j = torch.sum(torch.pow(output[:, :, 1:, :] - output[:, :, :-1, :], 2))
    tv_loss = true_divide(2*(true_divide(diff_i,count_w) + true_divide(diff_j,count_h)),b)
    return reg_weight * tv_loss


def get_reg(reg_name):
    if reg_name is None:
        return None
    if reg_name=='l1':
        return l1_reg
    elif reg_name=='l2':
        return l2_reg
    else:
        if '_reg' not in reg_name:
            reg_name=reg_name+'_reg'

        reg_modules = ['trident.optims.pytorch_regularizers']
        reg_fn = get_function(camel2snake(reg_name), reg_modules)
        return reg_fn