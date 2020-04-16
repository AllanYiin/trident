from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..backend.common import  get_session,get_function,camel2snake,snake2camel
import numpy as np
import tensorflow as tf


__all__ = ['l1_reg','l2_reg','orth_reg','get_reg','total_variation_norm_reg']


_device = "cuda" if tf.test.is_gpu_available() else "cpu"
_session=get_session()
_epsilon=_session.epsilon


def l1_reg(model:tf.keras.Model,reg_weight=1e-6):
    loss=0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            loss = loss + (reg_weight * torch.sum(torch.abs(param)))
    return loss


def l2_reg(model:tf.keras.Model,reg_weight=1e-6):
    loss = 0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            loss = loss + (0.5 * reg_weight * torch.sum(torch.pow(param, 2)))
    return loss


def orth_reg(model:tf.keras.Model,reg_weight=1e-6):
    loss = 0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param_flat = param.view(param.shape[0], -1)
            sym = torch.mm(param_flat, torch.t(param_flat))
            sym -= torch.eye(param_flat.shape[0])
            loss = loss + (reg_weight * sym.abs().sum())
    return loss

def total_variation_norm_reg(output:tf.Tensor,reg_weight=1e-6):
    return reg_weight * (tf.reduce_sum(tf.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) + tf.reduce_sum(tf.abs(output[:, :, :-1, :] - output[:, :, 1:, :])))


def get_reg(reg_name):
    if reg_name is None:
        return None
    if '_reg' not in reg_name:
        reg_name=reg_name+'_reg'
    reg_modules = ['trident.optims.pytorch_regularizers']
    reg_fn = get_function(camel2snake(reg_name), reg_modules)
    return reg_fn