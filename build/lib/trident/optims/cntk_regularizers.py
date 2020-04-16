from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import cntk as C


from ..backend.common import get_session,addindent,get_time_suffix,get_class,format_time,get_terminal_size,snake2camel,camel2snake,get_function

__all__ = ['l1_reg','l2_reg','orth_reg','get_reg','total_variation_norm_reg']


_device =  "cuda" if 'GPU' in str(C.all_devices()[0]) else "cpu"
_session=get_session()
_epsilon=_session.epsilon


def l1_reg(model:C.Function,reg_weight=1e-6):
    loss=0
    for param in model.parameters:
        loss = loss + (reg_weight * C.reduce_sum(C.abs(param)))
    return loss


def l2_reg(model:C.Function,reg_weight=1e-6):
    loss = 0
    for param in model.parameters:
        loss = loss + (reg_weight * C.reduce_sum(C.square(param)))
    return loss


def orth_reg(model:C.Function,reg_weight=1e-6):
    loss = 0
    for param in model.parameters:
        param_flat = C.reshape(param, -1)
        sym =C.times_transpose(param_flat,param_flat)
        sym -= C.eye_like(param_flat)
        loss =loss + (reg_weight * C.reduce_sum(C.abs(sym)))
    return loss



def total_variation_norm_reg(output:C.Variable,reg_weight=1e-6):
    return reg_weight * (C.reduce_sum(C.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) + C.reduce_sum(C.abs(output[:, :, :-1, :] - output[:, :, 1:, :])))


def get_reg(reg_name):
    if reg_name is None:
        return None
    if '_reg' not in reg_name:
        reg_name=reg_name+'_reg'
    reg_modules = ['trident.optims.pytorch_regularizers']
    reg_fn = get_function(camel2snake(reg_name), reg_modules)
    return reg_fn