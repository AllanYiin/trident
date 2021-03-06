from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from trident.backend.common import get_session, addindent, get_time_suffix, get_class, get_function, camel2snake
from trident.backend.pytorch_ops import *
from trident.data.mask_common import mask2trimap

__all__ = ['accuracy','recall','pixel_accuracy','alpha_pixel_accuracy','iou','psnr','mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','mae','mse','rmse','msle','get_metric']

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






@torch.no_grad()
def accuracy(output, target, topk=1,axis=1,ignore_index=-100, exclude_mask=False):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    input_tensor=output.copy().detach()
    target_tensor=target.copy().detach()
    num_classes = int_shape(output)[axis]
    if len(input_tensor)==0:
        return to_tensor(0.0)


    is_logsoftmax = None
    from_logits = None
    output_exp = exp(input_tensor)
    if (ndim(input_tensor) >= 1 and 'float' in str(input_tensor.dtype) and input_tensor.min() >= 0 and input_tensor.max() <= 1):
        is_logsoftmax = False
        from_logits = True
        input_tensor = clip(input_tensor, min=1e-8, max=1 - 1e-8)

    elif (ndim(output_exp) >= 1 and 'float' in str(output_exp.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1):
        is_logsoftmax = True
        from_logits = True
        input_tensor =  clip(output_exp, min=1e-8, max=1 - 1e-8)
    else:
        is_logsoftmax = False
        from_logits = False

    if input_tensor.dtype!=torch.int64 and topk==1:
        if len(input_tensor.size())==1: #binary
            input_tensor=input_tensor.gt(0.5).float()
        else:
            input_tensor=argmax(input_tensor,axis).squeeze()
    if target_tensor.dtype!=torch.int64:
        target_tensor=argmax(target_tensor,axis).squeeze()
    if input_tensor.shape!=target_tensor.shape and topk==1:
        raise  ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape,target_tensor.shape))

    input_mask=ones_like(input_tensor)
    if isinstance(ignore_index, int) and 0 <= ignore_index < num_classes:
        input_mask[input_tensor==ignore_index] = 0
    elif isinstance(ignore_index, (list, tuple)):
        for idx in ignore_index:
            if isinstance(idx, int) and 0 <= idx < int_shape(output)[axis]:
                input_mask[input_tensor == idx] = 0

    batch_size = target_tensor.size(0)
    if topk==1:
        return (input_tensor.eq(target_tensor).float()*input_mask).sum()/clip((input_mask).float().sum(),min=1)
    else:
        _, pred = input_tensor.topk(topk)
        pred = pred.t()
        correct = pred.eq(target_tensor.reshape((1, -1)).expand_as(pred))
        correct_k = reduce_sum(correct[:topk].reshape(-1).float(),axis=0,keepdims=True)
        return correct_k.mul_(1 / batch_size)



@torch.no_grad()
def recall(output, target, axis=1,ignore_index=-100):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    input_tensor=output.copy().detach()
    target_tensor=target.copy().detach()
    num_classes = int_shape(output)[axis]


    is_logsoftmax = None
    from_logits = None
    output_exp = exp(input_tensor)
    if (ndim(input_tensor) >= 1 and 'float' in str(input_tensor.dtype) and input_tensor.min() >= 0 and input_tensor.max() <= 1):
        is_logsoftmax = False
        from_logits = True
        input_tensor = clip(input_tensor, min=1e-8, max=1 - 1e-8)

    elif (ndim(output_exp) >= 1 and 'float' in str(output_exp.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1):
        is_logsoftmax = True
        from_logits = True
        input_tensor =  clip(output_exp, min=1e-8, max=1 - 1e-8)
    else:
        is_logsoftmax = False
        from_logits = False

    if input_tensor.dtype!=torch.int64 :
        if len(input_tensor.size())==1: #binary
            input_tensor=input_tensor.gt(0.5).float()
        else:
            input_tensor=argmax(input_tensor,axis).squeeze()
    if target_tensor.dtype!=torch.int64:
        target_tensor=argmax(target_tensor,axis).squeeze()
    if input_tensor.shape!=target_tensor.shape :
        raise  ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape,target_tensor.shape))
    target_mask=ones_like(target_tensor)
    if isinstance(ignore_index, int) and 0 <= ignore_index < num_classes:
        target_mask[target_tensor==ignore_index] = 0
    elif isinstance(ignore_index, (list, tuple)):
        for idx in ignore_index:
            if isinstance(idx, int) and 0 <= idx < int_shape(output)[axis]:
                target_mask[target_tensor == idx] = 0

    batch_size = target_tensor.size(0)
    return (input_tensor.eq(target_tensor).float()*target_mask).sum()/clip((target_mask).float().sum(),min=1)




@torch.no_grad()
def psnr(output, target):
    input_tensor = output.clone().detach()
    target_tensor = target.clone().detach()
    if input_tensor.shape != target_tensor.shape :
        raise ValueError(
            'input shape {0} is not competable with target shape {1}'.format(input_tensor.shape, target_tensor.shape))

    max_value = 255
    target_np = to_numpy(target_tensor)
    if  target_np.min() <0 :
        target_tensor = (target_tensor + 1) * 0.5
        input_tensor = (input_tensor + 1) * 0.5
        max_value = 1
    elif 0 <= target_np.min() <= 1 and 0 <= target_np.max() <= 1:
        max_value = 1
    rmse = ((input_tensor - target_tensor) ** 2).mean().sqrt()
    psnr = 20 * ((max_value/ rmse).log10_())
    return psnr

@torch.no_grad()
def mean_absolute_error(output, target):
    input_tensor = output.view(-1).clone().detach()
    target_tensor = target.view(-1).clone().detach()

    if input_tensor.shape != target_tensor.shape:
        raise ValueError(
            'input shape {0} is not competable with target shape {1}'.format(input_tensor.shape, target_tensor.shape))
    return torch.abs(input_tensor- target_tensor).mean()
mae=mean_absolute_error

@torch.no_grad()
def mean_squared_error(output, target):
    input_tensor = output.view(-1).clone().detach()
    target_tensor = target.view(-1).clone().detach()

    if input_tensor.shape != target_tensor.shape:
        raise ValueError(
            'input shape {0} is not competable with target shape {1}'.format(input_tensor.shape, target_tensor.shape))
    return F.mse_loss(input_tensor, target_tensor)
mse=mean_squared_error



@torch.no_grad()
def root_mean_squared_error(output, target):
    input_tensor=output.view(-1).clone().detach()
    target_tensor=target.view(-1).clone().detach()

    if input_tensor.shape!=target_tensor.shape :
        raise  ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape,target_tensor.shape))
    return torch.sqrt(F.mse_loss(input_tensor, target_tensor,reduction='mean'))
rmse=root_mean_squared_error


@torch.no_grad()
def mean_squared_logarithmic_error(output, target):
    input_tensor=output.view(-1).clone().detach()
    target_tensor=target.view(-1).clone().detach()

    if input_tensor.shape!=target_tensor.shape :
        raise  ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape,target_tensor.shape))
    return F.mse_loss(torch.log(1 + input_tensor), torch.log(1 + target_tensor))
msle=mean_squared_logarithmic_error


@torch.no_grad()
def pixel_accuracy(output, target):
    input_tensor = output.clone().detach()
    target_tensor = target.clone().detach()
    if input_tensor.dtype!=torch.int64 :
        input_tensor=argmax(input_tensor,axis=1).squeeze()
    pixel_labeled = (target_tensor > 0).sum().float()
    pixel_correct = ((input_tensor == target_tensor)*(target_tensor > 0)).sum().float()
    return pixel_correct/max(pixel_labeled,1)

@torch.no_grad()
def alpha_pixel_accuracy(output, alpha):
    output_tensor = to_numpy(output)
    alpha_tensor =  to_numpy(alpha)

    trimap=alpha_tensor.copy()
    trimap[(0<trimap)*(trimap<1)==True]=0.5
    if len(output_tensor.shape)==len(alpha_tensor.shape)+1 and output_tensor.shape[1]==2:
        # trimap_out=mask2trimap()(np.argmax(output_tensor,1))
        # trimap_out1=trimap_out.copy()
        # trimap_out1[trimap_out1==0.5]=0
        # trimap_out2=trimap_out.copy()
        # trimap_out2[trimap_out2 ==1] = 0
        output_tensor=output_tensor[:,1,:,:]*np.argmax(output_tensor,1)#trimap_out*trimap_out1+output_tensor[:,1,:,:]*trimap_out2
        output_tensor[output_tensor>0.95]=1
    pixel_labeled = (output_tensor > 0).sum()
    pixel_correct = ((output_tensor == alpha_tensor)*(trimap == 1)).sum()+ (np.less(np.abs(output_tensor - alpha_tensor),0.1).astype(np.float32)*(trimap == 0.5)).sum()
    return pixel_correct/max(pixel_labeled,1)

@torch.no_grad()
def iou(output, target):
    input_tensor = output.clone().detach()
    target_tensor = target.clone().detach()
    if input_tensor.dtype != torch.int64:
        input_tensor = argmax(input_tensor, axis=1).squeeze()

    intersection =( (input_tensor > 0) * (input_tensor == target_tensor)).sum().float()
    union=((input_tensor+target_tensor)>0).sum().float()

    return intersection/max(union,1)


def get_metric(metric_name):
    if metric_name is None:
        return None

    metric_modules = ['trident.optims.pytorch_metrics']
    if metric_name in __all__:
        metric_fn = get_function(metric_name, metric_modules)
    else:
        try:
            metric_fn = get_function(camel2snake(metric_name), metric_modules)
        except Exception :
            metric_fn = get_function(metric_name, metric_modules)
    return metric_fn



