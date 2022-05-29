from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import sys
from functools import partial
from  typing import Union,Optional,List,Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from trident.backend.common import get_session, addindent, get_time_suffix, get_class, get_function, camel2snake
from trident.backend.pytorch_ops import *
from trident.data.mask_common import mask2trimap
from trident.optims.losses import _check_logsoftmax_logit
__all__ = ['accuracy','recall','pixel_accuracy','alpha_pixel_accuracy','iou','psnr','mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','mae','mse','rmse','msle','get_metric']

# def accuracy(input, target,axis=1):
#     output_tensor=input.clone().detach()
#     target_tensor=target.clone().detach()
#     if output_tensor.dtype!=torch.int64:
#         output_tensor=argmax(output_tensor,axis).squeeze()
#     if target_tensor.dtype!=torch.int64:
#         target_tensor=argmax(target_tensor,axis).squeeze()
#     if output_tensor.shape!=target_tensor.shape:
#         raise  ValueError('input shape {0} is not competable with target shape {1}'.format(output_tensor.shape,target_tensor.shape))
#     else:
#         return output_tensor.eq(target_tensor).float().mean()






@torch.no_grad()
def accuracy(output:Tensor, target:Tensor, topk:int=1,axis:int=1,ignore_index:Union[int,list]=-100, exclude_mask:bool=False):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    output_tensor=output.copy().detach()
    target_tensor=target.copy().detach()
    num_classes = int_shape(output)[axis]
    if len(output_tensor)==0:
        return to_tensor(0.0)


    is_logsoftmax = None
    from_logits = None

    if _check_logsoftmax_logit(output, axis):
        output_tensor=exp(output_tensor)
    elif not _check_logit(output, axis):
        output_tensor=sigmoid(output_tensor)

    if output_tensor.dtype!=torch.int64 and topk==1:
        if len(output_tensor.size())==1: #binary
            output_tensor=output_tensor.gt(0.5).float()
        else:
            output_tensor=argmax(output_tensor,axis).squeeze()
    if target_tensor.dtype!=torch.int64:
        target_tensor=argmax(target_tensor,axis).squeeze()
    if output_tensor.shape!=target_tensor.shape and topk==1:
        output_tensor = argmax(output_tensor, axis).squeeze()
        #raise  ValueError('input shape {0} is not competable with target shape {1}'.format(output_tensor.shape,target_tensor.shape))

    input_mask=ones_like(output_tensor)
    if isinstance(ignore_index, int) and 0 <= ignore_index < num_classes:
        input_mask[output_tensor==ignore_index] = 0
    elif isinstance(ignore_index, (list, tuple)):
        for idx in ignore_index:
            if isinstance(idx, int) and 0 <= idx < int_shape(output)[axis]:
                input_mask[output_tensor == idx] = 0

    batch_size = target_tensor.size(0)
    if topk==1:
        return (output_tensor.eq(target_tensor).float()*input_mask).sum()/clip((input_mask).float().sum(),min=1)
    else:
        _, pred = output_tensor.topk(topk)
        pred = pred.t()
        correct = pred.eq(target_tensor.reshape((1, -1)).expand_as(pred))
        correct_k = reduce_sum(correct[:topk].reshape(-1).float(),axis=0,keepdims=True)
        return correct_k.mul_(1 / batch_size)



@torch.no_grad()
def recall(output:Tensor, target:Tensor, axis=1,ignore_index=0):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    output_tensor=output.copy().detach()
    target_tensor=target.copy().detach()
    num_classes = int_shape(output)[axis]


    is_logsoftmax = None
    from_logits = None
    output_exp = exp(output_tensor)
    if (ndim(output_tensor) >= 1 and 'float' in str(output_tensor.dtype) and output_tensor.min() >= 0 and output_tensor.max() <= 1):
        is_logsoftmax = False
        from_logits = True
        output_tensor = clip(output_tensor, min=1e-7, max=1 - 1e-7)

    elif (ndim(output_exp) >= 1 and 'float' in str(output_exp.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1):
        is_logsoftmax = True
        from_logits = True
        output_tensor =  clip(output_exp, min=1e-7, max=1 - 1e-7)
    else:
        is_logsoftmax = False
        from_logits = False

    if output_tensor.dtype!=torch.int64 :
        if len(output_tensor.size())==1: #binary
            output_tensor=output_tensor.gt(0.5).float()
        else:
            output_tensor=argmax(output_tensor,axis).squeeze()
    if target_tensor.dtype!=torch.int64:
        target_tensor=argmax(target_tensor,axis).squeeze()
    if output_tensor.shape!=target_tensor.shape :
        raise  ValueError('input shape {0} is not competable with target shape {1}'.format(output_tensor.shape,target_tensor.shape))
    target_mask=ones_like(target_tensor)
    if isinstance(ignore_index, int) and 0 <= ignore_index < num_classes:
        target_mask[target_tensor==ignore_index] = 0
    elif isinstance(ignore_index, (list, tuple)):
        for idx in ignore_index:
            if isinstance(idx, int) and 0 <= idx < int_shape(output)[axis]:
                target_mask[target_tensor == idx] = 0

    batch_size = target_tensor.size(0)
    return (output_tensor.eq(target_tensor).float()*target_mask).sum()/clip((target_mask).float().sum(),min=1)


from sklearn.metrics import roc_curve, auc, roc_auc_score


# make a prediction

@torch.no_grad()
def auc(output:Tensor, target:Tensor):
    output_np = to_numpy(exp(output))[:, 1]
    target_np = to_numpy(target)
    return roc_auc_score(target_np, output_np)

@torch.no_grad()
def psnr(output:Tensor, target:Tensor):
    output_tensor = output.clone().detach()
    target_tensor = target.clone().detach()
    if output_tensor.shape != target_tensor.shape :
        raise ValueError(
            'input shape {0} is not competable with target shape {1}'.format(output_tensor.shape, target_tensor.shape))

    max_value = 255
    target_np = to_numpy(target_tensor)
    if  target_np.min() <0 :
        target_tensor = (target_tensor + 1) * 0.5
        output_tensor = (output_tensor + 1) * 0.5
        max_value = 1
    elif 0 <= target_np.min() <= 1 and 0 <= target_np.max() <= 1:
        max_value = 1
    rmse = ((output_tensor - target_tensor) ** 2).mean().sqrt()
    psnr = 20 * ((max_value/ rmse).log10_())
    return psnr

@torch.no_grad()
def mean_absolute_error(output:Tensor, target:Tensor):
    output_tensor = output.view(-1).clone().detach()
    target_tensor = target.view(-1).clone().detach()

    if output_tensor.shape != target_tensor.shape:
        raise ValueError(
            'input shape {0} is not competable with target shape {1}'.format(output_tensor.shape, target_tensor.shape))
    return torch.abs(output_tensor- target_tensor).mean()
mae=mean_absolute_error

@torch.no_grad()
def mean_squared_error(output:Tensor, target:Tensor):
    output_tensor = output.view(-1).clone().detach()
    target_tensor = target.view(-1).clone().detach()

    if output_tensor.shape != target_tensor.shape:
        raise ValueError(
            'input shape {0} is not competable with target shape {1}'.format(output_tensor.shape, target_tensor.shape))
    return ((target_tensor-output_tensor)**2).mean()
mse=mean_squared_error



@torch.no_grad()
def root_mean_squared_error(output:Tensor, target:Tensor):
    output_tensor=output.view(-1).clone().detach()
    target_tensor=target.view(-1).clone().detach()

    if output_tensor.shape!=target_tensor.shape :
        raise  ValueError('input shape {0} is not competable with target shape {1}'.format(output_tensor.shape,target_tensor.shape))
    return ((target_tensor-output_tensor)**2).mean().sqrt()
rmse=root_mean_squared_error


@torch.no_grad()
def mean_squared_logarithmic_error(output:Tensor, target:Tensor):
    output_tensor=output.view(-1).clone().detach()
    target_tensor=target.view(-1).clone().detach()

    if output_tensor.shape!=target_tensor.shape :
        raise  ValueError('input shape {0} is not competable with target shape {1}'.format(output_tensor.shape,target_tensor.shape))
    return F.mse_loss(torch.log(1 + output_tensor), torch.log(1 + target_tensor))
msle=mean_squared_logarithmic_error


@torch.no_grad()
def pixel_accuracy(output:Tensor, target:Tensor):
    output_tensor = output.clone().detach()
    target_tensor = target.clone().detach()
    if output_tensor.dtype!=torch.int64 :
        output_tensor=argmax(output_tensor,axis=1).squeeze()
    pixel_labeled = (target_tensor > 0).sum().float()
    pixel_correct = ((output_tensor == target_tensor)*(target_tensor > 0)).sum().float()
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
def iou(output:Tensor, target:Tensor):
    output_tensor = output.clone().detach().view(output.size(0),output.size(1),-1)
    target_tensor=None
    if ndim(output)==ndim(target)+1:
        target_tensor = target.clone().detach().view(target.size(0),-1)
    elif ndim(output)==ndim(target):
        target_tensor = target.clone().detach().view(target.size(0),target.size(1),-1)
        target_tensor=argmax(target_tensor,1)

    if _check_logsoftmax_logit(output_tensor):
        output_tensor=exp(output_tensor)
    if output_tensor.dtype != torch.int64:
        output_tensor = argmax(output_tensor, axis=1).squeeze()
    intersection =( (output_tensor > 0) * (output_tensor == target_tensor)).sum().float()
    union=((output_tensor+target_tensor)>0).sum().float()

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



