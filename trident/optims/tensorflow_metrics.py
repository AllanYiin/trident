from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import itertools
import copy
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from trident.backend.common import get_session,addindent,get_time_suffix,get_function,get_class,format_time,get_terminal_size,snake2camel,camel2snake
from trident.backend.tensorflow_ops import *

__all__ = ['accuracy','pixel_accuracy','alpha_pixel_accuracy','iou','psnr','mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','mae','mse','rmse','msle','get_metric']


def accuracy(output, target, topk=1, axis=-1, exclude_mask=False):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    input_tensor = output.copy().detach()
    target_tensor = target.copy().detach()

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
        input_tensor = clip(output_exp, min=1e-8, max=1 - 1e-8)
    else:
        is_logsoftmax = False
        from_logits = False

    if input_tensor.dtype != tf.int64 and topk == 1:
        if len(int_shape(input_tensor)) == 1:  # binary
            input_tensor =greater_equal(input_tensor,0.5)
        else:
            input_tensor = argmax(input_tensor, axis).squeeze()
    if target_tensor.dtype != tf.int64:
        target_tensor = argmax(target_tensor, axis).squeeze()
    if input_tensor.shape != target_tensor.shape and topk == 1:
        raise ValueError('input shape {0} is not competable with target shape {1}'.format(input_tensor.shape, target_tensor.shape))

    batch_size = int_shape(target_tensor)[0]
    if topk == 1:
        return equal(input_tensor,target_tensor).mean()
    else:
        _,pred = input_tensor.topk(topk)
        pred = cast(tf.transpose(pred),'float32')
        target_tensor= cast(repeat_elements(expand_dims(target_tensor,0),topk,axis=0),'float32')
        correct = equal(pred,target_tensor).sum()
        return correct/batch_size


def pixel_accuracy(output, target):
    input_tensor = output.copy().detach()
    target_tensor = target.copy().detach()
    if input_tensor.dtype!=tf.int64 :
        input_tensor=argmax(input_tensor,axis=-1).squeeze()
    return equal(input_tensor,target_tensor).mean()

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



def iou(output, target):
    input_tensor = output.copy().detach()
    target_tensor = target.copy().detach()
    if input_tensor.dtype != tf.int64:
        input_tensor = argmax(input_tensor, axis=-1).squeeze()
    if target_tensor.dtype != tf.int64:
        target_tensor = argmax(target_tensor, axis=-1).squeeze()

    intersection =( greater(input_tensor ,0) * equal(input_tensor ,target_tensor)).sum()
    union=greater(( greater(input_tensor ,0) + greater(target_tensor ,0) ),0).sum()

    return intersection/maximum(union,1)

def psnr(output, target):
    if target.get_shape()!=output.get_shape() :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    diff =tf.math.reduce_mean(tf.math.square(output - target),[1,2,3])
    return tf.math.reduce_mean( (10 * tf.math.log(255**2 / diff)/tf.math.log(10)))


def mean_absolute_error(output, target):
    if target.get_shape()!=output.get_shape() :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return tf.math.reduce_mean(tf.math.abs( output -  target))
mae=mean_absolute_error


def mean_squared_error(output, target):
    if target.get_shape()!=output.get_shape() :
        raise ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return tf.math.reduce_mean(tf.math.square( output -  target))
mse=mean_squared_error




def root_mean_squared_error(output, target):
    if target.get_shape()!=output.get_shape() :
        raise  ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square( output -  target)))
rmse=root_mean_squared_error



def mean_squared_logarithmic_error(output, target):
    if target.get_shape()!=output.get_shape() :
        raise  ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape,target.shape))
    return tf.math.reduce_mean(tf.math.square(tf.math.log(1 + output)- tf.math.log(1 + target)))
msle=mean_squared_logarithmic_error


def get_metric(metric_name):
    if metric_name is None:
        return None
    metric_modules = ['trident.optims.tensorflow_metrics']
    try:
        metric_fn = get_function(camel2snake(metric_name), metric_modules)
    except Exception :
        metric_fn = get_function(metric_name, metric_modules)
    return metric_fn

