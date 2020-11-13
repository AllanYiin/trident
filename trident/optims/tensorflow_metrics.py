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

__all__ = ['accuracy','psnr','mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','mae','mse','rmse','msle','get_metric']


def accuracy(output, target, topk=1, axis=-1, exclude_mask=False):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    input_tensor = output.copy().detach()
    target_tensor = target.copy().detach()

    input_tensor_exp = exp(input_tensor)
    is_logsoftmax = None
    from_logits = None
    output_exp = exp(input_tensor)
    if (ndim(output) >= 1 and 'float' in str(output.dtype) and output.min() >= 0 and output.max() <= 1):
        is_logsoftmax = False
        from_logits = True
        output = clip(output, min=1e-8, max=1 - 1e-8)

    elif (ndim(output) >= 1 and 'float' in str(output.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1):
        is_logsoftmax = True
        from_logits = True
        output = clip(output, max=- 1e-8)
    else:
        is_logsoftmax = False
        from_logits = False

    if is_logsoftmax:
        input_tensor = exp(input_tensor)
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

