from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import itertools
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops
from trident.backend.common import get_session,addindent,get_time_suffix,get_function,get_class,format_time,get_terminal_size,snake2camel,camel2snake


__all__ = ['accuracy','psnr','mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','mae','mse','rmse','msle','get_metric']


def accuracy(output,target, topk=1, axis=1, **kwargs):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    if  target.get_shape()!=output.get_shape():
        raise  ValueError('output shape {0} is not competable with target shape {1}'.format(output.shape, target.shape))

    if topk==1:
        return tf.reduce_mean(tf.cast(tf.math.equal(tf.math.argmax(target, -1), tf.math.argmax(output, -1)), tf.float32))
    else:
        return tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=output, targets=target, k=topk), tf.float32))


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

