from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import collections
import itertools
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn_ops
from ..backend.common import get_session,addindent,get_time_suffix,get_function,get_class,format_time,get_terminal_size,snake2camel,camel2snake


__all__ = ['accuracy','psnr','mean_absolute_error','mean_squared_error','mean_squared_logarithmic_error','mae','mse','rmse','msle','get_metrics']


def accuracy(y_true, y_pred, topk=1,axis=1):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    if K.int_shape(y_true)!=K.int_shape(y_pred):
        raise  ValueError('y_pred shape {0} is not competable with y_true shape {1}'.format(y_pred.shape,y_true.shape))

    if topk==1:
        return tf.reduce_mean(tf.cast(tf.math.equal(tf.math.argmax(y_true,-1),tf.math.argmax(y_pred,-1)),tf.float32))
    else:
        return tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=y_pred,targets=y_true,k=topk),tf.float32))


def psnr(y_true, y_pred):
    if K.int_shape(y_true)!=K.int_shape(y_pred) :
        raise ValueError('y_pred shape {0} is not competable with y_true shape {1}'.format(y_pred.shape,y_true.shape))
    diff =tf.math.reduce_mean(tf.math.square(y_pred - y_true),[1,2,3])
    return tf.math.reduce_mean( (10 * tf.math.log(255**2 / diff)/tf.math.log(10)))


def mean_absolute_error(y_true, y_pred):
    if K.int_shape(y_true)!=K.int_shape(y_pred) :
        raise ValueError('y_pred shape {0} is not competable with y_true shape {1}'.format(y_pred.shape,y_true.shape))
    return tf.math.reduce_mean(tf.math.abs( y_pred -  y_true))
mae=mean_absolute_error


def mean_squared_error(y_true, y_pred):
    if K.int_shape(y_true)!=K.int_shape(y_pred) :
        raise ValueError('y_pred shape {0} is not competable with y_true shape {1}'.format(y_pred.shape,y_true.shape))
    return tf.math.reduce_mean(tf.math.square( y_pred -  y_true))
mse=mean_squared_error




def root_mean_squared_error(y_true, y_pred):
    if K.int_shape(y_true)!=K.int_shape(y_pred) :
        raise  ValueError('y_pred shape {0} is not competable with y_true shape {1}'.format(y_pred.shape,y_true.shape))
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square( y_pred -  y_true)))
rmse=root_mean_squared_error



def mean_squared_logarithmic_error(y_true, y_pred):
    if K.int_shape(y_true)!=K.int_shape(y_pred) :
        raise  ValueError('y_pred shape {0} is not competable with y_true shape {1}'.format(y_pred.shape,y_true.shape))
    return tf.math.reduce_mean(tf.math.square(tf.math.log(1 + y_pred)- tf.math.log(1 + y_true)))
msle=mean_squared_logarithmic_error


def get_metrics(metrics_name):
    if metrics_name is None:
        return None
    metrics_modules = ['trident.optimizers.tensorflow_metrics']
    try:
        metrics_fn = get_function(camel2snake(metrics_name), metrics_modules)
    except Exception :
        metrics_fn = get_function(metrics_name, metrics_modules)
    return metrics_fn

