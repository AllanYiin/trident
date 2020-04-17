import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from .tensorflow_backend import Layer, Sequential, is_tensor, to_numpy, to_tensor

__all__ = ['element_cosine_distance','is_nan','is_inf','is_abnormal_number','clip','reduce_mean','reduce_max','reduce_min','reduce_sum','sqrt','square','abs','exp','log','pow']




def is_nan(x):
    if isinstance(x, (tf.Tensor, tf.Variable)) or is_tensor(x):
        if x.ndim==0:
            return tf.math.is_nan(x)
        else:
            return tf.math.is_nan(x).numpy().any()
    elif isinstance(x,Layer):
        for para in x.weights:
            if tf.math.is_nan(para).numpy().any():
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isnan(x).any()
    else:
        raise NotImplementedError

def is_inf(x):
    if isinstance(x, (tf.Tensor, tf.Variable)) or is_tensor(x):
        if x.ndim==0:
            return tf.math.is_inf(x)
        else:
            return tf.math.is_inf(x).numpy().any()
    elif isinstance(x,Layer):
        for para in x.weights:
            if tf.math.is_inf(para).numpy().any():
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isinf(x).any()
    else:
        raise NotImplementedError

def is_abnormal_number(x):
    return is_nan(x) or is_inf(x)


def clip(x:EagerTensor,min_value=-np.inf,max_value=np.inf):
    return tf.clip_by_value(x,min,max)

def sqrt(x:EagerTensor):
    return tf.math.sqrt(x)

def square(x:EagerTensor):
    return tf.math.square(x)

def abs(x:EagerTensor):
    return tf.math.abs(x)

def pow(x:EagerTensor,y):
    return tf.math.pow(x,y)

def log(x:EagerTensor):
    return tf.math.log(x)

def exp(x:EagerTensor):
    return tf.math.exp(x)



############################
## reduce operation
###########################

def reduce_mean(x:EagerTensor,axis=None,keepdims=False):
    return tf.math.reduce_mean(x,axis=axis,keepdims=keepdims)

def reduce_sum(x:EagerTensor,axis=None,keepdims=False):
    return tf.math.reduce_sum(x,axis=axis,keepdims=keepdims)

def reduce_max(x:EagerTensor,axis=None,keepdims=False):
    return tf.math.reduce_max(x,axis=axis, keepdims=keepdims)

def reduce_min(x:EagerTensor,axis=None,keepdims=False):
    return tf.math.reduce_min(x,axis=axis,keepdims=keepdims)


def element_cosine_distance(v1, v2, axis=1):
    normalize_a = tf.nn.l2_normalize(v1, axis)
    normalize_b = tf.nn.l2_normalize(v2, axis)
    distance = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance
