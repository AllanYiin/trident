from typing import List

import numpy as np
import types
from contextlib import contextmanager
from functools import wraps
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from .tensorflow_backend import Layer, Sequential, is_tensor, to_numpy, to_tensor
from .common import _tensor_op,to_list

__all__ = ['element_cosine_distance','is_nan','is_inf','is_abnormal_number','is_sparse','ndim','is_sparse','int_shape','dot','clip','reduce_mean','reduce_max','reduce_min','reduce_sum','sqrt','square','abs','exp','log','pow','concate','ones','ones_like','zeros','zeros_like','meshgrid']

_context = []


@contextmanager
def tensor_context(**kwargs):
    r"""Context helper for computational graph building.
    Makes all elements within the with Block share the parameters.
    For example, in the following example, the default value of parameter `bn` will be set to True
    in the all layers within the with block.
    ```
    with tf.sg_context(bn=True):
        ...
        ...
    ```
    Args:
      **kwargs:
        in_dim: An integer. The size of input dimension, which is set to the last one by default.
        dim: An integer. The size of output dimension. Has the same value as in_dim by default.
        bn: Boolean. If True, batch normalization is applied.
        ln: Boolean. If True, layer normalization is applied.
        dout: A float of range [0, 100). A dropout rate. Default is 0..
        bias: Boolean. If True (Default), biases are added.
        name: A name for the layer. By default, the function name is assigned.
        act: A name of activation function. e.g., `sigmoid`, `tanh`, etc.
        reuse: `True` or `None`; if `True`, we go into reuse mode for this `layer` scope
          as well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
    Returns:
      None
    """
    global _context

    # set options when enter
    context_now =_tensor_op(kwargs)
    _context += [context_now]

    # if named context
    if context_now.name:
        context_now.scope_name = context_now.name
        context_now.name = None
        with tf.variable_scope(context_now.scope_name):
            yield
    else:
        yield

    # clear options when exit
    del _context[-1]


def get_op_context():
    r"""Get current context information
    Returns:
      tf.sg_opt class object which contains all context information
    """

    global _context

    # merge current context
    res = _tensor_op()
    for c in _context:
        res += c

    return res



def tensor_op(func):
    r""" Decorates a function `func` so that it can process a tensor operation.
    Tensor operation can be declare in a chainable behavior.
    Args:
        func: function to decorate
    Returns:
      A  tensor operation function.
    """
    @wraps(func)
    def wrapper(tensor, **kwargs):
        # call sugar function
        out = func(tensor, _tensor_op(kwargs))
        # save node info for reuse
        out._op = _tensor_op(func=func, arg=_tensor_op(kwargs)+get_op_context(), prev=tensor)
        # inject reuse function
        #out.sg_reuse = types.MethodType(sg_reuse, out)
        return out

    return wrapper

############################
## check operation
###########################
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

def is_sparse(x):
    return isinstance(x, tf.SparseTensor)

############################
## tensor attribute
###########################

def ndim(x):
    return x.shape.rank

def int_shape(x):
    return x.get_shape().as_list()

def clip(x:tf.Tensor,min_value=-np.inf,max_value=np.inf):
    return tf.clip_by_value(x,min,max)

############################
## basic math operation
###########################



def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.
    # Returns
        A tensor, dot product of `x` and `y`.

    ```
    {{np_implementation}}
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(int_shape(x), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(int_shape(y), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if is_sparse(x):
        out = tf.sparse.sparse_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

@tensor_op
def sqrt(x:tf.Tensor):
    return tf.math.sqrt(x)

@tensor_op
def square(x:tf.Tensor):
    return tf.math.square(x)

@tensor_op
def abs(x:tf.Tensor):
    return tf.math.abs(x)

@tensor_op
def pow(x:tf.Tensor,y):
    return tf.math.pow(x,y)

@tensor_op
def log(x:tf.Tensor):
    return tf.math.log(x)

@tensor_op
def exp(x:tf.Tensor):
    return tf.math.exp(x)



############################
## reduce operation
###########################

def reduce_mean(x:tf.Tensor,axis=None,keepdims=False):
    return tf.math.reduce_mean(x,axis=axis,keepdims=keepdims)

def reduce_sum(x:tf.Tensor,axis=None,keepdims=False):
    return tf.math.reduce_sum(x,axis=axis,keepdims=keepdims)

def reduce_max(x:tf.Tensor,axis=None,keepdims=False):
    return tf.math.reduce_max(x,axis=axis, keepdims=keepdims)

def reduce_min(x:tf.Tensor,axis=None,keepdims=False):
    return tf.math.reduce_min(x,axis=axis,keepdims=keepdims)


############################
## element-wise operation
###########################
def element_cosine_distance(v1, v2, axis=1):
    normalize_a = tf.nn.l2_normalize(v1, axis)
    normalize_b = tf.nn.l2_normalize(v2, axis)
    distance = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance



############################
## tensor shape operation
###########################

def reshape(x:tf.Tensor,shape=None)->tf.Tensor:
    if shape is None:
        return x
    elif isinstance(shape,(list,tuple,tf.TensorShape)):
        return tf.reshape(x,shape)
    else:
        shape=to_list(shape)
        return tf.reshape(x,shape)


def squeeze(x:tf.Tensor,axis=None):
    return tf.squeeze(x,axis=axis)


def expand_dims(x:tf.Tensor,axis=None):
    return tf.expand_dims(x,axis=axis)




############################
## tensor generation
###########################

def ones(shape,dtype=tf.float32,requires_grad=False):
    if requires_grad==True:
        return tf.ones(shape,dtype)
    else:
        return tf.no_gradient(tf.ones(shape,dtype))

def ones_like(a,dtype=tf.float32,requires_grad=False):
    if requires_grad==True:
        return tf.ones_like(a,dtype)
    else:
        return tf.no_gradient(tf.ones_like(a,dtype))

def zeros(shape,dtype=tf.float32,requires_grad=False):
    if requires_grad == True:
        return tf.zeros(shape, dtype)
    else:
        return tf.no_gradient(tf.zeros(shape, dtype))

def zeros_like(a,dtype=tf.float32,requires_grad=False):
    if requires_grad==True:
        return tf.zeros_like(a,dtype)
    else:
        return tf.no_gradient(tf.zeros_like(a,dtype))

def meshgrid(x, y, normalized_coordinates=False,requires_grad=False):
    '''Return meshgrid in range x & y.

    Args:
      requires_grad ():
      normalized_coordinates ():
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''

    if requires_grad==True:
        return tf.meshgrid(x,y)
    else:
        return tf.no_gradient(tf.meshgrid(x,y))





def concate(x:List[tf.Tensor],axis=1):
    return tf.concat(x,axis=axis)
