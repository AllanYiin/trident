from typing import List
import copy
import inspect
import math
import numpy as np
import types
from contextlib import contextmanager
from functools import wraps
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.framework.ops import EagerTensor
from trident.backend.common import _tensor_op,to_list

__all__ = ['to_numpy', 'to_tensor','is_tensor','element_cosine_distance','is_nan','is_inf','is_abnormal_number','any_nan','any_inf','any_abnormal_number','is_sparse','ndim','is_sparse','int_shape','dot','clip','reduce_mean','reduce_max','reduce_min','reduce_sum','sqrt','square','abs','exp','log','pow','round','ceil','floor','concate','reshape','transpose','permute','squeeze','expand_dims','ones','ones_like','zeros','zeros_like','meshgrid','identity','sigmoid','tanh','relu','relu6','leaky_relu','leaky_relu6','smooth_relu','p_relu','swish','elu','hard_sigmoid','hard_swish','selu','lecun_tanh','soft_sign','soft_plus','hard_tanh','logit','log_log','mish','softmax','bert_gelu','gpt_gelu','less','equal','greater','greater_equal','not_equal','less_equal']

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



def is_tensor(x):
    if hasattr(x, 'numpy'):
        with context.eager_mode() :
            return True
    elif x.__class__.__name__=='EagerTensor':
        return True
    elif tf.is_tensor(x):
        return True
    return False


def to_numpy(x) -> np.ndarray:
    """
    Convert whatever to numpy array
    :param x: List, tuple, PyTorch tensor or numpy array
    :return: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    # elif isinstance(x,EagerTensor):
    #     return x.numpy()
    elif hasattr(x, 'numpy'):
        with context.eager_mode():
            return x.__copy__().numpy()
    elif isinstance(x, tf.TensorShape):
        return np.array(copy.deepcopy(x.as_list()))
    elif isinstance(x, (tf.Tensor, tf.Variable)):
        return copy.deepcopy(x).value()
    # elif isinstance(x, tf.Variable):
    #     sess = tf.compat.v1.Session()
    #     x = sess.run(x.value())
    #     return x
    # elif isinstance(x, ops.Tensor):
    #     sess = tf.compat.v1.Session()
    #     x= sess.run(x)
    #     return x

    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        try:
            x = tf.keras.backend.get_value(x)
            if isinstance(x, np.ndarray):
                return x
        except:
            raise ValueError("Unsupported type")


def to_tensor(x, dtype=tf.float32,requires_grad=None) -> tf.Tensor:
    '''
     Convert input  to a tensor as possible
    Args:
        x (int,float,list,tuple,ndarray,tensor):
        dtype :
        requires_grad (bool): wheather need grade

    Returns: output tensor
    Examples:
        >>> to_tensor(2)
        <tf.Tensor: shape=(), dtype=int32, numpy=2>
        >>> to_tensor([1.0,2.0,3.0],requires_grad=True)
        <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor([1.0,2.0,3.0],requires_grad=False)
        <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor([1.0,2.0,3.0])
        <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor((1.0,2.0,3.0))
        <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor(np.arange(0,5))
        <tf.Tensor: shape=(5,), dtype=float32, numpy=
        array([0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00],
        dtype=float32)>

    '''
    if isinstance(x, int):
        return tf.constant(value=x,dtype=tf.int32)
    elif   isinstance(x, float):
        return tf.constant(value=x,dtype=tf.float32)
    else:
        if requires_grad == False:
            x =tf.constant(ops.convert_to_tensor_v2(x, dtype=dtype))
        elif requires_grad == True:
            x= tf.Variable(ops.convert_to_tensor_v2(x, dtype=dtype))

        else :
            x =ops.convert_to_tensor_v2(x, dtype=dtype)
        return x

############################
## tensor attribute
###########################

def ndim(x):
    return x.shape.rank

def int_shape(x):
    '''

    Args:
        x : input tensor

    Returns: tuple of integer as shape representation

    Examples:
    >>> int_shape(ones((3,3,7)))
    [3, 3, 7]

    '''
    return x.get_shape().as_list()

def is_sparse(x):
    return isinstance(x, tf.SparseTensor)

def cast(x, dtype):
    if isinstance(dtype, str):
        if 'float64' in dtype.lower() or 'double' in dtype.lower():
            dtype=tf.float64
        elif 'float16' in dtype.lower() or 'half' in dtype.lower():
            dtype=tf.float16
        elif 'float' in dtype.lower():
            dtype=tf.float32
        elif 'int64' in dtype.lower() or 'long' in dtype.lower():
            dtype=tf.int64
        elif 'int16' in dtype.lower() or 'short' in dtype.lower():
            dtype=tf.int16
        elif 'uint8' in dtype.lower() or 'byte' in dtype.lower():
            dtype=tf.uint8
        elif 'int8' in dtype.lower() or 'char' in dtype.lower():
            dtype=tf.int8
        elif 'int32' in dtype.lower() or 'int' in dtype.lower():
            dtype=tf.int32
        elif 'bool' in dtype.lower():
            dtype=tf.bool
    if isinstance(dtype,tf.dtypes):
        return tf.cast(x,dtype)
    else:
        return x


############################
## check operation
###########################
def is_nan(x):
    if isinstance(x, (tf.Tensor, tf.Variable)) or is_tensor(x):
        if x.ndim==0:
            return tf.math.is_nan(x)
        else:
            return tf.math.is_nan(x).numpy().any()
    elif 'Layer' in x.__class__.__name__:
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
    elif 'Layer' in x.__class__.__name__:
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

def any_nan(x):
    if isinstance(x, (tf.Tensor, tf.Variable)) or is_tensor(x):
        if x.ndim==0:
            return tf.math.is_nan(x).any()
        else:
            return tf.math.is_nan(x).numpy().any()
    elif isinstance(x,tf.Module):
        for para in x.weights:
            if tf.math.is_nan(para).numpy().any():
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isnan(x).any()
    else:
        raise NotImplementedError

def any_inf(x):
    if isinstance(x, (tf.Tensor, tf.Variable)) or is_tensor(x):
        if x.ndim==0:
            return tf.math.is_inf(x).any()
        else:
            return tf.math.is_inf(x).numpy().any()
    elif isinstance(x, tf.Module):
        for para in x.weights:
            if tf.math.is_inf(para).numpy().any():
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isinf(x).any()
    else:
        raise NotImplementedError

def any_abnormal_number(x):
    return any_nan(x) or any_inf(x)


############################
## compare operation
###########################


def less(left:tf.Tensor, right:tf.Tensor):
    '''
    Elementwise 'less' comparison of two tensors. Result is 1 if left < right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        Result is 1 if left < right else 0.

    Example:
   >>> less(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
   <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 0.0000e+00, 0.0000e+00], dtype=float32)>
   >>> less(to_tensor([-1,0,1]), 0)
   <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 0.0000e+00, 0.0000e+00], dtype=float32)>
    '''

    return tf.cast(tf.less(left,right),tf.float32)

def equal(left:tf.Tensor, right:tf.Tensor):
    '''
    Elementwise 'equal' comparison of two tensors. Result is 1 if values are equal 0 otherwise.
    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if values are equal 0 otherwise

    Example:
    >>> equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 1.0000e+00, 0.0000e+00], dtype=float32)>
    >>> equal(to_tensor([-1,0,1]), 1)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>
    '''
    return tf.cast(tf.equal(left,right),tf.float32)

def greater(left:tf.Tensor, right:tf.Tensor):
    '''
    Elementwise 'greater' comparison of two tensors. Result is 1 if left > right else 0.
    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left > right else 0.

    Example:
    >>> greater(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>
    >>> greater(to_tensor([-1,0,1]), 0)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>
    '''
    return tf.cast(tf.greater(left,right),tf.float32)

def greater_equal(left:tf.Tensor, right:tf.Tensor):
    '''
    Elementwise 'greater equal' comparison of two tensors. Result is 1 if left >= right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left >= right else 0

    Example:
    >>> greater_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 1.0000e+00, 1.0000e+00], dtype=float32)>
    >>> greater_equal(to_tensor([-1,0,1]), 0)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 1.0000e+00, 1.0000e+00], dtype=float32)>
    '''
    return tf.cast(tf.greater_equal(left,right),tf.float32)

def not_equal(left:tf.Tensor, right:tf.Tensor):
    '''
    Elementwise 'not equal' comparison of two tensors. Result is 1 if left != right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left != right else 0.

    Example:
    >>> not_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>
    >>> not_equal(to_tensor([-1,0,1]), 0)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>
    '''
    return tf.cast(tf.not_equal(left,right),tf.float32)

def less_equal(left:tf.Tensor, right:tf.Tensor):
    '''
    Elementwise 'less equal' comparison of two tensors. Result is 1 if left <= right else 0.

    Args:
        left: left side tensor
        right: right side tensor

    Returns:
        :Result is 1 if left <= right else 0.
    Example:
    >>> less_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 1.0000e+00, 0.0000e+00], dtype=float32)>
    >>> less_equal(to_tensor([-1,0,1]), 0)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 1.0000e+00, 0.0000e+00], dtype=float32)>

    '''
    return tf.cast(tf.less_equal(left,right),tf.float32)


def argmax(x:tf.Tensor,axis=-1)-> tf.Tensor:
    return tf.argmax(x,axis=axis)
def argmin(x:tf.Tensor,axis=-1)-> tf.Tensor:
    return tf.argmin(x,axis=axis)
def argsort(x:tf.Tensor,axis=-1,descending=True)-> tf.Tensor:
    return tf.argsort(x,axis=axis,descending=descending)

def maximum(x:tf.Tensor,other:(tf.Tensor,int,float))-> tf.Tensor:
    if isinstance(other,tf.Tensor):
        return tf.maximum(x,other)
    elif isinstance(other,(int,float)):
        return clip(x,min_value=other)

def minimum(x:tf.Tensor,other:(tf.Tensor,int,float))-> tf.Tensor:
    if isinstance(other,tf.Tensor):
        return tf.minimum(x,other)
    elif isinstance(other,(int,float)):
        return clip(x,max_value=other)




############################
## basic math operation
###########################

def floor(x:tf.Tensor):
    return tf.math.floor(x)

def ceil(x:tf.Tensor):
    return tf.math.ceil(x)

def round(x:tf.Tensor,digit:int=0):
    '''

    Args:
        x ():
        digit ():

    Returns:
    Examples;
    >>> round(to_tensor([[1,2,3,4,5]])/3,0)
    <tf.Tensor: shape=(1, 5), dtype=float32, numpy=
    array([[0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 2.0000e+00]],
          dtype=float32)>
    >>> round(to_tensor([[1,2,3,4,5]])/3,2)
    <tf.Tensor: shape=(1, 5), dtype=float32, numpy=
    array([[3.3000e-01, 6.7000e-01, 1.0000e+00, 1.3300e+00, 1.6700e+00]],
          dtype=float32)>
    >>> round(to_tensor([[11.6,24.3,35.2,14.4,23.5]])/3,-1)
    <tf.Tensor: shape=(1, 5), dtype=float32, numpy=
    array([[0.0000e+00, 1.0000e+01, 1.0000e+01, 0.0000e+00, 1.0000e+01]],
          dtype=float32)>

    '''
    if digit!=0:
        factor=float(math.pow(10,-1*digit))
        return tf.math.round(x/factor)*factor
    else:
        return tf.math.round(x)

def add(x, y):
    return tf.add(x,y)

def subtract(x, y):
    return tf.subtract(x,y)

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

def matmul(x,y,transpose_x=False,transpose_y=False):
    return tf.matmul(x,y,transpose_a=transpose_x,transpose_b=transpose_y)


def true_divide(x, y):
    return tf.truediv(x,y)

def pi():
    return to_tensor(np.pi)

def sqrt(x:tf.Tensor):
    return tf.math.sqrt(x)

def square(x:tf.Tensor):
    return tf.math.square(x)


def abs(x:tf.Tensor):
    return tf.math.abs(x)


def pow(x:tf.Tensor,y):
    return tf.math.pow(x,y)


def log(x:tf.Tensor):
    return tf.math.log(x)


def exp(x:tf.Tensor):
    return tf.math.exp(x)

def prod(x):
    return tf.math.reduce_prod(x)

def clip(x:tf.Tensor,min_value=-np.inf,max_value=np.inf):
    return tf.clip_by_value(x,min,max)


def sin(x:tf.Tensor):
    return tf.math.sin(x)
def cos(x:tf.Tensor):
    return tf.math.cos(x)
def tan(x:tf.Tensor):
    return tf.math.tan(x)


def asin(x:tf.Tensor):
    return tf.math.asin(x)
def acos(x:tf.Tensor):
    return tf.math.acos(x)
def atan(x:tf.Tensor):
    return tf.math.atan(x)


def sinh(x:tf.Tensor):
    '''
    Computes the element-wise sinh
    Args:
        x (tensor):input tensor

    Returns: element-wise sinh

    Examples
    >>> sinh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.1752e+00, 5.2110e-01],
           [-2.5261e-01, -8.2232e-01]], dtype=float32)>

    '''
    return tf.sinh(x)
def cosh(x:tf.Tensor):
    '''
    Computes the element-wise cosh
    Args:
        x (tensor):input tensor

    Returns: element-wise cosh

    Examples
    >>> cosh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.5431e+00, 1.1276e+00],
           [1.0314e+00, 1.2947e+00]], dtype=float32)>
    '''
    return tf.cosh(x)
def tanh(x:tf.Tensor):
    '''
    Computes the element-wise tanh
    Args:
        x (tensor):input tensor

    Returns: element-wise tanh

    Examples
    >>> tanh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[ 0.     ,  1.0472 ],
       [ 1.82348,  2.41886]])
    '''
    return tf.tanh(x)





############################
## element-wise operation
###########################
def element_times(left, right):
    '''
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Example:
    >>> element_times(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([5.0000e-01, 2.5000e-01, 1.2500e-01, 0.0000e+00], dtype=float32)>
    >>> element_times(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1.0000e+01, 2.0000e+01, 3.0000e+01, 6.0000e+01], dtype=float32)>
    >>> element_times(to_tensor([[5., 10.], [15., 30.]]), to_tensor([[1., 2.], [3.,1.]]))
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[5.0000e+00, 2.0000e+01],
           [4.5000e+01, 3.0000e+01]], dtype=float32)>
    '''
    return left*right

def element_max(left, right):
    '''
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Example:
    >>> element_max(to_tensor([1., 1., 0., -1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1.0000e+00, 1.0000e+00, 1.2500e-01, 0.0000e+00], dtype=float32)>
    >>> element_max(to_tensor([5., 10., 15., 30.]),to_tensor([20.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([2.0000e+01, 2.0000e+01, 2.0000e+01, 3.0000e+01], dtype=float32)>
    >>> element_max(to_tensor([5., 10., 15., 30.]), to_tensor([10., 2., 8., 2.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1.0000e+01, 1.0000e+01, 1.5000e+01, 3.0000e+01], dtype=float32)>
    '''
    return maximum(left,right)

def element_min (left, right):
    '''
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Example:
    >>> element_min(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([5.0000e-01, 2.5000e-01, 1.2500e-01, 0.0000e+00], dtype=float32)>
    >>> element_min(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([2.0000e+00, 2.0000e+00, 2.0000e+00, 2.0000e+00], dtype=float32)>
    >>> element_min(to_tensor([5., 10., 15., 30.]), to_tensor([1., 2., 1., 2.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 1.0000e+00, 2.0000e+00], dtype=float32)>
    '''
    return minimum(left, right)

def element_divide (left, right):
    '''
    The output of this operation is the element-wise divide of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise divide of the two  input

    Example:
    >>> element_divide(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([2.0000e+00, 4.0000e+00, 8.0000e+00, inf], dtype=float32)>
    >>> element_divide(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([2.5000e+00, 5.0000e+00, 7.5000e+00, 1.5000e+01], dtype=float32)>
    >>> element_divide(to_tensor([5., 10., 15., 30.]), to_tensor([1., 2., 1., 2.]))
    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([5.0000e+00, 5.0000e+00, 1.5000e+01, 1.5000e+01], dtype=float32)>
    '''
    return true_divide(left, right)


def element_cosine_distance(v1, v2, axis=1):
    normalize_a = tf.nn.l2_normalize(v1, axis)
    normalize_b = tf.nn.l2_normalize(v2, axis)
    distance = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance

def where(flag, value_if_true, value_if_false):
    '''
    return either ``value_if_true`` or ``value_if_false`` based on the value of ``flag``.
    If ``flag`` != 0 ``value_if_true`` is returned, otherwise ``value_if_false``.
    Behaves analogously to numpy.where(...).

    Args:
        flag: condition tensor
        value_if_true: true branch tensor
        value_if_false: false branch tensor
    Returns:
        :conditional selection

    Example:
    >>> x=to_tensor([0.1, 0.9, 0.8, 0.4, 0.5])
    >>> where(x>0.5, x, zeros_like(x))
    <tf.Tensor: shape=(5,), dtype=float32, numpy=
    array([0.0000e+00, 9.0000e-01, 8.0000e-01, 0.0000e+00, 0.0000e+00],
          dtype=float32)>
    '''
    return tf.where(flag, value_if_true, value_if_false)






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

def reduce_logsumexp(x:tf.Tensor,axis=None,keepdims=False):
    '''

    Args:
        x ():
        axis ():
        keepdims ():

    Returns:
    Examples
    >>> x = to_tensor(np.array([[0., 0., 0.], [0., 0., 0.]]))
    >>> reduce_logsumexp(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.7917595>
    >>> reduce_logsumexp(x, 0)
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([6.9315e-01, 6.9315e-01, 6.9315e-01], dtype=float32)>
    >>> reduce_logsumexp(x, [0, 1])
    <tf.Tensor: shape=(), dtype=float32, numpy=1.7917595>
    '''
    return tf.math.reduce_logsumexp(x,axis=axis,keepdims=keepdims)

def reduce_prod(x:tf.Tensor,axis=None,keepdims=False):
    return tf.math.reduce_prod(x,axis=axis,keepdims=keepdims)





#reduce_l1
#reduce_l2
#reduce_sum_square

mean=reduce_mean
sum=reduce_sum
max=reduce_max
min=reduce_min










############################
## activationoperation
###########################


def identity(x):
    return x


def sigmoid(x):
    return tf.nn.sigmoid(x)

def tanh(x):
    return tf.nn.tanh(x)



def relu(x,upper_limit=None):
    if upper_limit is not None and upper_limit<=0:
        raise ValueError('Upper limit should greater than 0!')
    elif upper_limit is not None:
        return clip(tf.nn.relu(x),0,upper_limit)
    return tf.nn.relu(x)


def relu6(x):
    return clip(tf.nn.relu(x),0,6)



def leaky_relu(x,alpha=0.02,upper_limit=None):
    if upper_limit is not None:
        return clip(tf.nn.leaky_relu(x,alpha), -np.inf, upper_limit)
    return tf.nn.leaky_relu(x,alpha)


def leaky_relu6(x,alpha=0.01):
    return clip(tf.nn.leaky_relu(x,alpha), -6, 6)


def elu(x,alpha=0.01,upper_limit=None):
    if upper_limit is not None:
        return clip(tf.nn.elu(x,alpha),-np.inf,upper_limit)
    return tf.nn.elu(x,alpha)


lrelu=leaky_relu


def smooth_relu(x,upper_limit=None):
    if upper_limit is not None:
        return clip(tf.math.log(1 + tf.math.exp(x)),-np.inf,upper_limit)
    return tf.math.log(1 + tf.math.exp(x))


def p_relu(x,upper_limit=None):
    if upper_limit is not None:
        return clip(tf.keras.layers.PReLU()(x),-np.inf,upper_limit)
    return tf.keras.layers.PReLU()(x)


def swish(x):
    return tf.nn.sigmoid(x) * x

def selu(x):
    return tf.nn.selu(x)

def soft_sign(x):
    return tf.nn.softsign(x)


def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(2/3 * x)

def soft_plus(x):
    return tf.nn.softplus(x)

def hard_sigmoid(x):
    return relu6(x+3)/6

def hard_tanh(x):
    return tf.keras.backend.clip(x,-1,1)

def hard_swish(x):
    return  x * hard_sigmoid(x)

def logit(x):
        return tf.math.log(x / (1 - x))



def log_log(x):
    return  1-tf.math.exp(-tf.math.exp(x))



def softmax(x,axis=-1):
    return tf.nn.softmax(x,axis=axis)

def log_softmax(x,axis=-1,keepdims=False):
    """Activation function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        keepdims ():
        axis ():
        x : input tensor
    """

    return x-reduce_logsumexp(x,axis=axis,keepdims=True)


def mish(x):
    return x*tf.nn.tanh(tf.nn.softplus(x))


def bert_gelu(x):

  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  return x *  0.5 * (1.0 + tf.nn.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))


def gpt_gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 /np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))





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


def transpose(x:tf.Tensor,pattern=None)->tf.Tensor:
    return tf.transpose(x,pattern)


def permute(x:tf.Tensor,pattern=None)->tf.Tensor:
    return tf.transpose(x,pattern)

def depth_to_space(x:tf.Tensor,block_size=2):
    '''
    Rearranges elements in the input tensor from the depth dimension into spatial blocks.
    The equivalent to Pixel-Shuffle
    Args:
        x (tensor): Input tensor, with dimensions CHW or NCHW
        block_size (int):

    Returns: resized tensor

    Examples
    >>> x = to_tensor(np.tile(np.array(np.reshape(range(8), (8,1,1)), dtype=np.float32), (1, 2, 3)).transpose([1,2,0]))
    >>> x
    <tf.Tensor: shape=(2, 3, 8), dtype=float32, numpy=
    array([[[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00]],
    <BLANKLINE>
           [[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00]]], dtype=float32)>


    >>> arr=depth_to_space(x,block_size=2)
    >>> print(arr.shape)
    (4, 6, 2)
    >>> arr
    <tf.Tensor: shape=(2, 4, 6), dtype=float32, numpy=
    array([[[0.0000e+00, 2.0000e+00, 0.0000e+00, 2.0000e+00, 0.0000e+00,
             2.0000e+00],
            [4.0000e+00, 6.0000e+00, 4.0000e+00, 6.0000e+00, 4.0000e+00,
             6.0000e+00],
            [0.0000e+00, 2.0000e+00, 0.0000e+00, 2.0000e+00, 0.0000e+00,
             2.0000e+00],
            [4.0000e+00, 6.0000e+00, 4.0000e+00, 6.0000e+00, 4.0000e+00,
             6.0000e+00]],
    <BLANKLINE>
           [[1.0000e+00, 3.0000e+00, 1.0000e+00, 3.0000e+00, 1.0000e+00,
             3.0000e+00],
            [5.0000e+00, 7.0000e+00, 5.0000e+00, 7.0000e+00, 5.0000e+00,
             7.0000e+00],
            [1.0000e+00, 3.0000e+00, 1.0000e+00, 3.0000e+00, 1.0000e+00,
             3.0000e+00],
            [5.0000e+00, 7.0000e+00, 5.0000e+00, 7.0000e+00, 5.0000e+00,
             7.0000e+00]]], dtype=float32)>
    '''
    if ndim(x)  not in (3,4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    elif x.shape[-1]%(block_size*block_size)!=0:
        raise ValueError('Input tensort channel must be divisible by square of block_size')
    else:
        orig_ndim=ndim(x)
        if orig_ndim ==3:
            x=expand_dims(x,0)
        x=tf.nn.depth_to_space(x,block_size=block_size,data_format='NHWC')
        if orig_ndim == 3:
            return x[0]
        return x



def space_to_depth(x:tf.Tensor,block_size=2):
    '''
        Rearranges elements in the input tensor from the spatial dimensions to the depth dimension.

        This is the reverse transformation of depth_to_space. This operation is useful for implementing and testing
        sub-pixel convolution that is part of models for image super-resolution .
        It rearranges elements of an input tensor of shape (N, C, H, W) to a tensor of shape (N, C*b*b, H/b, W/b),
        where b is the block_size,
        by rearranging non-overlapping spatial blocks of size block_size x block_size into the depth/channel
        dimension at each location.

        Args:
            x (tensor): Input tensor, with dimensions CHW or NCHW
            block_size (int):

        Returns: resized tensor
        Examples
        >>> arr=space_to_depth( to_tensor([[[0.,1. ],[2., 3.],[0.,1. ],[2., 3.],[0.,1. ],[2., 3.]],[[4., 5.],[6., 7.],[4., 5.],[6., 7.],[4., 5.],[6., 7.]],[[0.,1. ],[2., 3.],[0.,1. ],[2., 3.],[0.,1. ],[2., 3.]],[[4., 5.],[6., 7.],[4., 5.],[6., 7.],[4., 5.],[6., 7.]]]),block_size=2)
        >>> arr
        <tf.Tensor: shape=(2, 3, 8), dtype=float32, numpy=
        array([[[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00]],
        <BLANKLINE>
           [[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00]]], dtype=float32)>
        >>> print(arr.shape)
        (2, 3, 8)
        '''
    if ndim(x)  not in (3,4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    elif x.shape[-2]%block_size!=0 or  x.shape[-3]%block_size!=0:
        raise ValueError('Input tensort channel must be divisible by square of block_size')
    else:
        orig_ndim=ndim(x)
        if orig_ndim ==3:
            x=expand_dims(x,0)
        orig_shape=list(int_shape(x))
        x= tf.nn.space_to_depth(x,block_size=block_size,data_format='NHWC')
        if orig_ndim==3:
            return x[0]
        else:
            return x





############################
## tensor generation
###########################

def ones(shape,dtype=tf.float32,requires_grad=False):
    return tf.ones(shape,dtype)


def ones_like(a,dtype=tf.float32,requires_grad=False):
    return tf.ones_like(a,dtype)


def zeros(shape,dtype=tf.float32,requires_grad=False):
    return tf.zeros(shape, dtype)


def zeros_like(a,dtype=tf.float32,requires_grad=False):
    return tf.zeros_like(a,dtype)

def meshgrid(x, y, normalized_coordinates=False,requires_grad=False):
    '''Return meshgrid in range x & y.

    Args:
      requires_grad ():
      normalized_coordinates ():
      x: (int) first dim range.
      y: (int) second dim range.

    Returns:
      (tensor) meshgrid, sized [y,x,2]

    Example:
    >>> grid=meshgrid(3,2)
    >>> grid
    <tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
    array([[[0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00]],
    <BLANKLINE>
           [[1.0000e+00, 0.0000e+00],
            [1.0000e+00, 1.0000e+00]],
    <BLANKLINE>
           [[2.0000e+00, 0.0000e+00],
            [2.0000e+00, 1.0000e+00]]], dtype=float32)>
    >>> print(grid[0,0,:])
    tf.Tensor([0.0000e+00 0.0000e+00], shape=(2,), dtype=float32)
    >>> print(grid[:,0,0])
    tf.Tensor([0.0000e+00 1.0000e+00 2.0000e+00], shape=(3,), dtype=float32)
    >>> print(grid.shape)
    (3, 2, 2)
    >>> x = to_tensor([1, 2, 3])
    >>> y = to_tensor([4, 5, 6])
    >>> grid_x, grid_y = tf.meshgrid(x, y)
    >>> grid_x
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[1.0000e+00, 2.0000e+00, 3.0000e+00],
           [1.0000e+00, 2.0000e+00, 3.0000e+00],
           [1.0000e+00, 2.0000e+00, 3.0000e+00]], dtype=float32)>

    >>> grid_y
    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[4.0000e+00, 4.0000e+00, 4.0000e+00],
           [5.0000e+00, 5.0000e+00, 5.0000e+00],
           [6.0000e+00, 6.0000e+00, 6.0000e+00]], dtype=float32)>
    >>> meshgrid(3,2,normalized_coordinates=True)
    <tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
    array([[[0.0000e+00, 0.0000e+00],
            [0.0000e+00, 1.0000e+00]],
    <BLANKLINE>
           [[5.0000e-01, 0.0000e+00],
            [5.0000e-01, 1.0000e+00]],
    <BLANKLINE>
           [[1.0000e+00, 0.0000e+00],
            [1.0000e+00, 1.0000e+00]]], dtype=float32)>

    '''

    grid_list = tf.meshgrid(np.arange(0, x), np.arange(0, y))
    if normalized_coordinates==True:
        grid_list = tf.meshgrid(np.linspace(0, 1, int(x)),np.linspace(0, 1, int(y)))

    return transpose(tf.cast(tf.stack(grid_list, -1), tf.float32),[1,0,2])


def concate(x:List[tf.Tensor],axis=1):
    return tf.concat(concat_dim=axis,values=x)
