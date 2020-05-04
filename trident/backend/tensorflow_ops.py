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

__all__ = ['to_numpy', 'to_tensor','is_tensor','element_cosine_distance','is_nan','is_inf','is_abnormal_number','any_nan','any_inf','any_abnormal_number','is_sparse','ndim','is_sparse','int_shape','dot','clip','reduce_mean','reduce_max','reduce_min','reduce_sum','sqrt','square','abs','exp','log','pow','round','ceil','floor','concate','reshape','transpose','permute','squeeze','expand_dims','ones','ones_like','zeros','zeros_like','meshgrid','identity','sigmoid','tanh','relu','relu6','leaky_relu','leaky_relu6','smooth_relu','p_relu','swish','elu','hard_sigmoid','hard_swish','selu','lecun_tanh','soft_sign','soft_plus','hard_tanh','logit','log_log','mish','softmax','log_sum_exp','bert_gelu','gpt_gelu']

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
    if isinstance(x, int):
        return tf.constant(value=x,dtype=tf.int32)
    elif   isinstance(x, float):
        return tf.constant(value=x,dtype=tf.float32)
    else:
        if requires_grad == False:
            x =tf.stop_gradient(ops.convert_to_tensor_v2(x, dtype=dtype))
        elif requires_grad == True:
            x= ops.convert_to_tensor_v2(x, dtype=dtype)
        else :
            x =ops.convert_to_tensor_v2(x, dtype=dtype)
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
    (3, 3, 7)

    '''
    return x.get_shape().as_list()

def is_sparse(x):
    return isinstance(x, tf.SparseTensor)

def clip(x:tf.Tensor,min_value=-np.inf,max_value=np.inf):
    return tf.clip_by_value(x,min,max)

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
    >>> round(to_tensor([[1,2,3,4,5]])/3,-2)
    <tf.Tensor: shape=(1, 5), dtype=float32, numpy=
    array([[3.3000e-01, 6.7000e-01, 1.0000e+00, 1.3300e+00, 1.6700e+00]],
          dtype=float32)>

    '''
    if digit!=0:
        factor=float(math.pow(10,digit))
        return tf.math.round(x/factor)*factor
    else:
        return tf.math.round(x)


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

def log_sum_exp(x):
    """Activation function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x : input tensor
    """
    x_max = x.data.max()
    return log(reduce_sum(exp(x-x_max), 1, keepdims=True)) + x_max


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


def transpose(x:tf.Tensor,pattern=None)->tf.Tensor:
    return tf.transpose(x,pattern)


def permute(x:tf.Tensor,pattern=None)->tf.Tensor:
    return tf.transpose(x,pattern)






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
      (tensor) meshgrid, sized [x,y,2]

    Example:
    >>> grid=meshgrid(3,2)
    >>> grid
    <tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
    array([[[0.0000e+00, 0.0000e+00],
            [1.0000e+00, 0.0000e+00],
            [2.0000e+00, 0.0000e+00]],
    <BLANKLINE>
           [[0.0000e+00, 1.0000e+00],
            [1.0000e+00, 1.0000e+00],
            [2.0000e+00, 1.0000e+00]]], dtype=float32)>
    >>> print(grid[0,0,:])
     tensor([0., 0.])  tf.Tensor([0.0000e+00 0.0000e+00], shape=(2,), dtype=float32)
    >>> print(grid[:,0,0])
    tensor([0., 1., 2.]) tf.Tensor([0.0000e+00 0.0000e+00], shape=(2,), dtype=float32)
    >>> print(grid.shape)
    torch.Size([3, 2, 2])
    >>> print(grid.shape)
    torch.Size([3, 2, 2])

    >>> meshgrid(3,2,normalized_coordinates=True)
    tensor([[[0.0000, 0.0000],
             [0.0000, 1.0000]],
    <BLANKLINE>
            [[0.5000, 0.0000],
             [0.5000, 1.0000]],
    <BLANKLINE>
            [[1.0000, 0.0000],
             [1.0000, 1.0000]]])

    '''
    grid_list = tf.meshgrid(np.arange(0, x), np.arange(0, y))

    return tf.cast(tf.stack([grid_list[0], grid_list[1]], -1), tf.float32)


def concate(x:List[tf.Tensor],axis=1):
    return tf.concat(x,axis=axis)
