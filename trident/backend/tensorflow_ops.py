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
from trident.backend.common import to_list,unpack_singleton,epsilon


__all__ = ['is_tensor', 'to_numpy', 'to_tensor', 'ndim', 'int_shape','cast', 'is_sparse', 'is_nan', 'is_inf',
           'is_abnormal_number', 'any_nan', 'any_inf', 'any_abnormal_number', 'less', 'equal', 'greater',
           'greater_equal', 'not_equal', 'less_equal', 'argmax', 'argmin', 'argsort', 'maximum', 'minimum', 'floor',
           'ceil', 'round', 'dot', 'sqrt', 'square', 'abs', 'pow', 'log', 'exp', 'clip', 'add', 'subtract',
           'true_divide', 'pi', 'matmul', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
           'element_times', 'element_max', 'element_min', 'element_divide', 'element_cosine_distance', 'where',
           'reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min', 'mean', 'sum', 'max', 'min', 'reduce_logsumexp',
           'reduce_prod', 'depth_to_space', 'space_to_depth', 'identity', 'sigmoid', 'relu', 'relu6', 'leaky_relu',
           'leaky_relu6', 'smooth_relu', 'p_relu', 'swish', 'elu', 'hard_sigmoid', 'hard_swish', 'selu', 'lecun_tanh',
           'soft_sign', 'soft_plus', 'hard_tanh', 'logit', 'log_log', 'mish', 'softmax', 'log_softmax', 'gelu',
           'gpt_gelu','moments','l2_normalize', 'ones', 'ones_like', 'zeros', 'zeros_like','eye_like','arange', 'meshgrid', 'reshape', 'permute', 'transpose',
           'squeeze', 'expand_dims', 'concate', 'stack', 'gram_matrix', 'shuffle', 'random_choice','binary_crossentropy']



def is_tensor(x):
    """Checks whether `x` is exactly a tensor

      If `is_tensor(x)` returns `True`, it is safe to assume that `x` is a tensor or
      can be converted to a tensor using `ops.convert_to_tensor(x)`.

    Examples:

      >>> is_tensor(tf.constant([[1,2,3],[4,5,6],[7,8,9]]))
      True
      >>> is_tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]))
      False
      >>> is_tensor("Hello World")
      False

      Args:
        x: A python object to check.

      Returns:
        `True` if `x` is a tensor or "tensor-like", `False` if not.

    """
    if hasattr(x, 'numpy'):
        with context.eager_mode():
            return True
    elif x.__class__.__name__ == 'EagerTensor':
        return True
    elif isinstance(x, tf.Tensor):
        return True
    return False


def is_tensor_like(x):
    """Checks whether `x` is a "tensor-like".

      If `is_tensor_like(x)` returns `True`, it is safe to assume that `x` is a tensor or
      can be converted to a tensor using `ops.convert_to_tensor(x)`.

    Examples:

      >>> is_tensor_like(tf.constant([[1,2,3],[4,5,6],[7,8,9]]))
      True
      >>> is_tensor_like([[1,2,3],[4,5,6],[7,8,9]])
      True
      >>> is_tensor_like(np.array([[1,2,3],[4,5,6],[7,8,9]]))
      True
      >>> is_tensor_like ("Hello World")
      False

    Args:
        x: A python object to check.

    Returns:
        True` if `x` is a tensor or "tensor-like", `False` if not.

    """
    return tf.is_tensor(to_tensor(x))


def to_numpy(x) -> np.ndarray:
    """
     Convert whatever to numpy array

     Args
        x: List, tuple, PyTorch tensor or numpy array

     Returns
        Numpy array

     Examples
      >>> to_numpy(5)
      array([5])
      >>> to_numpy([1,2,3])
      array([1, 2, 3])
      >>> to_numpy((2,4),(1,3))
      array([[2, 4],
           [1, 3]])

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
    elif isinstance(x, (list,tuple)):
        return np.asarray(x)
    elif hasattr(x, '__len__') and len(x) > 1 and all( [isinstance(k, (list, tuple, int, float, np.ndarray)) for k in x]):
        x=unpack_singleton(x)
        return np.array([x])
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


def to_tensor(x, dtype=tf.float32, requires_grad=None) -> tf.Tensor:
    """
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
        <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00],
        dtype=float32)>
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

    """
    if isinstance(x, int):
        return tf.constant(value=x, dtype=tf.int32)
    elif isinstance(x, float):
        return tf.constant(value=x, dtype=tf.float32)
    else:
        try:

            if requires_grad == False:
                x = tf.constant(ops.convert_to_tensor_v2(x, dtype=dtype))
            else:
                x = ops.convert_to_tensor_v2(x, dtype=dtype)
            return x
        except:
            return x


############################
## tensor attribute
###########################

def ndim(x):
    """The number of dimensions of input tensor.

    Args:
        x (tf.Tensor): input tensor

    Returns: The number of dimensions

    """
    return x.shape.rank


def int_shape(x):
    """ Shape of input tensor in tuple of integer format

    Args:
        x : input tensor

    Returns: tuple of integer as shape representation

    Examples:
    >>> int_shape(ones((3,3,7)))
    [3, 3, 7]

    """
    return x.get_shape().as_list()


def is_sparse(x):
    """

    Args:
        x (tf.Tensor):

    Returns: if True, mean the input tensor is a sparse tensor.

    """
    return isinstance(x, tf.SparseTensor)


def cast(x, dtype):
    """

    Args:
        x (tf.Tensor): input tensor
        dtype (dtype or string):

    Returns:

    """
    if isinstance(dtype, str):
        if 'float64' in dtype.lower() or 'double' in dtype.lower():
            dtype = tf.float64
        elif 'float16' in dtype.lower() or 'half' in dtype.lower():
            dtype = tf.float16
        elif 'float' in dtype.lower():
            dtype = tf.float32
        elif 'int64' in dtype.lower() or 'long' in dtype.lower():
            dtype = tf.int64
        elif 'int16' in dtype.lower() or 'short' in dtype.lower():
            dtype = tf.int16
        elif 'uint8' in dtype.lower() or 'byte' in dtype.lower():
            dtype = tf.uint8
        elif 'int8' in dtype.lower() or 'char' in dtype.lower():
            dtype = tf.int8
        elif 'int32' in dtype.lower() or 'int' in dtype.lower():
            dtype = tf.int32
        elif 'bool' in dtype.lower():
            dtype = tf.bool
    if isinstance(dtype, tf.DType):
        return tf.cast(x, dtype)
    else:
        return x


############################
## check operation
###########################
def is_nan(x):
    if isinstance(x, (tf.Tensor, tf.Variable)) or is_tensor(x):
        if x.ndim == 0:
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
        if x.ndim == 0:
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
        if x.ndim == 0:
            return tf.math.is_nan(x).any()
        else:
            return tf.math.is_nan(x).numpy().any()
    elif isinstance(x, tf.Module):
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
        if x.ndim == 0:
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


def less(left: tf.Tensor, right: tf.Tensor):
    """
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
    """

    return tf.cast(tf.less(left, right), tf.float32)


def equal(left: tf.Tensor, right: tf.Tensor):
    """
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
    """
    return tf.cast(tf.equal(left, right), tf.float32)


def greater(left: tf.Tensor, right: tf.Tensor):
    """
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
    """
    return tf.cast(tf.greater(left, right), tf.float32)


def greater_equal(left: tf.Tensor, right: tf.Tensor):
    """
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
    """
    return tf.cast(tf.greater_equal(left, right), tf.float32)


def not_equal(left: tf.Tensor, right: tf.Tensor):
    """
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
    """
    return tf.cast(tf.not_equal(left, right), tf.float32)


def less_equal(left: tf.Tensor, right: tf.Tensor):
    """
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

    """
    return tf.cast(tf.less_equal(left, right), tf.float32)


def argmax(x: tf.Tensor, axis=-1) -> tf.Tensor:
    return tf.argmax(x, axis=axis)


def argmin(x: tf.Tensor, axis=-1) -> tf.Tensor:
    return tf.argmin(x, axis=axis)


def argsort(x: tf.Tensor, axis=-1, descending=True) -> tf.Tensor:
    return tf.argsort(x, axis=axis, descending=descending)


def maximum(x: tf.Tensor, other: (tf.Tensor, int, float)) -> tf.Tensor:
    if isinstance(other, tf.Tensor):
        return tf.maximum(x, other)
    elif isinstance(other, (int, float)):
        return clip(x, min=other)


def minimum(x: tf.Tensor, other: (tf.Tensor, int, float)) -> tf.Tensor:
    if isinstance(other, tf.Tensor):
        return tf.minimum(x, other)
    elif isinstance(other, (int, float)):
        return clip(x, max=other)


############################
## basic math operation
###########################

def floor(x: tf.Tensor):
    return tf.math.floor(x)


def ceil(x: tf.Tensor):
    return tf.math.ceil(x)


def round(x: tf.Tensor, digit: int = 0):
    """

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

    """
    if digit != 0:
        factor = float(math.pow(10, -1 * digit))
        return tf.math.round(x / factor) * factor
    else:
        return tf.math.round(x)


def add(x, y):
    return tf.add(x, y)


def subtract(x, y):
    return tf.subtract(x, y)


def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
 Args
        x: Tensor or variable.
        y: Tensor or variable.
 Returns
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
        return tf.reshape(tf.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if is_sparse(x):
        out = tf.sparse.sparse_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out


def matmul(a, b, transpose_a=False, transpose_b=False):
    return tf.matmul(a, b, transpose_a=transpose_b, transpose_b=transpose_b)


def true_divide(x, y):
    return tf.truediv(x, y)


def pi():
    return to_tensor(np.pi)


def sqrt(x: tf.Tensor):
    return tf.math.sqrt(x)


def square(x: tf.Tensor):
    return tf.math.square(x)


def abs(x: tf.Tensor):
    return tf.math.abs(x)


def pow(x: tf.Tensor, y):
    return tf.math.pow(x, y)


def log(x: tf.Tensor):
    return tf.math.log(x)


def exp(x: tf.Tensor):
    return tf.math.exp(x)


def prod(x):
    return tf.math.reduce_prod(x)


def clip(x: tf.Tensor, min=-np.inf, max=np.inf):
    return tf.clip_by_value(x, float(min), float(max))


def sin(x: tf.Tensor):
    return tf.math.sin(x)


def cos(x: tf.Tensor):
    return tf.math.cos(x)


def tan(x: tf.Tensor):
    return tf.math.tan(x)


def asin(x: tf.Tensor):
    return tf.math.asin(x)


def acos(x: tf.Tensor):
    return tf.math.acos(x)


def atan(x: tf.Tensor):
    return tf.math.atan(x)


def sinh(x: tf.Tensor):
    """
    Computes the element-wise sinh
    Args:
        x (tensor):input tensor

    Returns: element-wise sinh

    Examples
    >>> sinh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.1752e+00, 5.2110e-01],
           [-2.5261e-01, -8.2232e-01]], dtype=float32)>

    """
    return tf.sinh(x)


def cosh(x: tf.Tensor):
    """
    Computes the element-wise cosh
    Args:
        x (tensor):input tensor

    Returns: element-wise cosh

    Examples
    >>> cosh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[1.5431e+00, 1.1276e+00],
           [1.0314e+00, 1.2947e+00]], dtype=float32)>
    """
    return tf.cosh(x)


def tanh(x: tf.Tensor):
    """
    Computes the element-wise tanh
    Args:
        x (tensor):input tensor

    Returns: element-wise tanh

    Examples
    >>> tanh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
    tensor([[ 0.     ,  1.0472 ],
       [ 1.82348,  2.41886]])
    """
    return tf.tanh(x)


############################
## element-wise operation
###########################
def element_times(left, right):
    """
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
    """
    return left * right


def element_max(left, right):
    """
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
    """
    return maximum(left, right)


def element_min(left, right):
    """
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
    """
    return minimum(left, right)


def element_divide(left, right):
    """
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
    """
    return true_divide(left, right)


def element_cosine_distance(v1, v2, axis=1):
    normalize_a = tf.nn.l2_normalize(v1, axis)
    normalize_b = tf.nn.l2_normalize(v2, axis)
    distance = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance


def where(flag, value_if_true, value_if_false):
    """
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
    """
    return tf.where(flag, value_if_true, value_if_false)


############################
## reduce operation
###########################

def reduce_mean(x: tf.Tensor, axis=None, keepdims=False):
    return tf.math.reduce_mean(x, axis=axis, keepdims=keepdims)


def reduce_sum(x: tf.Tensor, axis=None, keepdims=False):
    return tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)


def reduce_max(x: tf.Tensor, axis=None, keepdims=False):
    return tf.math.reduce_max(x, axis=axis, keepdims=keepdims)


def reduce_min(x: tf.Tensor, axis=None, keepdims=False):
    return tf.math.reduce_min(x, axis=axis, keepdims=keepdims)


def reduce_logsumexp(x: tf.Tensor, axis=None, keepdims=False):
    """

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
    """
    return tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)


def reduce_prod(x: tf.Tensor, axis=None, keepdims=False):
    return tf.math.reduce_prod(x, axis=axis, keepdims=keepdims)


# reduce_l1
# reduce_l2
# reduce_sum_square

mean = reduce_mean
sum = reduce_sum
max = reduce_max
min = reduce_min


############################
## activationoperation
###########################


def identity(x):
    """Identity activation Layer
    A placeholder identity operator that is argument-insensitive.
    Examples:
        >>> Identity()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-3.0, -1.0, 0.0, 2.0])

    """
    return x


def sigmoid(x):
    return tf.nn.sigmoid(x)


def tanh(x):
    return tf.nn.tanh(x)


def relu(x, upper_limit=None):
    """Rectified Linear Unit activation function.
      With default values, it returns element-wise `max(x, 0)`.
      Otherwise, it follows:
      ```
        f(x) = max_value if x >= max_value
        f(x) = x if threshold <= x < max_value
        f(x) = negative_slope * (x - threshold) otherwise
      ```
    """
    if upper_limit is not None and upper_limit <= 0:
        raise ValueError('Upper limit should greater than 0!')
    elif upper_limit is not None:
        return clip(tf.nn.relu(x), 0, upper_limit)
    return tf.nn.relu(x)


def relu6(x):
    """Rectified Linear Unit  6 activation function.
      With default values, it returns element-wise `min(max(x, 0)`,6).
      Otherwise, it follows:
      ```
        f(x) = 6 if x >= 6
        f(x) = x if threshold <= x < 6
        f(x) = negative_slope * (x - threshold) otherwise
      ```
    """
    return relu(x,6)


def leaky_relu(x, alpha=0.02, upper_limit=None):
    """Leaky version of a Rectified Linear Unit.
        It allows a small gradient when the unit is not active:
        ```
        f(x) = alpha * x if x < 0
        f(x) = x if x >= 0
        ```
    """
    if upper_limit is not None:
        return clip(tf.nn.leaky_relu(x, alpha), -np.inf, upper_limit)
    return tf.nn.leaky_relu(x, alpha)


def leaky_relu6(x, alpha=0.01):
    """Leaky version of a Rectified Linear Unit.6
          It allows a small gradient when the unit is not active:
          ```
            f(x) = alpha * x if x < 0
            f(x) = x if  6>=x >= 0
            f(x) = 6 if  x > 6
          ```
    """
    return clip(tf.nn.leaky_relu(x, alpha), -6, 6)


def elu(x, alpha=0.01, upper_limit=None):
    """Exponential Linear Unit.
         It follows:
         ```
           f(x) =  alpha * (exp(x) - 1.) for x < 0
           f(x) = x for x >= 0
         ```
    """
    if upper_limit is not None:
        return clip(tf.nn.elu(x, alpha), -np.inf, upper_limit)
    return tf.nn.elu(x, alpha)


lrelu = leaky_relu


def smooth_relu(x, upper_limit=None):
    if upper_limit is not None:
        return clip(tf.math.log(1 + tf.math.exp(x)), -np.inf, upper_limit)
    return tf.math.log(1 + tf.math.exp(x))


def p_relu(x, upper_limit=None):
    if upper_limit is not None:
        return clip(tf.keras.layers.PReLU()(x), -np.inf, upper_limit)
    return tf.keras.layers.PReLU()(x)


def swish(x):
    """Self-Gated Activation Function.
      it follows:
      ```
        f(x) =  x * sigmoid(x)

      ```
    References:
        Swish: a Self-Gated Activation Function
        https://arxiv.org/abs/1710.05941v1

    Examples:
        >>> swish(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00], dtype=float32)>

    """
    return tf.nn.sigmoid(x) * x


def selu(x):
    return tf.nn.selu(x)


def soft_sign(x):
    return tf.nn.softsign(x)


def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(2 / 3 * x)


def soft_plus(x):
    return tf.nn.softplus(x)


def hard_sigmoid(x):
    return relu6(x + 3) / 6


def hard_tanh(x):
    return clip(x, -1, 1)


def hard_swish(x):
    return x * hard_sigmoid(x)


def logit(x):
    return tf.math.log(x / (1 - x))


def log_log(x):
    """LogLog Activation Function
      it follows:
      ```
        f(x) =  1 - exp(-exp(x))

      ```
    References:
        "Complementary Log-Log and Probit: Activation Functions Implemented in Artificial Neural Networks"
        https://ieeexplore.ieee.org/document/4626755/

    Examples:
        >>> LogLog()(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    """
    return 1 - tf.math.exp(-tf.math.exp(x))


def softmax(x, axis=-1):
    return tf.nn.softmax(x, axis=axis)


def log_softmax(x, axis=-1, keepdims=False):
    """Activation function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        keepdims ():
        axis ():
        x : input tensor
    """

    return x - reduce_logsumexp(x, axis=axis, keepdims=True)


def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


def hard_mish(x):
    return  x * hard_tanh(tf.nn.softplus(x))


def gelu(x):
    """
    Gaussian Error Linear Unit.
    it follows:
        ```
        f(x) =x∗Φ(x)
        where \Phi(x)Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

        ```

    References:
        Gaussian Error Linear Units (GELUs)
        https://arxiv.org/abs/1606.08415

    Examples:
        >>> gelu(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00], dtype=float32)>

    """
    return x * 0.5 * (1.0 + tf.nn.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))


def gpt_gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))


############################
## normalization operation
###########################

def moments(x:tf.Tensor, axis,  keepdims=True):
    """Calculates the mean and variance of `x`.

      The mean and variance are calculated by aggregating the contents of `x`
      across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
      and variance of a vector.

      Note: shift is currently not used; the true mean is computed and used.

      When using these moments for batch normalization (see
      `tf.nn.batch_normalization`):

       * for so-called "global normalization", used with convolutional filters with
         shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
       * for simple batch normalization pass `axes=[0]` (batch only).

      Args:
        x: A `Tensor`.
        axis: Array of ints.  Axes along which to compute mean and
          variance.
        shift: Not used in the current implementation.
        keepdims: produce moments with the same dimensionality as the input.

      Returns:
        Two `Tensor` objects: `mean` and `variance`.
      """
    return tf.nn.moments(x,axes=axis,keepdims=keepdims)



def l2_normalize(x:tf.Tensor,axis,  keepdims=True, eps=epsilon()):
    return x / (tf.norm(x,keepdims=keepdims)+eps)

############################
## tensor shape operation
###########################

def reshape(x: tf.Tensor, shape=None) -> tf.Tensor:
    if shape is None:
        return x
    elif isinstance(shape, tf.TensorShape):
        return tf.reshape(x, shape.as_list())
    elif isinstance(shape, (list, tuple)):
        return tf.reshape(x, to_list(shape))
    else:
        shape = to_list(shape)
        return tf.reshape(x, shape)


def squeeze(x: tf.Tensor, axis=None):
    return tf.squeeze(x, axis=axis)


def expand_dims(x: tf.Tensor, axis=None):
    return tf.expand_dims(x, axis=axis)


def transpose(x: tf.Tensor, perm=None) -> tf.Tensor:
    """
    Transposes a. Permutes the dimensions according to perm.
    The returned tensor's dimension i will correspond to the input dimension perm[i]. If perm is not given,
    it is set to (n-1...0), where n is the rank of the input tensor. Hence by default, this operation performs a
    regular matrix transpose on 2-D input Tensors.

    Example:
        >>> transpose(to_tensor( [[1 ,2 ,3],[4 ,5 ,6]]))
        <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
         array([[1.0000e+00, 4.0000e+00],
           [2.0000e+00, 5.0000e+00],
           [3.0000e+00, 6.0000e+00]], dtype=float32)>
        >>> transpose(to_tensor( [[1 ,2 ,3],[4 ,5 ,6]]),perm = to_tensor([1, 0],dtype=tf.int32))
        <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
         array([[1.0000e+00, 4.0000e+00],
           [2.0000e+00, 5.0000e+00],
           [3.0000e+00, 6.0000e+00]], dtype=float32)>
        >>> x1=to_tensor([[[1 ,2 ,3],[4 ,5 ,6]], [[7 ,8 ,9], [10,11,12]]])
        >>> transpose(x1, perm=to_tensor([0, 2, 1],dtype=tf.int32))
        <tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
        array([[[1.0000e+00, 4.0000e+00],
            [2.0000e+00, 5.0000e+00],
            [3.0000e+00, 6.0000e+00]],
        <BLANKLINE>
           [[7.0000e+00, 1.0000e+01],
            [8.0000e+00, 1.1000e+01],
            [9.0000e+00, 1.2000e+01]]], dtype=float32)>

    Args:
        x: A Tensor.
        perm: A permutation of the dimensions of ax

    Returns:
        A transposed Tensor.
    """
    if isinstance(perm, (list, tuple)):
        return tf.transpose(x, to_tensor(perm, dtype=tf.int32))
    elif perm is None:
        return tf.transpose(x)
    return tf.transpose(x, to_tensor(perm, dtype=tf.int32))


def permute(x: tf.Tensor, perm=None) -> tf.Tensor:
    if isinstance(perm, (list, tuple)):
        return tf.transpose(x, to_tensor(perm, dtype=tf.int32))
    elif perm is None:
        return tf.transpose(x)
    return tf.transpose(x, to_tensor(perm, dtype=tf.int32))


def depth_to_space(x: tf.Tensor, block_size=2):
    """
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
    """
    if ndim(x) not in (3, 4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    elif x.shape[-1] % (block_size * block_size) != 0:
        raise ValueError('Input tensort channel must be divisible by square of block_size')
    else:
        orig_ndim = ndim(x)
        if orig_ndim == 3:
            x = expand_dims(x, 0)
        x = tf.nn.depth_to_space(x, block_size=block_size, data_format='NHWC')
        if orig_ndim == 3:
            return x[0]
        return x


def space_to_depth(x: tf.Tensor, block_size=2):
    """
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
        >>> arr=space_to_depth( to_tensor([[[0.,1. ],[2., 3.],[0.,1. ],[2., 3.],[0.,1. ],[2., 3.]],[[4., 5.],[6.,
        7.],[4., 5.],[6., 7.],[4., 5.],[6., 7.]],[[0.,1. ],[2., 3.],[0.,1. ],[2., 3.],[0.,1. ],[2., 3.]],[[4., 5.],
        [6., 7.],[4., 5.],[6., 7.],[4., 5.],[6., 7.]]]),block_size=2)
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
        """
    if ndim(x) not in (3, 4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    elif x.shape[-2] % block_size != 0 or x.shape[-3] % block_size != 0:
        raise ValueError('Input tensort channel must be divisible by square of block_size')
    else:
        orig_ndim = ndim(x)
        if orig_ndim == 3:
            x = expand_dims(x, 0)
        orig_shape = list(int_shape(x))
        x = tf.nn.space_to_depth(x, block_size=block_size, data_format='NHWC')
        if orig_ndim == 3:
            return x[0]
        else:
            return x


############################
## tensor generation
###########################

def ones(shape, dtype=tf.float32, requires_grad=None):
    t= tf.ones(shape, dtype)
    if requires_grad==False:
        return tf.constant(t)
    else:
        return t



def ones_like(a, dtype=tf.float32, requires_grad=None):
    t= tf.ones_like(a, dtype)
    if requires_grad==False:
        return tf.constant(t)
    else:
        return t


def zeros(shape, dtype=tf.float32, requires_grad=None):
    t= tf.zeros(shape, dtype)
    if requires_grad==False:
        return tf.constant(t)
    else:
        return t


def zeros_like(a, dtype=tf.float32, requires_grad=None):
    t= tf.zeros_like(a, dtype)
    if requires_grad==False:
        return tf.constant(t)
    else:
        return t

def eye_like(a, dtype=tf.float32, requires_grad=None):
    """
    Creates a matrix with diagonal set to 1s and of the same shape and the same dynamic axes as ``x``. To be a
    matrix,
     ``x`` must have exactly two axes (counting both dynamic and static axes).

    Args:
        a: numpy array or  that outputs a tensor of rank 2
        requires_grad ():
        dtype ():

    Returns:
        tensor

    Example:
    >>> eye_like(to_tensor(np.random.standard_normal((3,4))))
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]])

    """
    if a.ndim == 2:
        t= tf.eye(a.shape[0], a.shape[1], dtype=dtype, requires_grad=requires_grad)
        if requires_grad == False:
            return tf.constant(t)
        else:
            return t
    else:
        raise ValueError('input tensor must have exactly two axe.')


def arange( *args,dtype=tf.int32, requires_grad=None):
    """

    Args:
        *args (int): the start, end, step
        dtype (dtype): dtype of the tensor
        requires_grad (bool): wheather need require gradient.

    Returns:

    """
    t=None
    if len(args)==1:
        t= tf.range(start=0,limit=args[0],dtype=dtype)
    elif len(args) == 2:
        t= tf.range(start=args[0],limit=args[1],dtype=dtype)
    elif len(args) == 3:
        t= tf.range(start=args[0],limit=args[1],delta=args[2],dtype=dtype)
    else:
        raise ValueError('only maximum  3 args in arange function ')
    if requires_grad==False:
        return tf.constant(t)
    return t


def meshgrid(x, y, normalized_coordinates=False, requires_grad=None):
    """Return meshgrid in range x & y.

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

    """

    grid_x, grid_y = tf.meshgrid(np.arange(0, x), np.arange(0, y))
    if normalized_coordinates == True:
        grid_x, grid_y = tf.meshgrid(np.linspace(0, 1, int(x)), np.linspace(0, 1, int(y)))

    t= transpose(tf.cast(tf.stack([grid_y, grid_x], -1), tf.float32), [1, 0, 2])
    if requires_grad==False:
        return tf.constant(t)
    elif requires_grad==True:
        return tf.Variable(t)
    else:
        return t


############################
## tensor manipulation
###########################

def concate(x: List[tf.Tensor], axis=1):
    return tf.concat(x, axis=axis)


def stack(x: List[tf.Tensor], axis=1):
    return tf.stack(x, axis=axis)


def gram_matrix(x: tf.Tensor):
    temp = x
    temp = squeeze(temp)
    fun = reshape(temp, [temp.shape[2], temp.shape[0] * temp.shape[1]])
    result = matmul(temp, temp, transpose_b=True)
    gram = expand_dims(result, axis=0)
    return gram


############################
## random
###########################


def shuffle(x: tf.Tensor):
    return tf.random.shuffle(x)


def random_choice(x: tf.Tensor):
    idx = np.random.choice(np.array(range(x.size(0))))
    return x[idx]




def binary_crossentropy(target, output, from_logits=False):
  """
  Binary crossentropy between an output tensor and a target tensor.
  Args:
      target: A tensor with the same shape as `output`.
      output: A tensor.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.
  Returns:
      A tensor.
  """

  # Compute cross entropy from probabilities.
  bce = target *log(output + epsilon())
  bce += (1 - target) *log(1 - output + epsilon())
  return -bce