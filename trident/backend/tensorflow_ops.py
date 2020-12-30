"""trident tensorflow basic operation."""
from copy import deepcopy
import math
import builtins
import numbers
from enum import Enum
from functools import wraps
from typing import List, Optional,Tuple
from types import MethodType
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import state_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops, dtypes
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops import math_ops

from trident.backend.common import to_list, unpack_singleton, epsilon, OrderedDict, get_function, get_session, TensorShape

__all__ = ['Tensor','is_tensor',  'is_tensor_like','to_numpy', 'to_tensor', 'ndim','numel', 'int_shape','tensor_to_shape','str2dtype','cast', 'is_sparse', 'is_nan', 'is_inf',
           'is_abnormal_number', 'any_nan', 'any_inf', 'any_abnormal_number', 'less', 'equal', 'greater',
           'greater_equal', 'not_equal', 'less_equal', 'argmax', 'argmin', 'argsort','topk', 'maximum', 'minimum', 'floor',
           'ceil', 'round', 'dot', 'sqrt','rsqrt' ,'square', 'abs', 'pow', 'log', 'exp', 'clip', 'add', 'subtract',
           'true_divide', 'pi', 'matmul', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
           'element_times', 'element_max', 'element_min', 'element_divide', 'element_cosine_distance', 'where',
           'reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min', 'mean', 'sum', 'max', 'min', 'reduce_logsumexp',
           'reduce_prod', 'reduce_any', 'depth_to_space', 'space_to_depth', 'identity', 'sigmoid', 'relu', 'relu6', 'leaky_relu',
           'leaky_relu6', 'smooth_relu', 'p_relu', 'swish', 'elu', 'hard_sigmoid', 'hard_swish', 'selu', 'lecun_tanh',
           'soft_sign', 'soft_plus', 'hard_tanh', 'logit', 'log_log', 'mish','hard_mish', 'softmax', 'log_softmax', 'gelu',
           'gpt_gelu','moments','norm','l2_normalize','spectral_norm', 'ones', 'ones_like', 'zeros', 'zeros_like','eye','eye_like','arange','make_onehot', 'meshgrid', 'reshape', 'permute', 'transpose',
           'squeeze', 'expand_dims', 'concate', 'stack','split','repeat_elements','gather','scatter_add','scatter_sub','scatter_max','scatter_min','assign','assign_add','assign_sub','gram_matrix','set_seed',
           'shuffle', 'random_choice','random_normal','random_normal_like','random_uniform','random_uniform_like','multinomial','binary_crossentropy']

Tensor=tf.Tensor


def numpy_compatible(func):
    """decorator for function to support non-tensor input

    Args:
        func : wrapped function

    Returns:

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if func.__name__ in ('max','min','abs','round','pow') and isinstance(args[0],tuple):
            args=unpack_singleton(args)

        x = args[0] if hasattr(args,'__len__') else args
        new_args = []
        new_kwargs = OrderedDict()


        if all([isinstance(arg, numbers.Number) for arg in args]) and (len(kwargs)==0 or all([isinstance(kv[1], numbers.Number) for kv in kwargs.items()])) and  func.__name__ in ('max','min','abs','round','pow'):
            builtins_funcs = get_function(func.__name__, ['builtins'])
            y = builtins_funcs(*args, **kwargs)
            return y
        elif all([isinstance(arg,numbers.Number) for arg in args]) and (len(kwargs)==0 or all([isinstance(kv[1],numbers.Number) for kv in kwargs.items()]) )and  get_function(func.__name__, ['math','numpy','trident.backend.numpy_ops']) is not None:
            mathfuncs=get_function(func.__name__, ['math','numpy','trident.backend.numpy_ops'])
            y = mathfuncs(*args, **kwargs)
            return y
        elif isinstance(x, np.ndarray):
            numpy_func = get_function(func.__name__, ['trident.backend.numpy_ops'])
            if numpy_func is not None:
                for arg in args:
                    if is_tensor(arg):
                        new_args.append(to_numpy(arg))
                    else:
                        new_args.append(arg)
                for k, v in kwargs.items():
                    if is_tensor(v):
                        new_kwargs[k] = to_numpy(v)
                    else:
                        new_kwargs[k] = v
                y = numpy_func(*new_args, **new_kwargs)
                return y
            else:
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        new_args.append(to_tensor(arg))
                    else:
                        new_args.append(arg)
                for k, v in kwargs.items():
                    if isinstance(v, np.ndarray):
                        new_kwargs[k] = to_tensor(v)
                    else:
                        new_kwargs[k] = v
                y = func(*new_args, **new_kwargs)
                return y
        else:
            for arg in args:
                if isinstance(arg, np.ndarray):
                    new_args.append(to_tensor(arg))
                else:
                    new_args.append(arg)
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    new_kwargs[k] = to_tensor(v)
                else:
                    new_kwargs[k] = v
            y = func(*new_args, **new_kwargs)
            return y

    return wrapper


def _get_device():
    return get_session().device


def detach(x:EagerTensor):
    return x._copy_nograd()


def is_tensor(x):
    """Checks whether `x` is exactly a tensor

    If `is_tensor(x)` returns `True`, that `x` is a EagerTensor .

    Args:
        x: A python object to check.

    Returns:
        `True` if `x` is exactly a tensor, `False` if not.

    Examples:
        >>> is_tensor(tf.constant([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        False
        >>> is_tensor("Hello World")
        False

    """

    if isinstance(x,EagerTensor):
        return True
    elif x.__class__.__name__ == 'EagerTensor':
        return True
    elif isinstance(x, Tensor):
        return True
    elif isinstance(x, tf.Variable):
        return True
    return False


def is_tensor_like(x):
    """Checks whether `x` is a "tensor-like".

    If `is_tensor_like(x)` returns `True`, it is safe to assume that `x` is a tensor or can be converted to a tensor using `ops.convert_to_tensor(x)`.

    Args:
        x: A python object to check.

    Returns:
        True` if `x` is a tensor or "tensor-like", `False` if not.


    Examples:
        >>> is_tensor_like(tf.constant([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor_like([[1,2,3],[4,5,6],[7,8,9]])
        True
        >>> is_tensor_like(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor_like ("Hello World")
        False

    """
    return tf.is_tensor(to_tensor(x))


def to_numpy(x) -> np.ndarray:
    """Convert whatever to numpy array

     Args:
        x: List, tuple, PyTorch tensor or numpy array

     Returns:
        Numpy array

     Examples:
          >>> to_numpy(5)
          array([5])
          >>> to_numpy([1,2,3])
          array([1, 2, 3])
          >>> to_numpy((2,4),(1,3))
          array([[2, 4],
               [1, 3]])

     """

    if x is None:
        return x
    elif isinstance(x, TensorShape):
        return np.array(x.dims)
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, tf.Variable):
        return x.value()._copy_nograd().numpy()
    elif context.executing_eagerly() and isinstance(x, EagerTensor):
        return x._copy_nograd().numpy()
    elif isinstance(x, tf.TensorShape):
        return np.array(deepcopy(x).as_list())
    elif isinstance(x, Tensor):
        return  ops.convert_to_tensor_v2(x)._copy_nograd().numpy()
    elif isinstance(x, (list,tuple)):
        return np.asarray(x)
    elif hasattr(x, '__len__') and len(x) > 1 and all( [isinstance(k, (list, tuple,numbers.Number, np.ndarray)) for k in x]):
        x=unpack_singleton(x)
        return np.array([x])
    # elif isinstance(x, ops.Tensor):
    #     sess = tf.compat.v1.Session()
    #     x= sess.run(x)
    #     return x

    elif isinstance(x, (list, tuple,numbers.Number)):
        return np.array(x)
    else:
        try:

            if not getattr(x, '_in_graph_mode', True):
                # This is a variable which was created in an eager context, but is being
                # evaluated from a Graph.
                with context.eager_mode():
                    return x.numpy()
            with x.graph.as_default():
                return x.eval(session=get_session((x,)))

        except:
            raise ValueError("Unsupported type")

def to_tensor(x, dtype=None,device=None, requires_grad=None) -> Tensor:
    """Convert the input `x` to a tensor of type `dtype`.

    Args:

        device ():
        x: An object to be converted (ex.numpy array, list, tensors).
        dtype (str or tf.Dtype): The destination type or type string.
        requires_grad (None or bool): whether need grade

    Returns:
        A tensor.

    Examples:
        >>> to_tensor(2)
        <Tensor: shape=(), dtype=int32, numpy=2>
        >>> to_tensor([1.0,2.0,3.0],requires_grad=True)
        <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00],
        dtype=float32)>
        >>> to_tensor([1.0,2.0,3.0],requires_grad=False)
        <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor([1.0,2.0,3.0])
        <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor((1.0,2.0,3.0))
        <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor(np.arange(0,5))
        <Tensor: shape=(5,), dtype=float32, numpy=
        array([0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00],
        dtype=float32)>

    """
    if device is None:
        device=_get_device()

    if isinstance(dtype, str):
        dtype = str2dtype(dtype)
    elif isinstance(x, np.ndarray) and (x.dtype==np.int64 or x.dtype==np.int32):
        dtype = tf.int64
    elif isinstance(x, (tuple, list)) and all([isinstance(item, numbers.Integral)  for item in x]):
        dtype = str2dtype('int')
        x=to_numpy(x)
    elif isinstance(x, (tuple, list)) and all([isinstance(item, numbers.Number)  for item in x]):
        x=to_numpy(x)
    elif dtype is None:
        if is_tensor(x):
            dtype = x.dtype
        else:
            dtype = tf.float32
    with tf.device(device):
        return tf.convert_to_tensor(x, dtype=dtype,)




    # if isinstance(x, int):
    #     x= tf.constant(value=x, dtype=tf.int32)
    # elif is_tensor(x):
    #     if x.dtype!=dtype:
    #         x=tf.cast(x,dtype)
    #
    # elif isinstance(x, float):
    #     x= tf.constant(value=x, dtype=tf.float32)
    # else:
    #     try:
    #
    #         if requires_grad == False:
    #             x =tf.constant(x, dtype=dtype)
    #         else:
    #             x = ops.convert_to_tensor(x, dtype=dtype)
    #     except:
    #         pass
    #
    # if dtype is not None :
    #     x = cast(x, dtype)
    # return x

def copy(x:Tensor):
    return tf.identity(x)

############################
## tensor attribute
###########################

def ndim(x):
    """The number of dimensions of input tensor.

    Args:
        x (Tensor): input tensor

    Returns:
        (int) The number of dimensions

    """
    return x.shape.rank


@numpy_compatible
def numel(x,name='numel'):
    """The number of elements of input tensor.

    Args:
        x (Tensor): input tensor
        name (str):Name

    Returns:
        (int) The number of elements

    """
    return tf.size(x,name=name)

@numpy_compatible
def int_shape(x):
    """ Shape of input tensor in tuple of integer format

    Args:
        x : input tensor

    Returns: tuple of integer as shape representation

    Examples:
    >>> int_shape(ones((3,3,7)))
    [3, 3, 7]

    """
    return tuple(x.get_shape().as_list())


def tensor_to_shape(x:Tensor,need_exclude_batch_axis=True):
    if need_exclude_batch_axis:
        shp=list(int_shape(x))
        shp[0]=None
        return TensorShape(shp)
    else:
        return TensorShape(int_shape(x))



def is_sparse(x):
    """

    Args:
        x (Tensor):

    Returns: if True, mean the input tensor is a sparse tensor.

    """
    return isinstance(x, tf.SparseTensor)



def str2dtype(dtype):
    """string to dtype
    Args:
        dtype ():
    """
    if isinstance(dtype, tf.DType):
        return dtype
    if isinstance(dtype, str):
        if 'float64' in dtype.lower() or 'double' in dtype.lower():
            dtype = tf.float64
        elif 'float32' in dtype.lower() or 'single' in dtype.lower():
            dtype = tf.float32
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
    if dtype is None:
        return tf.float32
    return dtype

@numpy_compatible
def cast(x, dtype):
    """Casts a tensor to a new type.

    The operation casts `x` (in case of `Tensor`) or `x.values`
    (in case of `SparseTensor` or `IndexedSlices`) to `dtype`.

    The operation supports data types (for `x` and `dtype`) of
    `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,
    `float16`, `float32`, `float64`, `complex64`, `complex128`, `bfloat16`.
    In case of casting from complex types (`complex64`, `complex128`) to real
    types, only the real part of `x` is returned. In case of casting from real
    types to complex types (`complex64`, `complex128`), the imaginary part of the
    returned value is set to `0`. The handling of complex types here matches the
    behavior of numpy.

    Args:
        x: A `Tensor` or `SparseTensor` or `IndexedSlices` of numeric type.
        dtype: The destination type. The list of supported dtypes and string is the same as
        `x`.


    Returns:
        A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` and
        same type as `dtype`.

    Examples:
        >>> x = to_tensor([1.8, 2.2])
        >>>cast(x, tf.int32)
        <tensor=array([1, 2], dtype=int32)>

    Raises:
        TypeError: If `x` cannot be cast to the `dtype`.

    """
    dtype=str2dtype(dtype)
    if isinstance(dtype, tf.DType):
        return tf.cast(x, dtype)
    else:
        return x


def float(x):
    return cast(x,tf.float32)

def int(x):
    return cast(x,tf.int32)

def long(x):
    return cast(x,tf.int64)

def cpu(x):
    with tf.device('/CPU:0'):
        x_new=x
        return x_new

def cuda(x, device=None):
    r"""Moves all model parameters and buffers to the GPU.

    This also makes associated parameters and buffers different objects. So
    it should be called before constructing optimizer if the module will
    live on GPU while being optimized.

    Args:
        x (Tensor): input tensor
        device (int, optional): if specified, all parameters will be
            copied to that device

    Returns:
        Module: self
    """
    if tf.test.is_gpu_available:
        if device is None:
            device = '/gpu:0'
        with tf.device(device):
            x_new = x
            return x_new
    else:
        return x

def to(x, *args):
    if 'cpu' in args:
        return cpu(x)
    elif 'gpu' in args or 'cuda' in args:
        return cuda(x)
    elif 'float' in args:
        return cast(x,tf.float32)
    elif 'long' in args:
        return cast(x,tf.int64)
    elif 'int' in args:
        return cast(x,tf.int32)
    else:
        return x


############################
## check operation
###########################

def is_nan(x):
    if isinstance(x, tf.Variable):
        x = x.value()
    if is_tensor(x):
        return tf.math.is_nan(x)
    elif 'Layer' in x.__class__.__name__:
        return [tf.math.is_inf(para.value()) for para in x.weights]
    elif isinstance(x, np.ndarray):
        return np.isnan(x)
    else:
        raise NotImplementedError


def is_inf(x):
    if isinstance(x, tf.Variable):
        x = x.value()
    if is_tensor(x):
        return tf.math.is_inf(x)
    elif 'Layer' in x.__class__.__name__:
        return [tf.math.is_inf(para.value())for para in x.weights]
    elif isinstance(x, np.ndarray):
        return np.isinf(x)
    else:
        raise NotImplementedError


def is_abnormal_number(x):
    return cast(greater(cast(is_nan(x), tf.int8) + cast(is_inf(x), tf.int8), 0),tf.bool)


def any_nan(x):
    if isinstance(x,tf.Variable):
        x=x.value()
    if is_tensor(x):
        if x.ndim == 0:
            return tf.math.is_nan(x)
        else:
            return tf.math.reduce_any(tf.math.is_nan(x))
    elif isinstance(x, tf.Module):
        for para in x.weights:
            if tf.math.reduce_any(tf.math.is_nan(para.value())):
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isnan(x).any()
    else:
        raise NotImplementedError

def any_inf(x):
    if isinstance(x, tf.Variable):
        x = x.value()
    if is_tensor(x):
        if x.ndim == 0:
            return tf.math.is_inf(x)
        else:
            return tf.math.reduce_any(tf.math.is_inf(x))

    elif isinstance(x, tf.Module):
        for para in x.weights:
            if tf.math.reduce_any(tf.math.is_inf(para.value())):
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isinf(x).any()
    else:
        raise NotImplementedError

def any_abnormal_number(x):
    return any_nan(x) |any_inf(x)


############################
## compare operation
###########################
@numpy_compatible
def less(left: Tensor, right:(Tensor,np.ndarray,numbers.Number),name='less'):
    """Elementwise 'less' comparison of two tensors. Result is 1 if left < right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        name(str):op name

    Returns:
        Result is 1 if left < right else 0.

    Examples:
       >>> less(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
       <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 0.0000e+00, 0.0000e+00], dtype=float32)>
       >>> less(to_tensor([-1,0,1]), 0)
       <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 0.0000e+00, 0.0000e+00], dtype=float32)>

    """

    return tf.cast(tf.less(left, right,name=name), tf.float32,name='cast')

@numpy_compatible
def equal(left: Tensor, right:(Tensor,np.ndarray,numbers.Number),name='equal'):
    """
    Elementwise 'equal' comparison of two tensors. Result is 1 if values are equal 0 otherwise.

    Args:
        left: left side tensor
        right: right side tensor
        name(str):op name
    Returns:
        :Result is 1 if values are equal 0 otherwise

    Examples:
        >>> equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        <Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 1.0000e+00, 0.0000e+00], dtype=float32)>
        >>> equal(to_tensor([-1,0,1]), 1)
        <Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>

    """
    return tf.cast(tf.equal(left, right,name=name), tf.float32,name='cast')

@numpy_compatible
def greater(left: Tensor, right:(Tensor,np.ndarray,numbers.Number),name='greater'):
    """
    Elementwise 'greater' comparison of two tensors. Result is 1 if left > right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        name(str):op name

    Returns:
        :Result is 1 if left > right else 0.

    Examples:
        >>> greater(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        <Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>
        >>> greater(to_tensor([-1,0,1]), 0)
        <Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>

    """
    return tf.cast(tf.greater(left, right,name=name), tf.float32,name='cast')

@numpy_compatible
def greater_equal(left: Tensor, right:(Tensor,np.ndarray,numbers.Number),name='greater_equal'):
    """Elementwise 'greater equal' comparison of two tensors. Result is 1 if left >= right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        name(str):op name

    Returns:
        :Result is 1 if left >= right else 0

    Examples:
        >>> greater_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        <Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 1.0000e+00, 1.0000e+00], dtype=float32)>
        >>> greater_equal(to_tensor([-1,0,1]), 0)
        <Tensor: shape=(3,), dtype=float32, numpy=array([0.0000e+00, 1.0000e+00, 1.0000e+00], dtype=float32)>

    """
    return tf.cast(tf.greater_equal(left, right,name=name), tf.float32,name='cast')

@numpy_compatible
def not_equal(left: Tensor, right:(Tensor,np.ndarray,numbers.Number),name='not_equal'):
    """Elementwise 'not equal' comparison of two tensors. Result is 1 if left != right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        name(str):op name

    Returns:
        :Result is 1 if left != right else 0.

    Examples:
        >>> not_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>
        >>> not_equal(to_tensor([-1,0,1]), 0)
        <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 0.0000e+00, 1.0000e+00], dtype=float32)>

    """
    return tf.cast(tf.not_equal(left, right,name=name), tf.float32,name='cast')

@numpy_compatible
def less_equal(left: Tensor, right:(Tensor,np.ndarray,numbers.Number),name='less_equal'):
    """Elementwise 'less equal' comparison of two tensors. Result is 1 if left <= right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        name(str):op name

    Returns:
        :Result is 1 if left <= right else 0.

    Examples:
        >>> less_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 1.0000e+00, 0.0000e+00], dtype=float32)>
        >>> less_equal(to_tensor([-1,0,1]), 0)
        <Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 1.0000e+00, 0.0000e+00], dtype=float32)>

    """
    return tf.cast(tf.less_equal(left, right,name=name), tf.float32,name='cast')

@numpy_compatible
def argmax(x: Tensor, axis=-1,name='argmax') -> Tensor:
    """Returns the index with the largest value across axes of a tensor.

    In case of identity returns the smallest index.

    For example:

    >>> A = tf.constant([2, 20, 30, 3, 6])
    >>> argmax(A)  # A[2] is maximum in tensor A
    <tf.Tensor: shape=(), dtype=int64, numpy=2>
    >>> B = tf.constant([[2, 20, 30, 3, 6], [3, 11, 16, 1, 8],
    ...                  [14, 45, 23, 5, 27]])
    >>> argmax(B, 0)
    <tf.Tensor: shape=(5,), dtype=int64, numpy=array([2, 2, 0, 2, 2])>
    >>> argmax(B, 1)
    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 2, 1])>
    >>> C = tf.constant([0, 0, 0, 0])
    >>> argmax(C) # Returns smallest index in case of ties
    <tf.Tensor: shape=(), dtype=int64, numpy=0>

    Args:
      x: A `Tensor`.
      axis: An integer, the axis to reduce across. Default to 0.
      output_type: An optional output dtype (`tf.int32` or `tf.int64`). Defaults
        to `tf.int64`.
      name: An optional name for the operation.

    Returns:
      A `Tensor` of type `output_type`.
    """
    return tf.math.argmax(x, axis=axis,name=name)

@numpy_compatible
def argmin(x: Tensor, axis=-1,name='argmin') -> Tensor:
    """Returns the index with the smallest value across axes of a tensor.

     Returns the smallest index in case of ties.

     Args:
       x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
         `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`,
         `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`,
         `uint64`.
       axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
         int32 or int64, must be in the range `-rank(input), rank(input))`.
         Describes which axis of the input Tensor to reduce across. For vectors,
         use axis = 0.
       output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to
         `tf.int64`.
       name: A name for the operation (optional).

     Returns:
       A `Tensor` of type `output_type`.

     Usage:
     ```python
     import tensorflow as tf
     a = [1, 10, 26.9, 2.8, 166.32, 62.3]
     b = argmin(input = a)
     c = tf.keras.backend.eval(b)
     # c = 0
     # here a[0] = 1 which is the smallest element of a across axis 0
     ```
     """
    return tf.math.argmin(x, axis=axis,name=name)

@numpy_compatible
def argsort(x: Tensor, axis=-1, descending=True,name='argsort') -> Tensor:
    """Returns the indices of a tensor that give its sorted order along an axis.

    For a 1D tensor, `tf.gather(values, tf.argsort(values))` is equivalent to
    `tf.sort(values)`. For higher dimensions, the output has the same shape as
    `values`, but along the given axis, values represent the index of the sorted
    element in that slice of the tensor at the given position.

    Usage:

    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = argsort(a,axis=-1,direction='ASCENDING',stable=False,name=None)
    c = tf.keras.backend.eval(b)
    # Here, c = [0 3 1 2 5 4]
    ```

    Args:
      x: 1-D or higher numeric `Tensor`.
      axis: The axis along which to sort. The default is -1, which sorts the last
        axis.
      descending: The direction in which to sort the values (`'ASCENDING'` or
        `'DESCENDING'`).
      name: Optional name for the operation.

    Returns:
      An int32 `Tensor` with the same shape as `values`. The indices that would
          sort each slice of the given `values` along the given `axis`.

    Raises:
      ValueError: If axis is not a constant scalar, or the direction is invalid.
    """
    return tf.argsort(x, axis=axis,direction='DESCENDING' if descending else  'ASCENDING',name=name)

@numpy_compatible
def topk(x: Tensor,  k=1, name='topk') -> Tensor:
    """Finds values and indices of the `k` largest entries for the last dimension.

     If the input is a vector (rank=1), finds the `k` largest entries in the vector
     and outputs their values and indices as vectors.  Thus `values[j]` is the
     `j`-th largest entry in `input`, and its index is `indices[j]`.

     For matrices (resp. higher rank input), computes the top `k` entries in each
     row (resp. vector along the last dimension).  Thus,

         values.shape = indices.shape = input.shape[:-1] + [k]

     If two elements are equal, the lower-index element appears first.

     Args:
       x: 1-D or higher `Tensor` with last dimension at least `k`.
       k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last
         dimension (along each row for matrices).

       name: Optional name for the operation.

     Returns:
       values: The `k` largest elements along each last dimensional slice.
       indices: The indices of `values` within the last dimension of `input`.
     """
    return tf.math.top_k(input=x,k=k,sorted=True,name=name)



@numpy_compatible
def maximum(x: Tensor, other: (Tensor, int, float)) -> Tensor:
    if isinstance(other, Tensor):
        return tf.maximum(x, other)
    elif isinstance(other, numbers.Number):
        return clip(x, min=other)

@numpy_compatible
def minimum(x: Tensor, other: (Tensor, int, float)) -> Tensor:
    if isinstance(other, Tensor):
        return tf.minimum(x, other)
    elif isinstance(other, numbers.Number):
        return clip(x, max=other)


############################
## basic math operation
###########################

@numpy_compatible
def add(x, y):
    """Returns x + y element-wise.

    *NOTE*: `math.add` supports broadcasting. `AddN` does not. More about broadcasting
    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    Args:
      x: A tensor, np.ndarray or a number
      y: A tensor, np.ndarray or a number

    Returns:
      A tensor, np.ndarray or a number

    """

    return tf.add(x, y)

@numpy_compatible
def subtract(x, y):
    """Returns x - y element-wise.

    *NOTE*: `Subtract` supports broadcasting. More about broadcasting
    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    Args:
      x: A `Tensor`.
      y: A `Tensor`.


    Returns:
      A `Tensor`. Has the same type as `x`.
    """
    return tf.subtract(x, y)

@numpy_compatible
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

@numpy_compatible
def matmul(a, b, transpose_a=False, transpose_b=False):
    """Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

     The inputs must, following any transpositions, be tensors of rank >= 2
     where the inner 2 dimensions specify valid matrix multiplication dimensions,
     and any further outer dimensions specify matching batch size.

     Both matrices must be of the same type. The supported types are:
     `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

     Either matrix can be transposed or adjointed (conjugated and transposed) on
     the fly by setting one of the corresponding flag to `True`. These are `False`
     by default.

     If one or both of the matrices contain a lot of zeros, a more efficient
     multiplication algorithm can be used by setting the corresponding
     `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
     This optimization is only available for plain matrices (rank-2 tensors) with
     datatypes `bfloat16` or `float32`.

     A simple 2-D tensor matrix multiplication:


     >>> a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
     >>> a  # 2-D tensor
     <Tensor: shape=(2, 3), dtype=int32, numpy=
     array([[1, 2, 3],
            [4, 5, 6]], dtype=int32)>
     >>> b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
     >>> b  # 2-D tensor
     <Tensor: shape=(3, 2), dtype=int32, numpy=
     array([[ 7,  8],
            [ 9, 10],
            [11, 12]], dtype=int32)>
     >>> c = matmul(a, b)
     >>> c  # `a` * `b`
     <Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[ 58,  64],
            [139, 154]], dtype=int32)>

     A batch matrix multiplication with batch shape [2]:

     >>> a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
     >>> a  # 3-D tensor
     <Tensor: shape=(2, 2, 3), dtype=int32, numpy=
     array([[[ 1,  2,  3],
             [ 4,  5,  6]],
            [[ 7,  8,  9],
             [10, 11, 12]]], dtype=int32)>
     >>> b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
     >>> b  # 3-D tensor
     <Tensor: shape=(2, 3, 2), dtype=int32, numpy=
     array([[[13, 14],
             [15, 16],
             [17, 18]],
            [[19, 20],
             [21, 22],
             [23, 24]]], dtype=int32)>
     >>> c = matmul(a, b)
     >>> c  # `a` * `b`
     <Tensor: shape=(2, 2, 2), dtype=int32, numpy=
     array([[[ 94, 100],
             [229, 244]],
            [[508, 532],
             [697, 730]]], dtype=int32)>

     Since python >= 3.5 the @ operator is supported
     (see [PEP 465](https://www.python.org/dev/peps/pep-0465/)). In TensorFlow,
     it simply calls the `tf.matmul()` function, so the following lines are
     equivalent:

     >>> d = a @ b @ [[10], [11]]
     >>> d = matmul(tf.matmul(a, b), [[10], [11]])

     Args:
        a: `Tensor` and rank > 1.
        b: `Tensor` with same type and rank as `a`.
        transpose_a: If `True`, `a` is transposed before multiplication.
        transpose_b: If `True`, `b` is transposed before multiplication.


     Returns:
         A `Tensor` of the same type as `a` and `b` where each inner-most matrix is the product of the corresponding matrices in `a` and `b`, e.g. if all transpose or adjoint attributes are `False`:

       `output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
       for all indices `i`, `j`.

       Note: This is matrix product, not element-wise product.


     Raises:
         ValueError: If `transpose_a` and `adjoint_a`, or `transpose_b` and
         `adjoint_b` are both set to `True`.

     """
    return tf.matmul(a, b, transpose_a=transpose_b, transpose_b=transpose_b)

@numpy_compatible
def true_divide(x, y):
    """Divides x / y elementwise (using Python 3 division operator semantics).

    NOTE: Prefer using the Tensor operator or tf.divide which obey Python
    division operator semantics.

    This function forces Python 3 division operator semantics where all integer
    arguments are cast to floating types first.   This op is generated by normal
    `x / y` division in Python 3 and in Python 2.7 with
    `from __future__ import division`.  If you want integer division that rounds
    down, use `x // y` or `tf.math.floordiv`.

    `x` and `y` must have the same numeric type.  If the inputs are floating
    point, the output will have the same type.  If the inputs are integral, the
    inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
    and `int64` (matching the behavior of Numpy).

    Args:
      x: `Tensor` numerator of numeric type.
      y: `Tensor` denominator of numeric type.


    Returns:
      `x / y` evaluated in floating point.

    Raises:
      TypeError: If `x` and `y` have different dtypes.
    """
    if isinstance(x,(numbers.Number)) and isinstance(y,(numbers.Number)):
        if y==0:
            return 1
        else:
            return x/y
    elif isinstance(x,(np.ndarray)) and isinstance(y,(numbers.Number,np.ndarray)):
        if isinstance(y, numbers.Number):
            return x.astype(np.float32) / y
        else:
            return x.astype(np.float32) / y.astype(np.float32)
    else:
        if not is_tensor(x):
            x=cast(to_tensor(x),'float32')

        if not is_tensor(y):
            y=cast(to_tensor(y),x.dtype.base_dtype)
        else:
            y = cast(y,x.dtype.base_dtype)
    return tf.math.divide_no_nan(x, y)

@numpy_compatible
def floor(x: Tensor):
    """Returns element-wise largest integer not greater than x.

    Args:
      x: A `Tensor`.

    Returns:
      A `Tensor`. Has the same type as `x`.

    """
    return tf.math.floor(x)

@numpy_compatible
def ceil(x: Tensor):
    """Return the ceiling of the input, element-wise.

    For example:

    >>> tf.math.ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    <Tensor: shape=(7,), dtype=float32,
    numpy=array([-1., -1., -0.,  1.,  2.,  2.,  2.], dtype=float32)>

    Args:
      x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`,
        `float32`, `float64`. `int32`
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `x`.

    @compatibility(numpy)
    Equivalent to np.ceil
    @end_compatibility
    """
    return tf.math.ceil(x)

@numpy_compatible
def round(x: Tensor, digit: int = 0):
    """Rounds the values of a tensor to the nearest integer, element-wise.

    Rounds half to even.  Also known as bankers rounding. If you want to round
    according to the current system rounding mode use tf::cint.

    Args:
        x: A `Tensor`
        digit: number of digit

    Returns:
        A `Tensor` of same shape and type as `x`.

    Examples;
        >>> round(to_tensor([[1,2,3,4,5]])/3,0)
        <Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 2.0000e+00]],
              dtype=float32)>
        >>> round(to_tensor([[1,2,3,4,5]])/3,2)
        <Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[3.3000e-01, 6.7000e-01, 1.0000e+00, 1.3300e+00, 1.6700e+00]],
              dtype=float32)>
        >>> round(to_tensor([[11.6,24.3,35.2,14.4,23.5]])/3,-1)
        <Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[0.0000e+00, 1.0000e+01, 1.0000e+01, 0.0000e+00, 1.0000e+01]],
              dtype=float32)>

    """
    if digit != 0:
        factor = float(math.pow(10, -1 * digit))
        return tf.math.round(x / factor) * factor
    else:
        return tf.math.round(x)



def pi():
    """ The number π (/paɪ/)
    The number π (/paɪ/) is a mathematical constant. It is defined as the ratio of a circle's circumference to its diameter

    Returns:
        The number π (/paɪ/)

    """
    return to_tensor(np.pi)


@numpy_compatible
def sqrt(x: Tensor):
    r"""Computes element-wise square root of the input tensor.

    Note: This operation does not support integer types.

    >>> x = tf.constant([[4.0], [16.0]])
    >>> tf.sqrt(x)
    <Tensor: shape=(2, 1), dtype=float32, numpy=
      array([[2.],
             [4.]], dtype=float32)>
    >>> y = tf.constant([[-4.0], [16.0]])
    >>> sqrt(y)
    <Tensor: shape=(2, 1), dtype=float32, numpy=
      array([[nan],
             [ 4.]], dtype=float32)>
    >>> z = tf.constant([[-1.0], [16.0]], dtype=tf.complex128)
    >>> sqrt(z)
    <Tensor: shape=(2, 1), dtype=complex128, numpy=
      array([[0.0+1.j],
             [4.0+0.j]])>

    Note: In order to support complex complex, please provide an input tensor
    of `complex64` or `complex128`.

    Args:
        x: A `Tensor`


    Returns:
      A `Tensor` of same size, type and sparsity as `x`.

    """

    return tf.math.sqrt(tf.clip_by_value(x,clip_value_min=0,clip_value_max=np.inf))

@numpy_compatible
def rsqrt(x: Tensor):
    """Computes reciprocal of square root of x element-wise.

    Args:
      x: input tensor

    Returns:
      output tensor


    Examples:
        >>> x = tf.constant([2., 0., -2.])
        >>> rsqrt(x)
        <Tensor: shape=(3,), dtype=float32,
        numpy=array([0.707, inf, nan], dtype=float32)>

    """

    return tf.math.rsqrt(x)

@numpy_compatible
def square(x: Tensor):
    r"""Computes square of x element-wise.

    I.e., \\(y = x * x = x^2\\).

    >>> tf.math.square([-2., 0., 3.])
    <Tensor: shape=(3,), dtype=float32, numpy=array([4., 0., 9.], dtype=float32)>

    Args:
      x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`,
      `complex64`, `complex128`.


    Returns:
      A `Tensor`. Has the same type as `x`.


    """

    return tf.math.square(x)

@numpy_compatible
def abs(x: Tensor):
    r"""Computes the absolute value of a tensor.

    Given a tensor of integer or floating-point values, this operation returns a
    tensor of the same type, where each element contains the absolute value of the
    corresponding element in the input.

    Given a tensor `x` of complex numbers, this operation returns a tensor of type
    `float32` or `float64` that is the absolute value of each element in `x`. For
    a complex number \\(a + bj\\), its absolute value is computed as \\(\sqrt{a^2
    + b^2}\\).  For example:

    >>> x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
    >>> tf.abs(x)
    <Tensor: shape=(2, 1), dtype=float64, numpy=
    array([[5.25594901],
           [6.60492241]])>

    Args:
      x: A `Tensor` or `SparseTensor` of type `float16`, `float32`, `float64`,
        `int32`, `int64`, `complex64` or `complex128`.


    Returns:
      A `Tensor` or `SparseTensor` of the same size, type and sparsity as `x`,
        with absolute values. Note, for `complex64` or `complex128` input, the
        returned `Tensor` will be of type `float32` or `float64`, respectively.
    """

    return tf.math.abs(x)

@numpy_compatible
def pow(x: Tensor, y):
    r"""Computes the power of one value to another.

    Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
    corresponding elements in `x` and `y`. For example:

    ```python
    x = tf.constant([[2, 2], [3, 3]])
    y = tf.constant([[8, 16], [2, 3]])
    tf.pow(x, y)  # [[256, 65536], [9, 27]]
    ```

    Args:
      x: A `Tensor`
      y: A `Tensor`


    Returns:
      A `Tensor`.
    """

    return tf.math.pow(x, y)

@numpy_compatible
def log(x: Tensor):
    r"""Computes natural logarithm of x element-wise.

    I.e., \\(y = \log_e x\\).

    See: https://en.wikipedia.org/wiki/Logarithm

    Args:
      x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`,
      `complex128`.


    Returns:
      A `Tensor`. Has the same type as `x`.

    Examples:
        >>> x = to_tensor([0, 0.5, 1, 5])
        >>> log(x)
        <Tensor: shape=(4,), dtype=float32, numpy=array([      -inf, -0.6931472,  0.       ,  1.609438 ], dtype=float32)>


    """

    return tf.math.log(x)

@numpy_compatible
def exp(x: Tensor):
    r"""Computes exponential of x element-wise.  \\(y = e^x\\).

    This function computes the exponential of the input tensor element-wise.
    i.e. `math.exp(x)` or \\(e^x\\), where `x` is the input tensor.
    \\(e\\) denotes Euler's number and is approximately equal to 2.718281.
    Output is positive for any real input.

    >>> x = tf.constant(2.0)
    >>> tf.math.exp(x)
    <Tensor: shape=(), dtype=float32, numpy=7.389056>

    >>> x = tf.constant([2.0, 8.0])
    >>> tf.math.exp(x)
    <Tensor: shape=(2,), dtype=float32,
    numpy=array([   7.389056, 2980.958   ], dtype=float32)>

    For complex numbers, the exponential value is calculated as
    \\(e^{x+iy}={e^x}{e^{iy}}={e^x}{\\cos(y)+i\\sin(y)}\\)

    For `1+1j` the value would be computed as:
    \\(e^1{\\cos(1)+i\\sin(1)} = 2.7182817 \\times (0.5403023+0.84147096j)\\)

    >>> x = tf.constant(1 + 1j)
    >>> tf.math.exp(x)
    <Tensor: shape=(), dtype=complex128,
    numpy=(1.4686939399158851+2.2873552871788423j)>

    Args:
      x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`,
        `float32`, `float64`, `complex64`, `complex128`.


    Returns:
      A `Tensor`. Has the same type as `x`.

    @compatibility(numpy)
    Equivalent to np.exp
    @end_compatibility

    """

    return tf.math.exp(x)

@numpy_compatible
def prod(x: Tensor):
    """Computes the product of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
      x: The tensor to reduce. Should have numeric type.



    Returns:
      The reduced tensor.

    @compatibility(numpy)
    Equivalent to np.prod
    @end_compatibility

    """

    return tf.math.reduce_prod(x,axis=None,keepdims=False)

@numpy_compatible
def clip(x: Tensor, min=-np.inf, max=np.inf):
    """Clips tensor values to a specified min and max.

    Given a tensor `t`, this operation returns a tensor of the same type and
    shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
    Any values less than `clip_value_min` are set to `clip_value_min`. Any values
    greater than `clip_value_max` are set to `clip_value_max`.

    Note: `clip_value_min` needs to be smaller or equal to `clip_value_max` for
    correct results.

    For example:

    Basic usage passes a scalar as the min and max value.

    >>> t = tf.constant([[-10., -1., 0.], [0., 2., 10.]])
    >>> t2 = tf.clip_by_value(t, clip_value_min=-1, clip_value_max=1)
    >>> t2.numpy()
    array([[-1., -1.,  0.],
           [ 0.,  1.,  1.]], dtype=float32)

    The min and max can be the same size as `t`, or broadcastable to that size.

    >>> t = tf.constant([[-1, 0., 10.], [-1, 0, 10]])
    >>> clip_min = [[2],[1]]
    >>> t3 = tf.clip_by_value(t, clip_value_min=clip_min, clip_value_max=100)
    >>> t3.numpy()
    array([[ 2.,  2., 10.],
           [ 1.,  1., 10.]], dtype=float32)

    Broadcasting fails, intentionally, if you would expand the dimensions of `t`

    >>> t = tf.constant([[-1, 0., 10.], [-1, 0, 10]])
    >>> clip_min = [[[2, 1]]] # Has a third axis
    >>> t4 = tf.clip_by_value(t, clip_value_min=clip_min, clip_value_max=100)
    Traceback (most recent call last):
    ...
    InvalidArgumentError: Incompatible shapes: [2,3] vs. [1,1,2]

    It throws a `TypeError` if you try to clip an `int` to a `float` value
    (`tf.cast` the input to `float` first).

    >>> t = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
    >>> t5 = tf.clip_by_value(t, clip_value_min=-3.1, clip_value_max=3.1)
    Traceback (most recent call last):
    ...
    TypeError: Cannot convert ...


    Args:
      x: A `Tensor` or `IndexedSlices`.
      min: The minimum value to clip to. A scalar `Tensor` or one that
        is broadcastable to the shape of `t`.
      max: The minimum value to clip to. A scalar `Tensor` or one that
        is broadcastable to the shape of `t`.


    Returns:
      A clipped `Tensor` or `IndexedSlices`.

    Raises:
      `tf.errors.InvalidArgumentError`: If the clip tensors would trigger array
        broadcasting that would make the returned tensor larger than the input.
      TypeError: If dtype of the input is `int32` and dtype of
        the `clip_value_min` or `clip_value_max` is `float32`
    """
    return tf.clip_by_value(x, min, max)






@numpy_compatible
def sin(x: Tensor):
    return tf.math.sin(x)

@numpy_compatible
def cos(x: Tensor):
    return tf.math.cos(x)

@numpy_compatible
def tan(x: Tensor):
    return tf.math.tan(x)

@numpy_compatible
def asin(x: Tensor):
    return tf.math.asin(x)

@numpy_compatible
def acos(x: Tensor):
    return tf.math.acos(x)

@numpy_compatible
def atan(x: Tensor):
    return tf.math.atan(x)

@numpy_compatible
def sinh(x: Tensor):
    """Computes the element-wise sinh

    Args:
        x (tensor):input tensor

    Returns: element-wise sinh

    Examples:
        >>> sinh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        <Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[1.1752e+00, 5.2110e-01],
               [-2.5261e-01, -8.2232e-01]], dtype=float32)>

    """
    return tf.sinh(x)

@numpy_compatible
def cosh(x: Tensor):
    """Computes the element-wise cosh

    Args:
        x (tensor):input tensor

    Returns: element-wise cosh

    Examples:
        >>> cosh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        <Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[1.5431e+00, 1.1276e+00],
               [1.0314e+00, 1.2947e+00]], dtype=float32)>

    """
    return tf.cosh(x)

@numpy_compatible
def tanh(x: Tensor):
    """Computes the element-wise tanh

    Args:
        x (tensor):input tensor

    Returns: element-wise tanh

    Examples:
        >>> tanh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 0.     ,  1.0472 ],
           [ 1.82348,  2.41886]])

    """
    return tf.tanh(x)


############################
## element-wise operation
###########################
@numpy_compatible
def element_times(left: Tensor, right:[Tensor,numbers.Number]):
    """
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Examples:
        >>> element_times(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([5.0000e-01, 2.5000e-01, 1.2500e-01, 0.0000e+00], dtype=float32)>
        >>> element_times(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([1.0000e+01, 2.0000e+01, 3.0000e+01, 6.0000e+01], dtype=float32)>
        >>> element_times(to_tensor([[5., 10.], [15., 30.]]), to_tensor([[1., 2.], [3.,1.]]))
        <Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[5.0000e+00, 2.0000e+01],
               [4.5000e+01, 3.0000e+01]], dtype=float32)>

    """
    return left * right

@numpy_compatible
def element_max(left: Tensor, right:[Tensor,numbers.Number]):
    """
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Examples:
        >>> element_max(to_tensor([1., 1., 0., -1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([1.0000e+00, 1.0000e+00, 1.2500e-01, 0.0000e+00], dtype=float32)>
        >>> element_max(to_tensor([5., 10., 15., 30.]),to_tensor([20.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([2.0000e+01, 2.0000e+01, 2.0000e+01, 3.0000e+01], dtype=float32)>
        >>> element_max(to_tensor([5., 10., 15., 30.]), to_tensor([10., 2., 8., 2.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([1.0000e+01, 1.0000e+01, 1.5000e+01, 3.0000e+01], dtype=float32)>

    """
    return maximum(left, right)

@numpy_compatible
def element_min(left: Tensor, right:[Tensor,numbers.Number]):
    """
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Examples:
        >>> element_min(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([5.0000e-01, 2.5000e-01, 1.2500e-01, 0.0000e+00], dtype=float32)>
        >>> element_min(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([2.0000e+00, 2.0000e+00, 2.0000e+00, 2.0000e+00], dtype=float32)>
        >>> element_min(to_tensor([5., 10., 15., 30.]), to_tensor([1., 2., 1., 2.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 1.0000e+00, 2.0000e+00], dtype=float32)>

    """
    return minimum(left, right)

@numpy_compatible
def element_divide(left: Tensor, right:[Tensor,numbers.Number]):
    """
    The output of this operation is the element-wise divide of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise divide of the two  input

    Examples:
        >>> element_divide(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([2.0000e+00, 4.0000e+00, 8.0000e+00, inf], dtype=float32)>
        >>> element_divide(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([2.5000e+00, 5.0000e+00, 7.5000e+00, 1.5000e+01], dtype=float32)>
        >>> element_divide(to_tensor([5., 10., 15., 30.]), to_tensor([1., 2., 1., 2.]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([5.0000e+00, 5.0000e+00, 1.5000e+01, 1.5000e+01], dtype=float32)>

    """
    return true_divide(left, right)

@numpy_compatible
def element_cosine_distance(v1: Tensor, v2:[Tensor,numbers.Number], axis=-1):
    """    The output of this operation is the element-wise cosine_distance of the two  input
    tensors. It supports broadcasting.

    Args:
        v1 ():
        v2 ():
        axis ():

    Returns:

    """
    normalize_a = tf.nn.l2_normalize(v1, axis)
    normalize_b = tf.nn.l2_normalize(v2, axis)
    distance = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance

@numpy_compatible
def where(flag, value_if_true, value_if_false, name='where'):
    """
    return either ``value_if_true`` or ``value_if_false`` based on the value of ``flag``.
    If ``flag`` != 0 ``value_if_true`` is returned, otherwise ``value_if_false``.
    Behaves analogously to numpy.where(...).

    Args:
        flag: condition tensor
        value_if_true: true branch tensor
        value_if_false: false branch tensor
        name (str): op name
    Returns:
        :conditional selection

    Examples:
    >>> x=to_tensor([0.1, 0.9, 0.8, 0.4, 0.5])
    >>> where(x>0.5, x, zeros_like(x))
    <Tensor: shape=(5,), dtype=float32, numpy=
    array([0.0000e+00, 9.0000e-01, 8.0000e-01, 0.0000e+00, 0.0000e+00],
          dtype=float32)>
    """
    return tf.where(flag, value_if_true, value_if_false,name=name)


############################
## reduce operation
###########################
@numpy_compatible
def reduce_mean(x: Tensor, axis=None, keepdims=False, name='reduce_mean'):
    """Computes the mean of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis` by computing the
    mean of elements across the dimensions in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions are retained
    with length 1.

    If `axis` is None, all dimensions are reduced, and a tensor with a single
    element is returned.

    For example:

    >>> x = tf.constant([[1., 1.], [2., 2.]])
    >>> reduce_mean(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.5>
    >>> reduce_mean(x, 0)
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.5, 1.5], dtype=float32)>
    >>> reduce_mean(x, 1)
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>

    Args:
      x: The tensor to reduce. Should have numeric type.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `[-rank(input_tensor),
        rank(input_tensor))`.
      keepdims: If true, retains reduced dimensions with length 1.
      name: A name for the operation (optional).

    Returns:
      The reduced tensor.

    @compatibility(numpy)
    Equivalent to np.mean

    Please note that `np.mean` has a `dtype` parameter that could be used to
    specify the output type. By default this is `dtype=float64`. On the other
    hand, `tf.reduce_mean` has an aggressive type inference from `input_tensor`,
    for example:

    >>> x = tf.constant([1, 0, 1, 0])
    >>> reduce_mean(x)
    <tf.Tensor: shape=(), dtype=int32, numpy=0>
    >>> y = tf.constant([1., 0., 1., 0.])
    >>> reduce_mean(y)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.5>

    @end_compatibility
    """
    return tf.math.reduce_mean(x, axis=axis, keepdims=keepdims,name=name)

@numpy_compatible
def reduce_sum(x: Tensor, axis=None, keepdims=False, name='reduce_sum'):
    """Computes the sum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.

    For example:

    >>> # x has a shape of (2, 3) (two rows and three columns):
    >>> x = tf.constant([[1, 1, 1], [1, 1, 1]])
    >>> x.numpy()
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)
    >>> # sum all the elements
    >>> # 1 + 1 + 1 + 1 + 1+ 1 = 6
    >>> reduce_sum(x).numpy()
    6
    >>> # reduce along the first dimension
    >>> # the result is [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> reduce_sum(x, 0).numpy()
    array([2, 2, 2], dtype=int32)
    >>> # reduce along the second dimension
    >>> # the result is [1, 1] + [1, 1] + [1, 1] = [3, 3]
    >>> reduce_sum(x, 1).numpy()
    array([3, 3], dtype=int32)
    >>> # keep the original dimensions
    >>> reduce_sum(x, 1, keepdims=True).numpy()
    array([[3],
           [3]], dtype=int32)
    >>> # reduce along both dimensions
    >>> # the result is 1 + 1 + 1 + 1 + 1 + 1 = 6
    >>> # or, equivalently, reduce along rows, then reduce the resultant array
    >>> # [1, 1, 1] + [1, 1, 1] = [2, 2, 2]
    >>> # 2 + 2 + 2 = 6
    >>> reduce_sum(x, [0, 1]).numpy()
    6


    Args:
      x: The tensor to reduce. Should have numeric type.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `[-rank(input_tensor),
        rank(input_tensor)]`.
      keepdims: If true, retains reduced dimensions with length 1.
      name: A name for the operation (optional).

    Returns:
      The reduced tensor, of the same dtype as the input_tensor.

    @compatibility(numpy)
    Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
    int64 while tensorflow returns the same dtype as the input.
    @end_compatibility
    """
    return tf.math.reduce_sum(x, axis=axis, keepdims=keepdims,name=name)

@numpy_compatible
def reduce_max(x: Tensor, axis=None, keepdims=False, name='reduce_max'):
    """Computes the maximum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.


    See the numpy docs for `np.amax` and `np.nanmax` behavior.

    Args:
      x: The tensor to reduce. Should have real numeric type.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `[-rank(input_tensor),
        rank(input_tensor))`.
      keepdims: If true, retains reduced dimensions with length 1.
      name (str): op name

    Returns:
      The reduced tensor.

    Examples:
        >>> x = tf.constant([5, 1, 2, 4])
        >>> print(reduce_max(x))
        Tensor(5, shape=(), dtype=int32)
        >>> x = tf.constant([-5, -1, -2, -4])
        >>> print(reduce_max(x))
        Tensor(-1, shape=(), dtype=int32)
        >>> x = tf.constant([4, float('nan')])
        >>> print(reduce_max(x))
        Tensor(4.0, shape=(), dtype=float32)
        >>> x = tf.constant([float('nan'), float('nan')])
        >>> print(reduce_max(x))
        Tensor(-inf, shape=(), dtype=float32)
        >>> x = tf.constant([float('-inf'), float('inf')])
        >>> print(reduce_max(x))
        Tensor(inf, shape=(), dtype=float32)

    """
    return tf.math.reduce_max(x, axis=axis, keepdims=keepdims,name=name)

@numpy_compatible
def reduce_min(x: Tensor, axis=None, keepdims=False, name='reduce_min'):
    """Computes the minimum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.


    See the numpy docs for `np.amax` and `np.nanmax` behavior.

    Args:
      x: The tensor to reduce. Should have real numeric type.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `[-rank(input_tensor),
        rank(input_tensor))`.
      keepdims: If true, retains reduced dimensions with length 1.
      name (str): op name

    Returns:
      The reduced tensor.

    Examples:
        >>> x = tf.constant([5, 1, 2, 4])
        >>> print(reduce_min(x))
        Tensor(5, shape=(), dtype=int32)
        >>> x = tf.constant([-5, -1, -2, -4])
        >>> print(reduce_min(x))
        Tensor(-1, shape=(), dtype=int32)
        >>> x = tf.constant([4, float('nan')])
        >>> print(reduce_min(x))
        Tensor(4.0, shape=(), dtype=float32)
        >>> x = tf.constant([float('nan'), float('nan')])
        >>> print(reduce_min(x))
        Tensor(-inf, shape=(), dtype=float32)
        >>> x = tf.constant([float('-inf'), float('inf')])
        >>> print(reduce_min(x))
        Tensor(inf, shape=(), dtype=float32)

    """
    return tf.math.reduce_min(x, axis=axis, keepdims=keepdims,name=name)

@numpy_compatible
def reduce_logsumexp(x: Tensor, axis=None, keepdims=False, name='reduce_logsumexp'):
    """Computes log(sum(exp(elements across dimensions of a tensor))).

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.

    This function is more numerically stable than log(sum(exp(input))). It avoids
    overflows caused by taking the exp of large inputs and underflows caused by
    taking the log of small inputs.

    Examples:
        >>> x = tf.constant([[0., 0., 0.], [0., 0., 0.]])
        >>> reduce_logsumexp(x)  # log(6)
        >>> reduce_logsumexp(x, 0)  # [log(2), log(2), log(2)]
        >>> reduce_logsumexp(x, 1)  # [log(3), log(3)]
        >>> reduce_logsumexp(x, 1, keepdims=True)  # [[log(3)], [log(3)]]
        >>> reduce_logsumexp(x, [0, 1])  # log(6)


    Args:
        x (tf.tensor): Input_tensor,the tensor to reduce. Should have numeric type.
        axis (int, list, tuple): The dimensions to reduce. If `None` (the default), reduces all dimensions. Must be in the range `[-rank(input_tensor), rank(input_tensor))`.
        keepdims (bool): If true, retains reduced dimensions with length 1.
        name (str): op name

    Returns:
      The reduced tensor.

    """
    return tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims,name=name)

@numpy_compatible
def reduce_prod(x: Tensor, axis=None, keepdims=False, name='reduce_prod'):
    """Computes the product of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
      x: Input_tensor, the tensor to reduce. Should have numeric type.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `[-rank(input_tensor),
        rank(input_tensor))`.
      keepdims: If true, retains reduced dimensions with length 1.
      name (str): op name

    Returns:
      The reduced tensor.

    @compatibility(numpy)
    Equivalent to np.prod
    @end_compatibility
    """
    return tf.math.reduce_prod(x, axis=axis, keepdims=keepdims,name=name)

@numpy_compatible
def reduce_any(x: Tensor, axis=None, keepdims=False, name='reduce_prod'):
    x=tf.greater(x,0)
    return tf.math.reduce_any(x, axis=axis, keepdims=keepdims,name=name)

# reduce_l1
# reduce_l2
# reduce_sum_square

mean = reduce_mean
sum = reduce_sum
max = reduce_max
min = reduce_min


mean = reduce_mean
sum = reduce_sum

@numpy_compatible
def max(*args, **kwargs):
    """

    Args:
        *args ():

    Returns:

    """
    if len(args) > 1 and all([str(a).isnumeric() for a in args]):
        args=[float(arg) for arg in args]
        return builtins.max(*args)
    elif len(args) > 1:
        new_args = [to_tensor(a).float() for a in args]
        return tf.max(*new_args)
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return np.max(args[0], axis=axis, keepdims=keepdims)
    elif len(args) == 1 and is_tensor(args[0]):
        if len(args[0]) == 0:
            return args[0]
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return reduce_max(args[0], axis=axis, keepdims=keepdims)

@numpy_compatible
def min(*args, **kwargs):
    """

    Args:
        *args ():

    Returns:

    """
    if len(args) > 1 and all([str(a).isnumeric() for a in args]):
        args = [float(arg) for arg in args]
        return builtins.min(*args)
    elif len(args) > 1:
        new_args = [to_tensor(a).float() for a in args]
        return tf.min(*new_args)
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return np.min(args[0], axis=axis, keepdims=keepdims)
    elif len(args) == 1 and is_tensor(args[0]):
        if len(args[0]) == 0:
            return args[0]
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return reduce_min(args[0], axis=axis, keepdims=keepdims)


@numpy_compatible
def max(*args, **kwargs):
    """General function for max operation

    Examples:
        >>> max(to_tensor([0.1, 0.9, 0.8, 0.4, 0.5])).cpu()
        tensor(0.9000)
        >>> max(to_tensor([0.1, 0.9, 0.8, 0.4, 0.5]),0.5).cpu()
        tensor([0.5000, 0.9000, 0.8000, 0.5000, 0.5000])
        >>> max(3,7)
        7
        >>> max(to_numpy([0.1, 0.9, 0.8, 0.4, 0.5]),0.5)
        array([5.0000e-01, 9.0000e-01, 8.0000e-01, 5.0000e-01, 5.0000e-01])
        >>> print(int_shape(to_tensor([[0.1, 0.9, 0.8],[0.3, 0.4, 0.5]])))
        (2, 3)
        >>> max(to_tensor([[0.1, 0.9, 0.8],[0.3, 0.4, 0.5]]),axis=0).cpu()
        tensor([0.3000, 0.9000, 0.8000])
        >>> max(to_tensor([[0.1, 0.9, 0.8],[0.3, 0.4, 0.5]]),dim=0).cpu()
        tensor([0.3000, 0.9000, 0.8000])
        >>> max(to_tensor([[0.1, 0.9, 0.8],[0.3, 0.4, 0.5]]),axis=0,keepdims=True).cpu()
        tensor([[0.3000, 0.9000, 0.8000]])
        >>> max(to_tensor([[0.1, 0.9, 0.8],[0.3, 0.4, 0.5]]),dim=0,keepdim=True).cpu()
        tensor([[0.3000, 0.9000, 0.8000]])



    """
    allargs=args+tuple(list(kwargs.values()))
    if len(allargs) == 1 and is_tensor(allargs[0]) and numel(allargs[0]) == 0:
        return allargs[0]
    elif len(allargs)  == 1 and is_tensor(allargs[0]) and numel(allargs[0]) >0:
        return tf.math.reduce_max(allargs[0], name='reduce_max')
    elif len(allargs) > 1 and is_tensor(allargs[0]) and not is_tensor(allargs[1]) and ('axis' in kwargs or 'dim' in kwargs or 'keepdims' in kwargs or 'keepdim' in kwargs):
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return  tf.math.reduce_max(allargs[0], axis=axis, keepdims=keepdims, name='reduce_max')
    elif len(args) ==2 and is_tensor(args[0]) and isinstance(args[1],numbers.Number):
        return tf.clip_by_value(args[0], args[1],np.inf)
    elif len(args) > 1 and is_tensor(args[0]) and all([is_tensor(arg) or isinstance(arg,(np.ndarray,numbers.Number))for arg in args]):
        new_args = [to_tensor(a).float() for a in args]
        return tf.math.maximum(*new_args, name='maximum')
    else:
        raise NotImplementedError('Max({0},{1}) is not implemented yet '.format(*args,**kwargs))



@numpy_compatible
def min(*args, **kwargs):
    """

    Args:
        *args ():

    Returns:

    """
    allargs = args + tuple(list(kwargs.values()))
    if len(allargs) == 1 and is_tensor(allargs[0]) and numel(allargs[0]) == 0:
        return allargs[0]
    elif len(allargs) == 1 and is_tensor(allargs[0]) and numel(allargs[0]) > 0:
        return tf.math.reduce_min(allargs[0], name='reduce_min')
    elif len(allargs) > 1 and is_tensor(allargs[0]) and not is_tensor(allargs[1]) and ('axis' in kwargs or 'dim' in kwargs or 'keepdims' in kwargs or 'keepdim' in kwargs):
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return tf.math.reduce_min(allargs[0], axis=axis, keepdims=keepdims, name='reduce_min')
    elif len(args) > 1 and is_tensor(args[0]) and all([is_tensor(arg) or isinstance(arg, (np.ndarray,numbers.Number)) for arg in args]):
        new_args = [to_tensor(a).float() for a in args]
        return tf.math.minimum(*new_args, name='minimum')
    else:
        raise NotImplementedError('Min({0},{1}) is not implemented yet '.format(*args, **kwargs))

############################
## activationoperation
###########################

@numpy_compatible
def identity(x:Tensor,name='identity'):
    r"""Return a Tensor with the same shape and contents as input.

      The return value is not the same Tensor as the original, but contains the same
      values.  This operation is fast when used on the same device.

    Examples:

      >>> a = tf.constant([0.78])
      >>> a_identity = tf.identity(a)
      >>> a.numpy()
      array([0.78], dtype=float32)
      >>> a_identity.numpy()
      array([0.78], dtype=float32)

      Calling `identity` on a variable will make a Tensor that represents the
      value of that variable at the time it is called. This is equivalent to calling
      `<variable>.read_value()`.

      >>> a = tf.Variable(5)
      >>> a_identity = tf.identity(a)
      >>> a.assign_add(1)
      <tf.Variable ... shape=() dtype=int32, numpy=6>
      >>> a.numpy()
      6
      >>> a_identity.numpy()
      5

      Args:
        x: A `Tensor`.
        name (str):op name


      Returns:
        A `Tensor`. Has the same type as `input`.

      """
    return  tf.identity(x,name=name)

@numpy_compatible
def sigmoid(x:Tensor,name='sigmoid'):
    return tf.nn.sigmoid(x,name=name)

@numpy_compatible
def tanh(x:Tensor,name='tanh'):
    return tf.nn.tanh(x,name=name)

@numpy_compatible
def relu(x:Tensor, upper_limit=None,name='relu'):
    """Rectified Linear Unit activation function.

    With default values, it returns element-wise `max(x, 0)`.
    Otherwise, it follows:

        f(x) = max_value if x >= max_value
        f(x) = x if threshold <= x < max_value
        f(x) = negative_slope * (x - threshold) otherwise

    """
    if upper_limit is not None and upper_limit <= 0:
        raise ValueError('Upper limit should greater than 0!')
    elif upper_limit is not None:
        return clip(tf.nn.relu(x,name=name), 0, upper_limit)
    return tf.nn.relu(x,name=name)

@numpy_compatible
def relu6(x:Tensor,name='relu6'):
    """Rectified Linear Unit  6 activation function.

    With default values, it returns element-wise `min(max(x, 0)`,6).
    Otherwise, it follows:

        f(x) = 6 if x >= 6
        f(x) = x if threshold <= x < 6
        f(x) = negative_slope * (x - threshold) otherwise


    """
    return relu(x,6)

@numpy_compatible
def leaky_relu(x:Tensor, alpha:float=0.02, upper_limit:Optional[float]=None,name='leaky_relu'):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:

        f(x) = alpha * x if x < 0
        f(x) = x if x >= 0

    Args:
        x (tensor): input tensor
        alpha (float):multiplier
        upper_limit (float):upper limit
    """
    if upper_limit is not None:
        return clip(tf.nn.leaky_relu(x, alpha,name=name), -np.inf, upper_limit)
    return tf.nn.leaky_relu(x, alpha,name=name)

@numpy_compatible
def leaky_relu6(x:Tensor,alpha:Tensor=0.02,name='leaky_relu6'):
    """Leaky version of a Rectified Linear Unit.6

    It allows a small gradient when the unit is not active:

        f(x) = alpha * x if x < 0
        f(x) = x if  6>=x >= 0
        f(x) = 6 if  x > 6

    """
    return clip(tf.nn.leaky_relu(x, alpha,name=name), -6, 6)

@numpy_compatible
def elu(x:Tensor, alpha=1.0, upper_limit:Optional[float]=None,name='elu'):
    """ Exponential Linear Unit.
    It follows:

        f(x) =  alpha * (exp(x) - 1.) for x < 0
        f(x) = x for x >= 0


    Args:

        x (tensor): input tensor
        alpha (float):multiplier
        upper_limit (float):upper limit
        name (str): op name

    Returns:
        (tensor) transformed tensor with the same shape as input tensor.

    Examples:
        >>> elu(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    """
    x=tf.nn.elu(x, name=name)*alpha
    if upper_limit is not None:
        return clip(x, -np.inf, upper_limit)
    else:
        return x


lrelu = leaky_relu

@numpy_compatible
def smooth_relu(x:Tensor, upper_limit=None,name='smooth_relu'):
    if upper_limit is not None:
        return clip(tf.math.log(1 + tf.math.exp(x)), -np.inf, upper_limit)
    return tf.math.log(1 + tf.math.exp(x))

@numpy_compatible
def p_relu(x:Tensor, upper_limit=None,name='p_relu'):
    if upper_limit is not None:
        return clip(tf.keras.layers.PReLU()(x), -np.inf, upper_limit)
    return tf.keras.layers.PReLU()(x)

@numpy_compatible
def swish(x:Tensor,name='swish'):
    """Self-Gated Activation Function.

    it follows:

        f(x) =  x * sigmoid(x)


    References:
        Swish: a Self-Gated Activation Function
        https://arxiv.org/abs/1710.05941v1

    Examples:
        >>> swish(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00], dtype=float32)>

    """
    return tf.nn.sigmoid(x) * x

@numpy_compatible
def selu(x:Tensor,name='selu'):
    """
    selu activation function

    .. math::
            \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.


    Scaled exponential linear unit operation. Computes the element-wise exponential linear
    of ``x``: ``scale * x`` for ``x >= 0`` and ``x``: ``scale * alpha * (exp(x)-1)`` otherwise.
    scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717

    paper: https://arxiv.org/abs/1706.02515
    Self-Normalizing Neural Networks
    Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter

    Args:
        x (tensor): input tensor

    Returns:The output tensor has the same shape as ``x``
    Examples:
        >>> selu(to_tensor([[-1, -0.5, 0, 1, 2]]))
        tensor([[-1.1113, -0.6918,  0.0000,  1.0507,  2.1014]])
    """
    return tf.nn.selu(x)

@numpy_compatible
def soft_sign(x:Tensor,name='soft_sign'):
    return tf.nn.softsign(x)

@numpy_compatible
def lecun_tanh(x:Tensor,name='lecun_tanh'):
    return 1.7159 * tf.nn.tanh(2 / 3 * x)

@numpy_compatible
def soft_plus(x:Tensor,name='soft_plus'):
    return tf.nn.softplus(x)

@numpy_compatible
def hard_sigmoid(x:Tensor,name='hard_sigmoid'):
    """Hard sigmoid Activation Function.

    Memory saving version of sigmoid
    it follows:

        f(x) =  relu6(x+3)/6


    Args:
        x (tensor): input tensor

    Returns:
        (tensor) transformed tensor with the same shape as input tensor.


    Examples:
        >>> hard_sigmoid(to_tensor([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([-0.0000, -0.3333,  0.0000,  1.6667])


    """
    return relu6(x + 3) / 6

@numpy_compatible
def hard_tanh(x:Tensor,name='hard_tanh'):
    """Hard Tanh Activation Function.

    Memory saving version of sigmoid
    it follows:

        f(x) =  clip(x, -1, 1)


    Args:
        x (tensor): input tensor

    Returns:
        (tensor) transformed tensor with the same shape as input tensor.


    Examples:
        >>> hard_tanh(to_tensor([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([-0.0000, -0.3333,  0.0000,  1.6667])


    """
    return clip(x, -1, 1)

@numpy_compatible
def hard_swish(x:Tensor,name='hard_swish'):
    """Hard swish Activation Function.

    Memory saving version of swish
    it follows:

        f(x) =  x * hard_sigmoid(x)


    Args:
        x (tensor): input tensor

    Returns:
        (tensor) transformed tensor with the same shape as input tensor.


    Examples:
        >>> hard_swish(to_tensor([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([-0.0000, -0.3333,  0.0000,  1.6667])

    References:
        Searching for MobileNetV3
        https://arxiv.org/abs/1905.02244

    """
    return x * hard_sigmoid(x)

@numpy_compatible
def logit(x:Tensor,name='logit'):
    return tf.math.log(x / (1 - x))

@numpy_compatible
def log_log(x:Tensor,name='log_log'):
    """LogLog Activation Function

    it follows:

        f(x) =  1 - exp(-exp(x))


    Examples:
        >>> log_log(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]


    References:
        "Complementary Log-Log and Probit: Activation Functions Implemented in Artificial Neural Networks"
        https://ieeexplore.ieee.org/document/4626755/

    """
    return 1 - tf.math.exp(-tf.math.exp(x))

@numpy_compatible
def softmax(x:Tensor, axis=-1,name='softmax'):
    return tf.nn.softmax(x, axis=axis)

@numpy_compatible
def log_softmax(x:Tensor, axis=-1, keepdims=False,name='log_softmax'):
    """Activation function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        keepdims ():
        axis ():
        x : input tensor
    """

    return x - reduce_logsumexp(x, axis=axis, keepdims=True)

@numpy_compatible
def mish(x:Tensor,name='mish'):
    """mish activation function

    it follows:

        f(x) =  x * tanh(softplus(x))



    Args:
        x (tensor): input tensor

    Returns:
        (tensor) transformed tensor with the same shape as input tensor.


    Examples:
        >>> mish(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    References:
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1

    """
    return x * tf.nn.tanh(tf.nn.softplus(x))

@numpy_compatible
def hard_mish(x:Tensor,name='hard_mish'):
    """hard mish activation function

    it follows:

        f(x) =  x * hard_tanh(softplus(x))



    Args:
        x (tensor): input tensor

    Returns:
        (tensor) transformed tensor with the same shape as input tensor.


    Examples:
        >>> hard_mish(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    References:
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1

    """
    return  x * hard_tanh(tf.nn.softplus(x))

@numpy_compatible
def gelu(x:Tensor,name='gelu'):
    """Gaussian Error Linear Unit.
    it follows:

        f(x) =x∗Φ(x)
        where \Phi(x)Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.


    References:
        Gaussian Error Linear Units (GELUs)
        https://arxiv.org/abs/1606.08415

    Examples:
        >>> gelu(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([-3.6374e-03, -1.5881e-01, 0.0000e+00, 1.9546e+00], dtype=float32)>

    """
    return x * 0.5 * (1.0 + tf.nn.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

@numpy_compatible
def gpt_gelu(x:Tensor,name='gpt_gelu'):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))


############################
## normalization operation
###########################
@numpy_compatible
def moments(x:Tensor, axis,  keepdims=True):
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

def norm(x:Tensor, order=None, axis=-1,  keepdims=False):
    if order is None:
        order='euclidean'
    return tf.norm(x,ord=order,axis=axis,keepdims=keepdims)

@numpy_compatible
def l2_normalize(x:Tensor,axis=-1,  keepdims=True, eps=epsilon()):
    """

    Args:
        x ():
        axis ():
        keepdims ():
        eps ():

    Returns:

    Examples:
        >>> a=cast(arange(9),'float32')-4.0
        >>> b=a.reshape((3, 3))
        >>> l2_normalize(a)
        >>> reduce_mean(l2_normalize(a)-tf.nn.l2_normalize(a)).numpy()
        0.0
        >>> l2_normalize(b)
        >>> reduce_mean(l2_normalize(b)-tf.nn.l2_normalize(b)).numpy()
        0.0
    """
    return x / (tf.norm(x,keepdims=keepdims)+eps)

@numpy_compatible
def spectral_norm(module, n_iterations=1,axis=-1):
    """Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module

        n_iterations (int, optional): number of power iterations to
            calculate spectral norm

        axis (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Examples::

        >>> m = spectral_norm(Dense(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> int_shape(m.weight)
        ([40])

    """
    w_shape = module.shape.as_list()
    module = tf.reshape(module, [-1, w_shape[axis]])

    u = tf.get_variable("u", [1, w_shape[axis]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(n_iterations):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(module))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, module)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, module), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = module / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

############################
## tensor shape operation
###########################
@numpy_compatible
def reshape(x: Tensor, shape=None) -> Tensor:
    if shape is None:
        return x
    elif isinstance(shape, tf.TensorShape):
        return tf.reshape(x, shape.as_list())
    elif isinstance(shape, TensorShape):
        return tf.reshape(x, shape.dims)
    elif isinstance(shape, (list, tuple)):
        return tf.reshape(x, to_list(shape))
    else:
        shape = to_list(shape)
        return tf.reshape(x, shape)

@numpy_compatible
def squeeze(x: Tensor, axis=None):
    return tf.squeeze(x, axis=axis)

@numpy_compatible
def expand_dims(x: Tensor, axis=None):
    return tf.expand_dims(x, axis=axis)

@numpy_compatible
def transpose(x: Tensor, perm=None) -> Tensor:
    """
    Transposes a. Permutes the dimensions according to perm.
    The returned tensor's dimension i will correspond to the input dimension perm[i]. If perm is not given,
    it is set to (n-1...0), where n is the rank of the input tensor. Hence by default, this operation performs a
    regular matrix transpose on 2-D input Tensors.

    Examples:
        >>> transpose(to_tensor( [[1 ,2 ,3],[4 ,5 ,6]]))
        <Tensor: shape=(3, 2), dtype=float32, numpy=
         array([[1.0000e+00, 4.0000e+00],
           [2.0000e+00, 5.0000e+00],
           [3.0000e+00, 6.0000e+00]], dtype=float32)>
        >>> transpose(to_tensor( [[1 ,2 ,3],[4 ,5 ,6]]),perm = to_tensor([1, 0],dtype=tf.int32))
        <Tensor: shape=(3, 2), dtype=float32, numpy=
         array([[1.0000e+00, 4.0000e+00],
           [2.0000e+00, 5.0000e+00],
           [3.0000e+00, 6.0000e+00]], dtype=float32)>
        >>> x1=to_tensor([[[1 ,2 ,3],[4 ,5 ,6]], [[7 ,8 ,9], [10,11,12]]])
        >>> transpose(x1, perm=to_tensor([0, 2, 1],dtype=tf.int32))
        <Tensor: shape=(2, 3, 2), dtype=float32, numpy=
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

@numpy_compatible
def permute(x: Tensor, perm=None) -> Tensor:
    if isinstance(perm, (list, tuple)):
        return tf.transpose(x, to_tensor(perm, dtype=tf.int32))
    elif perm is None:
        return tf.transpose(x)
    return tf.transpose(x, to_tensor(perm, dtype=tf.int32))

@numpy_compatible
def depth_to_space(x: Tensor, block_size=2):
    """Rearranges elements in the input tensor from the depth dimension into spatial blocks.

    The equivalent to Pixel-Shuffle

    Args:
        x (tensor): Input tensor, with dimensions CHW or NCHW
        block_size (int):

    Returns: resized tensor

    Examples:
        >>> x = to_tensor(np.tile(np.array(np.reshape(range(8), (8,1,1)), dtype=np.float32), (1, 2, 3)).transpose([1,2,0]))
        >>> x
        <Tensor: shape=(2, 3, 8), dtype=float32, numpy=array([[[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
                 5.0000e+00, 6.0000e+00, 7.0000e+00],
                [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
                 5.0000e+00, 6.0000e+00, 7.0000e+00],
                [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
                 5.0000e+00, 6.0000e+00, 7.0000e+00]],<BLANKLINE>
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
        <Tensor: shape=(2, 4, 6), dtype=float32, numpy=
        array([[[0.0000e+00, 2.0000e+00, 0.0000e+00, 2.0000e+00, 0.0000e+00,
                 2.0000e+00],
                [4.0000e+00, 6.0000e+00, 4.0000e+00, 6.0000e+00, 4.0000e+00,
                 6.0000e+00],
                [0.0000e+00, 2.0000e+00, 0.0000e+00, 2.0000e+00, 0.0000e+00,
                 2.0000e+00],
                [4.0000e+00, 6.0000e+00, 4.0000e+00, 6.0000e+00, 4.0000e+00,
                 6.0000e+00]],<BLANKLINE>
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

@numpy_compatible
def space_to_depth(x: Tensor, block_size=2):
    """Rearranges elements in the input tensor from the spatial dimensions to the depth dimension.

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

    Examples:
        >>> arr=space_to_depth( to_tensor([[[0.,1. ],[2., 3.],[0.,1. ],[2., 3.],[0.,1. ],[2., 3.]],[[4., 5.],[6.,7.],[4., 5.],[6., 7.],[4., 5.],[6., 7.]],[[0.,1. ],[2., 3.],[0.,1. ],[2., 3.],[0.,1. ],[2., 3.]],[[4., 5.],[6., 7.],[4., 5.],[6., 7.],[4., 5.],[6., 7.]]]),block_size=2)
        >>> arr
        <Tensor: shape=(2, 3, 8), dtype=float32, numpy=array([[[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00],
            [0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
             5.0000e+00, 6.0000e+00, 7.0000e+00]],<BLANKLINE>
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
    """Instantiates an all-ones tensor and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        A tensor, filled with `1.0`.

    Example
        >>> ones((3,4))
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]], dtype=float32)

    {{np_implementation}}
    """
    t= tf.ones(shape, dtype)
    if requires_grad==False:
        return tf.constant(t)
    else:
        return t


@numpy_compatible
def ones_like(a:Tensor, dtype=tf.float32, requires_grad=None):
    """Instantiates an all-ones variable of the same shape as another tensor.

    Args
        a (Tensor):  another tensor
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        A tensor, filled with `1.0` and shape is the same as another tensor.

    Example
        >>> ones_like(tf.random.normal((3,4)))
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]], dtype=float32)

    {{np_implementation}}
    """
    return tf.ones_like(a, dtype)



def zeros(shape, dtype=tf.float32, requires_grad=None):
    """Instantiates an all-zeros tensor and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        A tensor, filled with `0.0`.

    Example
        >>> zeros((3,4))
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)

    {{np_implementation}}
    """
    t= tf.zeros(shape, dtype)
    if requires_grad==False:
        return tf.constant(t)
    else:
        return t

@numpy_compatible
def zeros_like(a:Tensor, dtype=tf.float32, requires_grad=None):
    """Instantiates an all-zeros variable of the same shape as another tensor.

    Args
        a (Tensor):  another tensor
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        A tensor, filled with `0.0` and shape is the same as another tensor.

    Example
        >>> zeros_like(tf.random.normal((3,4)))
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)

    {{np_implementation}}
    """
    t= tf.zeros_like(a, dtype)
    if requires_grad==False:
        return tf.constant(t)
    else:
        return t


def eye(shape, dtype=tf.float32, requires_grad=None):
    """Instantiate an identity matrix and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        an identity matrix.

    Examples:
        >>> eye((3,4))
        tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]])

    """
    if len(shape) == 2:
        t= tf.eye(shape[0], shape[1], dtype=dtype)
        if requires_grad == False:
            return tf.constant(t)
        else:
            return t
    else:
        raise ValueError('input tensor must have exactly two axe.')

@numpy_compatible
def eye_like(a:Tensor, dtype=tf.float32, requires_grad=None):
    """
    Creates a matrix with diagonal set to 1s and of the same shape and the same dynamic axes as ``x``. To be a
    matrix, ``x`` must have exactly two axes (counting both dynamic and static axes).

    Args:
        a (Tensor):  another tensor of rank 2
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns:
        tensor

    Examples:
        >>> eye_like(to_tensor(np.random.standard_normal((3,4))))
        tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]])

    """
    if a.ndim == 2:
        t= tf.eye(a.shape[0], a.shape[1], dtype=dtype)
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
        requires_grad (bool): whether need require gradient.

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

@numpy_compatible
def make_onehot(label, num_classes, axis=-1):
    """
    Create one hot tensor based on the input tensor
    Args:
        label: input tensor, the value must be positive integer and less than num_class
        num_classes: the number of class in one hot tensor
        axis: The axis to fill (default: -1, a new inner-most axis).
    Returns:
        :onehot tensor
    Examples:
    >>> make_onehot(to_tensor([[1, 2],[1, 3]]).long(), 4, axis=-1)
    tensor([[[0., 1., 1., 0.],
             [0., 1., 0., 1.]],
    <BLANKLINE>
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.]]])

    """
    return tf.one_hot(indices=cast(label,'int64'), depth=num_classes, on_value=1.0, off_value=0.0, axis=axis)

@numpy_compatible
def meshgrid(x, y, normalized_coordinates=False, requires_grad=None):
    """Return meshgrid in range x & y.

    Args:
      requires_grad ():
      normalized_coordinates ():
      x: (int) first dim range.
      y: (int) second dim range.

    Returns:
      (tensor) meshgrid, sized [y,x,2]

    Examples:
        >>> grid=meshgrid(3,2)
        >>> grid
        <Tensor: shape=(3, 2, 2), dtype=float32, numpy=array([[[0.0000e+00, 0.0000e+00],
            [1.0000e+00, 0.0000e+00]],<BLANKLINE>
           [[0.0000e+00, 1.0000e+00],
            [1.0000e+00, 1.0000e+00]],<BLANKLINE>
           [[0.0000e+00, 2.0000e+00],
            [1.0000e+00, 2.0000e+00]]], dtype=float32)>
        >>> print(grid[0,0,:])
        Tensor([0.0000e+00 0.0000e+00], shape=(2,), dtype=float32)
        >>> print(grid[:,0,0])
        Tensor([0.0000e+00 1.0000e+00 2.0000e+00], shape=(3,), dtype=float32)
        >>> print(grid.shape)
        (3, 2, 2)
        >>> x = to_tensor([1, 2, 3])
        >>> y = to_tensor([4, 5, 6])
        >>> grid_x, grid_y = tf.meshgrid(x, y)
        >>> grid_x
        <Tensor: shape=(3, 3), dtype=float32, numpy=
        array([[1.0000e+00, 2.0000e+00, 3.0000e+00],
               [1.0000e+00, 2.0000e+00, 3.0000e+00],
               [1.0000e+00, 2.0000e+00, 3.0000e+00]], dtype=float32)>

        >>> grid_y
        <Tensor: shape=(3, 3), dtype=float32, numpy=
        array([[4.0000e+00, 4.0000e+00, 4.0000e+00],
               [5.0000e+00, 5.0000e+00, 5.0000e+00],
               [6.0000e+00, 6.0000e+00, 6.0000e+00]], dtype=float32)>
        >>> meshgrid(3,2,normalized_coordinates=True)
        <Tensor: shape=(3, 2, 2), dtype=float32, numpy=
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
@numpy_compatible
def concate(x: List[Tensor], axis=-1):
    return tf.concat(x, axis=axis)

@numpy_compatible
def stack(x: List[Tensor], axis=-1):
    return tf.stack(x, axis=axis)

@numpy_compatible
def split(x: Tensor, num_splits=2,axis=-1):
    """Splits a tensor `value` into a list of sub tensors.

      See also `tf.unstack`.

      If `num_or_size_splits` is an integer,  then `value` is split along the
      dimension `axis` into `num_or_size_splits` smaller tensors. This requires that
      `value.shape[axis]` is divisible by `num_or_size_splits`.

      If `num_or_size_splits` is a 1-D Tensor (or list), then `value` is split into
      `len(num_or_size_splits)` elements. The shape of the `i`-th
      element has the same size as the `value` except along dimension `axis` where
      the size is `num_or_size_splits[i]`.

      For example:

      >>> x = tf.Variable(tf.random.uniform([5, 30], -1, 1))
      >>>
      >>> # Split `x` into 3 tensors along dimension 1
      >>> s0, s1, s2 = tf.split(x, num_or_size_splits=3, axis=1)
      >>> tf.shape(s0).numpy()
      array([ 5, 10], dtype=int32)
      >>>
      >>> # Split `x` into 3 tensors with sizes [4, 15, 11] along dimension 1
      >>> split0, split1, split2 = tf.split(x, [4, 15, 11], 1)
      >>> tf.shape(split0).numpy()
      array([5, 4], dtype=int32)
      >>> tf.shape(split1).numpy()
      array([ 5, 15], dtype=int32)
      >>> tf.shape(split2).numpy()
      array([ 5, 11], dtype=int32)

      Args:
        x: The `Tensor` to split.
        num_splits: Either an integer indicating the number of splits along
          `axis` or a 1-D integer `Tensor` or Python list containing the sizes of
          each output tensor along `axis`. If a scalar, then it must evenly divide
          `value.shape[axis]`; otherwise the sum of sizes along the split axis
          must match that of the `value`.
        axis: An integer or scalar `int32` `Tensor`. The dimension along which to
          split. Must be in the range `[-rank(value), rank(value))`. Defaults to 0.
        num: Optional, used to specify the number of outputs when it cannot be
          inferred from the shape of `size_splits`.


      Returns:
        if `num_or_size_splits` is a scalar returns a list of `num_or_size_splits`
        `Tensor` objects; if `num_or_size_splits` is a 1-D Tensor returns
        `num_or_size_splits.get_shape[0]` `Tensor` objects resulting from splitting
        `value`.

      Raises:
        ValueError: If `num` is unspecified and cannot be inferred.
      """
    return tf.split(x, axis=axis,num_or_size_splits=num_splits)


@numpy_compatible
def repeat_elements(x: Tensor, multiples:int,axis=-1):
    """Repeat elements of a tensor.

    Args:
        x (Tensor):the input tensor.
        multiples(Tensor or int):The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
        axis (int): The dimension along which to repeat values. By default, use the flattened input array, and return a flat output array.

    Returns (Tensor)::Repeated tensor which has the same shape as input, except along the given axis.

    """
    x_shape = x.shape.as_list()
    # For static axis
    if x_shape[axis] is not None:
        # slices along the repeat axis
        splits = array_ops.split(value=x,
                                 num_or_size_splits=x_shape[axis],
                                 axis=axis)
        # repeat each slice the given number of reps
        x_rep = [s for s in splits for _ in range(multiples)]
        return concate(x_rep, axis)

    # Here we use tf.tile to mimic behavior of np.repeat so that
    # we can handle dynamic shapes (that include None).
    # To do that, we need an auxiliary axis to repeat elements along
    # it and then merge them along the desired axis.

    # Repeating
    auxiliary_axis = axis + 1
    x_shape = array_ops.shape(x)
    x_rep = array_ops.expand_dims(x, axis=auxiliary_axis)
    reps = np.ones(len(x.shape) + 1)
    reps[auxiliary_axis] = multiples
    x_rep = array_ops.tile(x_rep, reps)

    # Merging
    reps = np.delete(reps, auxiliary_axis)
    reps[axis] = multiples
    reps = array_ops.constant(reps, dtype='int32')
    x_shape *= reps
    x_rep = array_ops.reshape(x_rep, x_shape)

    # Fix shape representation
    x_shape = x.shape.as_list()
    x_rep.set_shape(x_shape)
    x_rep._keras_shape = tuple(x_shape)
    return x_rep


@numpy_compatible
def gather(x: Tensor,gather_axis, indices):
    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])
    # splice in our pytorch style index at the correct axis
    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(gather_locations)
        else:
            gather_indices.append(all_indices[:, axis])

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped

@numpy_compatible
def scatter_add(x: Tensor,indices: Tensor,updates: Tensor):
    return tf.tensor_scatter_nd_add(x, indices, updates)

@numpy_compatible
def scatter_sub(x: Tensor,indices: Tensor,updates: Tensor):
    return tf.tensor_scatter_nd_sub(x, indices, updates)


@numpy_compatible
def scatter_max(x: Tensor,indices: Tensor,updates: Tensor):
    return tf.tensor_scatter_nd_max(x, indices, updates)


@numpy_compatible
def scatter_min(x: Tensor,indices: Tensor,updates: Tensor):
    return tf.tensor_scatter_nd_min(x, indices, updates)



def assign(x:tf.Variable, new_x):
  return state_ops.assign(x, new_x)



def assign_add(x:tf.Variable, increment):
  """Update the value of `x` by adding `increment`.
  Arguments:
      x: A Variable.
      increment: A tensor of same shape as `x`.
  Returns:
      The variable `x` updated.
  """
  return state_ops.assign_add(x, increment)



def assign_sub(x:tf.Variable, decrement):
  """Update the value of `x` by subtracting `decrement`.
  Arguments:
      x: A Variable.
      decrement: A tensor of same shape as `x`.
  Returns:
      The variable `x` updated.
  """
  return state_ops.assign_sub(x, decrement)



def moving_average_assign(x:tf.Variable, value, momentum):
  """Compute the exponential moving average of a value.
  The moving average 'x' is updated with 'value' following:
  ```
  x = x * momentum + value * (1 - momentum)
  ```
  For example:
  >>> x = tf.Variable(0.0)
  >>> momentum=0.9
  >>> moving_average_assign(x, value = 2.0, momentum=momentum).numpy()
  >>> x.numpy()
  0.2
  The result will be biased towards the initial value of the variable.
  If the variable was initialized to zero, you can divide by
  `1 - momentum ** num_updates` to debias it (Section 3 of
  [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)):
  >>> num_updates = 1.0
  >>> x_zdb = x/(1 - momentum**num_updates)
  >>> x_zdb.numpy()
  2.0
  Arguments:
      x: A Variable, the moving average.
      value: A tensor with the same shape as `x`, the new value to be
        averaged in.
      momentum: The moving average momentum.
  Returns:
      The updated variable.
  """

  return moving_averages.assign_moving_average(
      x, value, momentum, zero_debias=False)



@numpy_compatible
def gram_matrix(x: Tensor):
    temp = x
    temp = squeeze(temp)
    fun = reshape(temp, [temp.shape[2], temp.shape[0] * temp.shape[1]])
    result = matmul(temp, temp, transpose_b=True)
    gram = expand_dims(result, axis=0)
    return gram


############################
## random
###########################

def set_seed(seed: int) -> None:
    """Setup random state from a seed for `tf.random`, `random` and  `numpy` (if can be imported).
    Args:
        seed (int): Random state seed

    """
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


@numpy_compatible
def shuffle(x: Tensor,seed=None):
    """Randomly shuffles a tensor along its first dimension.

    The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
    to one and only one `output[i]`. For example, a mapping that might occur for a
    3x2 tensor is:

    ```python
    [[1, 2],       [[5, 6],
     [3, 4],  ==>   [1, 2],
     [5, 6]]        [3, 4]]
    ```

    Args:
        x (Tensor): input tensor (1-D  tensor).
        seed (None or int): random seed.

    Returns:
        A tensor of same shape and type as `value`, shuffled along its first dimension.

    """
    if seed is not None:
        return tf.random.shuffle(x,seed)
    return tf.random.shuffle(x)

@numpy_compatible
def random_choice(x: Tensor,n:int=1):
    """Generates a random sample from a given 1-D array

    Args:
        x (Tensor): input tensor (1-D  tensor).
        n (int): how many items

    Returns:
        (Tensor) : single item ,the generated random samples

    """
    idxes = np.arange(len(x))
    np.random.shuffle(idxes)
    idx = idxes[:n]
    return unpack_singleton(x[idx])



def random_normal(shape, mean=0.0, std=1.0, dtype='float32', seed=None):
    """Outputs random values from a normal distribution.

    In this case, we are setting both the global and operation-level seed to
    ensure this result is reproducible.  See `tf.random.set_seed` for more
    information.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
      mean: A Tensor or Python value of type `dtype`, broadcastable with `stddev`.
        The mean of the normal distribution.
      std: A Tensor or Python value of type `dtype`, broadcastable with `mean`.
        The standard deviation of the normal distribution.
      dtype: The type of the output.
      seed: A Python integer. Used to create a random seed for the distribution.
        See
        `tf.random.set_seed`
        for behavior.

    Returns:
      A tensor of the specified shape filled with random normal values.


    Example :
        >>> #that generates a new set of random values every time
        >>> random_normal([4],dtype='float32' ,mean=0, stddev=1,seed=5)
        <Tensor: shape=(4,), dtype=float32, numpy=..., dtype=float32)>
        >>> #that outputs a reproducible result:
        >>> random_normal([2,2],dtype='float32' ,mean=0, stddev=1,seed=5)
        <Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[-1.3768897 , -0.01258316],
              [-0.169515   ,  1.0824056 ]], dtype=float32)>


    """
    return tf.random.normal(to_list(shape),mean=mean,stddev=std,dtype=str2dtype(dtype))

@numpy_compatible
def random_normal_like(x, mean=0.0, std=1.0, dtype='float32', seed=None):
    """Outputs random values from a normal distribution.

    In this case, we are setting both the global and operation-level seed to
    ensure this result is reproducible.  See `tf.random.set_seed` for more
    information.

    Args:
      x: input tensor.
      mean: A Tensor or Python value of type `dtype`, broadcastable with `stddev`.
        The mean of the normal distribution.
      std: A Tensor or Python value of type `dtype`, broadcastable with `mean`.
        The standard deviation of the normal distribution.
      dtype: The type of the output.
      seed: A Python integer. Used to create a random seed for the distribution.
        See
        `tf.random.set_seed`
        for behavior.

    Returns:
      A tensor of the specified shape filled with random normal values.


    Example :
        >>> #that generates a new set of random values every time
        >>> random_normal([4],dtype='float32' ,mean=0, stddev=1,seed=5)
        <Tensor: shape=(4,), dtype=float32, numpy=..., dtype=float32)>
        >>> #that outputs a reproducible result:
        >>> random_normal([2,2],dtype='float32' ,mean=0, stddev=1,seed=5)
        <Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[-1.3768897 , -0.01258316],
              [-0.169515   ,  1.0824056 ]], dtype=float32)>


    """
    return tf.random.normal(to_list(int_shape(x)),mean=mean,stddev=std,dtype=str2dtype(dtype))

def random_uniform(shape, min_value=0.0, max_value=None, dtype='float32', seed=None):
    """Outputs random values from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[min, max)`. The lower bound `minval` is included in the range, while
    the upper bound `maxval` is excluded.

    For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
    be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `max_value - min_value` is an exact power of two.  The bias is small for values of
    `max_value - min_value` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    Examples:

    >>> tf.random.uniform(shape=[2])
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([..., ...], dtype=float32)>
    >>> tf.random.uniform(shape=[], minval=-1., maxval=0.)
    <tf.Tensor: shape=(), dtype=float32, numpy=-...>
    >>> tf.random.uniform(shape=[], minval=5, maxval=10, dtype=tf.int64)
    <tf.Tensor: shape=(), dtype=int64, numpy=...>

    The `seed` argument produces a deterministic sequence of tensors across
    multiple calls. To repeat that sequence, use `tf.random.set_seed`:

    >>> tf.random.set_seed(5)
    >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
    <tf.Tensor: shape=(), dtype=int32, numpy=2>
    >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
    <tf.Tensor: shape=(), dtype=int32, numpy=0>
    >>> tf.random.set_seed(5)
    >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
    <tf.Tensor: shape=(), dtype=int32, numpy=2>
    >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
    <tf.Tensor: shape=(), dtype=int32, numpy=0>

    Without `tf.random.set_seed` but with a `seed` argument is specified, small
    changes to function graphs or previously executed operations will change the
    returned value. See `tf.random.set_seed` for details.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
      min_value: A Tensor or Python value of type `dtype`, broadcastable with
        `shape` (for integer types, broadcasting is not supported, so it needs to
        be a scalar). The lower bound on the range of random values to generate
        (inclusive).  Defaults to 0.
      max_value: A Tensor or Python value of type `dtype`, broadcastable with
        `shape` (for integer types, broadcasting is not supported, so it needs to
        be a scalar). The upper bound on the range of random values to generate
        (exclusive). Defaults to 1 if `dtype` is floating point.
      dtype: The type of the output: `float16`, `float32`, `float64`, `int32`,
        or `int64`.
      seed: A Python integer. Used in combination with `tf.random.set_seed` to
        create a reproducible sequence of tensors across multiple calls.


    Returns:
      A tensor of the specified shape filled with random uniform values.

    Raises:
      ValueError: If `dtype` is integral and `maxval` is not specified.
    """

    return tf.random.uniform(shape=to_list(shape),
    minval = min_value,
    maxval = max_value,
    dtype = str2dtype(dtype),
    seed = seed,
    name = 'random_uniform')

@numpy_compatible
def random_uniform_like(x,  min_value=0.0, max_value=None, dtype='float32', seed=None):
    """Outputs random values from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[min, max)`. The lower bound `minval` is included in the range, while
    the upper bound `maxval` is excluded.

    For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
    be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `max_value - min_value` is an exact power of two.  The bias is small for values of
    `max_value - min_value` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    Examples:

    >>> tf.random.uniform(shape=[2])
    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([..., ...], dtype=float32)>
    >>> tf.random.uniform(shape=[], minval=-1., maxval=0.)
    <tf.Tensor: shape=(), dtype=float32, numpy=-...>
    >>> tf.random.uniform(shape=[], minval=5, maxval=10, dtype=tf.int64)
    <tf.Tensor: shape=(), dtype=int64, numpy=...>

    The `seed` argument produces a deterministic sequence of tensors across
    multiple calls. To repeat that sequence, use `tf.random.set_seed`:

    >>> tf.random.set_seed(5)
    >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
    <tf.Tensor: shape=(), dtype=int32, numpy=2>
    >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
    <tf.Tensor: shape=(), dtype=int32, numpy=0>
    >>> tf.random.set_seed(5)
    >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
    <tf.Tensor: shape=(), dtype=int32, numpy=2>
    >>> tf.random.uniform(shape=[], maxval=3, dtype=tf.int32, seed=10)
    <tf.Tensor: shape=(), dtype=int32, numpy=0>

    Without `tf.random.set_seed` but with a `seed` argument is specified, small
    changes to function graphs or previously executed operations will change the
    returned value. See `tf.random.set_seed` for details.

    Args:
      x: inutput tensor.
      min_value: A Tensor or Python value of type `dtype`, broadcastable with
        `shape` (for integer types, broadcasting is not supported, so it needs to
        be a scalar). The lower bound on the range of random values to generate
        (inclusive).  Defaults to 0.
      max_value: A Tensor or Python value of type `dtype`, broadcastable with
        `shape` (for integer types, broadcasting is not supported, so it needs to
        be a scalar). The upper bound on the range of random values to generate
        (exclusive). Defaults to 1 if `dtype` is floating point.
      dtype: The type of the output: `float16`, `float32`, `float64`, `int32`,
        or `int64`.
      seed: A Python integer. Used in combination with `tf.random.set_seed` to
        create a reproducible sequence of tensors across multiple calls.


    Returns:
      A tensor of the specified shape filled with random uniform values.

    Raises:
      ValueError: If `dtype` is integral and `maxval` is not specified.
    """
    return tf.random.uniform(shape=to_list(int_shape(x)),
                             minval=min_value,
                             maxval=max_value,
                             dtype=str2dtype(dtype),
                             seed=seed,
                             name='random_uniform')


@numpy_compatible
def multinomial(x:Tensor,num_samples: int=1):
    """Draws samples from a categorical distribution.

  Example:

  ```python
  # samples has shape [1, 5], where each value is either 0 or 1 with equal
  # probability.
  samples = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)
  ```

  Args:
    x: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
      `[i, :]` represents the unnormalized log-probabilities for all classes.
    num_samples: 0-D.  Number of independent samples to draw for each row slice.

  Returns:
    The drawn samples of shape `[batch_size, num_samples]`.
  """
    return  tf.random.categorical(x,num_samples)


@numpy_compatible
def binary_crossentropy(output,target,  from_logits=False):
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








_FUN_NAMES = [
    # source_fun, target_fun
    ('to_numpy', to_numpy),
    ('detach', detach),
    ('cpu', cpu),
    ('cuda', cuda),
    ('copy', copy),
    ('numel', numel),
    ('ndim', ndim),
    ('int_shape', int_shape),
    ('cast', cast),
    ('is_sparse', is_sparse),
    ('is_nan', is_nan),
    ('is_inf', is_inf),
    ('is_abnormal_number', is_abnormal_number),
    ('any_nan', any_nan),
    ('any_inf', any_inf),
    ('any_abnormal_number', any_abnormal_number),
    ('less', less),
    ('equal', equal),
    ('greater', greater),
    ('greater_equal', greater_equal),
    ('not_equal', not_equal),
    ('less_equal', less_equal),
    ('argmax', argmax),
    ('argmin', argmin),
    ('argsort', argsort),
    ('topk', topk),
    ('maximum', maximum),
    ('minimum', minimum),
    ('floor', floor),
    ('ceil', ceil),
    ('round', round),
    ('dot', dot),
    ('sqrt', sqrt),
    ('rsqrt', rsqrt),
    ('square', square),
    ('abs', abs),
    ('pow', pow),
    ('log', log),
    ('exp', exp),
    ('clip', clip),
    ('add', add),
    ('subtract', subtract),
    ('true_divide', true_divide),
    ('matmul', matmul),
    ('sin', sin),
    ('cos', cos),
    ('tan', tan),
    ('asin', asin),
    ('acos', acos),
    ('atan', atan),
    ('sinh', sinh),
    ('cosh', cosh),
    ('tanh', tanh),
    ('element_times', element_times),
    ('element_max', element_max),
    ('element_min', element_min),
    ('element_divide', element_divide),
    ('element_cosine_distance', element_cosine_distance),
    ('where', where),
    ('reduce_mean', reduce_mean),
    ('reduce_sum', reduce_sum),
    ('reduce_max', reduce_max),
    ('reduce_min', reduce_min),
    ('mean', mean),
    ('sum', sum),
    ('max', max),
    ('min', min),
    ('reduce_logsumexp', reduce_logsumexp),
    ('reduce_prod', reduce_prod),
    ('depth_to_space', depth_to_space),
    ('space_to_depth', space_to_depth),
    ('identity', identity),
    ('sigmoid', sigmoid),
    ('relu', relu),
    ('relu6', relu6),
    ('leaky_relu', leaky_relu),
    ('leaky_relu6', leaky_relu6),
    ('smooth_relu', smooth_relu),
    ('p_relu', p_relu),
    ('swish', swish),
    ('elu', elu),
    ('hard_sigmoid', hard_sigmoid),
    ('hard_swish', hard_swish),
    ('selu', selu),
    ('lecun_tanh', lecun_tanh),
    ('soft_sign', soft_sign),
    ('soft_plus', soft_plus),
    ('hard_tanh', hard_tanh),
    ('logit', logit),
    ('log_log', log_log),
    ('mish', mish),
    ('hard_mish', hard_mish),
    ('softmax', softmax),
    ('log_softmax', log_softmax),
    ('gelu', gelu),
    ('gpt_gelu', gpt_gelu),
    ('l2_normalize', l2_normalize),
    ('ones_like', ones_like),
    ('zeros_like', zeros_like),
    ('eye_like', eye_like),
    ('arange', arange),
    ('make_onehot', make_onehot),
    ('meshgrid', meshgrid),
    ('reshape', reshape),
    ('permute', permute),
    ('transpose', transpose),
    ('squeeze', squeeze),
    ('expand_dims', expand_dims),
    ('repeat_elements', repeat_elements),
    ('gather', gather),
    ('gram_matrix', gram_matrix),
    ('shuffle', shuffle),
    ('random_choice', random_choice),
    ('random_normal_like', random_normal_like)
    ]
for target_fun_name,source_fun in _FUN_NAMES:
    if not hasattr(Tensor,target_fun_name):
        setattr(Tensor, target_fun_name, source_fun)
    elif target_fun_name in ["float","int","long"]:
        setattr(Tensor, target_fun_name, source_fun)
del _FUN_NAMES


#setattr(EagerTensor, 'detach', detach)