from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import random
import math
import os
import sys
import builtins
from scipy import special
from collections import Sized, Iterable
from functools import partial
from typing import Tuple, List, Optional, Union, Sequence

import numpy as np
from trident.backend.common import *
#
# __all__ = [ 'ndim', 'cast', 'int_shape', 'is_nan', 'is_inf',
#            'is_abnormal_number', 'any_nan', 'any_inf', 'any_abnormal_number', 'less', 'equal', 'greater',
#            'greater_equal', 'not_equal', 'less_equal', 'argmax', 'argmin', 'argsort', 'maximum', 'minimum', 'floor',
#            'ceil', 'round', 'dot', 'sqrt', 'rsqrt', 'prod', 'square', 'abs', 'pow', 'log', 'exp', 'clip', 'add', 'subtract',
#            'true_divide', 'pi', 'matmul', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
#            'element_times', 'element_max', 'element_min', 'element_divide', 'element_cosine_distance', 'where',
#            'reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min', 'mean', 'sum', 'max', 'min', 'reduce_logsumexp',
#            'reduce_prod', 'identity', 'sigmoid', 'relu', 'relu6', 'leaky_relu',
#            'leaky_relu6', 'smooth_relu', 'swish', 'elu', 'hard_sigmoid', 'hard_swish', 'selu', 'lecun_tanh',
#            'soft_sign', 'soft_plus', 'hard_tanh', 'logit', 'log_log', 'mish', 'hard_mish', 'softmax', 'log_softmax', 'gelu',
#            'gpt_gelu', 'moments', 'l2_normalize', 'ones', 'ones_like', 'zeros', 'zeros_like', 'eye', 'eye_like','arange', 'meshgrid', 'reshape',
#            'permute', 'transpose', 'squeeze', 'expand_dims', 'concate', 'stack', 'gram_matrix', 'set_seed', 'shuffle',
#            'random_choice', 'random_normal', 'random_normal_like', 'binary_crossentropy']
#



############################
## tensor attribute
###########################


def ndim(x:np.ndarray):
    """Number of dimension of a tensor

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (int) Number of dimension

    """
    return x.ndim

def numel(x: np.ndarray):
    """Number of elements of a tensor

    Args:
        x (Tensor): input tensor.

    Returns:
        (int) Number of elements

    """
    return x.size


def int_shape(x:np.ndarray):
    """Shape of a tensor in tuple of integer format

    Args:
        x : input tensor

    Returns:
        tuple of integer as shape representation

    Examples:
        >>> int_shape(ones((3,3,7)))
        (3, 3, 7)

    """
    return x.shape




def str2dtype(dtype):
    """ Mapping string to dtype

    Args:
        dtype (str): dtype representation string

    Returns:
        dtype

    """
    if isinstance(dtype, np.dtype):
        return dtype
    elif isinstance(dtype, str):
        if 'float64' in dtype.lower() or 'double' in dtype.lower():
            return np.float64
        elif 'float16' in dtype.lower() or 'half' in dtype.lower():
            return np.float16
        elif 'float' in dtype.lower():
            return np.float32
        elif 'int64' in dtype.lower() or 'long' in dtype.lower():
            return np.int64
        elif 'int16' in dtype.lower() or 'short' in dtype.lower():
            return np.int16
        elif 'uint8' in dtype.lower() or 'byte' in dtype.lower():
            return np.uint8
        elif 'int8' in dtype.lower() or 'char' in dtype.lower():
            return np.int8
        elif 'int32' in dtype.lower() or 'int' in dtype.lower():
            return np.int32
        elif 'bool' in dtype.lower():
            return np.bool
    return np.float32


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
        >>> x = np.array([1.8, 2.2])
        >>>cast(x, torch.int32)
        tensor([1, 2], dtype=int32)

    Raises:
        TypeError: If `x` cannot be cast to the `dtype`.

    """
    if isinstance(dtype, np.dtype):
        if dtype == np.float64:
            return x.astype(np.float64)
        elif dtype == np.float16 :
            return x.astype(np.float16)
        elif dtype == np.float32:
            return x.astype(np.float32)
        elif dtype == np.int64:
            return x.astype(np.int64)
        elif dtype == np.int32:
            return x.astype(np.int32)
        elif dtype == np.int16:
            return x.astype(np.int16)
        else:
            return x.astype(np.float32)


############################
## check operation
###########################

def is_nan(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:

    """
    if isinstance(x, np.ndarray):
        return np.isnan(x)
    else:
        return math.isnan(x)


def is_inf(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:

    """
    if isinstance(x, np.ndarray):
        return np.isinf(x)
    else:
        return math.isinf(x)



def is_abnormal_number(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:

    """
    return is_nan(x) or is_inf(x)


def any_nan(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:

    """
    if isinstance(x, np.ndarray):
        return np.isnan(x).any()
    elif isinstance(x, Iterable):
        return any([is_nan(n) for n in x])
    else:
        return is_nan(x)


def any_inf(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:

    """
    if isinstance(x, np.ndarray):
        return np.isinf(x).any()
    elif isinstance(x, Iterable):
        return any([is_inf(n) for n in x])
    else:
        return is_inf(x)


def any_abnormal_number(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:

    """
    return any_nan(x) or any_inf(x)





############################
## logical  operation
###########################


def logical_and(left, right):
    """Element-wise `logical and: x && y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor

    Returns:
        A Tensor of type bool with the same size as that of left or right.

    """
    return np.logical_and(left, right)


def logical_not(x:np.ndarray):
    """Element-wise `logical not: ~x`
    Args:
        x (Tensor): input boolean tensor
    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return np.logical_not(x)


def logical_or(left, right):
    """Element-wise `logical or: x || y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor
    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return np.logical_or(left, right)


def logical_xor(left, right):
    """Element-wise `logical xor: x ^ y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor

    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return np.logical_xor(left, right)



############################
## compare operation
###########################


def less(left:np.ndarray, right:(np.ndarray,float,int)):
    """
    Elementwise 'less' comparison of two tensors. Result is 1 if left < right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        Result is 1 if left < right else 0.

    Examples:
       >>> less(np.array([41., 42., 43.]), np.array([42., 42., 42.]))
       tensor([1., 0., 0.])
       >>> less(np.array([-1,0,1]), 0)
       tensor([1., 0., 0.])

    """

    return np.less(left,right).astype(np.float32)


def equal(left:np.ndarray, right:(np.ndarray,float,int)):
    """
    Elementwise 'equal' comparison of two tensors. Result is 1 if values are equal 0 otherwise.
    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if values are equal 0 otherwise

    Examples:
        >>> equal(np.array([41., 42., 43.]), np.array([42., 42., 42.]))
        tensor([0., 1., 0.])
        >>> equal(np.array([-1,0,1]), 1)
        tensor([0., 0., 1.])

    """
    return np.equal(left,right).astype(np.float32)


def greater(left:np.ndarray, right:(np.ndarray,float,int)):
    """
    Elementwise 'greater' comparison of two tensors. Result is 1 if left > right else 0.
    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left > right else 0.

    Examples:
        >>> greater(np.array([41., 42., 43.]), np.array([42., 42., 42.]))
        tensor([0., 0., 1.])
        >>> greater(np.array([-1,0,1]), 0)
        tensor([0., 0., 1.])

    """
    return np.greater(left,right).astype(np.float32)


def greater_equal(left:np.ndarray, right:(np.ndarray,float,int)):
    """
    Elementwise 'greater equal' comparison of two tensors. Result is 1 if left >= right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left >= right else 0

    Examples:
        >>> greater_equal(np.array([41., 42., 43.]), np.array([42., 42., 42.]))
        tensor([0., 1., 1.])
        >>> greater_equal(np.array([-1,0,1]), 0)
        tensor([0., 1., 1.])

    """
    return np.greater_equal(left,right).astype(np.float32)


def not_equal(left:np.ndarray, right:(np.ndarray,float,int)):
    """
    Elementwise 'not equal' comparison of two tensors. Result is 1 if left != right else 0.

    Args:
        left: left side tensor
        right: right side tensor
    Returns:
        :Result is 1 if left != right else 0.

    Examples:
        >>> not_equal(np.array([41., 42., 43.]), np.array([42., 42., 42.]))
        tensor([1., 0., 1.])
        >>> not_equal(np.array([-1,0,1]), 0)
        tensor([1., 0., 1.])

    """
    return np.not_equal(left,right).astype(np.float32)


def less_equal(left:np.ndarray, right:(np.ndarray,float,int)):
    """
    Elementwise 'less equal' comparison of two tensors. Result is 1 if left <= right else 0.

    Args:
        left: left side tensor
        right: right side tensor

    Returns:
        Result is 1 if left <= right else 0.
    Examples:
        >>> less_equal(np.array([41., 42., 43.]), np.array([42., 42., 42.]))
        tensor([1., 1., 0.])
        >>> less_equal(np.array([-1,0,1]), 0)
        tensor([1., 1., 0.])

    """
    return np.less_equal(left,right).astype(np.float32)


def argmax(x:np.ndarray, axis=1) ->np.ndarray:
    return np.argmax(x,axis=axis)


def argmin(x:np.ndarray, axis=1) ->np.ndarray:
    return np.argmin(x,axis=axis)


def argsort(x:np.ndarray, axis=1, descending=True) ->np.ndarray:
    if descending:
        return np.argsort(-x, axis=axis)
    else:
        return np.argsort(x,axis=axis)


def topk(x: np.ndarray,  k=1) -> np.ndarray:
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



     Returns:
       values: The `k` largest elements along each last dimensional slice.
       indices: The indices of `values` within the last dimension of `input`.
     """
    full_sort = np.argsort(x)
    return full_sort.take(np.arange(k))

def maximum(x:np.ndarray, other: (np.ndarray, int, float)) ->np.ndarray:
    if isinstance(other,np.ndarray):
        return np.maximum(x, other)
    elif isinstance(other, (int, float)):
        return np.clip(x,a_min=float(other))


def minimum(x:np.ndarray, other: (np.ndarray, int, float)) ->np.ndarray:
    if isinstance(other,np.ndarray):
        return np.minimum(x, other)
    elif isinstance(other, (int, float)):
        return np.clip(x,a_max=float(other))


############################
## basic math operation
###########################
def add(x, y):
    """Returns x + y element-wise.

    *NOTE*: `math.add` supports broadcasting. `AddN` does not. More about broadcasting
    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    Args:
        x (np.ndarray): input tensor.
        y (np.ndarray): another tensor.


    Returns:
      A `Tensor`. Has the same type as `x`.

    """
    return np.add(x, y)


def subtract(x, y):
    """Returns x - y element-wise.

    *NOTE*: `Subtract` supports broadcasting. More about broadcasting
    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    Args:
        x (np.ndarray): input tensor.
        y (np.ndarray): another tensor.


    Returns:
      A `Tensor`. Has the same type as `x`.

    """
    return np.subtract(x, y)


def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

     Args
        x (np.ndarray): input tensor.
        y (np.ndarray): another tensor.

     Returns
            A tensor, dot product of `x` and `y`.


    """
    return np.dot(x, y)


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
        x (np.ndarray): input tensor.
        y (np.ndarray): another tensor.


    Returns:
      `x / y` evaluated in floating point.

    Raises:
      TypeError: If `x` and `y` have different dtypes.

    """
    return np.true_divide(x, y)


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


     >>> a =reshape(np.array([1, 2, 3, 4, 5, 6]),[2, 3])
     >>> a  # 2-D tensor
     tensor([[1, 2, 3],
            [4, 5, 6]])
     >>> b = reshape(np.array([7, 8, 9, 10, 11, 12]), [3, 2])
     >>> b  # 2-D tensor
     tensor([[ 7,  8],
            [ 9, 10],
            [11, 12]])
     >>> c = matmul(a, b)
     >>> c  # `a` * `b`
     tensor([[ 58,  64],
            [139, 154]])

     A batch matrix multiplication with batch shape [2]:

     >>> a =  reshape(np.array(np.arange(1, 13, dtype=np.int32)),[2, 2, 3])
     >>> a  # 3-D tensor
     tensor([[[ 1,  2,  3],
             [ 4,  5,  6]],
            [[ 7,  8,  9],
             [10, 11, 12]]])
     >>> b =  reshape(np.array(np.arange(13, 25, dtype=np.int32)),[2, 3, 2])
     >>> b  # 3-D tensor
     tensor([[[13, 14],
             [15, 16],
             [17, 18]],
            [[19, 20],
             [21, 22],
             [23, 24]]])
     >>> c = matmul(a, b)
     >>> c  # `a` * `b`
     tensor([[[ 94, 100],
             [229, 244]],
            [[508, 532],
             [697, 730]]])

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
       A `Tensor` of the same type as `a` and `b` where each inner-most matrix
       is the product of the corresponding matrices in `a` and `b`, e.g. if all
       transpose or adjoint attributes are `False`:

       `output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j])`,
       for all indices `i`, `j`.

       Note: This is matrix product, not element-wise product.


     Raises:
       ValueError: If `transpose_a` and `adjoint_a`, or `transpose_b` and
         `adjoint_b` are both set to `True`.

     """
    if transpose_a:
        a = a.T
    if transpose_b:
        b = b.T
    return np.matmul(a, b)


def prod(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:

    """
    return np.prod(x)


def floor(x: (np.ndarray, float)):
    """Returns element-wise largest integer not greater than x.

    Args:
        x (np.ndarray): input tensor.

    Returns:
      A `Tensor`. Has the same type as `x`.

    """

    return np.floor(x)


def ceil(x: (np.ndarray, float)):
    """Return the ceiling of the input, element-wise.

    Example:

    >>> ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    <tf.Tensor: shape=(7,), dtype=float32,
    numpy=array([-1., -1., -0.,  1.,  2.,  2.,  2.], dtype=float32)>

    Args:
        x (np.ndarray): input tensor.


    Returns:
      A `tf.Tensor`. Has the same type as `x`.

    @compatibility(numpy)
    Equivalent to np.ceil
    @end_compatibility
    """
    return np.ceil(x)


def round(x: (np.ndarray, float), digit: int = 0):
    """Rounds the values of a tensor to the nearest integer, element-wise.

    Rounds half to even.  Also known as bankers rounding. If you want to round
    according to the current system rounding mode use tf::cint.

    Args:
        x (np.ndarray): input tensor.
        digit: number of digit

    Returns:
        A `Tensor` of same shape and type as `x`.

    Examples;
        >>> round(np.array([[1,2,3,4,5]])/3,0)
        <tf.Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 2.0000e+00]],
              dtype=float32)>
        >>> round(np.array([[1,2,3,4,5]])/3,2)
        <tf.Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[3.3000e-01, 6.7000e-01, 1.0000e+00, 1.3300e+00, 1.6700e+00]],
              dtype=float32)>
        >>> round(np.array([[11.6,24.3,35.2,14.4,23.5]])/3,-1)
        <tf.Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[0.0000e+00, 1.0000e+01, 1.0000e+01, 0.0000e+00, 1.0000e+01]],
              dtype=float32)>

    """

    return np.round(x,decimals=digit)




def pi():
    """ The number π (/paɪ/)
    The number π (/paɪ/) is a mathematical constant. It is defined as the ratio of a circle's circumference to its diameter

    Returns:
        The number π (/paɪ/)

    """
    return np.pi


def sqrt(x:np.ndarray):
    r"""Computes element-wise square root of the input tensor.

    Note: This operation does not support integer types.

    >>> x = np.array([[4.0], [16.0]])
    >>> tf.sqrt(x)
    <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
      array([[2.],
             [4.]], dtype=float32)>
    >>> y = np.array([[-4.0], [16.0]])
    >>> sqrt(y)
    <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
      array([[nan],
             [ 4.]], dtype=float32)>
    >>> z = np.array([[-1.0], [16.0]], dtype=tf.complex128)
    >>> sqrt(z)
    <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
      array([[0.0+1.j],
             [4.0+0.j]])>

    Note: In order to support complex complex, please provide an input tensor
    of `complex64` or `complex128`.

    Args:
        x: A `tf.Tensor`


    Returns:
      A `tf.Tensor` of same size, type and sparsity as `x`.

    """
    return np.sqrt(x)


def rsqrt(x:np.ndarray):
    """Computes reciprocal of square root of x element-wise.

    Args:
      x: input tensor

    Returns:
      output tensor


    Examples:
        >>> x = np.array([2., 0., -2.])
        >>> rsqrt(x)
        <tf.Tensor: shape=(3,), dtype=float32,
        numpy=array([0.707, inf, nan], dtype=float32)>

    """

    return 1/np.sqrt(x)


def square(x:np.ndarray):
    r"""Computes square of x element-wise.

    I.e., \\(y = x * x = x^2\\).

    >>> tf.math.square([-2., 0., 3.])
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([4., 0., 9.], dtype=float32)>

    Args:
      x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`,
      `complex64`, `complex128`.


    Returns:
      A `Tensor`. Has the same type as `x`.


    """
    return np.square(x)


def abs(x:np.ndarray):
    r"""Computes the absolute value of a tensor.

    Given a tensor of integer or floating-point values, this operation returns a
    tensor of the same type, where each element contains the absolute value of the
    corresponding element in the input.

    Given a tensor `x` of complex numbers, this operation returns a tensor of type
    `float32` or `float64` that is the absolute value of each element in `x`. For
    a complex number \\(a + bj\\), its absolute value is computed as \\(\sqrt{a^2
    + b^2}\\).  For example:

    >>> x = np.array([[-2.25 + 4.75j], [-3.25 + 5.75j]])
    >>> tf.abs(x)
    <tf.Tensor: shape=(2, 1), dtype=float64, numpy=
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

    return np.abs(x)


def pow(x:np.ndarray, y):
    r"""Computes the power of one value to another.

    Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
    corresponding elements in `x` and `y`. For example:

    ```python
    x = np.array([[2, 2], [3, 3]])
    y = np.array([[8, 16], [2, 3]])
    tf.pow(x, y)  # [[256, 65536], [9, 27]]
    ```

    Args:
        x (np.ndarray): input tensor.
        y (np.ndarray): another tensor.

    Returns:
      A `Tensor`.

    """

    return np.power(x,y)


def log(x:np.ndarray):
    r"""Computes natural logarithm of x element-wise.

    I.e., \\(y = \log_e x\\).

    See: https://en.wikipedia.org/wiki/Logarithm

    Args:
        x (np.ndarray): input tensor.


    Returns:
        A `Tensor`. Has the same type as `x`.

    Examples:
        >>> x = np.array([0, 0.5, 1, 5])
        >>> log(x)
        array([      -inf, -0.6931472,  0.       ,  1.609438 ])




    """

    return np.log(x)


def exp(x:np.ndarray):
    r"""Computes exponential of x element-wise.  \\(y = e^x\\).

    This function computes the exponential of the input tensor element-wise.
    i.e. `math.exp(x)` or \\(e^x\\), where `x` is the input tensor.
    \\(e\\) denotes Euler's number and is approximately equal to 2.718281.
    Output is positive for any real input.

    >>> x = np.array(2.0)
    >>> exp(x)
    <tf.Tensor: shape=(), dtype=float32, numpy=7.389056>

    >>> x = np.array([2.0, 8.0])
    >>> exp(x)
    tensor([   7.389056, 2980.958   ])

    For complex numbers, the exponential value is calculated as
    \\(e^{x+iy}={e^x}{e^{iy}}={e^x}{\\cos(y)+i\\sin(y)}\\)

    For `1+1j` the value would be computed as:
    \\(e^1{\\cos(1)+i\\sin(1)} = 2.7182817 \\times (0.5403023+0.84147096j)\\)

    >>> x =np.array(1 + 1j)
    >>> exp(x)
    tensor(1.4686939399158851+2.2873552871788423j)>

    Args:
      x: A `tf.Tensor`. Must be one of the following types: `bfloat16`, `half`,
        `float32`, `float64`, `complex64`, `complex128`.


    Returns:
      A `tf.Tensor`. Has the same type as `x`.

    @compatibility(numpy)
    Equivalent to np.exp
    @end_compatibility

    """

    return np.exp(x)


def clip(x:np.ndarray, min=-np.inf, max=np.inf):
    """

    Args:
        x (np.ndarray): input tensor.
        min ():
        max ():

    Returns:

    """
    return np.clip(x,min, max)


def sin(x:np.ndarray):
    """Computes the element-wise sine

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise sine

    Examples:
        >>> sin(np.array([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 0.8415,  0.4794],
                [-0.2474, -0.6816]])

    """
    return np.sin(x.astype(np.float32))


def cos(x:np.ndarray):
    """Computes the element-wise cosine

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise cosine

    Examples:
        >>> cos(np.array([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[0.5403, 0.8776],
                [0.9689, 0.7317]])

    """
    return np.cos(x.astype(np.float32))


def tan(x:np.ndarray):
    """Computes the element-wise tan

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise tan

    Examples:
        >>> tan(np.array([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 1.5574,  0.5463],
                [-0.2553, -0.9316]])

    """
    return np.tan(x.astype(np.float32))


def asin(x:np.ndarray):
    """Computes the element-wise arcsin (inverse sine)

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise arcsin

    Examples:
        >>> asin(np.array([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 1.5708,  0.5236],
                [-0.2527, -0.8481]])

    """
    return np.arcsin(x.astype(np.float32))


def acos(x:np.ndarray):
    """Computes the element-wise arccos (inverse cosine)

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> acos(np.array([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[0.0000, 1.0472],
                [1.8235, 2.4189]])

    """
    return np.arccos(x.astype(np.float32))


def atan(x:np.ndarray):
    """Computes the element-wise arctan (inverse tan)

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> atan(np.array([-1, 0, 1])).cpu()
        tensor([-0.7854,  0.0000,  0.7854])

    """
    return np.arctan(x.astype(np.float32))


def sinh(x:np.ndarray):
    """Computes the element-wise sinh

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise sinh

    Examples:
        >>> sinh(np.array([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 1.1752,  0.5211],
                [-0.2526, -0.8223]])

    """
    return np.sinh(x.astype(np.float32))


def cosh(x:np.ndarray):
    """Computes the element-wise cosh

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise cosh

    Examples:
        >>> cosh(np.array([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[1.5431, 1.1276],
                [1.0314, 1.2947]])

    """
    return np.cosh(x.astype(np.float32))


def tanh(x:np.ndarray):
    """Computes the element-wise tanh

    Args:
        x (np.ndarray): input tensor.

    Returns: element-wise tanh

    Examples:
        >>> tanh(np.array([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 0.7616,  0.4621],
                [-0.2449, -0.6351]])

    """
    return np.tanh(x.astype(np.float32))


############################
## elementwise operation
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

    Examples:
    >>> element_times(np.array([1., 1., 1., 1.]), np.array([0.5, 0.25, 0.125, 0.]))
    tensor([0.5000, 0.2500, 0.1250, 0.0000])
    >>> element_times(np.array([5., 10., 15., 30.]),np.array([2.]))
    tensor([10., 20., 30., 60.])
    >>> element_times(np.array([[5., 10.], [15., 30.]]), np.array([[1., 2.], [3.,1.]]))
    tensor([[ 5., 20.],
            [45., 30.]])
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

    Examples:
    >>> element_max(np.array([1., 1., 0., -1.]), np.array([0.5, 0.25, 0.125, 0.]))
    tensor([1.0000, 1.0000, 0.1250, 0.0000])
    >>> element_max(np.array([5., 10., 15., 30.]),np.array([20.]))
    tensor([20., 20., 20., 30.])
    >>> element_max(np.array([5., 10., 15., 30.]), np.array([10., 2., 8., 2.]))
    tensor([10., 10., 15., 30.])
    """
    return np.max(left, right)


def element_min(left, right):
    """
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Examples:
    >>> element_min(np.array([1., 1., 1., 1.]), np.array([0.5, 0.25, 0.125, 0.]))
    tensor([0.5000, 0.2500, 0.1250, 0.0000])
    >>> element_min(np.array([5., 10., 15., 30.]),np.array([2.]))
    tensor([2., 2., 2., 2.])
    >>> element_min(np.array([5., 10., 15., 30.]), np.array([1., 2., 1., 2.]))
    tensor([1., 2., 1., 2.])
    """
    return np.min(left, right)


def element_divide(left, right):
    """
    The output of this operation is the element-wise divide of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise divide of the two  input

    Examples:
    >>> element_divide(np.array([1., 1., 1., 1.]), np.array([0.5, 0.25, 0.125, 0.]))
    tensor([2., 4., 8., inf])
    >>> element_divide(np.array([5., 10., 15., 30.]),np.array([2.]))
    tensor([ 2.5000,  5.0000,  7.5000, 15.0000])
    >>> element_divide(np.array([5., 10., 15., 30.]), np.array([1., 2., 1., 2.]))
    tensor([ 5.,  5., 15., 15.])
    """
    return np.true_divide(left, right)


def element_cosine_distance(v1, v2, axis=-1):
    """

    Args:
        v1 ():
        v2 ():
        axis ():

    Returns:

    """
    reduce_dim = -1
    cos = np.sum((v1 * v2),axis=reduce_dim,keepdims=False) / (
            np.sqrt(np.sum((v1 * v1),axis=reduce_dim, keepdims=False))* np.sqrt(np.sum((v2 * v2),axis=reduce_dim,
                                                                                 keepdims=False)))
    return cos


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

    Examples:
    >>> x=np.array([0.1, 0.9, 0.8, 0.4, 0.5])
    >>> where(x>0.5, x, zeros_like(x))
    tensor([0.0000, 0.9000, 0.8000, 0.0000, 0.0000])
    """
    return np.where(flag, value_if_true, value_if_false)


############################
## reduce operation
###########################

def reduce_mean(x:np.ndarray, axis=None, keepdims=False, **kwargs):
    """Computes the mean of the input tensor's elements across a specified axis or a list of specified axes.

    Args:
        x (np.ndarray): input tensor.
        axis (int,list):  axis along which the reduction will be performed
        keepdims (bool): Keep the reduced dimension or not, default True mean keep reduced dimension
        **kwargs ():

    Returns:


    Examples:
        >>> data = np.array(np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32))
        >>> print(reduce_mean(data, 0).cpu())
        tensor([[30.,  1.],
                [40.,  2.]])
        >>> print(reduce_mean(data, axis=0).cpu())
        tensor([[30.,  1.],
                [40.,  2.]])
        >>> print(reduce_mean(data, axis=[0,2]).cpu())
        tensor([15.5000, 21.0000])

    """

    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if axis is None:
        return np.mean(x)
    elif isinstance(axis, int):
        return np.mean(x,axis=axis,keepdims=keepdims)
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            x =np.mean(x,axis=a, keepdims=keepdims)
        return x


def reduce_sum(x:np.ndarray, axis=None, keepdims=False, **kwargs):
    """Computes the sum of the input tensor's elements across a specified axis or a list of specified axes.

    Args:
        x (np.ndarray): input tensor.
        axis (int,list):  axis along which the reduction will be performed
        keepdims (bool): Keep the reduced dimension or not, default True mean keep reduced dimension
        **kwargs ():

    Returns:
        The sum of the input tensor's elements across a specified axis or a list of specified axes.

    Examples:
        >>> data = np.array(np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32))
        >>> print(reduce_sum(data).cpu())
        tensor(219.)
        >>> print(reduce_sum(data, 0).cpu())
        tensor([[ 90.,   3.],
                [120.,   6.]])
        >>> print(reduce_sum(data, axis=0).cpu())
        tensor([[ 90.,   3.],
                [120.,   6.]])
        >>> print(reduce_sum(data, axis=[0,2]).cpu())
        tensor([ 93., 126.])

    """
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if  x.size== 0:
        return zeros(1).astype(np.float32)
    if axis is None:
        return None
    elif isinstance(axis, int):
        return np.sum(x,axis=axis, keepdims=keepdims)
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            x = np.sum(x,axis=a, keepdims=keepdims)
        return x


def reduce_max(x:np.ndarray, axis=None, keepdims=False, **kwargs):
    """Computes the maximum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.


    See the numpy docs for `np.amax` and `np.nanmax` behavior.

    Args:
        x (np.ndarray): input tensor.
        axis: The dimensions to reduce. If `None` (the default), reduces all dimensions. Must be in the range `[-rank(input_tensor),
        rank(input_tensor)`.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
      The reduced tensor.

    Examples:
        >>> x = np.array([5, 1, 2, 4])
        >>> print(reduce_max(x))
        tensor(5, shape=(), dtype=int32)
        >>> x = np.array([-5, -1, -2, -4])
        >>> print(reduce_max(x))
        tensor(-1, shape=(), dtype=int32)
        >>> x = np.array([4, float('nan')])
        >>> print(reduce_max(x))
        tensor(4.0, shape=(), dtype=float32)
        >>> x = np.array([float('nan'), float('nan')])
        >>> print(reduce_max(x))
        tensor(-inf, shape=(), dtype=float32)
        >>> x =np.array([float('-inf'), float('inf')])
        >>> print(reduce_max(x))
        tensor(inf, shape=(), dtype=float32)

    """
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if  x.size== 0:
        return zeros(1).astype(np.float32)
    if axis is None:
        return None
    elif isinstance(axis, int):
        arr, idx = np.max(x,axis=axis, keepdims=keepdims)
        return arr
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = np.max(x,axis=a, keepdims=keepdims)
            x = arr
        return x


def reduce_min(x:np.ndarray, axis=None, keepdims=False, **kwargs):
    """Computes the minimum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.


    See the numpy docs for `np.amin` and `np.nanmin` behavior.

    Args:
        x (np.ndarray): input tensor.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `[-rank(input_tensor),
        rank(input_tensor))`.
      keepdims: If true, retains reduced dimensions with length 1.

    Returns:
      The reduced tensor.

    Examples:
        >>> x = np.array([5, 1, 2, 4])
        >>> print(reduce_min(x))
        tensor(5, shape=(), dtype=int32)
        >>> x = np.array([-5, -1, -2, -4])
        >>> print(reduce_min(x))
        tensor(-1, shape=(), dtype=int32)
        >>> x = np.array([4, float('nan')])
        >>> print(reduce_min(x))
        tensor(4.0, shape=(), dtype=float32)
        >>> x = np.array([float('nan'), float('nan')])
        >>> print(reduce_min(x))
        tensor(-inf, shape=(), dtype=float32)
        >>> x =np.array([float('-inf'), float('inf')])
        >>> print(reduce_min(x))
        tensor(inf, shape=(), dtype=float32)

    """
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if  x.size== 0:
        return zeros(1).astype(np.float32)
    if axis is None:
        return None
    elif isinstance(axis, int):
        arr, idx = np.min(x,axis=axis, keepdims=keepdims)
        return arr
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = np.min(x,axis=a, keepdims=keepdims)
            x = arr
        return x


def reduce_logsumexp(x:np.ndarray, axis=None, keepdims=False, **kwargs):
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
        >>> x =np.array([[0., 0., 0.], [0., 0., 0.]])
        >>> reduce_logsumexp(x)  # log(6)
        >>> reduce_logsumexp(x, 0)  # [log(2), log(2), log(2)]
        >>> reduce_logsumexp(x, 1)  # [log(3), log(3)]
        >>> reduce_logsumexp(x, 1, keepdims=True)  # [[log(3)], [log(3)]]
        >>> reduce_logsumexp(x, [0, 1])  # log(6)


    Args:
        x (np.ndarray): input tensor.
        axis (int, list, tuple): The dimensions to reduce. If `None` (the default), reduces all dimensions. Must be
        in the range `[-rank(input_tensor), rank(input_tensor))`.
        keepdims (bool): If true, retains reduced dimensions with length 1.

    Returns:
      The reduced tensor.

    """
    if  x.size== 0:
        return zeros(1).astype(np.float32)
    if axis is None:
        return None
    else:
        return log(reduce_sum(exp(x), axis=axis, keepdims=keepdims))


def reduce_prod(x:np.ndarray, axis=None, keepdims=False, **kwargs):
    """Computes the product of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
        x (np.ndarray): input tensor.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `[-rank(input_tensor),
        rank(input_tensor))`.
      keepdims: If true, retains reduced dimensions with length 1.

    Returns:
      The reduced tensor.

    @compatibility(numpy)
    Equivalent to np.prod
    @end_compatibility
    """
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if  x.size== 0:
        return zeros(1).astype(np.float32)
    if axis is None:
        return None
    if isinstance(axis, int):
        arr, idx = np.prod(x,axis=axis, keepdims=keepdims)
        return arr
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = np.prod(x,axis=a, keepdims=keepdims)
            x = arr
        return x


    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if  x.size== 0:
        return zeros(1).astype(np.float32)
    x=greater(x,0)
    if axis is None:
        return np.any(x,keepdims=keepdims)
    if isinstance(axis, int):
        arr, idx = np.any(x,axis=axis, keepdims=keepdims)
        return arr
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = np.any(x,axis=a, keepdims=keepdims)
            x = arr
        return x

# reduce_log_sum_exp
# reduce_prod
# reduce_l1
# reduce_l2
# reduce_sum_square

mean = reduce_mean
sum = reduce_sum


def max(*args, **kwargs):
    """

    Args:
        *args ():

    Returns:

    """
    allargs = args + tuple(list(kwargs.values()))
    if len(allargs) == 1 and is_numpy(allargs[0]) and numel(allargs[0])== 0:
        return allargs[0]
    elif len(allargs) == 1 and is_numpy(allargs[0]) and numel(allargs[0]) > 0:
        return np.max(allargs[0])
    elif len(allargs) > 1 and is_numpy(allargs[0]) and not is_numpy(allargs[1]) and ('axis' in kwargs or 'dim' in kwargs or 'keepdims' in kwargs or 'keepdim' in kwargs):
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return reduce_max(allargs[0], axis=axis, keepdims=keepdims)
    elif len(args) > 1 and is_numpy(args[0]) and isinstance(args[1],np.ndarray) :
        return np.maximum(x1=args[0],x2=args[1])
    elif len(args) > 1 and is_numpy(args[0]) and isinstance(args[1],(int,float)) :
        return np.clip(args[0],a_min=args[1],a_max=np.inf)
    else:
        raise NotImplementedError('Max({0},{1}) is not implemented yet '.format(*args, **kwargs))



def min(*args, **kwargs):
    """

    Args:
        *args ():

    Returns:

    """
    allargs = args +  tuple(list(kwargs.values()))
    if len(allargs) == 1 and is_numpy(allargs[0]) and numel(allargs[0]) == 0:
        return allargs[0]
    elif len(allargs) == 1 and is_numpy(allargs[0]) and numel(allargs[0]) > 0:
        return np.min(allargs[0])
    elif len(allargs) > 1 and is_numpy(allargs[0]) and not is_numpy(allargs[1]) and ('axis' in kwargs or 'dim' in kwargs or 'keepdims' in kwargs or 'keepdim' in kwargs):
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return reduce_min(allargs[0], axis=axis, keepdims=keepdims)
    elif len(args) > 1 and is_numpy(args[0]) and isinstance(args[1], np.ndarray):
        return np.minimum(x1=args[0], x2=args[1])
    elif len(args) > 1 and is_numpy(args[0]) and isinstance(args[1], (int, float)):
        return np.clip(args[0],a_min=-np.inf, a_max=args[1])
    else:
        raise NotImplementedError('Min({0},{1}) is not implemented yet '.format(*args, **kwargs))


############################
## activationoperation
###########################


def identity(x):
    """Identity activation Layer
    A placeholder identity operator that is argument-insensitive.

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.

    Examples:
        >>> identity(np.array([-3.0, -1.0, 0.0, 2.0]))
        tensor([-3.0, -1.0, 0.0, 2.0])

    """
    return x


def relu(x):
    """Rectified Linear Unit activation function.
      With default values, it returns element-wise `max(x, 0)`.
      Otherwise, it follows:
      ```
        f(x) = max_value if x >= max_value
        f(x) = x if threshold <= x < max_value
        f(x) = negative_slope * (x - threshold) otherwise
      ```

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return np.clip(x,a_min=0)


def relu6(x):
    """Rectified Linear Unit  6 activation function.
      With default values, it returns element-wise `min(max(x, 0)`,6).
      Otherwise, it follows:
      ```
        f(x) = 6 if x >= 6
        f(x) = x if threshold <= x < 6
        f(x) = negative_slope * (x - threshold) otherwise
      ```
    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return np.clip(x,a_min=0,a_max=6)


def leaky_relu(x, slope=0.2,**kwargs):
    """Leaky version of a Rectified Linear Unit.
        It allows a small gradient when the unit is not active:
        ```
        f(x) = alpha * x if x < 0
        f(x) = x if x >= 0
        ```
    Args:
        x (np.ndarray): input tensor.
        slope (float): slope

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    slope=kwargs.get('alpha',slope)
    return np.clip(x,a_min=0)+np.clip(-x*slope,a_max=0)


def leaky_relu6(x, slope=0.2,**kwargs):
    """Leaky version of a Rectified Linear Unit.6
          It allows a small gradient when the unit is not active:
          ```
            f(x) = alpha * x if x < 0
            f(x) = x if  6>=x >= 0
            f(x) = 6 if  x > 6
          ```

    Args:
        x (np.ndarray): input tensor.
        slope (float): slope

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    slope = kwargs.get('alpha', slope)
    return np.clip(x,a_min=0,a_max=6)+np.clip(-x*slope,a_min=-6,a_max=0)


def smooth_relu(x):
    """smooth_relu activation function


    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return np.log(1 + np.exp(x))


def crelu(x,axis=1):
    """Computes Concatenated ReLU.

    Concatenates a ReLU which selects only the positive part of the activation
    with a ReLU which selects only the *negative* part of the activation.
    Note that as a result this non-linearity doubles the depth of the activations.
    Source: [Understanding and Improving Convolutional Neural Networks via
    Concatenated Rectified Linear Units. W. Shang, et
    al.](https://arxiv.org/abs/1603.05201)

    Args:
        x (Tensor): input tensor.
        axis: The axis that the output values are concatenated along. Default is 1.

    Returns:
      A `Tensor` with the same type as `x`.

    References:
      Understanding and Improving Convolutional Neural Networks via Concatenated
      Rectified Linear Units:
        [Shang et al., 2016](http://proceedings.mlr.press/v48/shang16)
        ([pdf](http://proceedings.mlr.press/v48/shang16.pdf))
    """
    x=np.concatenate([x,-x],axis=axis)
    return relu(x)



def sigmoid(x):
    """softmax activation function

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return 1/(1+np.exp(-x))


def swish(x):
    """Self-Gated Activation Function.
    it follows:
      ```
        f(x) =  x * sigmoid(x)

      ```
    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    References:
        Swish: a Self-Gated Activation Function
        https://arxiv.org/abs/1710.05941v1

    Examples:
        >>> swish(np.array([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00])

    """
    return x * sigmoid(x)


def hard_sigmoid(x):
    """Hard sigmoid Activation Function.

    Memory saving version of sigmoid
    it follows:

        f(x) =  relu6(x+3)/6

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.



    Examples:
        >>> hard_sigmoid(np.array([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([-0.0000, -0.3333,  0.0000,  1.6667])


    """
    return relu6(x + 3) / 6


def hard_swish(x):
    """Hard swish Activation Function.

    Memory saving version of swish
    it follows:

        f(x) =  x * hard_sigmoid(x)


    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.

    Examples:
        >>> hard_swish(np.array([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([-0.0000, -0.3333,  0.0000,  1.6667])

    References:
        Searching for MobileNetV3
        https://arxiv.org/abs/1905.02244

    """
    return x * hard_sigmoid(x)


def hard_tanh(x):
    """Hard Tanh Activation Function.

    Memory saving version of sigmoid
    it follows:

        f(x) =  clip(x, -1, 1)

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    Examples:
        >>> hard_tanh(np.array([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([-0.0000, -0.3333,  0.0000,  1.6667])


    """
    return np.clip(x, -1, 1)

def elu(x,alpha):
    """ Exponential Linear Unit.
    It follows:

        f(x) =  alpha * (exp(x) - 1.) for x < 0
        f(x) = x for x >= 0

    Args:
        x (np.ndarray): input tensor.
        alpha (float): multiplier

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    Examples:
        >>> elu(np.array([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    """
    return relu(x)+alpha*(np.exp(np.clip(x,a_max=0))-1)


def selu(x):
    """
    selu activation function


    .. math::
            \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.


    Scaled exponential linear unit operation. Computes the element-wise exponential linear
    of ``x``: ``scale * x`` for ``x >= 0`` and ``x``: ``scale * alpha * (exp(x)-1)`` otherwise.
    scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717

    References:
        paper: https://arxiv.org/abs/1706.02515
        Self-Normalizing Neural Networks
        Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    Examples:
        >>> selu(np.array([[-1, -0.5, 0, 1, 2]]))
        tensor([[-1.1113, -0.6918,  0.0000,  1.0507,  2.1014]])

    """
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    temp1 =scale *relu(x)
    temp2 = scale * elu(-1 * relu(-1 * x),alpha)
    return temp1 + temp2




def lecun_tanh(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return 1.7159 * np.tanh(2 / 3 * x)


def soft_sign(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return np.log(np.exp(x)+1)


def soft_plus(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return np.log(1+np.exp(x))


def logit(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return np.log(x / (1 - x))


def log_log(x):
    """LogLog Activation Function

    it follows:

        f(x) =  1 - exp(-exp(x))

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    Examples:
        >>> loglog(np.array([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    References:
        "Complementary Log-Log and Probit: Activation Functions Implemented in Artificial Neural Networks"
        https://ieeexplore.ieee.org/document/4626755/


    """
    return 1 - np.exp(-np.exp(x))


def mish(x):
    """mish activation function

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.



    Examples:
        >>> mish(np.array([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    References:
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1

    """
    return x * (np.tanh(soft_plus(x)))


def hard_mish(x):
    """hard mish activation function

    it follows:

        f(x) =  x * hard_tanh(softplus(x))

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.



    Examples:
        >>> hard_mish(np.array([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    References:
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1

    """
    return x * hard_tanh(soft_plus(x))


def softmax(x, axis=1):
    """
    Computes the gradient of :math:`f(z)=\log\sum_i\exp(z_i)` at ``z = x``. Concretely,
    :math:`\mathrm{softmax}(x)=\left[\frac{\exp(x_1)}{\sum_i\exp(x_i)}\quad\frac{\exp(x_1)}{\sum_i\exp(
    x_i)}\quad\ldots\quad\frac{\exp(x_1)}{\sum_i\exp(x_i)}\right]`
    with the understanding that the implementation can use equivalent formulas
    for efficiency and numerical stability.
    The output is a vector of non-negative numbers that sum to 1 and can
    therefore be interpreted as probabilities for mutually exclusive outcomes
    as in the case of multiclass classification.
    If ``axis`` is given as integer, then the softmax will be computed along that axis.
    If the provided ``axis`` is -1, it will be computed along the last axis. Otherwise,
    softmax will be applied to all axes.

    Args:
        x (np.ndarray): input tensor.
        axis (int,list):  axis along which the reduction will be performed

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    Examples:
    >>> softmax(np.array([[1, 1, 2, 3]]))
    tensor([[0.0826, 0.0826, 0.2245, 0.6103]])
    >>> softmax(np.array([1., 1.]))
    tensor([0.5000, 0.5000])
    >>> softmax(np.array([[[1, 1], [3, 5]]]), axis=-1)
    tensor([[[0.5000, 0.5000],
             [0.1192, 0.8808]]])
    >>> softmax(np.array([[[1, 1], [3, 5]]]), axis=1)
    tensor([[[0.1192, 0.0180],
             [0.8808, 0.9820]]])

    """

    return np.true_divide(np.exp(x) ,np.clip(np.sum(np.exp(x),axis=axis),a_min=epsilon()))



def log_softmax(x, axis=1):
    """
    Computes the logsoftmax normalized values of x. That is, y = x - log(reduce_sum(exp(x), axis))
    (the implementation uses an equivalent formula for numerical stability).
    It is also possible to use `x - reduce_log_sum_exp(x, axis)` instead of log_softmax:
    this can be faster (one reduce pass instead of two), but can behave slightly differently numerically.

    Args:
        x (np.ndarray): input tensor.
        axis (int,list):  axis along which the reduction will be performed

    Returns:
        (np.ndarray): output tensor and have same shape with x.

    """
    return x - reduce_logsumexp(x, axis=1)


def gelu(x):
    """
    Gaussian Error Linear Unit.
    it follows:
        ```
        f(x) =x∗Φ(x)
        where \Phi(x)Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

        ```
    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.

    References:
        Gaussian Error Linear Units (GELUs)
        https://arxiv.org/abs/1606.08415

    Examples:
        >>> gelu(np.array([-3.0, -1.0, 0.0, 2.0]))
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00], dtype=float32)>

    """
    return x * 0.5 * (1.0 + special.erf(x / math.sqrt(2.0)))


def gpt_gelu(x):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.pow(x, 3))))


############################
## normalization operation
###########################

def moments(x:np.ndarray, axis, keepdims=True):
    """

    Args:
        keepdims ():
        x (np.ndarray): input tensor.
        axis (int) :

    Returns:
        (np.ndarray): output tensor and have same shape with x.


    """
    _axes = list(axis)
    norm_mean = reduce_mean(x, axis=_axes, keepdims=keepdims)
    norm_variance = reduce_mean(square(x - norm_mean), axis=_axes, keepdims=keepdims)
    return norm_mean, norm_variance


def l2_normalize(x:np.ndarray, eps=epsilon()):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:
        (np.ndarray): output tensor and have same shape with x.



    """
    return x / np.sqrt(np.square(x).sum() + eps)


############################
## tensor shape operation
###########################
def reshape(x, shape=None) ->np.ndarray:
    """

    Args:
        x (np.ndarray): input tensor.
        shape ():

    Returns:

    """
    if shape is None:
        return x
    elif isinstance(shape, list):
        return np.reshape(x, to_list(shape))
    elif isinstance(shape, tuple):
        shape = to_list(shape)
        return np.reshape(x, shape)
    else:
        return x


def transpose(x, pattern=None) ->np.ndarray:
    """

    Args:
        x (np.ndarray): input tensor.
        pattern ():

    Returns:

    """
    return np.transpose(x,*pattern)


def permute(x, pattern=None) ->np.ndarray:
    """

    Args:
        x (np.ndarray): input tensor.
        pattern ():

    Returns:

    """
    return np.transpose(x,*pattern)


def squeeze(x:np.ndarray, axis=0):
    """

    Args:
        x (np.ndarray): input tensor.
        axis ():

    Returns:

    """
    return np.squeeze(x,axis)


def expand_dims(x:np.ndarray, axis=0):
    """

    Args:
        x (np.ndarray): input tensor.
        axis ():

    Returns:

    """
    return np.expand_dims(x,axis)


# def depth_to_space(x:np.ndarray, block_size=2):
#     """
#     Rearranges elements in the input tensor from the depth dimension into spatial blocks.
#     The equivalent to Pixel-Shuffle
#
#     Args:
#         x (np.ndarray): input tensor.
#         block_size (int):
#
#     Returns: resized tensor
#
#     Examples:
#     >>> x = np.array(np.tile(np.array(np.reshape(range(8), (8, 1, 1)), dtype=np.float32), (1, 2, 3)))
#     >>> x
#     tensor([[[0., 0., 0.],
#              [0., 0., 0.]],
#     <BLANKLINE>
#             [[1., 1., 1.],
#              [1., 1., 1.]],
#     <BLANKLINE>
#             [[2., 2., 2.],
#              [2., 2., 2.]],
#     <BLANKLINE>
#             [[3., 3., 3.],
#              [3., 3., 3.]],
#     <BLANKLINE>
#             [[4., 4., 4.],
#              [4., 4., 4.]],
#     <BLANKLINE>
#             [[5., 5., 5.],
#              [5., 5., 5.]],
#     <BLANKLINE>
#             [[6., 6., 6.],
#              [6., 6., 6.]],
#     <BLANKLINE>
#             [[7., 7., 7.],
#              [7., 7., 7.]]])
#     >>> arr=depth_to_space(x,block_size=2)
#     >>> print(arr.shape)
#     torch.Size([2, 4, 6])
#     >>> arr
#     tensor([[[0., 1., 0., 1., 0., 1.],
#              [2., 3., 2., 3., 2., 3.],
#              [0., 1., 0., 1., 0., 1.],
#              [2., 3., 2., 3., 2., 3.]],
#     <BLANKLINE>
#             [[4., 5., 4., 5., 4., 5.],
#              [6., 7., 6., 7., 6., 7.],
#              [4., 5., 4., 5., 4., 5.],
#              [6., 7., 6., 7., 6., 7.]]])
#
#     """
#     if ndim(x) not in (3, 4):
#         raise ValueError('Input tensort length of shape should be 3 or 4 ')
#     elif x.shape[-3] % (block_size * block_size) != 0:
#         raise ValueError('Input tensort channel must be divisible by square of block_size')
#     else:
#         orig_ndim = ndim(x)
#         if orig_ndim == 3:
#             x = expand_dims(x, 0)
#         x = F.pixel_shuffle(x, block_size)
#         if orig_ndim == 3:
#             return x[0]
#         return x
#
#
# def space_to_depth(x:np.ndarray, block_size=2):
#     """
#     Rearranges elements in the input tensor from the spatial dimensions to the depth dimension.
#
#     This is the reverse transformation of depth_to_space. This operation is useful for implementing and testing
#     sub-pixel convolution that is part of models for image super-resolution .
#     It rearranges elements of an input tensor of shape (N, C, H, W) to a tensor of shape (N, C*b*b, H/b, W/b),
#     where b is the block_size,
#     by rearranging non-overlapping spatial blocks of size block_size x block_size into the depth/channel dimension at
#     each location.
#
#     Args:
#         x (np.ndarray): input tensor.
#         block_size (int):
#
#     Returns: resized tensor
#     Examples:
#     >>> arr=space_to_depth(np.array([[[0., 1., 0., 1., 0., 1.],[2., 3., 2., 3., 2., 3.],[0., 1., 0., 1., 0., 1.],
#     [2., 3., 2., 3., 2., 3.]],[[4., 5., 4., 5., 4., 5.],[6., 7., 6., 7., 6., 7.], [4., 5., 4., 5., 4., 5.],[6., 7.,
#     6., 7., 6., 7.]]]),block_size=2)
#     >>> arr
#     tensor([[[0., 0., 0.],
#              [0., 0., 0.]],
#     <BLANKLINE>
#             [[1., 1., 1.],
#              [1., 1., 1.]],
#     <BLANKLINE>
#             [[2., 2., 2.],
#              [2., 2., 2.]],
#     <BLANKLINE>
#             [[3., 3., 3.],
#              [3., 3., 3.]],
#     <BLANKLINE>
#             [[4., 4., 4.],
#              [4., 4., 4.]],
#     <BLANKLINE>
#             [[5., 5., 5.],
#              [5., 5., 5.]],
#     <BLANKLINE>
#             [[6., 6., 6.],
#              [6., 6., 6.]],
#     <BLANKLINE>
#             [[7., 7., 7.],
#              [7., 7., 7.]]])
#     >>> print(arr.shape)
#     torch.Size([8, 2, 3])
#     """
#     if ndim(x) not in (3, 4):
#         raise ValueError('Input tensort length of shape should be 3 or 4 ')
#     elif x.shape[-2] % block_size != 0 or x.shape[-1] % block_size != 0:
#         raise ValueError('Input tensort channel must be divisible by square of block_size')
#     else:
#         orig_ndim = ndim(x)
#         if orig_ndim == 3:
#             x = expand_dims(x, 0)
#         orig_shape = list(int_shape(x))
#         x = reshape(x, (
#             orig_shape[0], orig_shape[1], orig_shape[2] // block_size, block_size, orig_shape[3] // block_size, block_size))
#         x = permute(x, [0, 1, 3, 5, 2, 4])
#         x = reshape(x, (orig_shape[0], orig_shape[1] * block_size * block_size, orig_shape[2] // block_size,
#                         orig_shape[3] // block_size))
#         if orig_ndim == 3:
#             return x[0]
#         return x


############################
## tensor generation
###########################

def pad(x:np.ndarray, paddings: Sequence[int], mode='constant', value=0):
    r"""Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\_left}, \text{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\_left}, \text{padding\_right},`
        :math:`\text{padding\_top}, \text{padding\_bottom}`
        :math:`\text{padding\_front}, \text{padding\_back})`.

    Padding mode:
        See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
        :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate padding is implemented for padding the last 3 dimensions of 5D input
        tensor, or the last 2 dimensions of 4D input tensor, or the last dimension of
        3D input tensor. Reflect padding is only implemented for padding the last 2
        dimensions of 4D input tensor, or the last dimension of 3D input tensor.

    Note:
        When using the CUDA backend, this operation may induce nondeterministic
        behaviour in its backward pass that is not easily switched off.
        Please see the notes on :doc:`/notes/randomness` for background.

    Args:

        x (np.ndarray): N-dimensional tensor
        paddings (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

     Examples:
        >>> t=np.array([[0., 1., 2.],[3., 4., 5.],[6., 7., 8.]])
        >>> pad(t, ((1, 1),(2, 0)), mode='replicate')
        array([[0., 0., 1., 2., 2.],
          [0., 0., 1., 2., 2.],
          [0., 0., 1., 2., 2.],
          [3., 3., 4., 5., 5.],
          [6., 6., 7., 8., 8.]])

    """
    valid_items=['constant', 'reflect', 'replicate' ,'circular','symmetric','zero']
    if mode not in valid_items:
        raise ValueError('{0} is not valid for mode.'.format(mode))
    if mode=='zero':
        mode='constant'
        value=0
    if mode == 'replicate':
        mode = 'edge'
    if mode == 'circular':
        mode = 'symmetric'
    if mode=='constant':
        return np.pad(x,pad_width=paddings,mode=mode,constant_values=value)
    else:
        return np.pad(x, pad_width=paddings, mode=mode)


############################
## tensor generation
###########################





def ones(shape, dtype=np.float32):
    """Instantiates an all-ones tensor and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        A tensor, filled with `1.0`.

    Example
        >>> ones((3,4))
        tensor([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]])

    {{np_implementation}}
    """
    return np.ones(shape, dtype=dtype)


def ones_like(a, dtype=np.float32):
    """Instantiates an all-ones variable of the same shape as another tensor.

    Args
        a (np.ndarray):  another tensor
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        A tensor, filled with `1.0` and shape is the same as another tensor.

    Example
        >>> ones_like(torch.randn((3,4)))
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]], dtype=float32)

    {{np_implementation}}
    """
    return np.ones(a.shape, dtype=dtype)


def zeros(shape, dtype=np.float32):
    """Instantiates an all-zeros tensor and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        A tensor, filled with `0.0`.

    Example
        >>> zeros((3,4))
        tensor([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)

    {{np_implementation}}
    """
    return np.zeros(shape, dtype=dtype)


def zeros_like(a, dtype=np.float32):
    """Instantiates an all-zeros variable of the same shape as another tensor.

    Args
        a (np.ndarray):  another tensor
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
    return np.zeros(a.shape, dtype=dtype)


def eye(shape, dtype=np.float32):
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
        return np.eye(shape[0], shape[1], dtype=dtype)
    else:
        raise ValueError('input tensor must have exactly two axe.')


def eye_like(a, dtype=np.float32):
    """
    Creates a matrix with diagonal set to 1s and of the same shape and the same dynamic axes as ``x``. To be a
    matrix, ``x`` must have exactly two axes (counting both dynamic and static axes).

    Args:
        a (np.ndarray):  another tensor of rank 2
        dtype (String):  data type
        requires_grad (bool):  whether need gradient

    Returns
        an identity matrix.

    Examples:
    >>> eye_like(np.ndarray(3,4))
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]])

    """
    if a.ndim == 2:
        return np.eye(a.shape[0], a.shape[1], dtype=dtype)
    else:
        raise ValueError('input tensor must have exactly two axe.')

#
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
    >>> make_onehot(np.array([[1, 2],[1, 3]]).long(), 4, axis=-1)
    tensor([[[0., 1., 1., 0.],
             [0., 1., 0., 1.]],
    <BLANKLINE>
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.]]])

    """


    onehot=np.put(label, num_classes, 1)
    return onehot


def arange(*args, dtype=np.int32):
    """

    Args:
        *args (int): the start, end, step
        dtype (dtype): dtype of the tensor

    Returns:

    """
    if len(args) == 1:
        return np.arange(end=args[0], dtype=dtype)
    elif len(args) == 2:
        return np.arange(start=args[0], end=args[1], dtype=dtype)
    elif len(args) == 3:
        return np.arange(start=args[0], end=args[1], step=args[2], dtype=dtype)
    else:
        raise ValueError('only maximum  3 args in arange function ')


def meshgrid(x, y, normalized_coordinates=False, requires_grad=False):
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
    >>> grid.cpu()
    tensor([[[0., 0.],
             [0., 1.]],
    <BLANKLINE>
            [[1., 0.],
             [1., 1.]],
    <BLANKLINE>
            [[2., 0.],
             [2., 1.]]])
    >>> print(grid[0,0,:].cpu())
     tensor([0., 0.])
    >>> print(grid[:,0,0].cpu())
    tensor([0., 1., 2.])
    >>> print(grid.shape)
    torch.Size([3, 2, 2])


    >>> grid1=meshgrid(3,2,normalized_coordinates=True)
    >>> grid1.cpu()
    tensor([[[0.0000, 0.0000],
             [0.0000, 1.0000]],
    <BLANKLINE>
            [[0.5000, 0.0000],
             [0.5000, 1.0000]],
    <BLANKLINE>
            [[1.0000, 0.0000],
             [1.0000, 1.0000]]])
    >>> grid1.shape
    torch.Size([3, 2, 2])
    """
    xs = np.linspace(0, int(x - 1), int(x), dtype=np.float32)
    ys = np.linspace(0, int(y - 1), int(y), dtype=np.float32)
    if normalized_coordinates:
        xs = np.linspace(0, 1, int(x), dtype=np.float32)
        ys =np.linspace(0, 1, int(y), dtype=np.float32)
    grid_x, grid_y = np.meshgrid([xs, ys])

    grid = np.stack([grid_y, grid_x], -1)
    return grid

def reverse(x, axis):
  """Reverse a tensor along the specified axes.

  Arguments:
      x: Tensor to reverse.
      axis: Integer or iterable of integers.
          Axes to reverse.

  Returns:
      A tensor.
  """
  if isinstance(axis, int):
    axis = [axis]
  return np.flip(x,axis=axis)

############################
## tensor manipulation
###########################

def concate(x: List[np.ndarray], axis=1):
    """

    Args:
        x ():
        axis ():

    Returns:

    """
    return np.concatenate(x, axis=axis)


def stack(x: List[np.ndarray], axis=1):
    """

    Args:
        x ():
        axis ():

    Returns:

    """
    return np.stack(x, axis=axis)


def gram_matrix(x:np.ndarray):
    """

    Args:
        x (np.ndarray): input tensor.

    Returns:

    """
    features = np.reshape(x, (-1, x.shape[-2] * x.shape[-1])) - np.mean(x)
    return features.dot(features.T)


############################
## random
###########################

def set_seed(seed: int) -> None:
    """Setup random state from a seed for `torch.random`, `random` and  `numpy` (if can be imported).

    Args:
        seed (int): Random state seed

    """
    random.seed(seed)

    np.random.seed(seed)


def shuffle(t:np.ndarray):
    """

    Args:
        t (np.ndarray): input tensor.

    Returns:

    """

    return  np.random.shuffle(t)


def random_choice(t:np.ndarray,n:int=1):
    """Generates a random sample from a given 1-D array

    Args:
        t (np.ndarray): input tensor (1-D  tensor).
        n (int): how many items

    Returns:
        (np.ndarray) : single item ,the generated random samples

    """

    idxes = np.arange(len(t))
    np.random.shuffle(idxes)
    idx =idxes[:n]

    if isinstance(t, (list, tuple)):
        return [t[idx] for idx in idxes[:n]]
    else:
        return t[idx]


def random_normal(shape, dtype='float32', mean=0.0, std=1.0, seed=None):
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
        <tf.Tensor: shape=(4,), dtype=float32, numpy=..., dtype=float32)>
        >>> #that outputs a reproducible result:
        >>> random_normal([2,2],dtype='float32' ,mean=0, stddev=1,seed=5)
        <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[-1.3768897 , -0.01258316],
              [-0.169515   ,  1.0824056 ]], dtype=float32)>


    """
    if seed is not None:
        set_seed(seed)
    if dtype is not None:
        if isinstance(dtype,str):
            dtype=str2dtype(dtype)
        return np.random.normal(loc=mean, scale=std, size=shape).astype(dtype)
    return np.random.normal(loc=mean, scale=std, size=shape).astype(np.float32)


def random_normal_like(a, dtype='float32', mean=0.0, std=1.0, seed=None):
    """Outputs random values from a normal distribution.

    In this case, we are setting both the global and operation-level seed to
    ensure this result is reproducible.  See `tf.random.set_seed` for more
    information.

    Args:
      a: A 1-D integer Tensor or Python array. The shape of the output tensor.
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
        <tf.Tensor: shape=(4,), dtype=float32, numpy=..., dtype=float32)>
        >>> #that outputs a reproducible result:
        >>> random_normal([2,2],dtype='float32' ,mean=0, stddev=1,seed=5)
        <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[-1.3768897 , -0.01258316],
              [-0.169515   ,  1.0824056 ]], dtype=float32)>


    """
    if seed is not None:
        set_seed(seed)
    if dtype is not None:
        if isinstance(dtype,str):
            dtype=str2dtype(dtype)
        return np.random.normal(loc=mean, scale=std, size=a.shape).astype(dtype)
    return np.random.normal(loc=mean, scale=std, size=a.shape).astype(np.float32)


############################
## loss
###########################

def binary_crossentropy(target, output, from_logits=False):
    if  not from_logits:
        output = sigmoid(output)
    output = np.clip(output,epsilon(), 1.0 - epsilon())
    output = -target * np.log(output) - (1.0 - target) * np.log(1.0 - output)
    return output



def conv2d(input, weight,bias: Optional[np.ndarray]=None, strides=1, padding=0,dilation=1,groups=1):
    strides=(strides,strides)
    padding=(padding,padding)
    ish = input.shape
    fsh = weight.shape
    output = np.zeros([ish[0],(ish[1]-fsh[0])//strides[1]+1,(ish[2]-fsh[1])//strides[2]+1,fsh[3]])
    osh = output.shape
    for m in range(osh[0]):
        for i in range(osh[1]):
            for j in range(osh[2]):
                for di in range(fsh[0]):
                    for dj in range(fsh[1]):
                        t = np.dot(
                                input[m,strides[1]*i+di,strides[2]*j+dj,:],
                            weight[di, dj, :, :]
                            )
                        output[m,i,j] = np.sum(
                                [
                                    t,
                                    output[m,i,j]
                                ],
                                axis=0
                            )
    return output


############################
## color space
###########################

def rgb2gray(rgb:np.ndarray):
    """Compute luminance of an RGB image.

    Args:
        rgb (tensor):  rgb image (shape:(H,W,C))
    Returns:
        gray(tensor):  gray-scale image(shape:(H,W))

    """
    rgb=rgb.copy().astype(np.float32)
    if ndim(rgb)!=3 :
        raise ValueError('input rgb image ndim should equal 3 but get {0}'.format(ndim(rgb)))
    r,g,b=np.split(rgb,3,axis=-1)
    gray = 0.2125*r + 0.7154*g + 0.0721*b
    return gray


def rgb2hsv(rgb:np.ndarray):
    """Compute luminance of an RGB image.

    Args:
        rgb (tensor):  rgb image (shape:(H,W,C))
    Returns:
        gray(tensor):  gray-scale image(shape:(H,W))

    Examples:
        >>> import cv2
        >>> from skimage import color
        >>> img=cv2.cvtColor(cv2.imread('../../images/cat.jpg'),cv2.COLOR_BGR2RGB)
        >>> hsv_Img=rgb2hsv(np.array(img.copy()).astype(np.float32))
        >>> ground_truth_hsv=np.array(color.rgb2hsv(img.copy()/255.)*255.0).astype(np.float32)
        >>> abs(hsv_Img-ground_truth_hsv).mean().item()<2
        True
        >>> print( ground_truth_hsv)
        >>> print( hsv_Img)
        >>> print( abs(hsv_Img-ground_truth_hsv).mean().item())

    """
    rgb=rgb.astype(np.float32)/255.0
    if ndim(rgb)!=3 :
        raise ValueError('input rgb image ndim should equal 3 but get {0}'.format(ndim(rgb)))
    #rgb=rgb[np.newaxis, ...]
    out = zeros_like(rgb.copy())

    # -- V channel
    out_v = reduce_max(rgb.copy(),-1)

    # -- S channel
    delta =reduce_max(rgb.copy(),-1)-reduce_min(rgb.copy(),-1)
    delta_zero=zeros_like(delta,dtype=delta.dtype)
    out_s = where(delta==0,delta_zero,delta / out_v)

    # -- H channel
    # red is max
    out0=zeros_like(delta,dtype=delta.dtype)
    idx = (rgb[..., 0] == out_v)

    out0[idx] = (rgb[..., 1][idx] - rgb[..., 2][idx]) / delta[idx]

    # green is max
    idx = (rgb[..., 1] == out_v)
    out0[idx] =  2. + (rgb[..., 2][idx] - rgb[..., 1][idx]) / delta[idx]

    # blue is max
    idx = (rgb[..., 2] == out_v)
    out0[idx] =  4. + (rgb[..., 0][idx] - rgb[..., 1][idx]) / delta[idx]


    out_h = (out0/ 6.) % 1.
    out_h[delta == 0.] = 0.

    # -- output
    out[..., 0] = out_h
    out[..., 1] = out_s
    out[..., 2] = out_v


    return out*255.0


def xyz2rgb(xyz:np.ndarray):
    """
    input xyz as pytorch tensor of size (batch_size,  h, w, 3) or (h, w,3)
    """
    if len(xyz.shape) == 4:
        if int_shape(xyz)[-1]==3:
            xyz =xyz.transpose([0,3,1,2])
        elif int_shape(xyz)[1]==3:
            pass
    elif len(xyz.shape) == 3:
        if int_shape(xyz)[-1]==3:
            xyz =xyz.transpose([2,0,1])
        elif  int_shape(xyz)[0]==3:
            pass
    xyz=xyz/255.0
    transform_tensor = np.array([[3.2404542, - 1.5371385, - 0.4985314],
                                     [-0.9692660, 1.8760108, 0.0415560],
                                     [0.0556434, - 0.2040259, 1.0572252]],dtype=xyz.dtype)

    transform_tensor=np.expand_dims(np.expand_dims(transform_tensor,2),3)
    convolved=None
    if len(xyz.shape) == 4:
        convolved = conv2d(xyz, transform_tensor)
    else:
        convolved = np.expand_dims(conv2d(np.expand_dims(xyz,0), transform_tensor),0)
    # return convolved
    rgb=convolved*255.0
    if len(rgb.shape) == 4:
        if int_shape(rgb)[-1] == 3:
            return rgb
        elif  int_shape(rgb)[1] == 3:
            rgb = rgb.transpose([0, 2, 3, 1])
            return rgb

    elif len(rgb.shape) == 3:
        if int_shape(rgb)[-1] == 3:
            return rgb
        elif int_shape(rgb)[0] == 3:
            rgb = rgb.transpose([1, 2,0])
            return rgb

    raise ValueError('image should channel-last')


def rgb2xyz(rgb:np.ndarray):
    """
    input rgb as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if len(rgb.shape) == 4:
        if int_shape(rgb)[-1]==3:
            rgb =rgb.transpose([0,3,1,2])
        elif int_shape(rgb)[1]==3:
            pass
    elif len(rgb.shape) == 3:
        if int_shape(rgb)[-1]==3:
            rgb =rgb.transpose([2,0,1])
        elif  int_shape(rgb)[0]==3:
            pass
    rgb=rgb/255.0
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055)**2.4, rgb / 12.92)

    transform_tensor = np.array([[0.4124564, 0.3575761, 0.1804375],
                                     [0.2126729, 0.7151522, 0.0721750],
                                     [0.0193339, 0.1191920, 0.9503041]],dtype=rgb.dtype)

    transform_tensor=np.expand_dims(np.expand_dims(transform_tensor,2),3)
    xyz=None
    if len(rgb.shape) == 4:
        xyz=conv2d(rgb, transform_tensor)
    else:
        xyz= np.expand_dims(conv2d(np.expand_dims(rgb,0), transform_tensor),0)
    xyz=xyz*255.0
    if len(xyz.shape) == 4:
        if int_shape(xyz)[-1] == 3:
            return xyz
        elif int_shape(xyz)[1] == 3:
            xyz = xyz.transpose([0, 2, 3, 1])
            return xyz

    elif len(xyz.shape) == 3:
        if int_shape(xyz)[-1] == 3:
            return xyz
        elif int_shape(xyz)[0] == 3:
            xyz = xyz.transpose([1, 2, 0])
            return xyz

    raise ValueError('image should channel-last')



# LAB
# CIE-L*a*b*: A perceptually uniform color space,
# i.e. distances are meaningful. L* in [0..1] and a*, b* almost in [-1..1].
D65 = [0.95047, 1.00000, 1.08883]


def lab_f(t:np.ndarray):
    return where(t > 0.008856451679035631, cast(t.pow(1.0 / 3.0),cast_dtype=t.dtype), cast(t * 7.787037037037035 + 0.13793103448275862,cast_dtype=t.dtype))


def lab_finv(t:np.ndarray):
    return where(t > 0.20689655172413793, cast(t.pow(3.0),cast_dtype=t.dtype), cast(0.12841854934601665 * (t - 0.13793103448275862),cast_dtype=t.dtype))


def lab2xyz(lab:np.ndarray, wref=None):
    """
    input lab as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    l
    """
    if len(lab.shape) == 4:
        if int_shape(lab)[-1]==3:
            lab =lab.transpose([0,3,1,2])
        elif int_shape(lab)[1]==3:
            pass
        lab[:,0, :, :] = lab[:,0, :, :] / 100.0
        lab[:,1:, :, :] = (lab[:,1:, :, :] - 127) / 128
    elif len(lab.shape) == 3:
        if int_shape(lab)[-1]==3:
            lab =lab.transpose([2,0,1])
        elif  int_shape(lab)[0]==3:
            pass
        lab[0,:,:]=lab[0,:,:]/100.0
        lab[1:, :, :]=(lab[1:, :, :]-127)/128
    if wref is None:
        wref = D65
    dim = 1 if len(lab.shape) == 4 else 0
    l, a, b = np.split(lab,3, axis=dim)

    l2 = (l + 0.16) / 1.16
    x = wref[0] * lab_finv(l2 + a / 5)
    y = wref[1] * lab_finv(l2)
    z = wref[2] * lab_finv(l2 - b / 2)
    xyz = np.concatenate([x, y, z], axis=dim)
    xyz=xyz*255.0
    if len(xyz.shape) == 4:
        if int_shape(xyz)[-1] == 3:
            return xyz
        elif int_shape(xyz)[1] == 3:
            xyz = xyz.transpose([0, 2, 3, 1])
            return xyz
    elif len(xyz.shape) == 3:
        if int_shape(xyz)[-1] == 3:
            return xyz
        elif int_shape(xyz)[0] == 3:
            xyz = xyz.transpose([1, 2, 0])
            return xyz

    raise ValueError('image should channel-last')

def xyz2lab(xyz:np.ndarray, wref=None):
    """
    input xyz as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if len(xyz.shape) == 4:
        if int_shape(xyz)[-1]==3:
            xyz =xyz.transpose([0,3,1,2])
        elif int_shape(xyz)[1]==3:
            pass
    elif len(xyz.shape) == 3:
        if int_shape(xyz)[-1]==3:
            xyz =xyz.transpose([2,0,1])
        elif  int_shape(xyz)[0]==3:
            pass
    xyz=xyz/255.0
    if wref is None:
        wref = D65
    dim = 1 if len(xyz.shape) == 4 else 0
    x, y, z = xyz.chunk(3, dim=dim)

    fy = lab_f(y / wref[1])
    l = 1.16 * fy - 0.16
    a = 5.0 * (lab_f(x / wref[0]) - fy)
    b = 2.0 * (fy - lab_f(z / wref[2]))
    lab =np.concatenate([clip(l,0,1)*100, clip(a,-1,1)*128+127, clip(b,-1,1)*128+127], dim=dim)

    if len(lab.shape) == 4:
        if int_shape(lab)[-1] == 3:
            return lab
        elif int_shape(lab)[1] == 3:
            lab = lab.transpose([0, 2, 3, 1])
            return lab

    elif len(lab.shape) == 3:
        if int_shape(lab)[-1] == 3:
            return lab
        elif int_shape(lab)[0] == 3:
            lab = lab.transpose([1, 2, 0])
            return lab

    raise ValueError('image should channel-last')


def lab2rgb(lab:np.ndarray):
    """
    input lab as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if ndim(lab)==4:
        channel_idx=1
        if int_shape(lab)[1]==3:
            channel_idx = 1
        elif  int_shape(lab)[-1]==3:
            channel_idx = -1
        rgb=xyz2rgb(lab2xyz(lab))
        if channel_idx==1:
            rgb=rgb.transpose([0,3,1,2])
        return rgb
    else:
        rgb=xyz2rgb(lab2xyz(lab))
        return rgb


def rgb2lab(rgb:np.ndarray):
    """
    input rgb as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if ndim(rgb)==4:
        channel_idx=1
        if int_shape(rgb)[1]==3:
            channel_idx = 1
        elif  int_shape(rgb)[-1]==3:
            channel_idx = -1
        xyz=xyz2lab(rgb2xyz(rgb))
        if channel_idx==1:
            xyz=xyz.transpose([0,3,1,2])
        return xyz
    else:
        xyz=xyz2lab(rgb2xyz(rgb))
        return xyz

def gray2rgb(gray:np.ndarray):
    """Compute luminance of an RGB image.

    Args:
        gray(tensor):  gray-scale image(shape:(H,W))
    Returns:
        rgb (tensor):  rgb image (shape:(H,W,C))

    """
    gray=gray.copy().astype(np.float32)
    if ndim(gray) == 3 and int_shape(gray)[-1]==1:
        gray=gray[:,:,0]
    if ndim(gray)!=2 :
        raise ValueError('input gray image ndim should equal 2 but get {0}'.format(ndim(gray)))
    rgb=stack([gray,gray,gray],axis=-1)
    return rgb






#
# def torch_rot90_(x:np.ndarray):
#     return x.transpose_(2, 3).flip(2)
#
#
# def torch_rot90(x:np.ndarray):
#     return x.transpose(2, 3).flip(2)
#
#
# def torch_rot180(x:np.ndarray):
#     return x.flip(2).flip(3)
#
#
# def torch_rot270(x:np.ndarray):
#     return x.transpose(2, 3).flip(3)
#
#
# def torch_flipud(x:np.ndarray):
#     """
#     Flip image tensor vertically
#     :param x:
#     :return:
#     """
#     return x.flip(2)
#
#
# def torch_fliplr(x:np.ndarray):
#     """
#     Flip image tensor horizontally
#     :param x:
#     :return:
#     """
#     return x.flip(3)
#
#
# def pad_image_tensor(image_tensor:np.ndarray, pad_size: int = 32):
#     """Pad input tensor to make it's height and width dividable by @pad_size
#
#     :param image_tensor: Input tensor of shape NCHW
#     :param pad_size: Pad size
#     :return: Tuple of output tensor and pad params. Second argument can be used to reverse pad operation of model output
#     """
#     rows, cols = image_tensor.size(2), image_tensor.size(3)
#     if (isinstance(pad_size, Sized) and isinstance(pad_size, Iterable) and len(pad_size) == 2):
#         pad_height, pad_width = [int(val) for val in pad_size]
#     elif isinstance(pad_size, int):
#         pad_height = pad_width = pad_size
#     else:
#         raise ValueError(
#             "Unsupported pad_size: {pad_size}, must be either tuple(pad_rows,pad_cols) or single int scalar.")
#
#     if rows > pad_height:
#         pad_rows = rows % pad_height
#         pad_rows = pad_height - pad_rows if pad_rows > 0 else 0
#     else:
#         pad_rows = pad_height - rows
#
#     if cols > pad_width:
#         pad_cols = cols % pad_width
#         pad_cols = pad_width - pad_cols if pad_cols > 0 else 0
#     else:
#         pad_cols = pad_width - cols
#
#     if pad_rows == 0 and pad_cols == 0:
#         return image_tensor, (0, 0, 0, 0)
#
#     pad_top = pad_rows // 2
#     pad_btm = pad_rows - pad_top
#
#     pad_left = pad_cols // 2
#     pad_right = pad_cols - pad_left
#
#     pad = [pad_left, pad_right, pad_top, pad_btm]
#     image_tensor = torch.nn.functional.pad(image_tensor, pad)
#     return image_tensor, pad
#
#
# def unpad_image_tensor(image_tensor, pad):
#     pad_left, pad_right, pad_top, pad_btm = pad
#     rows, cols = image_tensor.size(2), image_tensor.size(3)
#     return image_tensor[..., pad_top: rows - pad_btm, pad_left: cols - pad_right]
#
#
# def unpad_xyxy_bboxes(bboxes_tensor:np.ndarray, pad, dim=-1):
#     pad_left, pad_right, pad_top, pad_btm = pad
#     pad =np.ndarray([pad_left, pad_top, pad_left, pad_top], dtype=bboxes_tensor.dtype).to(bboxes_tensor.device)
#
#     if dim == -1:
#         dim = len(bboxes_tensor.size()) - 1
#
#     expand_dims = list(set(range(len(bboxes_tensor.size()))) - {dim})
#     for i, dim in enumerate(expand_dims):
#         pad = pad.unsqueeze(dim)
#
#     return bboxes_tensor - pad
#
#
# def angle_to_rotation_matrix(angle) ->np.ndarray:
#     """
#     Creates a rotation matrix out of angles in degrees
#     Args:
#         angle: (np.ndarray): tensor of angles in degrees, any shape.
#
#     Returns:
#        np.ndarray: tensor of *x2x2 rotation matrices.
#
#     Shape:
#         - Input: :math:`(*)`
#         - Output: :math:`(*, 2, 2)`
#
#     Examples:
#         >>> input = torch.rand(1, 3)  # Nx3
#         >>> output = angle_to_rotation_matrix(input)  # Nx3x2x2
#     """
#     ang_rad = angle * np.pi / 180
#     cos_a = torch.cos(ang_rad)
#     sin_a = torch.sin(ang_rad)
#     return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)
#
#
# def get_rotation_matrix2d(center:np.ndarray, angle, scale) ->np.ndarray:
#     r"""Calculates an affine matrix of 2D rotation.
#
#     The function calculates the following matrix:
#
#     .. math::
#         \begin{bmatrix}
#             \alpha & \beta & (1 - \alpha) \cdot \text{x}
#             - \beta \cdot \text{y} \\
#             -\beta & \alpha & \beta \cdot \text{x}
#             + (1 - \alpha) \cdot \text{y}
#         \end{bmatrix}
#
#     where
#
#     .. math::
#         \alpha = \text{scale} \cdot cos(\text{angle}) \\
#         \beta = \text{scale} \cdot sin(\text{angle})
#
#     The transformation maps the rotation center to itself
#     If this is not the target, adjust the shift.
#
#     Args:
#         center (Tensor,tuple): center of the rotation in the source image.
#         angle (Tensor,float): rotation angle in degrees. Positive values mean
#             counter-clockwise rotation (the coordinate origin is assumed to
#             be the top-left corner).
#         scale (Tensor,float): isotropic scale factor.
#
#     Returns:
#         Tensor: the affine matrix of 2D rotation.
#
#     Shape:
#         - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
#         - Output: :math:`(B, 2, 3)`
#
#     Examples:
#         >>> center = torch.zeros(1, 2)
#         >>> scale = torch.ones(1)
#         >>> angle = 45. * torch.ones(1)
#         >>> get_rotation_matrix2d(center, angle, scale)
#         tensor([[[ 0.7071,  0.7071,  0.0000],
#                  [-0.7071,  0.7071,  0.0000]]])
#     """
#     center = np.array(center)
#     angle = np.array(angle)
#     scale = np.array(scale)
#
#     if len(center) == 2 and ndim(center) == 1:
#         center = center.unsqueeze(0)
#     if not (len(center.shape) == 2 and center.shape[1] == 2):
#         raise ValueError("Input center must be a Bx2 tensor. Got {}".format(center.shape))
#
#     # convert angle and apply scale
#     scaled_rotation = angle_to_rotation_matrix(angle) * scale.view(-1, 1, 1)
#     alpha = scaled_rotation[:, 0, 0]
#     beta = scaled_rotation[:, 0, 1]
#
#     # unpack the center to x, y coordinates
#     x = center[..., 0]
#     y = center[..., 1]
#
#     # create output tensor
#     batch_size = center.shape[0]
#     M = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
#     M[..., 0:2, 0:2] = scaled_rotation
#     M[..., 0, 2] = (np.ndarray(1.) - alpha) * x - beta * y
#     M[..., 1, 2] = beta * x + (np.ndarray(1.) - alpha) * y
#     return M
#
#
# def _compute_rotation_matrix(angle:np.ndarray, center:np.ndarray) ->np.ndarray:
#     """Computes a pure affine rotation matrix."""
#     scale_tensor = torch.ones_like(angle)
#     matrix_tensor = get_rotation_matrix2d(center, angle, scale)
#     return matrix_tensor
#
#
# def _compute_translation_matrix(translation:np.ndarray) ->np.ndarray:
#     """Computes affine matrix for translation."""
#     matrix_tensor = torch.eye(3, device=translation.device, dtype=translation.dtype)
#     matrix = matrix_tensor.repeat(translation.shape[0], 1, 1)
#
#     dx, dy = torch.chunk(translation, chunks=2, dim=-1)
#     matrix[..., 0, 2:3] += dx
#     matrix[..., 1, 2:3] += dy
#     return matrix
#
#
# def _compute_scaling_matrix(scale:np.ndarray, center:np.ndarray) ->np.ndarray:
#     """Computes affine matrix for scaling."""
#     angle_tensor = torch.zeros_like(scale)
#     matrix_tensor = get_rotation_matrix2d(center, angle_tensor, scale)
#     return matrix_tensor
#
#
# def _compute_shear_matrix(shear:np.ndarray) ->np.ndarray:
#     """Computes affine matrix for shearing."""
#     matrix_tensor = torch.eye(3, device=shear.device, dtype=shear.dtype)
#     matrix = matrix_tensor.repeat(shear.shape[0], 1, 1)
#
#     shx, shy = torch.chunk(shear, chunks=2, dim=-1)
#     matrix[..., 0, 1:2] += shx
#     matrix[..., 1, 0:1] += shy
#     return matrix
#
#
# # based on:
# # https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166
#
# def normal_transform_pixel(height, width):
#     """
#
#     Args:
#         height ():
#         width ():
#
#     Returns:
#
#     """
#     tr_mat =np.ndarray([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])  # 1x3x3
#     tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
#     tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)
#
#     tr_mat = tr_mat.unsqueeze(0)
#
#     return tr_mat
#
#
# def dst_norm_to_dst_norm(dst_pix_trans_src_pix, dsize_src, dsize_dst):
#     # source and destination sizes
#     src_h, src_w = dsize_src
#     dst_h, dst_w = dsize_dst
#     # the devices and types
#     device = dst_pix_trans_src_pix.device
#     dtype = dst_pix_trans_src_pix.dtype
#     # compute the transformation pixel/norm for src/dst
#     src_norm_trans_src_pix = normal_transform_pixel(src_h, src_w).to(device).to(dtype)
#     src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
#     dst_norm_trans_dst_pix = normal_transform_pixel(dst_h, dst_w).to(device).to(dtype)
#     # compute chain transformations
#     dst_norm_trans_src_norm = torch.matmul(dst_norm_trans_dst_pix,
#                                            torch.matmul(dst_pix_trans_src_pix, src_pix_trans_src_norm))
#     return dst_norm_trans_src_norm
#
#
# def transform_points(trans_01:np.ndarray, points_1:np.ndarray) ->np.ndarray:
#     r"""Function that applies transformations to a set of points.
#     Args:
#         trans_01 (np.ndarray): tensor for transformations of shape
#           :math:`(B, D+1, D+1)`.
#         points_1 (np.ndarray): tensor of points of shape :math:`(B, N, D)`.
#     Returns:
#        np.ndarray: tensor of N-dimensional points.
#     Shape:
#         - Output: :math:`(B, N, D)`
#
#     Examples:
#
#         >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
#         >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
#         >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
#     """
#     if not torch.is_tensor(trans_01) or not torch.is_tensor(points_1):
#         raise TypeError("Input type is not anp.ndarray")
#     if not trans_01.device == points_1.device:
#         raise TypeError("Tensor must be in the same device")
#     if not trans_01.shape[0] == points_1.shape[0]:
#         raise ValueError("Input batch size must be the same for both tensors")
#     if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
#         raise ValueError("Last input dimensions must differe by one unit")
#     # to homogeneous
#     points_1_h = torch.nn.functional.pad(points_1, [0, 1], "constant", 1.0)
#     # transform coordinates
#     points_0_h = torch.matmul(trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
#     points_0_h = torch.squeeze(points_0_h, dim=-1)
#     # to euclidean
#     z_vec_tensor = points_0_h[..., -1:]
#     mask_tensor = torch.abs(z_vec_tensor) > 1e-8
#     scale_tensor = torch.ones_like(z_vec_tensor).masked_scatter_(mask_tensor,
#                                                                 np.ndarray(1.0) / z_vec_tensor[mask_tensor])
#
#     return scale_tensor * points_0_h[..., :-1]
#
#
# def warp_grid(dst_homo_src:np.ndarray, dsize) ->np.ndarray:
#     r"""Computes the grid to warp the coordinates grid by an homography.
#
#     Args:
#         dst_homo_src (np.ndarray): Homography or homographies (stacked) to
#                           transform all points in the grid. Shape of the
#                           homography has to be :math:`(N, 3, 3)`.
#
#     Returns:
#        np.ndarray: the transformed grid of shape :math:`(N, H, W, 2)`.
#     """
#     height, width = dsize
#     grid = meshgrid(height, width, normalized_coordinates=True)
#
#     batch_size = dst_homo_src.shape[0]
#     device = dst_homo_src.device
#     dtype = dst_homo_src.dtype
#     # expand grid to match the input batch size
#     grid_tensor = grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
#     if len(dst_homo_src.shape) == 3:  # local homography case
#         dst_homo_src = dst_homo_src.view(batch_size, 1, 3, 3)  # NxHxWx3x3
#     # perform the actual grid transformation,
#     # the grid is copied to input device and casted to the same type
#     flow_tensor = transform_points(dst_homo_src, grid_tensor.to(device).to(dtype))  # NxHxWx2
#     return flow_tensor.view(batch_size, height, width, 2)  # NxHxWx2
#
#
# def warp_affine(src:np.ndarray, M:np.ndarray, dsize: Tuple[int, int], mode: Optional[str] = 'bilinear',
#                 padding_mode: Optional[str] = 'zeros') ->np.ndarray:
#     r"""Applies an affine transformation to a tensor.
#
#     The function warp_affine transforms the source tensor using
#     the specified matrix:
#
#     .. math::
#         \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
#         M_{21} x + M_{22} y + M_{23} \right )
#
#     Args:
#         src (np.ndarray): input tensor of shape :math:`(B, C, H, W)`.
#         M (np.ndarray): affine transformation of shape :math:`(B, 2, 3)`.
#         dsize (Tuple[int, int]): size of the output image (height, width).
#         mode (Optional[str]): interpolation mode to calculate output values
#           'bilinear' | 'nearest'. Default: 'bilinear'.
#         padding_mode (Optional[str]): padding mode for outside grid values
#           'zeros' | 'border' | 'reflection'. Default: 'zeros'.
#
#     Returns:
#        np.ndarray: the warped tensor.
#
#     Shape:
#         - Output: :math:`(B, C, H, W)`
#
#     .. note::
#        See a working example `here <https://github.com/arraiyopensource/
#        kornia/blob/master/docs/source/warp_affine.ipynb>`__.
#     """
#     if not torch.is_tensor(src):
#         raise TypeError("Input src type is not anp.ndarray. Got {}".format(type(src)))
#     if not torch.is_tensor(M):
#         raise TypeError("Input M type is not anp.ndarray. Got {}".format(type(M)))
#     if not len(src.shape) == 4:
#         raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))
#     if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
#         raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(src.shape))
#     try:
#         # we generate a 3x3 transformation matrix from 2x3 affine
#         M_3x3_tensor = F.pad(M, [0, 0, 0, 1, 0, 0], mode="constant", value=0)
#         M_3x3_tensor[:, 2, 2] += 1.0
#
#         dst_norm_trans_dst_norm = dst_norm_to_dst_norm(M_3x3_tensor, (src.shape[-2:]), dsize)
#         # launches the warper
#         return F.grid_sample(src, warp_grid(torch.inverse(dst_norm_trans_dst_norm), dsize=dsize), mode='bilinear',
#                              padding_mode='zeros')
#     except Exception:
#         PrintException()
#         return None
#
#
# def affine(tensor:np.ndarray, matrix:np.ndarray) ->np.ndarray:
#     r"""Apply an affine transformation to the image.
#
#     Args:
#         tensor (np.ndarray): The image tensor to be warped.
#         matrix (np.ndarray): The 2x3 affine transformation matrix.
#
#     Returns:
#        np.ndarray: The warped image.
#     """
#     # warping needs data in the shape of BCHW
#     is_unbatched = tensor.ndimension() == 3
#     if is_unbatched:
#         tensor = torch.unsqueeze(tensor, dim=0)
#
#     # we enforce broadcasting since by default grid_sample it does not
#     # give support for that
#     matrix = matrix.expand(tensor.shape[0], -1, -1)
#
#     # warp the input tensor
#     height = tensor.shape[-2]
#     width = tensor.shape[-1]
#     warped_tensor = warp_affine(tensor, matrix, (height, width))
#
#     # return in the original shape
#     if is_unbatched:
#         warped = torch.squeeze(warped_tensor, dim=0)
#
#     return warped_tensor
#
#
# # based on:
# # https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185
#
# def rotate(tensor:np.ndarray, angle:np.ndarray) ->np.ndarray:
#     r"""Rotate the image anti-clockwise about the centre.
#
#     See :class:`~kornia.Rotate` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not anp.ndarray. Got {}".format(type(tensor)))
#     if not torch.is_tensor(angle):
#         raise TypeError("Input angle type is not anp.ndarray. Got {}".format(type(angle)))
#
#     if len(tensor.shape) not in (3, 4,):
#         raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
#                          "Got: {}".format(tensor.shape))
#
#     # compute the rotation matrix
#     # TODO: add broadcasting to get_rotation_matrix2d for center
#     angle = angle.expand(tensor.shape[0])
#     center =np.ndarray([(tensor.size(4) - 1) / 2, (tensor.size(3) - 1) / 2]).expand(tensor.shape[0], -1).to(
#         tensor.device)
#     rotation_matrix = _compute_rotation_matrix(angle, center)
#
#     # warp using the affine transform
#     return affine(tensor, rotation_matrix[..., :2, :3])
#
#
# def translate(tensor:np.ndarray, translation:np.ndarray) ->np.ndarray:
#     r"""Translate the tensor in pixel units.
#
#     See :class:`~kornia.Translate` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not anp.ndarray. Got {}".format(type(tensor)))
#     if not torch.is_tensor(translation):
#         raise TypeError("Input translation type is not anp.ndarray. Got {}".format(type(translation)))
#     if len(tensor.shape) not in (3, 4,):
#         raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
#                          "Got: {}".format(tensor.shape))
#
#     # compute the translation matrix
#     translation_matrix = _compute_translation_matrix(translation)
#
#     # warp using the affine transform
#     return affine(tensor, translation_matrix[..., :2, :3])
#
#
# def scale(tensor:np.ndarray, scale_factor:np.ndarray) ->np.ndarray:
#     r"""Scales the input image.
#
#     See :class:`~kornia.Scale` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not anp.ndarray. Got {}".format(type(tensor)))
#     if not torch.is_tensor(scale_factor):
#         raise TypeError("Input scale_factor type is not anp.ndarray. Got {}".format(type(scale_factor)))
#
#     # compute the tensor center
#
#     # compute the rotation matrix
#     # TODO: add broadcasting to get_rotation_matrix2d for center
#     center =np.ndarray([(tensor.size(4) - 1) / 2, (tensor.size(3) - 1) / 2]).expand(tensor.shape[0], -1).to(
#         tensor.device)
#     scale_factor = scale_factor.expand(tensor.shape[0])
#     scaling_matrix = _compute_scaling_matrix(scale_factor, center)
#
#     # warp using the affine transform
#     return affine(tensor, scaling_matrix[..., :2, :3])
#
#
# def shear(tensor:np.ndarray, shear:np.ndarray) ->np.ndarray:
#     r"""Shear the tensor.
#
#     See :class:`~kornia.Shear` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not anp.ndarray. Got {}".format(type(tensor)))
#     if not torch.is_tensor(shear):
#         raise TypeError("Input shear type is not anp.ndarray. Got {}".format(type(shear)))
#     if len(tensor.shape) not in (3, 4,):
#         raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
#                          "Got: {}".format(tensor.shape))
#
#     # compute the translation matrix
#     shear_matrix = _compute_shear_matrix(shear)
#
#     # warp using the affine transform
#     return affine(tensor, shear_matrix[..., :2, :3])

#
# _FUN_NAMES = [
#     # source_fun, target_fun
#     ('int_shape', int_shape),
#     ('cast', cast),
#     ('is_nan', is_nan),
#     ('is_inf', is_inf),
#     ('is_abnormal_number', is_abnormal_number),
#     ('any_nan', any_nan),
#     ('any_inf', any_inf),
#     ('any_abnormal_number', any_abnormal_number),
#     ('less', less),
#     ('equal', equal),
#     ('greater', greater),
#     ('greater_equal', greater_equal),
#     ('not_equal', not_equal),
#     ('less_equal', less_equal),
#     ('argmax', argmax),
#     ('argmin', argmin),
#     ('argsort', argsort),
#     ('maximum', maximum),
#     ('minimum', minimum),
#     ('floor', floor),
#     ('ceil', ceil),
#     ('round', round),
#     ('dot', dot),
#     ('sqrt', sqrt),
#     ('rsqrt', rsqrt),
#     ('square', square),
#     ('abs', abs),
#     ('pow', pow),
#     ('log', log),
#     ('exp', exp),
#     ('clip', clip),
#     ('add', add),
#     ('subtract', subtract),
#     ('true_divide', true_divide),
#     ('matmul', matmul),
#     ('sin', sin),
#     ('cos', cos),
#     ('tan', tan),
#     ('asin', asin),
#     ('acos', acos),
#     ('atan', atan),
#     ('sinh', sinh),
#     ('cosh', cosh),
#     ('tanh', tanh),
#     ('element_times', element_times),
#     ('element_max', element_max),
#     ('element_min', element_min),
#     ('element_divide', element_divide),
#     ('element_cosine_distance', element_cosine_distance),
#     ('where', where),
#     ('reduce_mean', reduce_mean),
#     ('reduce_sum', reduce_sum),
#     ('reduce_max', reduce_max),
#     ('reduce_min', reduce_min),
#     ('mean', mean),
#     ('sum', sum),
#     ('max', max),
#     ('min', min),
#     ('reduce_logsumexp', reduce_logsumexp),
#     ('reduce_prod', reduce_prod),
#     ('identity', identity),
#     ('sigmoid', sigmoid),
#     ('relu', relu),
#     ('relu6', relu6),
#     ('leaky_relu', leaky_relu),
#     ('leaky_relu6', leaky_relu6),
#     ('smooth_relu', smooth_relu),
#     ('swish', swish),
#     ('elu', elu),
#     ('hard_sigmoid', hard_sigmoid),
#     ('hard_swish', hard_swish),
#     ('selu', selu),
#     ('lecun_tanh', lecun_tanh),
#     ('soft_sign', soft_sign),
#     ('soft_plus', soft_plus),
#     ('hard_tanh', hard_tanh),
#     ('logit', logit),
#     ('log_log', log_log),
#     ('mish', mish),
#     ('hard_mish', hard_mish),
#     ('softmax', softmax),
#     ('log_softmax', log_softmax),
#     ('gelu', gelu),
#     ('gpt_gelu', gpt_gelu),
#     ('l2_normalize', l2_normalize),
#     ('ones_like', ones_like),
#     ('zeros_like', zeros_like),
#     ('eye_like', eye_like),
#     ('arange', arange),
#     ('meshgrid', meshgrid),
#     ('reshape', reshape),
#     ('permute', permute),
#     ('transpose', transpose),
#     ('squeeze', squeeze),
#     ('expand_dims', expand_dims),
#     ('concate', concate),
#     ('stack', stack),
#     ('gram_matrix', gram_matrix),
#     ('shuffle', shuffle),
#     ('random_choice', random_choice),
#     ('random_normal_like', random_normal_like)
# ]
# for target_fun_name, source_fun in _FUN_NAMES:
#     if not hasattr(np.ndarray, target_fun_name):
#         setattr(np.ndarray, target_fun_name, source_fun)
# del _FUN_NAMES