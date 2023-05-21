import os
import builtins
os.environ['TRIDENT_BACKEND'] = 'jax'
from typing import List, Tuple, Union, Sequence
from functools import wraps
import numpy as np
import numbers
import collections
import jax
import jax.numpy as jnp
import jaxlib
from trident.backend.common import to_list, unpack_singleton, epsilon, OrderedDict, get_function, get_session, \
    TensorShape, get_session_value,is_instance
from trident.backend import dtype as Dtype
from trident import context

__all__ = ['Tensor', 'is_gpu_available', 'is_tpu_available', 'is_tensor', 'is_tensor_like', 'to_numpy', 'to_tensor',
           'tensor_to_shape', 'to', 'cuda', 'cpu', 'copy', 'detach',
           'ndim', 'numel', 'cast', 'str2dtype', 'int_shape', 'logical_and', 'logical_or',
           'logical_xor', 'logical_not', 'less', 'equal', 'greater',
           'greater_equal', 'not_equal', 'less_equal', 'argmax', 'argmin', 'argsort', 'topk', 'maximum', 'minimum',
           'floor',
           'ceil', 'round', 'dot', 'sign', 'sqrt', 'rsqrt', 'prod', 'square', 'abs', 'pow', 'log', 'exp', 'clip', 'add',
           'subtract',
           'true_divide', 'pi', 'matmul', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh',
           'asinh', 'acosh', 'atanh', 'where',
           'reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min', 'mean', 'sum', 'max', 'min', 'reduce_logsumexp',
           'reduce_prod', 'reduce_any', 'depth_to_space', 'space_to_depth', 'pad', 'identity', 'sigmoid', 'relu',
           'relu6', 'leaky_relu', 'celu',
           'leaky_relu6', 'smooth_relu', 'crelu', 'p_relu', 'swish', 'elu', 'hard_sigmoid', 'hard_swish', 'selu',
           'silu', 'lecun_tanh',
           'soft_sign', 'soft_plus', 'square_plus', 'hard_tanh', 'logit', 'log_log', 'mish', 'hard_mish', 'softmax',
           'log_softmax', 'gelu', 'reverse',
           'gpt_gelu', 'moments', 'norm', 'l2_normalize', 'broadcast_to', 'expand_as', 'ones',
           'ones_like', 'zeros', 'zeros_like', 'eye', 'eye_like', 'make_onehot', 'arange', 'meshgrid', 'reshape',
           'permute', 'transpose', 'squeeze', 'expand_dims', 'concate', 'stack', 'split', 'bbox_iou', 'bbox_diou']

from math import e, nan, inf, pi

__all__.extend(['e', 'pi', 'nan', 'inf'])

print(jax.default_backend())
Tensor = jaxlib.xla_extension.DeviceArray
if 'tpu' in jax.default_backend():
    arr = jnp.array(np.random.normal(0, 1, (3, 3)))
    print(arr.__class__)
else:
    try:
        jax.config.update('jax_array', True)
        Tensor =jax.experimental.array.Array
    except:
        Tensor = jaxlib.xla_extension.DeviceArray

ctx = context._context()

_bool = builtins.bool
_int = builtins.int


def numpy_compatible(func):
    """decorator for function to support non-tensor input

    Args:
        func : wrapped function

    Returns:

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if func.__name__ in ('max', 'min', 'abs', 'round', 'pow') and isinstance(args[0], tuple):
            args = unpack_singleton(args)

        x = args[0] if hasattr(args, '__len__') else args
        new_args = []
        new_kwargs = OrderedDict()

        if all([isinstance(arg, numbers.Number) for arg in args]) and (len(kwargs) == 0 or all(
                [isinstance(kv[1], numbers.Number) for kv in kwargs.items()])) and func.__name__ in (
                'max', 'min', 'abs', 'round', 'pow'):
            builtins_funcs = get_function(func.__name__, ['builtins'])
            y = builtins_funcs(*args, **kwargs)
            return y
        elif all([isinstance(arg, numbers.Number) for arg in args]) and (
                len(kwargs) == 0 or all([isinstance(kv[1], numbers.Number) for kv in kwargs.items()])) and get_function(func.__name__, ['math', 'numpy', 'trident.backend.numpy_ops']) is not None:
            mathfuncs = get_function(func.__name__, ['math', 'numpy', 'trident.backend.numpy_ops'])
            y = mathfuncs(*args, **kwargs)
            return y
        # elif isinstance(x, list) and all([isinstance(arg, np.ndarray) for arg in x]) and func.__name__ in ['concate','stack','vstack','hstack']:
        #     numpy_func = get_function(func.__name__, ['trident.backend.numpy_ops','numpy'])
        #     y = numpy_func(*args, **kwargs)
        #     return y
        # elif isinstance(x, list) and all([isinstance(arg, Tensor) for arg in x])  and func.__name__ in ['concate','stack','vstack','hstack']:
        #     tensor_func = get_function(func.__name__, ['trident.backend.jax_ops'])
        #     y = tensor_func(*args, **kwargs)
        #     return y
        #
        elif isinstance(x, np.ndarray):
            numpy_func = get_function(func.__name__, ['trident.backend.numpy_ops', 'numpy'])
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


def is_gpu_available():
    """
    Args:

    Returns:
        (Bool): is gpu available.
    Examples:
        >>> print(is_gpu_available())
        True

    """
    # devices = jax.local_devices()
    # print(devices)
    _default_backend = jax.default_backend()
    # print(_default_backend)
    if 'gpu' in _default_backend:
        return True
    return False


def is_tpu_available():
    """
     Args:

     Returns:
         (Bool): is gpu available.
     Examples:
         >>> print(is_tpu_available())
         False

     """
    from jax.lib import xla_bridge
    _default_backend = xla_bridge.get_backend().platform
    if 'tpu' in _default_backend:
        return True
    return False


# def _get_device():
#     """get current device
#
#     Returns: device string ('cpu', 'cuda)
#
#     """
#     if get_session().device is None:
#         _set_device(jax.default_backend())
#     return get_session().device
#
#
# def _set_device(device='cpu'):
#     device = device.lower().replace('cuda', 'gpu').replace('xpu', 'tpu')
#     if device == 'gpu' and not is_gpu_available():
#         raise ValueError('Gpu is not available...')
#     if device == 'tpu' and not is_tpu_available():
#         raise ValueError('Tpu is not available...')
#     try:
#         device_ = device
#         if device == ['xpu', 'tpu']:
#             import torch_xla.core.xla_model as xm
#             device_ = xm.xla_device()
#         elif device in ['cuda', 'gpu']:
#             device_ = jax.devices('gpu')[0]
#         elif device in ['cpu']:
#             device_ = jax.devices()[0]
#         set_session('device', device_)
#
#         gcitems = gc.get_objects()
#         for i in range(len(gcitems)):
#             obj = gcitems[i]
#             try:
#                 if is_tensor(obj):
#                     jax.device_put(obj, device=device_)
#                 elif is_instance(obj, 'Layer'):
#                     obj.to(device_)
#             except Exception:
#                 pass
#     except Exception as e:
#         print(e)


# def numpy_compatible(func):
#     """decorator for function to support non-tensor input
#
#     Args:
#         func : wrapped function
#
#     Returns:
#
#     """
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         if func.__name__ in ('max','min','abs','round','pow') and isinstance(args[0],tuple):
#             args=unpack_singleton(args)
#
#         x = args[0] if hasattr(args,'__len__') else args
#         new_args = []
#         new_kwargs = OrderedDict()
#
#
#         if all([isinstance(arg, numbers.Number) for arg in args]) and (len(kwargs)==0 or all([isinstance(kv[1], numbers.Number) for kv in kwargs.items()])) and  func.__name__ in ('max','min','abs','round','pow'):
#             builtins_funcs = get_function(func.__name__, ['builtins'])
#             y = builtins_funcs(*args, **kwargs)
#             return y
#         elif all([isinstance(arg,numbers.Number) for arg in args]) and (len(kwargs)==0 or all([isinstance(kv[1],numbers.Number) for kv in kwargs.items()]) )and  get_function(func.__name__, ['math','numpy','trident.backend.numpy_ops']) is not None:
#             mathfuncs=get_function(func.__name__, ['math','numpy','trident.backend.numpy_ops'])
#             y = mathfuncs(*args, **kwargs)
#             return y
#         # elif isinstance(x, list) and all([isinstance(arg, np.ndarray) for arg in x]) and func.__name__ in ['concate','stack','vstack','hstack']:
#         #     numpy_func = get_function(func.__name__, ['trident.backend.numpy_ops','numpy'])
#         #     y = numpy_func(*args, **kwargs)
#         #     return y
#         # elif isinstance(x, list) and all([isinstance(arg, Tensor) for arg in x])  and func.__name__ in ['concate','stack','vstack','hstack']:
#         #     tensor_func = get_function(func.__name__, ['trident.backend.jax_ops'])
#         #     y = tensor_func(*args, **kwargs)
#         #     return y
#         #
#         elif isinstance(x, np.ndarray):
#             numpy_func = get_function(func.__name__, ['trident.backend.numpy_ops','numpy'])
#             if numpy_func is not None:
#                 for arg in args:
#                     if is_tensor(arg):
#                         new_args.append(to_numpy(arg))
#                     else:
#                         new_args.append(arg)
#                 for k, v in kwargs.items():
#                     if is_tensor(v):
#                         new_kwargs[k] = to_numpy(v)
#                     else:
#                         new_kwargs[k] = v
#                 y = numpy_func(*new_args, **new_kwargs)
#                 return y
#             else:
#                 for arg in args:
#                     if isinstance(arg, np.ndarray):
#                         new_args.append(to_tensor(arg))
#                     else:
#                         new_args.append(arg)
#                 for k, v in kwargs.items():
#                     if isinstance(v, np.ndarray):
#                         new_kwargs[k] = to_tensor(v)
#                     else:
#                         new_kwargs[k] = v
#                 y = func(*new_args, **new_kwargs)
#                 return y
#         else:
#             for arg in args:
#                 if isinstance(arg, np.ndarray):
#                     new_args.append(to_tensor(arg))
#                 else:
#                     new_args.append(arg)
#             for k, v in kwargs.items():
#                 if isinstance(v, np.ndarray):
#                     new_kwargs[k] = to_tensor(v)
#                 else:
#                     new_kwargs[k] = v
#             y = func(*new_args, **new_kwargs)
#             return y
#
#     return wrapper

def is_tensor(x:Tensor)->_bool:
    """Checks whether `x` is exactly a tensor

    If `is_tensor(x)` returns `True`, that `x` is a EagerTensor .

    Args:
        x: A python object to check.

    Returns:
        `True` if `x` is exactly a tensor, `False` if not.

    Examples:
        >>> is_tensor(jnp.array([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        False
        >>> is_tensor("Hello World")
        False

    """

    if isinstance(x, jax.xla.DeviceArray):
        return True
    return False


def is_tensor_like(x:Tensor)->_bool:
    """Checks whether `x` is a "tensor-like".

    If `is_tensor_like(x)` returns `True`, it is safe to assume that `x` is a tensor or can be converted to a tensor using `ops.convert_to_tensor(x)`.

    Args:
        x: A python object to check.

    Returns:
        True` if `x` is a tensor or "tensor-like", `False` if not.


    Examples:
        >>> is_tensor_like(jnp.array([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor_like([[1,2,3],[4,5,6],[7,8,9]])
        True
        >>> is_tensor_like(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor_like ("Hello World")
        False

    """
    return is_tensor(to_tensor(x))


def to_numpy(x) -> np.ndarray:
    """Convert whatever to numpy array

     Args:
        x: List, tuple, jax tensor or numpy array

     Returns:
        Numpy array

     Examples:
          >>> to_numpy(5)
          array(5))
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
    elif isinstance(x, (list, tuple)):
        x = [to_numpy(item) if is_tensor(item) else item for item in x]
        return np.stack(x, 0)
    elif is_tensor(x):
        return np.array(x)
    # elif context.executing_eage
    elif hasattr(x, '__len__') and len(x) > 1 and all(
            [isinstance(k, (list, tuple, numbers.Number, np.ndarray)) for k in x]):
        x = unpack_singleton(x)
        return np.array([x])
    else:
        return x


def to_tensor(x, dtype=None, device=None, requires_grad=None) -> Tensor:
    """Convert the input `x` to a tensor of type `dtype`.

    Args:

        device ():
        x: An object to be converted (ex.numpy array, list, tensors).
        dtype (str or jax.Dtype): The destination type or type string.
        requires_grad (None or bool): whether need grade

    Returns:
        A tensor.

    Examples:
        >>> to_tensor(2)
        Array(2.0000e+00, dtype=float32)
        >>> to_tensor([1.0,2.0,3.0],requires_grad=True)
        Array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)
        >>> to_tensor([1.0,2.0,3.0],requires_grad=False)
        Array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)
        >>> to_tensor([1.0,2.0,3.0])
        Array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)
        >>> to_tensor((1.0,2.0,3.0))
        Array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)
        >>> to_tensor(np.arange(0,5))
        Array([0, 1, 2, 3, 4], dtype=int32)

    """
    if x is None:
        return x
    if device is not None and ('cuda' in device.lower() or 'gpu' in device.lower()):
        device = 'gpu:0'
    else:
        device = "cpu:0"
    input_dtype = dtype
    if dtype is None and isinstance(x, numbers.Integral):
        dtype = Dtype.int64
    elif dtype is None and isinstance(x, collections.Iterable) and all(
            [isinstance(item, numbers.Integral) for item in x]):
        dtype = Dtype.int64
    elif dtype is None:
        dtype = Dtype.float32
    elif isinstance(dtype, str):
        dtype = str2dtype(dtype)
    if device is None:
        device = get_session().device

    if isinstance(x, Tensor):
        if dtype is None:
            return x
        else:
            return x.astype(dtype)
    elif isinstance(x, np.ndarray):
        npdtype = x.dtype

        if 'int' in str(npdtype):
            x = jnp.array(x, dtype=jnp.int64)
        else:
            x = jnp.array(x, dtype=jnp.float32)
        return x
    else:
        x = jnp.array(x)
        if 'complex' in str(x.dtype):
            return x
        else:
            return jnp.array(x, dtype=jnp.float32)


def tensor_to_shape(x: Tensor, need_exclude_batch_axis=True, is_singleton=False) -> TensorShape:
    """Get tensor shape information ten convert to TensorShape

    Args:
        is_singleton ():
        x (Tensor):
        need_exclude_batch_axis (bool):

    Returns:
        shape (TensorShape):

    Examples:
        >>> tensor_to_shape(random_normal((2,64,32,32)))
        TensorShape([None, 64, 32, 32])

    """
    if isinstance(x, numbers.Number) or (is_tensor(x) and ndim(x) == 0):
        return TensorShape([None])
    elif isinstance(x, str) or (isinstance(x, list) and len(x) > 0 and isinstance(x[0], str)):
        return TensorShape([None])
    if need_exclude_batch_axis and is_singleton == False:
        shp = list(int_shape(x))
        if len(shp) == 0:
            print('')
        shp[0] = None
        return TensorShape(shp)
    elif need_exclude_batch_axis and is_singleton == True:
        return TensorShape([None] + list(int_shape(x)))
    else:
        return TensorShape(int_shape(x))


_float_dtype = Dtype.float16 if ctx.amp_available == True and ctx.is_autocast_enabled == True and get_session().device == 'cuda' else Dtype.float32


# if isinstance(x, int):
#     x= jax.constant(value=x, dtype=jax.int32)
# elif is_tensor(x):
#     if x.dtype!=dtype:
#         x=jax.cast(x,dtype)
#
# elif isinstance(x, float):
#     x= jax.constant(value=x, dtype=jax.float32)
# else:
#     try:
#
#         if requires_grad == False:
#             x =jax.constant(x, dtype=dtype)
#         else:
#             x = ops.convert_to_tensor(x, dtype=dtype)
#     except:
#         pass
#
# if dtype is not None :
#     x = cast(x, dtype)
# return x

# def to(x, *args):
#     args=unpack_singleton(args)
#     if isinstance(args,str):
#         if 'cpu' in args:
#             return cpu(x)
#         elif 'gpu' in args or 'cuda' in args:
#             return cuda(x)
#         elif 'float' in args:
#             return cast(x, jnp.float32)
#         elif 'long' in args:
#             return cast(x, jnp.int64)
#         elif 'int' in args:
#             return cast(x, jnp.int32)
#     elif isinstance(args,dtypes.DType):
#         return cast(x, args)
#     else:
#         return x

def str2dtype(dtype_str: (str, jnp.dtype)):
    """ Mapping string to dtype

    Args:
        dtype_str (str): dtype representation string

    Returns:
        dtype

    >>> str2dtype('float16')
     jnp.float16

    """

    if isinstance(dtype_str, jnp.dtype):
        return dtype_str
    elif isinstance(dtype_str, jax._src.numpy.lax_numpy._ScalarMeta):
        return dtype_str

    if dtype_str:
        return dtype_str
    elif isinstance(dtype_str, str):
        if 'float64' in dtype_str.lower() or 'double' in dtype_str.lower():
            return Dtype.float64
        elif 'float32' in dtype_str.lower() or 'single' in dtype_str.lower():
            return Dtype.float32
        elif 'float16' in dtype_str.lower() or 'half' in dtype_str.lower():
            return Dtype.float16
        elif 'float' in dtype_str.lower():
            return _float_dtype
        elif 'int64' in dtype_str.lower() or 'long' in dtype_str.lower():
            return Dtype.int64
        elif 'int16' in dtype_str.lower() or 'short' in dtype_str.lower():
            return Dtype.int16
        elif 'uint8' in dtype_str.lower() or 'byte' in dtype_str.lower():
            return Dtype.uint8
        elif 'int8' in dtype_str.lower() or 'char' in dtype_str.lower():
            return Dtype.int8
        elif 'int32' in dtype_str.lower() or 'int' in dtype_str.lower():
            return Dtype.int32
        elif 'bool' in dtype_str.lower():
            return Dtype.bool
    return _float_dtype


def cast(x: Tensor, cast_dtype):
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
        cast_dtype: The destination type. The list of supported dtypes and string is the same as
        `x`.


    Returns:
        A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` and
        same type as `dtype`.

    Examples:
        >>> x = to_tensor([1.8, 2.2])
        >>> cast(x,jnp.int64)
        Array([1, 2], dtype=int32)

    Raises:
        TypeError: If `x` cannot be cast to the `dtype`.

    """

    cast_dtype = str2dtype(cast_dtype)
    if isinstance(x, (jnp.ndarray, jax.xla.DeviceArray)) and isinstance(cast_dtype, (
            jnp.dtype, jax._src.numpy.lax_numpy._ScalarMeta)):
        return jnp.array(x, dtype=cast_dtype)
    elif isinstance(x, np.ndarray) and isinstance(cast_dtype, np.dtype):
        return jnp.array(x, dtype=cast_dtype)
    else:
        return x


def float(x: Tensor)->Tensor:
    return jnp.array(x, jnp.float32)


def int(x: Tensor)->Tensor:
    return jnp.array(x, jnp.int32)


def long(x: Tensor)->Tensor:
    return jnp.array(x, jnp.int64)


def cpu(x: Tensor)->Tensor:
    return jax.device_put(x, device=jax.devices("cpu")[0])


def cuda(x: Tensor, device: int = None):
    """Moves all model parameters and buffers to the GPU.

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
    if is_gpu_available:
        return jax.device_put(x, device=jax.devices("gpu")[0])
    else:
        return x


def to(x, *args):
    args = unpack_singleton(args)
    if isinstance(args, str):
        if 'cpu' in args:
            return cpu(x)
        elif 'gpu' in args or 'cuda' in args:
            return cuda(x)
        elif 'float' in args:
            return cast(x, jnp.float32)
        elif 'long' in args:
            return cast(x, jnp.int64)
        elif 'int' in args:
            return cast(x, jnp.int32)
    elif isinstance(args, (jnp.dtype, jax._src.numpy.lax_numpy._ScalarMeta)):
        return cast(x, args)
    else:
        return x


def copy(x: Tensor) -> Tensor:
    """Returns a copy of x.

    Args:
        x:: input tensor

    Returns:
        a copy of x..

    """
    return jnp.copy(x)


def detach(x: Tensor)->Tensor:
    """Make the tensor stop gradient calculation.

    Args:
        x:

    Returns:
        stop gradient Tensor

    """
    x = jax.lax.stop_gradient(x)
    return x


@numpy_compatible
def ndim(x: Tensor)->_int:
    """Number of dimension of a tensor

    Args:
        x (Tensor): input tensor.

    Returns:
        (int) Number of dimension

    """
    return len(int_shape(x))


def numel(x: Tensor)->_int:
    """Number of elements of a tensor

    Args:
        x (Tensor): input tensor.

    Returns:
        (int) Number of elements

    """
    return len(x.reshape(-1))


def int_shape(x):
    """Shape of a tensor in tuple of integer format

    Args:
        x : input tensor

    Returns:
        tuple of integer as shape representation

    Examples:
        >>> int_shape(ones((3,3,7)))
        [3, 3, 7]

    """

    if x is None:
        return []
    elif is_instance(x, 'Parameter'):
        return list(x.data.shape)
    elif isinstance(x, TensorShape):
        return x.dims
    elif hasattr(x, 'shape'):
        return [d for d in x.shape]  # if isinstance(x,np.ndarray)  else  [d for d in  x.size()]
    else:
        return []


def logical_and(left, right):
    """Element-wise `logical and: x && y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor

    Returns:
        A Tensor of type bool with the same size as that of left or right.

    """
    return jnp.logical_and(left, right)


def logical_not(x: np.ndarray):
    """Element-wise `logical not: ~x`
    Args:
        x (Tensor): input boolean tensor
    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return jnp.logical_not(x)


def logical_or(left, right):
    """Element-wise `logical or: x || y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor
    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return jnp.logical_or(left, right)


def logical_xor(left, right):
    """Element-wise `logical xor: x ^ y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor

    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return jnp.logical_xor(left, right)


############################
# compare operation
###########################


def less(left: np.ndarray, right: (np.ndarray, float, int)):
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

    return jnp.less(left, right).astype(np.float32)


def equal(left: Tensor, right: (Tensor, float, int)):
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
    return jnp.equal(left, right).astype(np.float32)


def greater(left: Tensor, right: (Tensor, float, int)):
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
    return jnp.greater(left, right).astype(np.float32)


def greater_equal(left: Tensor, right: (Tensor, float, int)):
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
    return jnp.greater_equal(left, right).astype(np.float32)


def not_equal(left: Tensor, right: (Tensor, float, int)):
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
    return jnp.not_equal(left, right).astype(np.float32)


def less_equal(left: Tensor, right: (Tensor, float, int)):
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
    return jnp.less_equal(left, right)


def argmax(x: Tensor, axis=-1) -> Tensor:
    return jnp.argmax(x, axis=axis)


def argmin(x: Tensor, axis=-1) -> Tensor:
    return jnp.argmin(x, axis=axis)


def argsort(x: np.ndarray, axis=-1, descending=True) -> np.ndarray:
    if descending:
        return jnp.argsort(-x, axis=axis)
    else:
        return jnp.argsort(x, axis=axis)


def topk(x: Tensor, k=1) -> Tensor:
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

    return jax.lax.top_k(x, k=k)


def maximum(x: Tensor, other: (Tensor, int, float)) -> np.ndarray:
    if isinstance(other, np.ndarray):
        return np.maximum(x, other)
    elif isinstance(other, (int, float)):
        return np.clip(x, a_min=float(other))


def minimum(x: np.ndarray, other: (np.ndarray, int, float)) -> np.ndarray:
    if isinstance(other, np.ndarray):
        return np.minimum(x, other)
    elif isinstance(other, (int, float)):
        return np.clip(x, a_max=float(other))


############################
# basic math operation
###########################
@numpy_compatible
def add(x, y):
    """Returns x + y element-wise.

    *NOTE*: `math.add` supports broadcasting. `AddN` does not. More about broadcasting
    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    Args:
        x (Tensor): input tensor.
        y (Tensor): another tensor.


    Returns:
      A `Tensor`. Has the same type as `x`.

    """
    return jnp.add(x, y)


@numpy_compatible
def subtract(x, y):
    """Returns x - y element-wise.

    *NOTE*: `Subtract` supports broadcasting. More about broadcasting
    [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    Args:
        x (Tensor): input tensor.
        y (Tensor): another tensor.


    Returns:
      A `Tensor`. Has the same type as `x`.

    """
    return jnp.subtract(x, y)


@numpy_compatible
def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

     Args
        x (Tensor): input tensor.
        y (Tensor): another tensor.

     Returns
            A tensor, dot product of `x` and `y`.


    """
    return jnp.dot(x, y)


@numpy_compatible
def true_divide(x, y):
    """Divides x / y elementwise (using Python 3 division operator semantics).

    NOTE: Prefer using the Tensor operator or divide which obey Python
    division operator semantics.

    This function forces Python 3 division operator semantics where all integer
    arguments are cast to floating types first.   This op is generated by normal
    `x / y` division in Python 3 and in Python 2.7 with
    `from __future__ import division`.  If you want integer division that rounds
    down, use `x // y` or `math.floordiv`.

    `x` and `y` must have the same numeric type.  If the inputs are floating
    point, the output will have the same type.  If the inputs are integral, the
    inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
    and `int64` (matching the behavior of Numpy).

    Args:
        x (Tensor): input tensor.
        y (Tensor): another tensor.


    Returns:
      `x / y` evaluated in floating point.

    Raises:
      TypeError: If `x` and `y` have different dtypes.

    """
    return jnp.true_divide(x, y)


@numpy_compatible
def matmul(a: Tensor, b: Tensor, transpose_a=False, transpose_b=False):
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


     >>> a =reshape(to_tensor([1, 2, 3, 4, 5, 6]),[2, 3])
     >>> a
     tensor([[1, 2, 3],
            [4, 5, 6]])
     >>> b = reshape(to_tensor([7, 8, 9, 10, 11, 12]), [3, 2])
     >>> b
     tensor([[ 7,  8],
            [ 9, 10],
            [11, 12]])
     >>> c = matmul(a, b)
     >>> c  # `a` * `b`
     tensor([[ 58,  64],
            [139, 154]])

     A batch matrix multiplication with batch shape [2]:

     >>> a =  reshape(to_tensor(np.arange(1, 13, dtype=np.int32)),[2, 2, 3])
     >>> a
     tensor([[[ 1,  2,  3],
             [ 4,  5,  6]],
            [[ 7,  8,  9],
             [10, 11, 12]]])
     >>> b =  reshape(to_tensor(np.arange(13, 25, dtype=np.int32)),[2, 3, 2])
     >>> b
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
     it simply calls the `matmul()` function, so the following lines are
     equivalent:

         >>> d = a @ b @ [[10], [11]]
         >>> d = matmul(matmul(a, b), [[10], [11]])

     Args:
       a: `Tensor` and rank > 1.
       b: `Tensor` with same type and rank as `a`.
       transpose_a: If `True`, `a` is transposed before multiplication.
       transpose_b: If `True`, `b` is transposed before multiplication.


     Returns:
       A `Tensor` of the same type as `a` and `b` where each innermost matrix
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
    return jnp.matmul(a, b)


@numpy_compatible
def floor(x: (Tensor, float)):
    """Returns element-wise greater integer not greater than x.

    Args:
        x (Tensor): input tensor.

    Returns:
      A `Tensor`. Has the same type as `x`.

    """

    return jnp.floor(x)


@numpy_compatible
def ceil(x: (Tensor, float)):
    """Return the ceiling of the input, element-wise.

    Example:

    >>> ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    <Tensor: shape=(7,), dtype=float32,
    numpy=array([-1., -1., -0.,  1.,  2.,  2.,  2.], dtype=float32>)

    Args:
        x (Tensor): input tensor.


    Returns:
      A `Tensor`. Has the same type as `x`.

    @compatibility(numpy)
    Equivalent to np.ceil
    @end_compatibility
    """

    return jnp.ceil(x)


@numpy_compatible
def round(x: Tensor, digit: int = 0):
    """Rounds the values of a tensor to the nearest integer, element-wise.

    Rounds half to even.  Also known as bankers rounding. If you want to round
    according to the current system rounding mode use tf::cint.

    Args:
        x (Tensor): input tensor.
        digit: number of digit

    Returns:
        A `Tensor` of same shape and type as `x`.

    Examples;
        >>> round(to_tensor([[1,2,3,4,5]])/3,0)
        <Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 2.0000e+00]],
              dtype=float32>
        >>> round(to_tensor([[1,2,3,4,5]])/3,2)
        <Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[3.3000e-01, 6.7000e-01, 1.0000e+00, 1.3300e+00, 1.6700e+00]],
              dtype=float32>
        >>> round(to_tensor([[11.6,24.3,35.2,14.4,23.5]])/3,-1)
        <Tensor: shape=(1, 5), dtype=float32, numpy=
        array([[0.0000e+00, 1.0000e+01, 1.0000e+01, 0.0000e+00, 1.0000e+01]],
              dtype=float32>

    """

    return jnp.round(x, decimals=digit)


@numpy_compatible
def prod(x: Tensor)->Tensor:
    """Computes the product of elements across dimensions of a tensor.

    Reduces `input_tensor` along all the dimensions


    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
        x (Tensor): input tensor.

    Returns:
        The reduced tensor.

    @compatibility(numpy)
    Equivalent to np.prod
    @end_compatibility
    """

    return jnp.prod(x)


def pi():
    """ The number π (/paɪ/)
    The number π (/paɪ/) is a mathematical constant. It is defined as the ratio of a circle's circumference to its diameter

    Returns:
        The number π (/paɪ/)

    """
    return jnp.pi


@numpy_compatible
def sign(x: Tensor) -> Tensor:
    """The output of this operation is the element-wise sign of the two  inputtensor.


    Args:
        x (Tensor): input tensor.

    Returns:
        The sign of the input tensor.

    """

    return jnp.sign(x)


@numpy_compatible
def sqrt(x: Tensor)->Tensor:
    """Computes element-wise square root of the input tensor.

    Note: This operation does not support integer types.

    >>> x = to_tensor([[4.0], [16.0]])
    >>> sqrt(x)
    <Tensor: shape=(2, 1), dtype=float32, numpy=
      array([[2.],
             [4.]], dtype=float32>
    >>> y = to_tensor([[-4.0], [16.0]])
    >>> sqrt(y)
    <Tensor: shape=(2, 1), dtype=float32, numpy=
      array([[nan],
             [ 4.]], dtype=float32>
    >>> z = to_tensor([[-1.0], [16.0]], dtype=complex128)
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
    return jnp.sqrt(x)


@numpy_compatible
def rsqrt(x: Tensor)->Tensor:
    """Computes reciprocal of square root of x element-wise.

    Args:
      x: input tensor

    Returns:
      output tensor


    Examples:
        >>> x = to_tensor([2., 0., -2.])
        >>> rsqrt(x)
        <Tensor: shape=(3,), dtype=float32,
        numpy=array([0.707, inf, nan], dtype=float32>

    """
    return jnp.reciprocal(jnp.sqrt(x))


@numpy_compatible
def square(x: Tensor)->Tensor:
    """Computes square of x element-wise.

    I.e., \\(y = x * x = x^2\\).

    >>> square([-2., 0., 3.])
    <Tensor: shape=(3,), dtype=float32, numpy=array([4., 0., 9.], dtype=float32>

    Args:
      x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`,
      `complex64`, `complex128`.


    Returns:
      A `Tensor`. Has the same type as `x`.


    """
    return x ** 2


@numpy_compatible
def abs(x: Tensor)->Tensor:
    """Computes the absolute value of a tensor.

    Given a tensor of integer or floating-point values, this operation returns a
    tensor of the same type, where each element contains the absolute value of the
    corresponding element in the input.

    Given a tensor `x` of complex numbers, this operation returns a tensor of type
    `float32` or `float64` that is the absolute value of each element in `x`. For
    a complex number \\(a + bj\\), its absolute value is computed as \\(\\sqrt{a^2
    + b^2}\\).  For example:

    >>> x = to_tensor([[-2.25 + 4.75j], [-3.25 + 5.75j]])
    >>> abs(x)
    Array([[5.2559e+00],
           [6.6049e+00]], dtype=float32)

    Args:
      x: A `Tensor` or `SparseTensor` of type `float16`, `float32`, `float64`,
        `int32`, `int64`, `complex64` or `complex128`.


    Returns:
      A `Tensor` or `SparseTensor` of the same size, type and sparsity as `x`,
        with absolute values. Note, for `complex64` or `complex128` input, the
        returned `Tensor` will be of type `float32` or `float64`, respectively.
    """
    return jnp.abs(x)


@numpy_compatible
def pow(x: Tensor, y: (Tensor, float)):
    """Computes the power of one value to another.

    Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
    corresponding elements in `x` and `y`. For example:

    ```python
    x = to_tensor([[2, 2], [3, 3]])
    y = to_tensor([[8, 16], [2, 3]])
    pow(x, y)  # [[256, 65536], [9, 27]]
    ```

    Args:
        x (Tensor): input tensor.
        y (Tensor): another tensor.

    Returns:
      A `Tensor`.

    """

    y = to_tensor(y, dtype=x.dtype, device=x.device)
    return jnp.power(x, y)


@numpy_compatible
def log(x: Tensor)->Tensor:
    """Computes natural logarithm of x element-wise.

    I.e., \\(y = \\log_e x\\).

    See: https://en.wikipedia.org/wiki/Logarithm

    Args:
        x (Tensor): input tensor.


    Returns:
        A `Tensor`. Has the same type as `x`.

    Examples:
        >>> x = to_tensor([0, 0.5, 1, 5])
        >>> log(x)
        Array([-inf, -6.9315e-01, 0.0000e+00, 1.6094e+00], dtype=float32)




    """
    return jnp.log(x)


@numpy_compatible
def exp(x: Tensor)->Tensor:
    """Computes exponential of x element-wise.  \\(y = e^x\\).

    This function computes the exponential of the input tensor element-wise.
    i.e. `math.exp(x)` or \\(e^x\\), where `x` is the input tensor.
    \\(e\\) denotes Euler's number and is approximately equal to 2.718281.
    Output is positive for any real input.

    >>> x = to_tensor(2.0)
    >>> exp(x)
    Array(7.3891e+00, dtype=float32)

    >>> x = to_tensor([2.0, 8.0])
    >>> exp(x)
    Array([7.3891e+00, 2.9810e+03], dtype=float32)

    For complex numbers, the exponential value is calculated as
    \\(e^{x+iy}={e^x}{e^{iy}}={e^x}{\\cos(y)+i\\sin(y)}\\)

    For `1+1j` the value would be computed as:
    \\(e^1{\\cos(1)+i\\sin(1)} = 2.7182817 \\times (0.5403023+0.84147096j)\\)

    >>> x =to_tensor(1 + 1j)
    >>> exp(x)
    Array(1.4687+2.2874j, dtype=complex64)

    Args:
      x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`,
        `float32`, `float64`, `complex64`, `complex128`.


    Returns:
      A `Tensor`. Has the same type as `x`.

    @compatibility(numpy)
    Equivalent to np.exp
    @end_compatibility

    """
    return jnp.exp(x)


@numpy_compatible
def clip(x: Tensor, min=None, max=None):
    """

    Args:
        x (Tensor): input tensor.
        min ():
        max ():

    Returns:

    """
    return jnp.clip(x, a_min=min, a_max=max)


############################
# trigonometric functions
###########################


@numpy_compatible
def sin(x: Tensor)->Tensor:
    """Computes the element-wise sine

    Args:
        x (Tensor): input tensor.

    Returns: element-wise sine

    Examples:
        >>> sin(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 0.8415,  0.4794],
                [-0.2474, -0.6816]])

    """
    return jnp.sin(x)


@numpy_compatible
def cos(x: Tensor)->Tensor:
    """Computes the element-wise cosine

    Args:
        x (Tensor): input tensor.

    Returns: element-wise cosine

    Examples:
        >>> cos(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
         Array([[5.4030e-01, 8.7758e-01],
           [9.6891e-01, 7.3169e-01]], dtype=float32)

    """
    return jnp.cos(x)


@numpy_compatible
def tan(x: Tensor)->Tensor:
    """Computes the element-wise tan

    Args:
        x (Tensor): input tensor.

    Returns: element-wise tan

    Examples:
        >>> tan(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 1.5574,  0.5463],
                [-0.2553, -0.9316]])

    """
    return jnp.tan(x)


@numpy_compatible
def asin(x: Tensor)->Tensor:
    """Computes the element-wise arcsin (inverse sine)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arcsin

    Examples:
        >>> asin(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        Array([[1.5708e+00, 5.2360e-01],
               [-2.5268e-01, -8.4806e-01]], dtype=float32)

    """
    return jnp.arcsin(x)


@numpy_compatible
def acos(x: Tensor)->Tensor:
    """Computes the element-wise arccos (inverse cosine)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> acos(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        Array([[0.0000e+00, 1.0472e+00],
               [1.8235e+00, 2.4189e+00]], dtype=float32)

    """
    return jnp.arccos(x)


@numpy_compatible
def atan(x: Tensor)->Tensor:
    """Computes the element-wise arctan (inverse tan)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> atan(to_tensor([-1, 0, 1])).cpu()
        Array([-7.8540e-01, 0.0000e+00, 7.8540e-01], dtype=float32)

    """
    return jnp.arctan(x)


def atan2(x: Tensor, other: Tensor) -> Tensor:
    """Computes the element-wise arctangent (angles in radians between x and other )

    Args:
        x (Tensor): input tensor.
        other (Tensor): second input tensor.

    Returns:  the output tensor.

     Examples:
         >>> atan2(to_tensor([-1, 0, 1]), to_tensor([2, 4, 6])).cpu()
         Array([-4.6365e-01, 0.0000e+00, 1.6515e-01], dtype=float32)

    """
    return jnp.arctan(x / (other + 1e-6))


@numpy_compatible
def sinh(x: Tensor)->Tensor:
    """Computes the element-wise sinh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise sinh

    Examples:
        >>> sinh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 1.1752,  0.5211],
                [-0.2526, -0.8223]])

    """
    return jnp.sinh(x)


@numpy_compatible
def cosh(x: Tensor)->Tensor:
    """Computes the element-wise cosh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise cosh

    Examples:
        >>> cosh(to_tensor([[1,0.5],[-0.25,-0.75]]))
        Array([[1.5431e+00, 1.1276e+00],
           [1.0314e+00, 1.2947e+00]], dtype=float32)

    """
    return jnp.cosh(x)


@numpy_compatible
def tanh(x: Tensor)->Tensor:
    """Computes the element-wise tanh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise tanh

    Examples:
        >>> tanh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 0.7616,  0.4621],
                [-0.2449, -0.6351]])

    """
    return jnp.tanh(x)


@numpy_compatible
def asinh(x: Tensor)->Tensor:
    """Computes the element-wise asinh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise asinh

    Examples:
        >>> asinh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        Array([[8.8137e-01, 4.8121e-01],
               [-2.4747e-01, -6.9315e-01]], dtype=float32)

    """
    return jnp.arcsinh(x)


@numpy_compatible
def acosh(x: Tensor)->Tensor:
    """Computes the element-wise acosh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise acosh

    Examples:
        >>> acosh(to_tensor([[1.5431, 1.1276],[1.0314, 1.2947]])).cpu()
        Array([[1.0000e+00, 4.9995e-01],
               [2.4995e-01, 7.5002e-01]], dtype=float32)

    """
    return jnp.arccosh(x)


@numpy_compatible
def atanh(x: Tensor)->Tensor:
    """Computes the element-wise atanh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise atanh

    Examples:
        >>> atanh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        Array([[inf, 5.4931e-01],
           [-2.5541e-01, -9.7296e-01]], dtype=float32)

    """
    return jnp.arctanh(x)


@numpy_compatible
def where(flag, value_if_true=None, value_if_false=None):
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
    >>> x=to_tensor([0.1, 0.9, 0.8, 0.4, 0.5])
    >>> where(x>0.5, x, zeros_like(x))
    tensor([0.0000, 0.9000, 0.8000, 0.0000, 0.0000])
    """
    if value_if_true is None and value_if_false is None:
        return jnp.where(flag)
    else:
        return jnp.where(flag.bool(), value_if_true, value_if_false)


############################
# reduce operation
###########################
@numpy_compatible
def reduce_mean(x: Tensor, axis=None, keepdims=False, **kwargs):
    """Computes the mean of the input tensor's elements across a specified axis or a list of specified axes.

    Args:
        x (Tensor): input tensor.
        axis (int,list):  axis along which the reduction will be performed
        keepdims (bool): Keep the reduced dimension or not, default True mean keep reduced dimension
        **kwargs ():

    Returns:


    Examples:
        >>> data = to_tensor(np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32))
        >>> print(reduce_mean(data, 0).cpu())
        tensor([[30.,  1.],
                [40.,  2.]])
        >>> print(reduce_mean(data, axis=0).cpu())
        tensor([[30.,  1.],
                [40.,  2.]])
        >>> print(reduce_mean(data, axis=[0,2]).cpu())
        tensor([15.5000, 21.0000])

    """

    return jnp.mean(x, axis=axis, keepdims=keepdims)


@numpy_compatible
def reduce_sum(x: Tensor, axis=None, keepdims=False, **kwargs):
    """Computes the sum of the input tensor's elements across a specified axis or a list of specified axes.

    Args:
        x (Tensor): input tensor.
        axis (int,list):  axis along which the reduction will be performed
        keepdims (bool): Keep the reduced dimension or not, default True mean keep reduced dimension
        **kwargs ():

    Returns:
        The sum of the input tensor's elements across a specified axis or a list of specified axes.

    Examples:
        >>> data = to_tensor(np.array([[[5,1], [20,2]],[[30,1], [40,2]],[[55,1], [60,2]]], dtype=np.float32))
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

    return jnp.sum(x, axis=axis, keepdims=keepdims)


@numpy_compatible
def reduce_max(x: Tensor, axis=None, keepdims=False, **kwargs):
    """Computes the maximum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.


    See the numpy docs for `np.amax` and `np.nanmax` behavior.

    Args:
        x (Tensor): input tensor.
        axis: The dimensions to reduce. If `None` (the default), reduces all dimensions. Must be in the range `[-rank(input_tensor),rank(input_tensor)]`.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
      The reduced tensor.

    Examples:
        >>> x = to_tensor([5, 1, 2, 4])
        >>> print(reduce_max(x))
        tensor(5, shape=(), dtype=int32)
        >>> x = to_tensor([-5, -1, -2, -4])
        >>> print(reduce_max(x))
        tensor(-1, shape=(), dtype=int32)
        >>> x = to_tensor([4, float('nan')])
        >>> print(reduce_max(x))
        tensor(4.0, shape=(), dtype=float32)
        >>> x = to_tensor([float('nan'), float('nan')])
        >>> print(reduce_max(x))
        tensor(-inf, shape=(), dtype=float32)
        >>> x =to_tensor([float('-inf'), float('inf')])
        >>> print(reduce_max(x))
        tensor(inf, shape=(), dtype=float32)

    """

    return jnp.max(x, axis=axis, keepdims=keepdims)


@numpy_compatible
def reduce_min(x: Tensor, axis=None, keepdims=False, **kwargs):
    """Computes the minimum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.


    See the numpy docs for `np.amin` and `np.nanmin` behavior.

    Args:
        x (Tensor): input tensor.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `(-rank(input_tensor), rank(input_tensor))`.
      keepdims: If true, retains reduced dimensions with length 1.

    Returns:
      The reduced tensor.

    Examples:
        >>> x = to_tensor([5, 1, 2, 4])
        >>> print(reduce_min(x))
        tensor(5, shape=(), dtype=int32)
        >>> x = to_tensor([-5, -1, -2, -4])
        >>> print(reduce_min(x))
        tensor(-1, shape=(), dtype=int32)
        >>> x = to_tensor([4, float('nan')])
        >>> print(reduce_min(x))
        tensor(4.0, shape=(), dtype=float32)
        >>> x = to_tensor([float('nan'), float('nan')])
        >>> print(reduce_min(x))
        tensor(-inf, shape=(), dtype=float32)
        >>> x =to_tensor([float('-inf'), float('inf')])
        >>> print(reduce_min(x))
        tensor(inf, shape=(), dtype=float32)

    """

    return jnp.min(x, axis=axis, keepdims=keepdims)


@numpy_compatible
def reduce_std(x: Tensor, axis=None, keepdims=False, **kwargs):
    return jnp.std(x, axis=axis, keepdims=keepdims)


@numpy_compatible
def reduce_logsumexp(x: Tensor, axis=None, keepdims=False, **kwargs):
    return jnp.log(jnp.sum(jnp.exp(x), axis=axis, keepdims=keepdims))


@numpy_compatible
def reduce_prod(x: Tensor, axis=None, keepdims=False, **kwargs):
    """Computes the product of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.
    Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
    entry in `axis`. If `keepdims` is true, the reduced dimensions
    are retained with length 1.

    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
        x (Tensor): input tensor.
      axis: The dimensions to reduce. If `None` (the default), reduces all
        dimensions. Must be in the range `(-rank(input_tensor),rank(input_tensor))`.
      keepdims: If true, retains reduced dimensions with length 1.

    Returns:
      The reduced tensor.

    @compatibility(numpy)
    Equivalent to np.prod
    @end_compatibility
    """
    return jnp.prod(x, axis=axis, keepdims=keepdims)


@numpy_compatible
def reduce_any(x: Tensor, axis=None, keepdims=False, **kwargs):
    return jnp.any(xaxis=axis, keepdims=keepdims)


# reduce_log_sum_exp
# reduce_prod
# reduce_l1
# reduce_l2
# reduce_sum_square

mean = reduce_mean
sum = reduce_sum


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
    allargs = args + tuple(list(kwargs.values()))
    if len(allargs) == 1 and is_tensor(allargs[0]) and allargs[0].element_size() == 0:
        return allargs[0]
    elif len(allargs) == 1 and is_tensor(allargs[0]) and allargs[0].element_size() > 0:
        value, idx = allargs[0].max()
        return value
    elif len(allargs) > 1 and is_tensor(allargs[0]) and not is_tensor(allargs[1]) and (
            'axis' in kwargs or 'keepdims' in kwargs):
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)
        return reduce_max(allargs[0], axis=axis, keepdims=keepdims)
    elif len(args) > 1 and is_tensor(args[0]) and all(
            [is_tensor(arg) or isinstance(arg, (np.ndarray, float, int)) for arg in args]):
        new_args = [to_tensor(a) for a in args]
        return jnp.max(*new_args)
    else:
        raise NotImplementedError('Max({0},{1}) is not implemented yet '.format(*args, **kwargs))


@numpy_compatible
def min(*args, **kwargs):
    """

    Args:
        *args ():

    Returns:

    """
    allargs = args + tuple(list(kwargs.values()))
    if len(allargs) == 1 and is_tensor(allargs[0]) and allargs[0].element_size() == 0:
        return allargs[0]
    elif len(allargs) == 1 and is_tensor(allargs[0]) and allargs[0].element_size() > 0:
        return jnp.min(allargs[0])
    elif len(allargs) > 1 and is_tensor(allargs[0]) and not is_tensor(allargs[1]) and (
            'axis' in kwargs or 'keepdims' in kwargs):
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)
        return reduce_min(allargs[0], axis=axis, keepdims=keepdims)
    elif len(args) > 1 and is_tensor(args[0]) and all(
            [is_tensor(arg) or isinstance(arg, (np.ndarray, float, int)) for arg in args]):
        new_args = [to_tensor(a) for a in args]
        return jnp.min(*new_args)
    else:
        raise NotImplementedError('Min({0},{1}) is not implemented yet '.format(*args, **kwargs))


@numpy_compatible
def identity(x:Tensor)->Tensor:
    """Identity activation Layer
    A placeholder identity operator that is argument-insensitive.

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.

    Examples:
        >>> identity(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-3.0, -1.0, 0.0, 2.0])

    """
    return x


@numpy_compatible
def relu(x: Tensor)->Tensor:
    r"""Rectified linear unit activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{relu}(x) = \max(x, 0)

    Args:
      x : input array
    """
    return jax.nn.relu(x)


@numpy_compatible
def sigmoid(x: Tensor)->Tensor:
    return 0.5 * (jnp.tanh(x / 2.) + 1)


@numpy_compatible
def relu6(x:Tensor)->Tensor:
    """Rectified Linear Unit  6 activation function.
      With default values, it returns element-wise `min(max(x, 0)`,6).
      Otherwise, it follows:
      ```
        f(x) = 6 if x >= 6
        f(x) = x if threshold <= x < 6
        f(x) = negative_slope * (x - threshold) otherwise
      ```
    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jax.nn.relu6(x)


@numpy_compatible
def leaky_relu(x, slope=0.2):
    """Leaky version of a Rectified Linear Unit.
        It allows a small gradient when the unit is not active:
        ```
        f(x) = alpha * x if x < 0
        f(x) = x if x >= 0
        ```
    Args:
        slope ():
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jax.nn.leaky_relu(x, negative_slope=slope)


@numpy_compatible
def leaky_relu6(x, slope=0.2):
    """Leaky version of a Rectified Linear Unit.6
          It allows a small gradient when the unit is not active:
          ```
            f(x) = alpha * x if x < 0
            f(x) = x if  6>=x >= 0
            f(x) = 6 if  x > 6
          ```

    Args:
        slope ():
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jnp.clip(jax.nn.leaky_relu(x, negative_slope=slope), -6, 6)


@numpy_compatible
def smooth_relu(x:Tensor)->Tensor:
    """smooth_relu activation function


    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jnp.log(1 + jnp.exp(x))


@numpy_compatible
def celu(x, alpha: Tensor = 1.0):
    """Continuously-differentiable exponential linear unit activation.

     Computes the element-wise function:

     .. math::
       \\mathrm{celu}(x) = \\begin{cases}
         x, & x > 0\\
         \\alpha \\left(\\exp(\\frac{x}{\\alpha}) - 1\\right), & x \\le 0
       \\end{cases}

     For more information, see
     `Continuously Differentiable Exponential Linear Units
     <https://arxiv.org/pdf/1704.07483.pdf>`_.

     Args:
       x : input array
       alpha : array or scalar (default: 1.0)
     """
    return jnp.where(x > 0, x, alpha * jnp.expm1(x / alpha))


@numpy_compatible
def crelu(x, axis=-1):
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
    return jnp.concatenate([jax.nn.relu(x), jax.nn.relu(-x)], axis=axis)


@numpy_compatible
def elu(x, alpha=1):
    """ Exponential Linear Unit.
    It follows:

        f(x) =  alpha * (exp(x) - 1.)  for x < 0
        f(x) = x for x >= 0

    :math:`\\text{ELU}(x) = \\max(0,x) + \\min(0, \\alpha * (\\exp(x) - 1))`.

    Args:
        x (Tensor): input tensor.
        alpha (float):multiplier

    Returns:
        (Tensor): output tensor and get same shape with x.


    Examples:
        >>> elu(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00]

    """
    return jax.nn.elu(x, alpha=alpha)


def p_relu(x, weight):
    """Parametric Rectified Linear Unit.
      It follows:
      ```
        f(x) = alpha * x for x < 0
        f(x) = x for x >= 0
      ```
      where `alpha` is a learned parameters , it's a 1-D array, the length equal 1 or input_filters.

    Args:
        x (Tensor): input tensor.
        weight: (1 or None)  if None num_parameters will equal to input_filters .

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jax.nn.prelu(x, weight=weight)


@numpy_compatible
def sigmoid(x:Tensor)->Tensor:
    """sigmoid activation function

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jax.nn.sigmoid(x)


@numpy_compatible
def swish(x:Tensor)->Tensor:
    """Self-Gated Activation Function.
    it follows:
      ```
        f(x) =  x * sigmoid(x)

      ```
    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    References:
        Swish: a Self-Gated Activation Function
        https://arxiv.org/abs/1710.05941v1

    Examples:
        >>> swish(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00])

    """
    return x * jax.nn.sigmoid(x)


@numpy_compatible
def selu(x:Tensor)->Tensor:
    """
    selu activation function


    .. math::
            \\text{SELU}(x) = \\text{scale} * (\\max(0,x) + \\min(0, \\alpha * (\\exp(x) - 1)))

    with :math:`\\alpha = 1.6732632423543772848170429916717` and
    :math:`\\text{scale} = 1.0507009873554804934193349852946`.


    Scaled exponential linear unit operation. Computes the element-wise exponential linear
    of ``x``: ``scale * x`` for ``x >= 0`` and ``x``: ``scale * alpha * (exp(x)-1)`` otherwise.
    scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717

    References:
        paper: https://arxiv.org/abs/1706.02515
        Self-Normalizing Neural Networks
        Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    Examples:
        >>> selu(to_tensor([[-1, -0.5, 0, 1, 2]]))
        tensor([[-1.1113, -0.6918,  0.0000,  1.0507,  2.1014]])

    """
    return jax.nn.selu(x)


@numpy_compatible
def soft_sign(x:Tensor)->Tensor:
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return x.exp().add(1).log()


@numpy_compatible
def lecun_tanh(x:Tensor)->Tensor:
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return 1.7159 * jax.nn.tanh(2 / 3 * x)


@numpy_compatible
def hard_sigmoid(x:Tensor)->Tensor:
    """Hard sigmoid Activation Function.

    Memory saving version of sigmoid
    it follows:

        Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
        In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.



    Examples:
        >>> hard_sigmoid(to_tensor([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([0.0000, 0.3000, 0.5000, 0.9000])


    """

    return jnp.clip(x * 0.2 + 0.5, 0., 1.)


@numpy_compatible
def hard_swish(x:Tensor)->Tensor:
    """Hard swish Activation Function.

    Memory saving version of swish
    it follows:

        f(x) =  x * hard_sigmoid(x)


    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.

    Examples:
        >>> hard_swish(to_tensor([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([-0.0000, -0.3333,  0.0000,  1.6667])

    References:
        Searching for MobileNetV3
        https://arxiv.org/abs/1905.02244

    """
    return jnp.dot(x, hard_sigmoid(x))


@numpy_compatible
def hard_tanh(x:Tensor)->Tensor:
    """Hard Tanh Activation Function.

    Memory saving version of sigmoid
    it follows:

        f(x) =  clip(x, -1, 1)

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    Examples:
        >>> hard_tanh(to_tensor([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([-0.0000, -0.3333,  0.0000,  1.6667])


    """
    return jnp.clip(x, -1, 1)


@numpy_compatible
def soft_plus(x:Tensor)->Tensor:
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jax.nn.softplus(x)


@numpy_compatible
def square_plus(x:Tensor)->Tensor:
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return (x + jnp.sqrt(x ** 2 + 4)) / 2.0


@numpy_compatible
def logit(x:Tensor)->Tensor:
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jnp.log(x / (1 - x))


@numpy_compatible
def log_log(x:Tensor)->Tensor:
    """LogLog Activation Function

    it follows:

        f(x) =  1 - exp(-exp(x))

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    Examples:
        >>> log_log(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00])

    References:
        "Complementary Log-Log and Probit: Activation Functions Implemented in Artificial Neural Networks"
        https://ieeexplore.ieee.org/document/4626755/


    """
    return 1 - jnp.exp(-jnp.exp(x))


@numpy_compatible
def mish(x:Tensor)->Tensor:
    """mish activation function

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.



    Examples:
        >>> mish(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00])

    References:
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1

    """
    return x * (jax.nn.tanh(jax.nn.softplus(x)))


@numpy_compatible
def hard_mish(x:Tensor)->Tensor:
    """hard mish activation function

    it follows:

        f(x) =  x * hard_tanh(softplus(x))

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.



    Examples:
        >>> hard_mish(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00])

    References:
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1

    """
    return x * hard_tanh(jax.nn.softplus(x))


@numpy_compatible
def softmax(x, axis=-1,temperature=1):
    """
     Computes the gradient of :math:`f(z)=\\log\\sum_i\\exp(z_i)` at ``z = x``. Concretely,
     :math:`\\mathrm{softmax}(x)=\\left[\\frac{\\exp(x_1)}{\\sum_i\\exp(x_i)}\\quad\\frac{\\exp(x_1)}{\\sum_i\\exp(
     x_i)}\\quad\\ldots\\quad\\frac{\\exp(x_1)}{\\sum_i\\exp(x_i)}\\right]`
     with the understanding that the implementation can use equivalent formulas
     for efficiency and numerical stability.
     The output is a vector of non-negative numbers that sum to 1 and can
     therefore be interpreted as probabilities for mutually exclusive outcomes
     as in the case of multiclass classification.
     If ``axis`` is given as integer, then the softmax will be computed along that axis.
     If the provided ``axis`` is -1, it will be computed along the last axis. Otherwise,
     softmax will be applied to all axes.

     Args:
         x (Tensor): input tensor.
         axis (int,list):  axis along which the reduction will be performed
         temperature(float): Temperature

     Returns:
         (Tensor): output tensor and get same shape with x.


     Examples:
     >>> softmax(to_tensor([[1, 1, 2, 3]]))
     tensor([[0.0826, 0.0826, 0.2245, 0.6103]])
     >>> softmax(to_tensor([1., 1.]))
     tensor([0.5000, 0.5000])
     >>> softmax(to_tensor([[[1, 1], [3, 5]]]), axis=-1)
     tensor([[[0.5000, 0.5000],
              [0.1192, 0.8808]]])
     >>> softmax(to_tensor([[[1, 1], [3, 5]]]), axis=-1)
     tensor([[[0.1192, 0.0180],
              [0.8808, 0.9820]]])

     """
    return jax.nn.softmax(x/temperature, axis=axis)


@numpy_compatible
def log_softmax(x, axis=-1,temperature=1):
    """
     Computes the logsoftmax normalized values of x. That is, y = x - log(reduce_sum(exp(x), axis))
     (the implementation uses an equivalent formula for numerical stability).
     It is also possible to use `x - reduce_log_sum_exp(x, axis)` instead of log_softmax:
     this can be faster (one reduce pass instead of two), but can behave slightly differently numerically.

     Args:
         x (Tensor): input tensor.
         axis (int,list):  axis along which the reduction will be performed
         temperature(float): Temperature

     Returns:
         (Tensor): output tensor and get same shape with x.

     """
    return jax.nn.log_softmax(x/temperature, axis=axis)


def gelu(x:Tensor)->Tensor:
    """
    Gaussian Error Linear Unit.
    it follows:
        ```
        f(x) =x∗Φ(x)
        where \\Phi(x)Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

        ```
    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.

    References:
        Gaussian Error Linear Units (GELUs)
        https://arxiv.org/abs/1606.08415

    Examples:
        >>> gelu(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        <Tensor: shape=(4,), dtype=float32, numpy=array([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00], dtype=float32>

    """
    return jax.nn.gelu(x)


def gpt_gelu(x:Tensor)->Tensor:
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return 0.5 * x * (1 + jax.n.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))


def silu(x:Tensor)->Tensor:
    return jax.nn.silu(x)


############################
# normalization operation
###########################

def moments(x: Tensor, axis, keepdims=True):
    """

    Args:
        keepdims ():
        x (Tensor): input tensor.
        axis (int) :

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    _axes = list(axis)
    norm_mean = reduce_mean(x, axis=_axes, keepdims=keepdims).detach()
    norm_variance = reduce_mean(square(x - norm_mean), axis=_axes, keepdims=keepdims)
    return norm_mean, norm_variance


def norm(x: Tensor, order=None, axis=-1, keepdims=False):
    """

    Args:

        x (Tensor): The input tensor. If dim is None, x must be 1-D or 2-D, unless :attr:`ord`
            is None. If both :attr:`dim` and :attr:`ord` are None, the 2-norm of the input flattened to 1-D
            will be returned.

        order (int, float, inf, -inf, 'fro', 'nuc', optional): The order of norm.
            inf refers to :attr:`float('inf')`, numpy's :attr:`inf` object, or any equivalent object.
            The following norms can be calculated:

            =====  ============================  ==========================
            ord    norm for matrices             norm for vectors
            =====  ============================  ==========================
            None   Frobenius norm                2-norm
            'fro'  Frobenius norm                -- not supported --
            'nuc'  nuclear norm                  -- not supported --
            inf    max(sum(abs(x), dim=1))       max(abs(x))
            -inf   min(sum(abs(x), dim=1))       min(abs(x))
            0      -- not supported --           sum(x != 0)
            1      max(sum(abs(x), dim=0))       as below
            -1     min(sum(abs(x), dim=0))       as below
            2      2-norm (the largest sing. value)  as below
            -2     smallest singular value       as below
            other  -- not supported --           sum(abs(x)**ord)**(1./ord)
            =====  ============================  ==========================

            Default: ``None``

        axis (int, 2-tuple of ints, 2-list of ints, optional): If :attr:`dim` is an int,
            vector norm will be calculated over the specified dimension. If :attr:`dim`
            is a 2-tuple of ints, matrix norm will be calculated over the specified
            dimensions. If :attr:`dim` is None, matrix norm will be calculated
            when the input tensor has two dimensions, and vector norm will be
            calculated when the input tensor has one dimension. Default: ``None``

        keepdims (bool, optional): If set to True, the reduced dimensions are retained
            in the result as dimensions with size one. Default: ``False``


    Returns:

    """
    if ndim(x) == 1:
        axis = 0
    # if pt_version >= version1_7:
    #     return  jnp.linalg.norm(x, ord=order,dim=axis, keepdim=keepdims)
    # else:
    return x.norm(p=order, dim=axis, keepdim=keepdims)


@numpy_compatible
def l2_normalize(x: Tensor, axis=-1, keepdims=True, eps=epsilon()):
    """

    Args:
        eps ():
        keepdims ():
        axis ():
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.

    Examples:
        >>> a=to_tensor(np.arange(9)-4.0)
        >>> b=a.reshape((3, 3))
        >>> l2_normalize(a)

        >>> l2_normalize(b)


    """
    return x / jnp.sqrt(jnp.maximum(jnp.sum(x ** 2, axis=axis, keepdims=keepdims), eps))


# @numpy_compatible
# def spectral_norm(module, n_iterations=1, axis=-1):
#     return nn.utils.spectral_norm(module, n_power_iterations=n_iterations, dim=axis)


############################
# tensor shape operation
###########################


def broadcast_to(x: Tensor, shape: Union[(List, Tuple, jnp.shape, TensorShape)] = None) -> Tensor:
    if shape is None:
        return x
    elif isinstance(shape, TensorShape):
        shape = shape.dims
    if len(shape) > 2 and int_shape(x)[-1] != shape[-1] and int_shape(x)[-1] == shape[1]:
        shape = to_list(shape)
        new_shape = shape[0:1] + shape[2:] + shape[1:2]
        x = jnp.broadcast_to(x, new_shape)
        return x.transpose(-1, 1)
    else:
        return jnp.broadcast_to(x, shape)


def expand_as(left: Tensor, right: Tensor) -> Tensor:
    left_shape = int_shape(left)
    right_shape = int_shape(right)

    if len(right_shape) > 2 and left_shape[-1] != right_shape[-1] and left_shape[-1] == right_shape[1]:
        new_shape = right_shape[0:1] + right_shape[2:] + right_shape[1:2]
        left = left.expand(new_shape)
        return left.transpose(-1, 1)
    else:
        return left.expand_as(right)


@numpy_compatible
def reshape(x: Tensor, shape: Union[(List, Tuple, jnp.shape, TensorShape)] = None) -> Tensor:
    """

    Args:
        x (Tensor): input tensor.
        shape ():

    Returns:

    """
    if shape is None:
        return x
    elif isinstance(shape, TensorShape):
        shape = shape.dims

    return jnp.reshape(x, shape)


@numpy_compatible
def transpose(x, dim0: int, dim1: int) -> Tensor:
    """

    Args:
        x (Tensor): input tensor.
        dim0 (int):the first dimension to be transposed
        dim1 (int):the second dimension to be transposed

    Returns:
        transposed tensor

    """
    x = jnp.transpose(x, dim0, dim1)
    if not x.is_contiguous():
        return x.contiguous()
    return x


@numpy_compatible
def permute(x, *dims) -> Tensor:
    """

    Args:
        x (Tensor): input tensor.
        dims ():

    Returns:

    """
    x = jnp.permute(x, *dims)
    if not x.is_contiguous():
        return x.contiguous()
    return x


@numpy_compatible
def squeeze(x: Tensor, axis=None):
    """

    Args:
        x (Tensor): input tensor.
        axis ():

    Returns:

    """
    return x.squeeze(dim=axis)


@numpy_compatible
def expand_dims(x: Tensor, axis: Union[int, Sequence[int]]):
    """

    Args:
        x (Tensor): input tensor.
        axis ():

    Returns:

    """
    return jnp.expand_dims(x,axis=axis)


@numpy_compatible
def depth_to_space(x: Tensor, block_size=2):
    """
    Rearranges elements in the input tensor from the depth dimension into spatial blocks.
    The equivalent to Pixel-Shuffle

    Args:
        x (Tensor): input tensor.
        block_size (int):

    Returns: resized tensor

    Examples:
    >>> x = to_tensor(np.tile(np.array(np.reshape(range(8), (1, 1, 8)), dtype=np.float32), (2, 3, 1)))
    >>> arr=depth_to_space(x,block_size=2)
    >>> print(arr.shape)
    (4, 6, 2)
    >>> arr
    Array([[[0.0000e+00, 1.0000e+00],
            [2.0000e+00, 3.0000e+00],
            [0.0000e+00, 1.0000e+00],
            [2.0000e+00, 3.0000e+00],
            [0.0000e+00, 1.0000e+00],
            [2.0000e+00, 3.0000e+00]],
    <BLANKLINE>
           [[4.0000e+00, 5.0000e+00],
            [6.0000e+00, 7.0000e+00],
            [4.0000e+00, 5.0000e+00],
            [6.0000e+00, 7.0000e+00],
            [4.0000e+00, 5.0000e+00],
            [6.0000e+00, 7.0000e+00]],
    <BLANKLINE>
           [[0.0000e+00, 1.0000e+00],
            [2.0000e+00, 3.0000e+00],
            [0.0000e+00, 1.0000e+00],
            [2.0000e+00, 3.0000e+00],
            [0.0000e+00, 1.0000e+00],
            [2.0000e+00, 3.0000e+00]],
    <BLANKLINE>
           [[4.0000e+00, 5.0000e+00],
            [6.0000e+00, 7.0000e+00],
            [4.0000e+00, 5.0000e+00],
            [6.0000e+00, 7.0000e+00],
            [4.0000e+00, 5.0000e+00],
            [6.0000e+00, 7.0000e+00]]], dtype=float32)

    """
    if x.ndim not in (3, 4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    if x.ndim == 4:  # Batched case.
        return jax.vmap(depth_to_space, in_axes=(0, None))(x, block_size)

    height, width, depth = x.shape
    if depth % (block_size ** 2) != 0:
        raise ValueError(
            f'Number of channels {depth} must be divisible by block_size ** 2 {block_size ** 2}.'
        )
    new_depth = depth // (block_size ** 2)
    outputs = jnp.reshape(x, [height, width, block_size, block_size, new_depth])
    outputs = jnp.transpose(outputs, [0, 2, 1, 3, 4])
    outputs = jnp.reshape(outputs, [height * block_size, width * block_size, new_depth])
    return outputs


@numpy_compatible
def space_to_depth(x: Tensor, block_size=2):
    """
    Rearranges elements in the input tensor from the spatial dimensions to the depth dimension.

    This is the reverse transformation of depth_to_space. This operation is useful for implementing and testing
    sub-pixel convolution that is part of models for image super-resolution .
    It rearranges elements of an input tensor of shape (N, C, H, W) to a tensor of shape (N, C*b*b, H/b, W/b),
    where b is the block_size,
    by rearranging non-overlapping spatial blocks of size block_size x block_size into the depth/channel dimension at
    each location.

    Args:
        x (Tensor): input tensor.
        block_size (int):

    Returns: resized tensor
    Examples:
    >>> arr=space_to_depth(to_tensor([[[0., 1., 0., 1., 0., 1.],[2., 3., 2., 3., 2., 3.],[0., 1., 0., 1., 0., 1.],
    [2., 3., 2., 3., 2., 3.]],[[4., 5., 4., 5., 4., 5.],[6., 7., 6., 7., 6., 7.], [4., 5., 4., 5., 4., 5.],[6., 7.,
    6., 7., 6., 7.]]]),block_size=2)
    >>> arr
    tensor([[[0., 0., 0.],
             [0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1.],
             [1., 1., 1.]],
    <BLANKLINE>
            [[2., 2., 2.],
             [2., 2., 2.]],
    <BLANKLINE>
            [[3., 3., 3.],
             [3., 3., 3.]],
    <BLANKLINE>
            [[4., 4., 4.],
             [4., 4., 4.]],
    <BLANKLINE>
            [[5., 5., 5.],
             [5., 5., 5.]],
    <BLANKLINE>
            [[6., 6., 6.],
             [6., 6., 6.]],
    <BLANKLINE>
            [[7., 7., 7.],
             [7., 7., 7.]]])
    >>> print(arr.shape)
     jnp.shape([8, 2, 3])
    """
    if x.ndim not in (3, 4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    if x.ndim == 4:  # Batched case.
        return jax.vmap(space_to_depth, in_axes=(0, None))(x, block_size)

    height, width, depth = x.shape
    if height % block_size != 0:
        raise ValueError(
            f'Height {height} must be divisible by block size {block_size}.')
    if width % block_size != 0:
        raise ValueError(
            f'Width {width} must be divisible by block size {block_size}.')
    new_depth = depth * (block_size ** 2)
    new_height = height // block_size
    new_width = width // block_size
    outputs = jnp.reshape(x,
                          [new_height, block_size, new_width, block_size, depth])
    outputs = jnp.transpose(outputs, [0, 2, 1, 3, 4])
    outputs = jnp.reshape(outputs, [new_height, new_width, new_depth])
    return outputs


def pad(x: Tensor, paddings: Sequence[int], mode='constant', value=0):
    """Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\\left\\lfloor\\frac{\\text{len(pad)}}{2}\\right\\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding\\_left}, \\text{padding\\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding\\_left}, \\text{padding\\_right},`
        :math:`\text{padding\\_top}, \\text{padding\\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding\\_left}, \\text{padding\\_right},`
        :math:`\\text{padding\\_top}, \\text{padding\\_bottom}`
        :math:`\\text{padding\\_front}, \\text{padding\\_back})`.

    Padding mode:
        See :class:` jnp.nn.ConstantPad2d`, :class:` jnp.nn.ReflectionPad2d`, and
        :class:` jnp.nn.ReplicationPad2d` for concrete examples on how each of the
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
        paddings ():
        x (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\\frac{m}{2} \\leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Examples::

        >>> t4d =  jnp.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
         jnp.shape([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
         jnp.shape([3, 3, 8, 4])
        >>> t4d =  jnp.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
         jnp.shape([3, 9, 7, 3])

    """
    valid_items = ['constant', 'reflect', 'replicate', 'circular', 'symmetric', 'zero']
    if mode not in valid_items:
        raise ValueError('{0} is not valid for mode.'.format(mode))
    if mode == 'zero':
        mode = 'constant'
        value = 0
    if mode == 'symmetric':
        mode = 'circular'
    return jnp.pad(x, pad_width=paddings, mode=mode, value=value)


############################
# tensor generation
###########################

def ones(shape: Union[(List, Tuple, jnp.shape, TensorShape)], dtype=None, requires_grad=True):
    """Instantiates an all-ones tensor and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether we need gradient

    Returns
        A tensor, filled with `1.0`.

    Example
        >>> ones((3,4))
        Array([[1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00],
           [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00],
           [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00]], dtype=float32)

    """
    if isinstance(shape, TensorShape):
        shape = shape.dims
    if dtype is None:
        dtype = _float_dtype
    x = jnp.ones(shape, dtype=dtype).to(get_session_value('device'))
    if not requires_grad:
        x = jax.lax.stop_gradient(x)
    return x


@numpy_compatible
def ones_like(a, dtype=None, requires_grad=True):
    """Instantiates an all-ones variable of the same shape as another tensor.

    Args
        a (Tensor):  another tensor
        dtype (String):  data type
        requires_grad (bool):  whether we need gradient

    Returns
        A tensor, filled with `1.0` and shape is the same as another tensor.

    Example
        >>> ones_like( jnp.randn((3,4)))
        Array([[1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00],
           [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00],
           [1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00]], dtype=float32)

    {{np_implementation}}
    """
    if dtype is None:
        dtype = a.dtype
    x = jnp.ones(a.shape, dtype=dtype)
    if not requires_grad:
        x = jax.lax.stop_gradient(x)
    return x


def zeros(shape: Union[(List, Tuple, jnp.shape, TensorShape)], dtype=None, requires_grad=True):
    """Instantiates an all-zeros tensor and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether we need gradient

    Returns
        A tensor, filled with `0.0`.

    Example
        >>> zeros((3,4))
        tensor([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)

    {{np_implementation}}
    """
    if isinstance(shape, TensorShape):
        shape = shape.dims
    if dtype is None:
        dtype = _float_dtype
    x = jnp.zeros(shape, dtype=dtype).to(get_session_value('device'))
    if not requires_grad:
        x = jax.lax.stop_gradient(x)
    return x


@numpy_compatible
def zeros_like(a, dtype=None, requires_grad=True):
    """Instantiates an all-zeros variable of the same shape as another tensor.

    Args
        a (Tensor):  another tensor
        dtype (String):  data type
        requires_grad (bool):  whether we need gradient

    Returns
        A tensor, filled with `0.0` and shape is the same as another tensor.

    Example
        >>> zeros_like(random_normal((3,4)))
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0.]], dtype=float32)

    """
    if dtype is None:
        dtype = a.dtype
    x = jnp.zeros(a.shape, dtype=dtype).to(get_session_value('device'))
    if not requires_grad:
        x = jax.lax.stop_gradient(x)
    return x


def eye(shape: Union[(List, Tuple, jnp.shape, TensorShape)], dtype=None, requires_grad=None):
    """Instantiate an identity matrix and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether we need gradient

    Returns
        an identity matrix.

    Examples:
        >>> eye((3,4))
        tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]])

    """
    if isinstance(shape, TensorShape):
        shape = shape.dims
    if dtype is None:
        dtype = _float_dtype

    if len(shape) == 2:
        return jnp.eye(shape[0], shape[1], dtype=dtype).to(get_session_value('device'))
    else:
        raise ValueError('input tensor must have exactly two axe.')


@numpy_compatible
def eye_like(a, dtype=None, requires_grad=True):
    """
    Creates a matrix with diagonal set to 1s and of the same shape and the same dynamic axes as ``x``. To be a
    matrix, ``x`` must have exactly two axes (counting both dynamic and static axes).

    Args:
        a (Tensor):  another tensor of rank 2
        dtype (String):  data type
        requires_grad (bool):  whether we need gradient

    Returns
        an identity matrix.

    Examples:
    >>> eye_like(Tensor(3,4))
    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]])

    """
    if dtype is None:
        dtype = a.dtype
    if a.ndim == 2:
        return jnp.eye(a.shape[0], a.shape[1], dtype=dtype).to(
            get_session_value('device'))
    else:
        raise ValueError('input tensor must have exactly two axe.')


@numpy_compatible
def make_onehot(label, num_classes, axis=-1):
    """
    Create one hot tensor based on the input tensor
    Args:
        label: input tensor, the value must be positive integer and less than num_class
        num_classes: the number of class in one hot tensor
        axis: The axis to fill (default: -1, a new innermost axis).
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

    onehot = jnp.nn.functional.one_hot(label.long(), num_classes).to(_float_dtype)
    last_index = ndim(onehot) - 1
    if axis < 0:
        axis += ndim(onehot)
    if axis is None or axis in [-1, last_index]:
        return onehot
    else:

        axes = list(range(len(onehot.shape)))
        axes.pop(-1)
        axes.insert(axis, -1)

        onehot = onehot.permute(axes)
        return onehot


def arange(*args, dtype=Dtype.int32, requires_grad=True):
    """

    Args:
        *args (int): the start, end, step
        dtype (dtype): dtype of the tensor
        requires_grad (bool): whether we need gradient

    Returns:

    """
    if len(args) == 1:
        return jnp.arange(end=args[0], dtype=dtype, requires_grad=requires_grad).to(get_session_value('device'))
    elif len(args) == 2:
        return jnp.arange(start=args[0], end=args[1], dtype=dtype, requires_grad=requires_grad).to(
            get_session_value('device'))
    elif len(args) == 3:
        return jnp.arange(start=args[0], end=args[1], step=args[2], dtype=dtype, requires_grad=requires_grad).to(
            get_session_value('device'))
    else:
        raise ValueError('only maximum  3 args in arange function ')


def meshgrid(x, y, normalized_coordinates=False, requires_grad=True):
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
     jnp.shape([3, 2, 2])


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
     jnp.shape([3, 2, 2])
    """
    xs = jnp.linspace(0, int(x - 1), int(x), device=get_session_value('device'), dtype=_float_dtype,
                      requires_grad=requires_grad)
    ys = jnp.linspace(0, int(y - 1), int(y), device=get_session_value('device'), dtype=_float_dtype,
                      requires_grad=requires_grad)
    if normalized_coordinates:
        xs = jnp.linspace(0, 1, int(x), device=get_session_value('device'), dtype=_float_dtype,
                          requires_grad=requires_grad)
        ys = jnp.linspace(0, 1, int(y), device=get_session_value('device'), dtype=_float_dtype,
                          requires_grad=requires_grad)
    grid_x, grid_y = jnp.meshgrid([xs, ys])

    grid = jnp.stack([grid_y, grid_x], -1).to(get_session_value('device'))
    return grid


@numpy_compatible
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
    return jnp.flip(x, dims=axis)


############################
# tensor manipulation
###########################

def concate(x: List[Tensor], axis=-1):
    """

    Args:
        x ():
        axis ():

    Returns:

    """
    return jnp.cat(x, dim=axis)


def stack(x: List[Tensor], axis=-1):
    """

    Args:
        x ():
        axis ():

    Returns:

    """
    return jnp.stack(x, dim=axis)


def split(x: Tensor, num_splits=2, axis=-1):
    """Splits a tensor `value` into a list of sub tensors.

      See also `unstack`.

      If `num_or_size_splits` is an integer,  then `value` is split along the
      dimension `axis` into `num_or_size_splits` smaller tensors. This requires that
      `value.shape[axis]` is divisible by `num_or_size_splits`.

      If `num_or_size_splits` is a 1-D Tensor (or list), then `value` is split into
      `len(num_or_size_splits)` elements. The shape of the `i`-th
      element has the same size as the `value` except along dimension `axis` where
      the size is `num_or_size_splits[i]`.

      For example:

      >>> x = to_tensor(np.random.uniform([5, 30]))
      >>> s0, s1, s2 = split(x, num_splits=3, axis=-1)
      >>> int_shape(s0)
      array([ 5, 10], dtype=int32)
      >>> split0, split1, split2 = split(x, [4, 15, 11], 1)
      >>> int_shape(split0)
      array([5, 4], dtype=int32)
      >>> int_shape(split1)
      array([ 5, 15], dtype=int32)
      >>> int_shape(split2)
      array([ 5, 11], dtype=int32)

      Args:
        x: The `Tensor` to split.
        num_splits: Either an integer indicating the number of splits along
          `axis` or a 1-D integer `Tensor` or Python list containing the sizes of
          each output tensor along `axis`. If a scalar, then it must evenly divide
          `value.shape[axis]`; otherwise the sum of sizes along the split axis
          must match that of the `value`.
        axis: An integer or scalar `int32` `Tensor`. The dimension along which to
          split. Must be in the range `[-rank(value), rank(value)]`. Defaults to 0.


      Returns:
        if `num_or_size_splits` is a scalar returns a list of `num_or_size_splits`
        `Tensor` objects; if `num_or_size_splits` is a 1-D Tensor returns
        `num_or_size_splits.get_shape[0]` `Tensor` objects resulting from splitting
        `value`.

      Raises:
        ValueError: If `num` is unspecified and cannot be inferred.
      """

    return jnp.chunk(x, dim=axis, chunks=num_splits)


# def make_onehot(label, num_classes, axis=-1):
#     """
#     Create a one-hot encoding of x of size k.
#
#     x: array
#         The array to be one hot encoded
#     k: interger
#         The number of classes
#     dtype: jnp.dtype, optional(default=float32)
#         The dtype to be used on the encoding
#     Examples:
#     >>> make_onehot(jnp.array([[1, 2],[1, 3]],dtype=jnp.int64), 4, axis=-1)
#     tensor([[[0., 1., 1., 0.],
#              [0., 1., 0., 1.]],
#     <BLANKLINE>
#             [[0., 0., 0., 0.],
#              [0., 0., 0., 0.]]])
#
#     """
#     onehot = jnp.array(label[:, None] == jnp.arange(num_classes), dtype=jnp.float32)
#     if axis != -1:
#         axes = np.arange(ndim(label))
#         axes[axis] = ndim(label) - 1
#         axes[-1] = axis
#         onehot = jnp.transpose(axes)
#     return onehot

@numpy_compatible
def binary_cross_entropy(output, target, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
      Args:
          target: A tensor with the same shape as `output`.
          output: A tensor.
          from_logits: Whether `output` is expected to be a logits tensor.
              By default, we consider that `output`
              encodes a probability distribution.
      Returns:
          A tensor.
      """
    if from_logits:
        pass
    else:
        output = sigmoid(output)
    if isinstance(target,(_float,_int )):
        target=(ones_like(output)*float(target)).to(output.dtype).to(output.device)
    bce = target *  jnp.log(jnp.clip(output,1e-7,1))
    bce =bce+ (1 - target) * jnp.log(jnp.clip(1 - output,1e-7,1))
    return -bce


def bbox_iou(bboxes1: Tensor, bboxes2: Tensor):
    """

    Args:
        bboxes1 (Tensor): shape (n, 4)
        bboxes2 (Tensor): shape (n, 4)

    Returns:
         ious(Tensor): shape (n)

    Examples;
    >>> boxes1=np.array([[39, 63, 203, 112], [49, 75, 203, 125],[31, 69, 201, 125],[50, 72, 197, 121],[35, 51, 196, 110]])
    >>> boxes2=np.array([[54, 66, 198, 114], [42, 78, 186, 126], [18, 63, 235, 135],[54, 72, 198, 120],[36, 60, 180, 108]])
    >>> bbox_iou(boxes1,boxes2)
    DeviceArray([0.79577124, 0.787838  , 0.609319  , 0.9466281 , 0.72765553],            dtype=float32)
    >>> iou_loss=(1-bbox_iou(boxes1,boxes2)).sum()/(boxes1.shape[0])
    >>> print(iou_loss)
    0.22655764
    """

    b1_y1, b1_x1, b1_y2, b1_x2 = jnp.split(bboxes1, 4, axis=-1)
    b2_y1, b2_x1, b2_y2, b2_x2 = jnp.split(bboxes2, 4, axis=-1)

    y1 = jnp.maximum(b1_y1, b2_y1)
    x1 = jnp.maximum(b1_x1, b2_x1)
    y2 = jnp.minimum(b1_y2, b2_y2)
    x2 = jnp.minimum(b1_x2, b2_x2)

    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    intersection = jnp.maximum(x2 - x1, 0) * jnp.maximum(y2 - y1, 0)

    union = b1_area + b2_area - intersection
    iou = intersection / union
    iou = jnp.squeeze(iou, -1)

    return iou.astype('float32')


def bbox_diou(bboxes1, bboxes2):
    """

    Args:
        bboxes1 (Tensor): shape (n, 4)
        bboxes2 (Tensor): shape (n, 4)

    Returns:
         ious(Tensor): shape (n)

    Examples;
    >>> boxes1=jnp.array([[39, 63, 203, 112], [49, 75, 203, 125],[31, 69, 201, 125],[50, 72, 197, 121],[35, 51, 196, 110]])
    >>> boxes2=jnp.array([[54, 66, 198, 114], [42, 78, 186, 126], [18, 63, 235, 135],[54, 72, 198, 120],[36, 60, 180, 108]])
    >>> bbox_diou(boxes1,boxes2)
    Array([7.9471e-01, 7.8265e-01, 6.0713e-01, 9.4636e-01, 7.2533e-01],      dtype=float32)
    >>> iou_loss=(1-bbox_diou(boxes1,boxes2)).sum()/(boxes1.shape[0])
    >>> print(iou_loss)
    Array(2.2876e-01, dtype=float32)

    """

    x1, y1, x2, y2  = jnp.split(bboxes1, 4, axis=-1)
    x1g, y1g, x2g, y2g = jnp.split(bboxes2, 4, axis=-1)
    x2 = jnp.maximum(x1, x2)
    y2 = jnp.maximum(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = jnp.maximum(x1, x1g)
    ykis1 = jnp.maximum(y1, y1g)
    xkis2 = jnp.minimum(x2, x2g)
    ykis2 = jnp.minimum(y2, y2g)

    xc1 = jnp.minimum(x1, x1g)
    yc1 = jnp.minimum(y1, y1g)
    xc2 = jnp.maximum(x2, x2g)
    yc2 = jnp.maximum(y2, y2g)

    #intsctk = jnp.zeros_like(x1,dtype=x1.dtype)
    mask = jnp.array((ykis2 > ykis1)*(xkis2 > xkis1), jnp.float32)
    intsctk = ((xkis2- xkis1) * (ykis2 - ykis1))*mask
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk =jnp.squeeze(iouk - u,axis=-1)
    return diouk


_FUN_NAMES = [
    # source_fun, target_fun
    ('to_numpy', to_numpy),
    ('to', to),
    ('cuda', cuda),
    ('cpu', cpu),
    ('numel', numel),
    ('ndim', ndim),
    ('int_shape', int_shape),
    ('cast', cast),
    ('logical_and', logical_and),
    ('logical_or', logical_or),
    ('logical_xor', logical_xor),
    ('logical_not', logical_not),
    ('less', less),
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
    ('square_plus', square_plus),
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
    ('reverse', reverse),
    ('reshape', reshape),
    ('transpose', transpose),
    ('squeeze', squeeze),
    ('expand_dims', expand_dims)
]

for target_fun_name, source_fun in _FUN_NAMES:
    if not hasattr(Tensor, target_fun_name):
        setattr(Tensor, target_fun_name, source_fun)
    elif not hasattr(Tensor, target_fun_name):
        setattr(Tensor, target_fun_name, source_fun)
    elif target_fun_name in ["to", "float", "int", "long", "sum", "mean"]:
        setattr(Tensor, target_fun_name, source_fun)
del _FUN_NAMES
