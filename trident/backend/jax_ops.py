from typing import List, Tuple, Optional, Union, Callable, Any, Iterable, Iterator, Mapping, TypeVar, overload
from functools import partial, wraps
import numpy as np
import numbers
import collections
import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
from trident.backend.common import to_list, unpack_singleton, epsilon, OrderedDict, get_function, get_session, \
    TensorShape
from trident.backend import dtype as Dtype
from trident import context

jax.config.update('jax_array', True)
Tensor = jax.Array

ctx = context._context()


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
                len(kwargs) == 0 or all([isinstance(kv[1], numbers.Number) for kv in kwargs.items()])) and get_function(
            func.__name__, ['math', 'numpy', 'trident.backend.numpy_ops']) is not None:
            mathfuncs = get_function(func.__name__, ['math', 'numpy', 'trident.backend.numpy_ops'])
            y = mathfuncs(*args, **kwargs)
            return y
        # elif isinstance(x, list) and all([isinstance(arg, np.ndarray) for arg in x]) and func.__name__ in ['concate','stack','vstack','hstack']:
        #     numpy_func = get_function(func.__name__, ['trident.backend.numpy_ops','numpy'])
        #     y = numpy_func(*args, **kwargs)
        #     return y
        # elif isinstance(x, list) and all([isinstance(arg, Tensor) for arg in x])  and func.__name__ in ['concate','stack','vstack','hstack']:
        #     tensor_func = get_function(func.__name__, ['trident.backend.pytorch_ops'])
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
    devices = jax.devices()
    if len(devices) == 1:
        return False
    for device in devices[1:]:
        if device.platform == "gpu":
            return True
    return False


def is_tpu_available():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        return True
    except:
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
#         #     tensor_func = get_function(func.__name__, ['trident.backend.pytorch_ops'])
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

def is_tensor(x):
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


def is_tensor_like(x):
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
    elif isinstance(x, (list, tuple, numbers.Number)):
        return np.array(x)
    elif isinstance(x, (list, tuple)):
        return np.asarray(x)
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
        <jax.Tensor: shape=(), dtype=int64, numpy=2>
        >>> to_tensor([1.0,2.0,3.0],requires_grad=True)
        <jax.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor([1.0,2.0,3.0],requires_grad=False)
        <jax.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor([1.0,2.0,3.0])
        <jax.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor((1.0,2.0,3.0))
        <jax.Tensor: shape=(3,), dtype=float32, numpy=array([1.0000e+00, 2.0000e+00, 3.0000e+00], dtype=float32)>
        >>> to_tensor(np.arange(0,5))
        <jax.Tensor: shape=(5,), dtype=int64, numpy=array([0, 1, 2, 3, 4], dtype=int64)>

    """
    if x is None:
        return x
    if device is not None and ('cuda' in device.lower() or 'gpu' in device.lower()):
        device = '/gpu:0'
    else:
        device = "/cpu:0"
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
        if x is not None:
            if input_dtype is None:
                dtype = x.dtype
            else:
                x = x.type(dtype)
            with jax.device(device):
                return jax.identity(x)
        else:
            return None
    elif isinstance(x, np.ndarray):
        npdtype = x.dtype

        if 'int' in str(npdtype):
            with jax.device(device):
                x = jax.convert_to_tensor(x, dtype=jax.int64)
        else:
            with jax.device(device):
                x = jax.convert_to_tensor(x, dtype=jax.float32)
        return x
    else:
        with jax.device(device):
            return jax.convert_to_tensor(x, dtype=dtype)


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
#             return cast(x,tf.float32)
#         elif 'long' in args:
#             return cast(x,tf.int64)
#         elif 'int' in args:
#             return cast(x,tf.int32)
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
    torch.float16

    """
    if isinstance(dtype_str, jnp.dtype):
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
        >>>cast(x, jax.int32)
        <tensor=array([1, 2], dtype=int32)>

    Raises:
        TypeError: If `x` cannot be cast to the `dtype`.

    """

    cast_dtype = str2dtype(cast_dtype)
    if isinstance(x, (jnp.ndarray, jax.xla.DeviceArray)) and isinstance(cast_dtype, jnp.dtype):
        return x.astype(cast_dtype)
    elif isinstance(x, np.ndarray) and isinstance(cast_dtype, np.dtype):
        return x.astype(cast_dtype)
    else:
        return x


@numpy_compatible
def ndim(x: Tensor):
    """Number of dimension of a tensor

    Args:
        x (Tensor): input tensor.

    Returns:
        (int) Number of dimension

    """
    return len(int_shape(x))


def numel(x: Tensor):
    """Number of elements of a tensor

    Args:
        x (Tensor): input tensor.

    Returns:
        (int) Number of elements

    """
    return len(x.reshape(-1))


def int_shape(x: Tensor):
    """Shape of a tensor in tuple of integer format

    Args:
        x : input tensor

    Returns:
        tuple of integer as shape representation

    Examples:
        >>> int_shape(ones((3,3,7)))
        (3, 3, 7)

    """

    if x is None:
        return []
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
## compare operation
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


def less_equal(left: Tensor, right: (np.Tensor, float, int)):
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
    return jnp.less_equal(left, right).astype(np.float32)


def argmax(x: Tensor, axis=1) -> Tensor:
    return jnp.argmax(x, axis=axis)


def argmin(x: Tensor, axis=1) -> Tensor:
    return jnp.argmin(x, axis=axis)


def argsort(x: np.ndarray, axis=1, descending=True) -> np.ndarray:
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
## basic math operation
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
def round(x: (Tensor, float), digit: int = 0):
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
def prod(x: Tensor):
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
def sqrt(x: Tensor):
    r"""Computes element-wise square root of the input tensor.

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
def rsqrt(x: Tensor):
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
def square(x: Tensor):
    r"""Computes square of x element-wise.

    I.e., \\(y = x * x = x^2\\).

    >>> math.square([-2., 0., 3.])
    <Tensor: shape=(3,), dtype=float32, numpy=array([4., 0., 9.], dtype=float32>

    Args:
      x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`,
      `complex64`, `complex128`.


    Returns:
      A `Tensor`. Has the same type as `x`.


    """
    return x ** 2


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

    >>> x = to_tensor([[-2.25 + 4.75j], [-3.25 + 5.75j]])
    >>> abs(x)
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
    return jnp.abs(x)


@numpy_compatible
def pow(x: Tensor, y: (Tensor, float)):
    r"""Computes the power of one value to another.

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
def log(x: Tensor):
    r"""Computes natural logarithm of x element-wise.

    I.e., \\(y = \log_e x\\).

    See: https://en.wikipedia.org/wiki/Logarithm

    Args:
        x (Tensor): input tensor.


    Returns:
        A `Tensor`. Has the same type as `x`.

    Examples:
        >>> x = to_tensor([0, 0.5, 1, 5])
        >>> log(x)
        array([      -inf, -0.6931472,  0.,  1.609438 ])




    """
    return jnp.log(x)


@numpy_compatible
def exp(x: Tensor):
    r"""Computes exponential of x element-wise.  \\(y = e^x\\).

    This function computes the exponential of the input tensor element-wise.
    i.e. `math.exp(x)` or \\(e^x\\), where `x` is the input tensor.
    \\(e\\) denotes Euler's number and is approximately equal to 2.718281.
    Output is positive for any real input.

    >>> x = to_tensor(2.0)
    >>> exp(x)
    <Tensor: shape=(), dtype=float32, numpy=7.389056>

    >>> x = to_tensor([2.0, 8.0])
    >>> exp(x)
    tensor([   7.389056, 2980.958   ])

    For complex numbers, the exponential value is calculated as
    \\(e^{x+iy}={e^x}{e^{iy}}={e^x}{\\cos(y)+i\\sin(y)}\\)

    For `1+1j` the value would be computed as:
    \\(e^1{\\cos(1)+i\\sin(1)} = 2.7182817 \\times (0.5403023+0.84147096j)\\)

    >>> x =to_tensor(1 + 1j)
    >>> exp(x)
    tensor(1.4686939399158851+2.2873552871788423j)>

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


@numpy_compatible
def sin(x: Tensor):
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
def cos(x: Tensor):
    """Computes the element-wise cosine

    Args:
        x (Tensor): input tensor.

    Returns: element-wise cosine

    Examples:
        >>> cos(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[0.5403, 0.8776],
                [0.9689, 0.7317]])

    """
    return jnp.cos(x)


@numpy_compatible
def tan(x: Tensor):
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
def asin(x: Tensor):
    """Computes the element-wise arcsin (inverse sine)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arcsin

    Examples:
        >>> asin(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 1.5708,  0.5236],
                [-0.2527, -0.8481]])

    """
    return jnp.asin(x)


@numpy_compatible
def acos(x: Tensor):
    """Computes the element-wise arccos (inverse cosine)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> acos(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[0.0000, 1.0472],
                [1.8235, 2.4189]])

    """
    return jnp.acos(x)


@numpy_compatible
def atan(x: Tensor):
    """Computes the element-wise arctan (inverse tan)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> atan(to_tensor([-1, 0, 1])).cpu()
        tensor([-0.7854,  0.0000,  0.7854])

    """
    return jnp.atan(x)


@numpy_compatible
def sinh(x: Tensor):
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
def cosh(x: Tensor):
    """Computes the element-wise cosh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise cosh

    Examples:
        >>> cosh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[1.5431, 1.1276],
                [1.0314, 1.2947]])

    """
    return jnp.cosh(x)


@numpy_compatible
def tanh(x: Tensor):
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
## reduce operation
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
def identity(x):
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
def relu(x: Tensor):
    return jax.nn.relu(x)


@numpy_compatible
def sigmoid(x: Tensor):
    return 0.5 * (jnp.tanh(x / 2.) + 1)


@numpy_compatible
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
def smooth_relu(x):
    """smooth_relu activation function


    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jnp.log(1 + jnp.exp(x))


@numpy_compatible
def celu(x, alpha: Tensor = 1.0):
    r"""Continuously-differentiable exponential linear unit activation.

     Computes the element-wise function:

     .. math::
       \mathrm{celu}(x) = \begin{cases}
         x, & x > 0\\
         \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
       \end{cases}

     For more information, see
     `Continuously Differentiable Exponential Linear Units
     <https://arxiv.org/pdf/1704.07483.pdf>`_.

     Args:
       x : input array
       alpha : array or scalar (default: 1.0)
     """
    return jnp.where(x > 0, x, alpha * jnp.expm1(x / alpha))


@numpy_compatible
def crelu(x, axis=1):
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

        f(x) =  alpha * (exp(x) - 1.) for x < 0
        f(x) = x for x >= 0

    :math:`\text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))`.

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
def sigmoid(x):
    """softmax activation function

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jax.nn.sigmoid(x)


@numpy_compatible
def swish(x):
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
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    Examples:
        >>> selu(to_tensor([[-1, -0.5, 0, 1, 2]]))
        tensor([[-1.1113, -0.6918,  0.0000,  1.0507,  2.1014]])

    """
    return jax.nn.selu(x)


@numpy_compatible
def soft_sign(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return x.exp().add(1).log()


@numpy_compatible
def lecun_tanh(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return 1.7159 * jax.nn.tanh(2 / 3 * x)


@numpy_compatible
def hard_sigmoid(x):
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
def hard_swish(x):
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
    return x * hard_sigmoid(x)


@numpy_compatible
def hard_tanh(x):
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
def soft_plus(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jax.nn.softplus(x)


@numpy_compatible
def square_plus(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return (x + jnp.sqrt(x ** 2 + 4)) / 2.0


@numpy_compatible
def logit(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return jnp.log(x / (1 - x))


@numpy_compatible
def log_log(x):
    """LogLog Activation Function

    it follows:

        f(x) =  1 - exp(-exp(x))

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    Examples:
        >>> loglog(to_tensor([-3.0, -1.0, 0.0, 2.0]))
        tensor([-1.4228e-01, -2.6894e-01, 0.0000e+00, 1.7616e+00])

    References:
        "Complementary Log-Log and Probit: Activation Functions Implemented in Artificial Neural Networks"
        https://ieeexplore.ieee.org/document/4626755/


    """
    return 1 - jnp.exp(-jnp.exp(x))


@numpy_compatible
def mish(x):
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
def hard_mish(x):
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
        x (Tensor): input tensor.
        axis (int,list):  axis along which the reduction will be performed

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
    >>> softmax(to_tensor([[[1, 1], [3, 5]]]), axis=1)
    tensor([[[0.1192, 0.0180],
             [0.8808, 0.9820]]])

    """
    return jax.nn.softmax(x, dim=axis)


@numpy_compatible
def log_softmax(x, axis=1):
    """
    Computes the logsoftmax normalized values of x. That is, y = x - log(reduce_sum(exp(x), axis))
    (the implementation uses an equivalent formula for numerical stability).
    It is also possible to use `x - reduce_log_sum_exp(x, axis)` instead of log_softmax:
    this can be faster (one reduce pass instead of two), but can behave slightly differently numerically.

    Args:
        x (Tensor): input tensor.
        axis (int,list):  axis along which the reduction will be performed

    Returns:
        (Tensor): output tensor and get same shape with x.

    """
    return x - reduce_logsumexp(x, axis=axis, keepdims=True)


def gelu(x):
    """
    Gaussian Error Linear Unit.
    it follows:
        ```
        f(x) =x∗Φ(x)
        where \Phi(x)Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.

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


def gpt_gelu(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return 0.5 * x * (1 + jax.n.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))))


def silu(x):
    return jax.nn.silu(x)


def make_onehot(label, num_classes, axis=-1):
    """
    Create a one-hot encoding of x of size k.

    x: array
        The array to be one hot encoded
    k: interger
        The number of classes
    dtype: jnp.dtype, optional(default=float32)
        The dtype to be used on the encoding
    Examples:
    >>> make_onehot(jnp.array([[1, 2],[1, 3]],dtype=jnp.int64), 4, axis=-1)
    tensor([[[0., 1., 1., 0.],
             [0., 1., 0., 1.]],
    <BLANKLINE>
            [[0., 0., 0., 0.],
             [0., 0., 0., 0.]]])

    """
    onehot = jnp.array(label[:, None] == jnp.arange(num_classes), dtype=jnp.float32)
    if axis != -1:
        axes = np.arange(ndim(label))
        axes[axis] = ndim(label) - 1
        axes[-1] = axis
        onehot = jnp.transpose(axes)
    return onehot


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

    b1_y1, b1_x1, b1_y2, b1_x2 = jnp.split(bboxes1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = jnp.split(bboxes2, 4, axis=1)

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
    >>> boxes1=np.array([[39, 63, 203, 112], [49, 75, 203, 125],[31, 69, 201, 125],[50, 72, 197, 121],[35, 51, 196, 110]])
    >>> boxes2=np.array([[54, 66, 198, 114], [42, 78, 186, 126], [18, 63, 235, 135],[54, 72, 198, 120],[36, 60, 180, 108]])
    >>> bbox_diou(boxes1,boxes2)
    DeviceArray([7.9471e-01, 7.8265e-01, 6.0713e-01, 9.4636e-01, 7.2533e-01],            dtype=float32)
    >>> iou_loss=(1-bbox_diou(boxes1,boxes2)).sum()/(boxes1.shape[0])
    >>> print(iou_loss)
    0.2287639

    """

    b1_y1, b1_x1, b1_y2, b1_x2 = jnp.split(bboxes1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = jnp.split(bboxes2, 4, axis=1)
    y1 = jnp.maximum(b1_y1, b2_y1)
    x1 = jnp.maximum(b1_x1, b2_x1)
    y2 = jnp.minimum(b1_y2, b2_y2)
    x2 = jnp.minimum(b1_x2, b2_x2)

    out_max_xy = jnp.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = jnp.minimum(bboxes1[:, :2], bboxes2[:, :2])
    c_h = jnp.maximum(out_max_xy[:, 0] - out_min_xy[:, 0], 0)
    c_w = jnp.maximum(out_max_xy[:, 1] - out_min_xy[:, 1], 0)

    center_x1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_y1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_x2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
    center_y2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2

    p2 = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    p2 = jnp.expand_dims(p2, axis=-1)

    c2 = c_w ** 2 + c_h ** 2
    c2 = jnp.expand_dims(c2, axis=-1)

    intersection = jnp.maximum(x2 - x1, 0) * jnp.maximum(y2 - y1, 0)
    # Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    diou = intersection / union - p2 / c2
    diou = jnp.squeeze(diou, -1)
    return diou


_FUN_NAMES = [
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
    ('tanh', tanh)
]
for target_fun_name, source_fun in _FUN_NAMES:
    if not hasattr(Tensor, target_fun_name):
        setattr(Tensor, target_fun_name, source_fun)
    elif target_fun_name in ["to", "float", "int", "long", "sum", "mean"]:
        setattr(Tensor, target_fun_name, source_fun)
del _FUN_NAMES
