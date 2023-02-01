from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TRIDENT_BACKEND'] = 'pytorch'

import builtins
import collections
import gc
import math
import numbers
import random
from collections import Sized, Iterable
from distutils.version import LooseVersion
from functools import wraps
from typing import Tuple, List, Optional, Union, Sequence, Any
from torch.types import _int, _float, _bool, Number, _dtype, Device
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.ops as visionop
from trident.backend import dtype as Dtype
from trident.backend.common import *
# from trident.backend.numpy_ops import DTYPE_MAPPING as numpy_DTYPE_MAPPING
from trident import context

# from trident.backend.tensorspec import TensorShape

version = torch.__version__
pt_version = LooseVersion(vstring=version)
version1_7 = LooseVersion(vstring='1.7.0')
ctx = context._context()


def is_gpu_available():
    return torch.cuda.is_available()


def is_tpu_available():
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        return True
    except:
        return False


def _get_device():
    """get current device

    Returns: device string ('cpu', 'cuda)

    """
    if get_session().device is None:
        _set_device("cuda" if torch.cuda.is_available() else 'xpu' if is_tpu_available() else "cpu")
    return get_session().device


def _set_device(device='cpu'):
    device = device.lower().replace('gpu', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Gpu is not available...')
    if device == 'xpu' and not is_tpu_available():
        raise ValueError('Tpu is not available...')
    try:
        device_ = device
        if device == 'xpu':
            import torch_xla.core.xla_model as xm
            device_ = xm.xla_device()
        set_session('device', device_)

        gcitems = gc.get_objects()
        for i in range(len(gcitems)):
            obj = gcitems[i]
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    obj.to(device_)
                elif isinstance(obj, nn.Module):
                    obj.to(device_)
            except Exception:
                pass
    except Exception as e:
        print(e)



_float_dtype = Dtype.float16 if ctx.amp_available == True and ctx.is_autocast_enabled == True and _get_device() == 'cuda' else Dtype.float32

Tensor = torch.Tensor

################################################################################
# Define numeric constants
################################################################################

# For Python Array API (https://data-apis.org/array-api/latest/API_specification/constants.html) and
# NumPy consistency (https://numpy.org/devdocs/reference/constants.html)



__all__ = ['Tensor', 'is_gpu_available', 'is_tpu_available', 'is_tensor', 'is_tensor_like', 'to_numpy', 'to_tensor',
           'to_scalar', 'ndim', 'numel', 'cast', 'str2dtype', 'int_shape', 'tensor_to_shape', 'is_sparse', 'is_nan',
           'is_inf',
           'is_abnormal_number', 'any_nan', 'any_inf', 'any_abnormal_number', 'logical_and', 'logical_or',
           'logical_xor', 'logical_not', 'less', 'equal', 'greater',
           'greater_equal', 'not_equal', 'less_equal', 'argmax', 'argmin', 'argsort', 'topk', 'maximum', 'minimum',
           'floor',
           'ceil', 'round', 'dot', 'sqrt', 'rsqrt', 'prod', 'square', 'abs', 'pow', 'log', 'exp', 'clip', 'add',
           'subtract',
           'true_divide', 'pi', 'matmul', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
           'element_times', 'element_max', 'element_min', 'element_divide', 'element_cosine_distance', 'where',
           'reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_min', 'mean', 'sum', 'max', 'min', 'reduce_logsumexp',
           'reduce_prod', 'reduce_any', 'depth_to_space', 'space_to_depth', 'pad', 'identity', 'sigmoid', 'relu',
           'relu6', 'leaky_relu', 'celu',
           'leaky_relu6', 'smooth_relu', 'crelu', 'p_relu', 'swish', 'elu', 'hard_sigmoid', 'hard_swish', 'selu',
           'silu', 'lecun_tanh',
           'soft_sign', 'soft_plus', 'square_plus', 'hard_tanh', 'logit', 'log_log', 'mish', 'hard_mish', 'softmax',
           'log_softmax', 'gelu', 'reverse',
           'gpt_gelu', 'moments', 'norm', 'l2_normalize', 'spectral_norm', 'broadcast_to', 'expand_as', 'ones',
           'ones_like', 'zeros', 'zeros_like', 'eye', 'eye_like', 'make_onehot', 'arange', 'meshgrid', 'reshape',
           'permute', 'transpose', 'squeeze', 'expand_dims', 'concate', 'stack', 'split', 'repeat_elements', 'gather',
           'index_select', 'scatter_add', 'scatter_sub', 'scatter_max', 'scatter_min', 'gram_matrix', 'set_seed',
           'shuffle',
           'random_choice', 'random_normal', 'random_normal_like', 'random_uniform', 'random_uniform_like',
           'multinomial', 'random_bernoulli', 'binary_cross_entropy',
           'rgb2xyz', 'rgb2hsv', 'rgb2lab', 'rgb2gray', 'xyz2lab', 'xyz2rgb', 'lab2xyz', 'lab2rgb', 'xywh2xyxy',
           'xyxy2xywh', 'bbox_iou',
           'bbox_giou', 'bbox_ciou', 'bbox_diou', 'nms']

from math import e , nan , inf , pi
__all__.extend(['e', 'pi', 'nan', 'inf'])


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


############################
# tensor attribute
###########################
def is_tensor(x) -> _bool:
    """Checks whether `x` is exactly a tensor .


    Args:
        x: A python object to check.

    Returns:
        'True` if `x` is exactly a tensor , `False` if not.

    Examples:
        >>> is_tensor(torch.tensor([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor(Parameter(to_tensor([[1,2,3],[4,5,6],[7,8,9]])))
        True
        >>> is_tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        False
        >>> is_tensor("Hello World")
        False`

    """
    return isinstance(x, torch.Tensor)


def is_tensor_like(x) -> _bool:
    """Checks whether `x` is a  tensor or "tensor-like".

    If `is_tensor_like(x)` returns `True`, it is safe to assume that `x` is a tensor or
    can be converted to a tensor`.

    Args:
        x: A python object to check.

    Returns:
        `True` if `x` is a tensor or "tensor-like", `False` if not.

    Examples:
        >>> is_tensor_like(torch.tensor([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor_like([[1,2,3],[4,5,6],[7,8,9]])
        True
        >>> is_tensor_like(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        True
        >>> is_tensor_like("Hello World")
        False

    """
    return is_tensor(x) or isinstance(x, np.ndarray)


def to_numpy(*x) -> np.ndarray:
    """Convert whatever to numpy array

    Args:
        x (List, tuple, tensor or numpy array): whatever you want to convert to numpy ndarray.

    Returns:
        a numpy ndarray

    Examples:
        >>> to_numpy(5)
        array([5])
        >>> to_numpy([1,2,3])
        array([1, 2, 3])
        >>> to_numpy((2,4),(1,3))
        array([[2, 4],
           [1, 3]])

    """
    x = unpack_singleton(x)
    if isinstance(x, np.ndarray):
        return x
    elif x is None:
        return None
    elif isinstance(x, TensorShape):
        return np.array(x.dims)
    elif isinstance(x, Tensor):
        if isinstance(x, torch.autograd.Variable):
            x = x.data
        return x.clone().cpu().detach().numpy()
    elif isinstance(x, (list, tuple)):
        x = [to_numpy(item) if is_tensor(item) else item for item in x]
        return np.stack(x, 0)

    elif isinstance(x, bool):
        return np.array([int(x)]).astype(np.bool)
    elif isinstance(x, numbers.Number):
        return np.asarray([x])
    else:
        raise ValueError("Unsupported type: {0}".format(x.__class__.__name__))


def to_tensor(x: Any, dtype: Optional[_dtype] = None, device: Device = None, requires_grad: _bool = None) -> Tensor:
    """ Convert input  to a tensor as possible

    Args:
        x: An object to be converted (ex.numpy array, list, tensors).
        dtype (str or torch.dtype):
        device (str or torch.device):
        requires_grad (None or bool): whether we need grade

    Returns:
        A tensor.

    Examples:
        >>> to_tensor(2)
        tensor(2)
        >>> to_tensor([1.0,2.0,3.0],requires_grad=True)
        tensor([1., 2., 3.], requires_grad=True)
        >>> to_tensor([1.0,2.0,3.0],requires_grad=False)
        tensor([1., 2., 3.])
        >>> to_tensor([1.0,2.0,3.0])
        tensor([1., 2., 3.])
        >>> to_tensor((1.0,2.0,3.0))
        tensor([1., 2., 3.])
        >>> to_tensor(np.arange(0,5))
        tensor([0, 1, 2, 3, 4])

    """
    if is_tpu_available() and (device == 'xpu' or device is None):
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    # input_dtype = dtype
    if dtype is None and isinstance(x, numbers.Integral):
        dtype = Dtype.int64
    # string data to tensor

    elif isinstance(x, np.ndarray):
        dtype = str2dtype(str(x.dtype).replace('numpy', 'Dtype'))

    elif isinstance(x, Tensor) and 'float' not in str(x.dtype):
        dtype = x.dtype
    elif isinstance(x, Tensor) and 'float' in str(x.dtype):
        dtype = _float_dtype
    elif dtype is None and not is_tensor(x) and isinstance(x, collections.Iterable) and all(
            [isinstance(item, numbers.Integral) for item in x]):
        dtype = Dtype.int64
    elif dtype is None:
        dtype = _float_dtype
    elif isinstance(dtype, str):
        dtype = str2dtype(dtype)
    if device is None:
        device = _get_device()

    if isinstance(x, Tensor):
        if x is not None:
            # dtype =None
            if dtype is None:
                dtype = x.dtype
            else:
                if dtype == Dtype.int32:
                    dtype = Dtype.int64
                x = x.type(dtype)
            x = x.to(device)
            if isinstance(requires_grad, bool) and requires_grad != x.requires_grad:
                x.requires_grad = requires_grad
            return x.to(dtype)
        return None
    else:
        if x is None:
            return None
        elif isinstance(x, str):
            return torch.ByteTensor(list(bytes(x, 'utf8'))).to(device)
        elif isinstance(x, np.ndarray) and x.dtype == np.string_:
            return torch.ByteTensor(x).to(device)
        elif isinstance(x, list) and all([isinstance(s, str) for s in x]):
            max_len = builtins.max([len(s) for s in x])
            w_t = torch.zeros((len(x), max_len), dtype=torch.uint8)
            for i, ts in enumerate(x):
                w_t[i, 0:ts.size()[0]] = ts
            return w_t.to(device)

        elif isinstance(x, numbers.Integral):
            if dtype is None:
                dtype = Dtype.int64

            return torch.tensor(x, dtype=dtype).to(device) if requires_grad is None else torch.tensor(x,
                                                                                                      dtype=dtype,
                                                                                                      requires_grad=requires_grad).to(
                device)

        elif isinstance(x, float):
            if dtype is None:
                dtype = _float_dtype
            return torch.tensor(x, dtype=dtype).to(device) if requires_grad is None else torch.tensor(x,
                                                                                                      dtype=dtype,
                                                                                                      requires_grad=requires_grad).to(
                device)
        elif isinstance(x, (list, tuple)):
            if all([isinstance(item, numbers.Integral) for item in x]):
                if dtype is None:
                    dtype = Dtype.int64
                x = torch.tensor(x, dtype=dtype).to(device) if requires_grad is None else torch.tensor(x,
                                                                                                       dtype=dtype,
                                                                                                       requires_grad=requires_grad).to(
                    device)
            elif len(x) == 1:
                x = unpack_singleton(x)
                if dtype is None:
                    dtype = _float_dtype
                x = torch.tensor(x, dtype=dtype).to(device) if requires_grad is None else torch.tensor(x,
                                                                                                       dtype=dtype,
                                                                                                       requires_grad=requires_grad).to(
                    device)
            else:
                if dtype is None:
                    dtype = _float_dtype
                x = torch.tensor(x, dtype=dtype).to(device) if requires_grad is None else torch.tensor(x, dtype=dtype,
                                                                                                       requires_grad=requires_grad).to(
                    device)
            x = x.to(device)
            return x
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
            if dtype is None:
                dtype = _float_dtype
            x = x.type(dtype)
            if not requires_grad:
                x.requires_grad = False
            elif requires_grad:
                x.requires_grad = True
            return x
        else:
            return x


def to_scalar(x: Any) -> Optional[Union[_int, _float]]:
    if x is None:
        return None
    elif is_tensor(x):
        x = x.squeeze()
        if ndim(x) == 0:
            return x.item()
        elif ndim(x) == 1:
            return x[-1].item()
        else:
            return x.mean().item()
    elif isinstance(x, np.ndarray):
        x = np.squeeze(x)
        if len(x.shape) == 0:
            return x.item()
        elif len(x.shape) == 1:
            return x[-1].item()
        else:
            return x.mean().item()
    elif isinstance(x, numbers.Number):
        return x
    elif isinstance(x, (list, tuple)) and len(x) > 0:
        return to_scalar(x[-1])
    else:
        return None


def copy(x: Tensor) -> Tensor:
    return x.clone()


@numpy_compatible
def ndim(x: Tensor) -> _int:
    """Number of dimension of a tensor

    Args:
        x (Tensor): input tensor.

    Returns:
        (int) Number of dimension

    """
    return len(int_shape(x))


def numel(x: Tensor) -> _int:
    """Number of elements of a tensor

    Args:
        x (Tensor): input tensor.

    Returns:
        (int) Number of elements

    """
    return x.numel()


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
    elif isinstance(x, torch.Size):
        return [d for d in x]
    elif hasattr(x, 'shape'):
        return [d for d in x.shape]  # if isinstance(x,np.ndarray)  else  [d for d in  x.size()]
    else:
        return []


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
            pass
        else:
            shp[0] = None
        return TensorShape(shp)
    elif need_exclude_batch_axis and is_singleton == True:
        return TensorShape([None] + int_shape(x))
    else:
        return TensorShape(int_shape(x))


def is_sparse(x):
    """ Check whether the tensor is sparse

    Args:
        x (Tensor): input tensor.

    Returns:
        if input tensor is sparse return True, else False.

    """
    return is_tensor(x) and 'sparse' in str(type(x))


def str2dtype(dtype_str: (str, torch.dtype)):
    """ Mapping string to dtype

    Args:
        dtype_str (str): dtype representation string

    Returns:
        dtype

    >>> str2dtype('float16')
    torch.float16

    """
    if isinstance(dtype_str, torch.dtype):
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


def cast(x, cast_dtype: (str, torch.dtype)):
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
        >>>cast(x, torch.int32)
        tensor([1, 2], dtype=int32)

    Raises:
        TypeError: If `x` cannot be cast to the `dtype`.

    """
    cast_dtype = str2dtype(cast_dtype)
    if isinstance(cast_dtype, torch.dtype):
        if cast_dtype == Dtype.float64 or cast_dtype == Dtype.double:
            return x.double()
        elif cast_dtype == Dtype.float16 or cast_dtype == Dtype.half:
            return x.half()
        elif cast_dtype == Dtype.float32:
            return x.float()
        elif cast_dtype == Dtype.int64:
            return x.long()
        elif cast_dtype == Dtype.int32:
            return x.int()
        elif cast_dtype == Dtype.int16:
            return x.short()
        elif cast_dtype == Dtype.int8:
            return x.char()
        elif cast_dtype == Dtype.uint8:
            return x.byte()
        elif cast_dtype == Dtype.bool:
            return x.bool()
        else:
            return x


def to(x, *args, **kwargs):
    if len(kwargs) == 0 and all([isinstance(arg, str) for arg in args]):
        for arg in args:
            if isinstance(arg, str):
                if 'cpu' in arg:
                    x = x.cpu()
                elif 'gpu' in arg or 'cuda' in arg:
                    if is_gpu_available():
                        x = x.cuda()
                    else:
                        x = x.cpu()
                elif 'tpu' in arg or 'xpu' in arg:
                    if is_tpu_available():
                        x = x.xpu()
                    else:
                        x = x.cpu()
                else:
                    try:
                        x = cast(x, str2dtype(arg))
                    except:
                        x = torch._C._TensorBase.to(x, arg)
    else:
        x = torch._C._TensorBase.to(x, *args, **kwargs)
    return x


############################
# check operation
###########################
@numpy_compatible
def is_nan(x)->_bool:
    """

    Args:
        x (Tensor): input tensor.

    Returns:

    """
    if isinstance(x, Tensor):
        return torch.isnan(x)
    elif isinstance(x, nn.Module):
        return [torch.isnan(para) for para in x.parameters()]
    elif isinstance(x, np.ndarray):
        return np.isnan(x)
    elif isinstance(x, numbers.Number):
        return math.isnan(x)
    else:
        raise NotImplementedError


@numpy_compatible
def is_inf(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:

    """
    if isinstance(x, Tensor):
        return torch.isinf(x)
    elif isinstance(x, nn.Module):
        return [torch.isinf(para) for para in x.parameters()]
    elif isinstance(x, np.ndarray):
        return np.isinf(x)
    elif isinstance(x, numbers.Number):
        return math.isinf(x)
    else:
        raise NotImplementedError


def is_abnormal_number(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:

    """
    return is_nan(x) | is_inf(x)


def any_nan(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:

    """
    if isinstance(x, Tensor):
        return torch.isnan(x).any()
    elif isinstance(x, nn.Module):
        for para in x.parameters():
            if torch.isnan(para).any():
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isnan(x).any()
    elif isinstance(x, numbers.Number):
        return math.isnan(x)
    else:
        raise NotImplementedError


@numpy_compatible
def any_inf(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:

    """
    if isinstance(x, Tensor):
        return torch.isinf(x).any()
    elif isinstance(x, nn.Module):
        for para in x.parameters():
            if torch.isinf(para).any():
                return True
        return False
    elif isinstance(x, np.ndarray):
        return np.isinf(x).any()
    elif isinstance(x, numbers.Number):
        return math.isinf(x)
    else:
        raise NotImplementedError


@numpy_compatible
def any_abnormal_number(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:

    """
    with torch.no_grad():
        return any_nan(x) | any_inf(x)


############################
# logical  operation
###########################

@numpy_compatible
def logical_and(left, right):
    """Element-wise `logical and: x && y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor

    Returns:
        A Tensor of type bool with the same size as that of left or right.

    """
    return torch.logical_and(left, right)


@numpy_compatible
def logical_not(x: Tensor):
    """Element-wise `logical not: ~x`
    Args:
        x (Tensor): input boolean tensor
    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return torch.logical_not(x)


@numpy_compatible
def logical_or(left, right):
    """Element-wise `logical or: x || y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor
    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return torch.logical_or(left, right)


@numpy_compatible
def logical_xor(left, right):
    """Element-wise `logical xor: x ^ y`.
    Args:
        left (Tensor): input boolean tensor
        right (Tensor): input boolean tensor

    Returns:
        A Tensor of type bool with the same size as that of x .
    """
    return torch.logical_xor(left, right)


############################
# comparison  operation
###########################

@numpy_compatible
def less(left: Tensor, right: Union[Tensor, np.ndarray, numbers.Number], dtype=None):
    """
    Elementwise 'less' comparison of two tensors. Result is 1 if left < right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        dtype (dtype): output tensor dtype.
    Returns:
        Result is 1 if left < right else 0.

    Examples:
       >>> less(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
       tensor([1., 0., 0.])
       >>> less(to_tensor([-1,0,1]), 0)
       tensor([1., 0., 0.])

    """
    if dtype is None:
        dtype = Dtype.bool
    if not is_tensor(left):
        left = to_tensor(left)
    right = to_tensor(right, dtype=left.dtype, device=left.device)
    return left.lt(right).to(dtype)


@numpy_compatible
def equal(left: Tensor, right: Union[Tensor, np.ndarray, numbers.Number], dtype=Dtype.bool):
    """
    Elementwise 'equal' comparison of two tensors. Result is 1 if values are equal 0 otherwise.
    Args:
        left: left side tensor
        right: right side tensor
        dtype (dtype): output tensor dtype.
    Returns:
        :Result is 1 if values are equal 0 otherwise

    Examples:
        >>> equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.])).cpu()
        tensor([False,  True, False])
        >>> equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.])).sum().cpu()
        tensor(1.)
        >>> reduce_mean(equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))).cpu()
        tensor(0.3333)
        >>> equal(to_tensor([-1,0,1]), 1).cpu()
        tensor([[False, False,  True]])
        >>> equal(to_tensor([1,2,3]), 3).cpu()
        tensor([[False, False,  True]])

    """
    if not is_tensor(left):
        left = to_tensor(left)
    right = to_tensor(right, dtype=left.dtype, device=left.device)
    return left.eq(right).to(dtype)


@numpy_compatible
def greater(left: Tensor, right: Union[Tensor, np.ndarray, numbers.Number], dtype=Dtype.bool):
    """
    Elementwise 'greater' comparison of two tensors. Result is 1 if left > right else 0.
    Args:
        left: left side tensor
        right: right side tensor
        dtype (dtype): output tensor dtype.
    Returns:
        :Result is 1 if left > right else 0.

    Examples:
        >>> greater(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        tensor([0., 0., 1.])
        >>> greater(to_tensor([-1,0,1]), 0)
        tensor([0., 0., 1.])

    """

    if not is_tensor(left):
        left = to_tensor(left)
    right = to_tensor(right, dtype=left.dtype, device=left.device)
    return left.gt(right).to(dtype)


@numpy_compatible
def greater_equal(left: Tensor, right: Union[Tensor, np.ndarray, numbers.Number], dtype=Dtype.bool):
    """
    Elementwise 'greater equal' comparison of two tensors. Result is 1 if left >= right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        dtype (dtype): output tensor dtype.
    Returns:
        :Result is 1 if left >= right else 0

    Examples:
        >>> greater_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        tensor([0., 1., 1.])
        >>> greater_equal(to_tensor([-1,0,1]), 0)
        tensor([0., 1., 1.])

    """
    if not is_tensor(left):
        left = to_tensor(left)
    right = to_tensor(right, dtype=left.dtype, device=left.device)
    return left.ge(right).to(dtype)


@numpy_compatible
def not_equal(left: Tensor, right: Union[Tensor, np.ndarray, numbers.Number], dtype=Dtype.bool):
    """
    Elementwise 'not equal' comparison of two tensors. Result is 1 if left != right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        dtype (dtype): output tensor dtype.
    Returns:
        :Result is 1 if left != right else 0.

    Examples:
        >>> not_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        tensor([1., 0., 1.])
        >>> not_equal(to_tensor([-1,0,1]), 0)
        tensor([1., 0., 1.])

    """
    if not is_tensor(left):
        left = to_tensor(left)
    right = to_tensor(right, dtype=left.dtype, device=left.device) if not is_tensor(right) else right
    return left.ne(right).to(dtype)


@numpy_compatible
def less_equal(left: Tensor, right: Union[Tensor, np.ndarray, numbers.Number], dtype=Dtype.bool):
    """
    Elementwise 'less equal' comparison of two tensors. Result is 1 if left <= right else 0.

    Args:
        left: left side tensor
        right: right side tensor
        dtype (dtype): output tensor dtype.

    Returns:
        Result is 1 if left <= right else 0.
    Examples:
        >>> less_equal(to_tensor([41., 42., 43.]), to_tensor([42., 42., 42.]))
        tensor([1., 1., 0.])
        >>> less_equal(to_tensor([-1,0,1]), 0)
        tensor([1., 1., 0.])

    """
    if not is_tensor(left):
        left = to_tensor(left)
    right = to_tensor(right, dtype=left.dtype, device=left.device)
    return left.le(right).to(dtype)


@numpy_compatible
def argmax(x: Tensor, axis=1) -> Tensor:
    if ndim(x) == 1:
        axis = 0
    if len(x.shape) > axis:
        _, idx = x.max(dim=axis)
    else:
        _, idx = x.max()
    return idx


@numpy_compatible
def argmin(x: Tensor, axis=1) -> Tensor:
    if ndim(x) == 1:
        axis = 0
    if len(x.shape) > axis:
        _, idx = x.min(dim=axis)
    else:
        _, idx = x.min()
    return idx


@numpy_compatible
def argsort(x: Tensor, axis=1, descending=True) -> Tensor:
    if ndim(x) == 1:
        axis = 0
    return torch.argsort(x, dim=axis, descending=descending)


@numpy_compatible
def topk(x: Tensor, k=1) -> Tensor:
    axis = 1 if len(x.size()) > 1 else 0
    _, idx = torch.topk(x, k=k, dim=axis, largest=True, sorted=True)
    return idx


@numpy_compatible
def maximum(x: Tensor, other: (Tensor, int, float)) -> Tensor:
    if isinstance(other, Tensor):
        return torch.max(x, other)
    elif isinstance(other, (int, float)):
        return torch.clamp(x, min=other)


@numpy_compatible
def minimum(x: Tensor, other: (Tensor, int, float)) -> Tensor:
    if isinstance(other, Tensor):
        return torch.min(x, other)
    elif isinstance(other, (int, float)):
        return torch.clamp(x, max=other)


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
    return torch.add(x, y)


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
    return torch.sub(x, y)


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
    return torch.dot(x, y)


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
    return torch.true_divide(x, y)


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
    return torch.matmul(a, b)


@numpy_compatible
def floor(x: Tensor) -> Tensor:
    """Returns element-wise greater integer not greater than x.

    Args:
        x (Tensor): input tensor.

    Returns:
      A `Tensor`. Has the same type as `x`.

    """

    return x.floor()


@numpy_compatible
def ceil(x: Tensor) -> Tensor:
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

    return x.ceil()


@numpy_compatible
def round(x: (Tensor, float), digit: int = 0) -> Tensor:
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

    if digit != 0:
        factor = to_tensor(float(math.pow(10.0, -1 * digit)))
        return (x / factor).round() * factor
    else:
        return torch.round(x)


@numpy_compatible
def prod(x: Tensor) -> Tensor:
    """Computes the product of elements across dimensions of a tensor.

    Reduces `input_tensor` along all the dimensions


    If `axis` is None, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
        x (Tensor): input tensor.

    Returns:
        The reduced tensor.

    """

    return torch.prod(x)


def pi() -> Tensor:
    """ The number π (/paɪ/)
    The number π (/paɪ/) is a mathematical constant. It is defined as the ratio of a circle's circumference to its diameter

    Returns:
        The number π (/paɪ/)

    """
    return to_tensor(np.pi)


@numpy_compatible
def sqrt(x: Tensor) -> Tensor:
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

    Note: In order to support complex, please provide an input tensor
    of `complex64` or `complex128`.

    Args:
        x: A `Tensor`


    Returns:
      A `Tensor` of same size, type and sparsity as `x`.

    """
    return x.sqrt()


@numpy_compatible
def rsqrt(x: Tensor) -> Tensor:
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
    return x.rsqrt()


@numpy_compatible
def square(x: Tensor) -> Tensor:
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
def abs(x: Tensor) -> Tensor:
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
    return x.abs()


@numpy_compatible
def pow(x: Tensor, y: Number) -> Tensor:
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
    return torch.pow(x, y)


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
    return torch.log(x)


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
    return torch.exp(x)


@numpy_compatible
def clip(x: Tensor, min=None, max=None):
    """

    Args:
        x (Tensor): input tensor.
        min ():
        max ():

    Returns:

    """
    return torch.clamp(x, min=min, max=max)


############################
# trigonometric functions
###########################


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
    return torch.sin(x)


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
    return torch.cos(x)


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
    return torch.tan(x)


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
    return torch.asin(x)


@numpy_compatible
def acos(x: Tensor) -> Tensor:
    """Computes the element-wise arccos (inverse cosine)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> acos(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[0.0000, 1.0472],
                [1.8235, 2.4189]])

    """
    return torch.acos(x)


@numpy_compatible
def atan(x: Tensor) -> Tensor:
    """Computes the element-wise arctan (inverse tan)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> atan(to_tensor([-1, 0, 1])).cpu()
        tensor([-0.7854,  0.0000,  0.7854])

    """
    return torch.atan(x)


@numpy_compatible
def atan(x: Tensor) -> Tensor:
    """Computes the element-wise   arctan (inverse tan)

    Args:
        x (Tensor): input tensor.

    Returns: element-wise arccos

    Examples:
        >>> atan(to_tensor([-1, 0, 1])).cpu()
        tensor([-0.7854,  0.0000,  0.7854])

    """
    return torch.atan(x)


def atan2(x: Tensor, other: Tensor) -> Tensor:
    """"Computes the element-wise arctangent (angles in radians between x and other )

    Args:
        x (Tensor): input tensor.
        other (Tensor): second input tensor.

    Returns:  the output tensor.

     Examples:
         >>> atan2(to_tensor([-1, 0, 1]), to_tensor([2, 4, 6])).cpu()
         tensor([-0.4636,  0.0000,  0.1651])

    """
    return atan(x / (other + 1e-6))


@numpy_compatible
def sinh(x: Tensor) -> Tensor:
    """Computes the element-wise sinh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise sinh

    Examples:
        >>> sinh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
        tensor([[ 1.1752,  0.5211],
                [-0.2526, -0.8223]])

    """
    return torch.sinh(x)


@numpy_compatible
def cosh(x: Tensor) -> Tensor:
    """Computes the element-wise cosh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise cosh

    Examples:
        >>> cosh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
         tensor([[1.5431, 1.1276],
                [1.0314, 1.2947]])

    """
    return torch.cosh(x)


@numpy_compatible
def tanh(x: Tensor) -> Tensor:
    """Computes the element-wise tanh

    Args:
        x (Tensor): input tensor.

    Returns: element-wise tanh

    Examples:
        >>> tanh(to_tensor([[1,0.5],[-0.25,-0.75]])).cpu()
         tensor([[ 0.7616,  0.4621],
                [-0.2449, -0.6351]])

    """
    return torch.tanh(x)


############################
# elementwise operation
###########################
@numpy_compatible
def element_times(left, right) -> Tensor:
    """
    The output of this operation is the element-wise product of the two  input
    tensors. It supports broadcasting.

    Args:
        right: right side tensor
        left: left side tensor

    Returns:
        :the element-wise product of the two  input

    Examples:
        >>> element_times(to_tensor([1., 1., 1., 1.]), to_tensor([0.5, 0.25, 0.125, 0.])).cpu()
        tensor([0.5000, 0.2500, 0.1250, 0.0000])
        >>> element_times(to_tensor([5., 10., 15., 30.]),to_tensor([2.])).cpu()
        tensor([10., 20., 30., 60.])
        >>> element_times(to_tensor([[5., 10.], [15., 30.]]), to_tensor([[1., 2.], [3.,1.]])).cpu()
        tensor([[ 5., 20.],
                [45., 30.]])
    """
    return left * right


@numpy_compatible
def element_max(left, right) -> Tensor:
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
        tensor([1.0000, 1.0000, 0.1250, 0.0000])
        >>> element_max(to_tensor([5., 10., 15., 30.]),to_tensor([20.]))
        tensor([20., 20., 20., 30.])
        >>> element_max(to_tensor([5., 10., 15., 30.]), to_tensor([10., 2., 8., 2.]))
        tensor([10., 10., 15., 30.])
    """
    return torch.max(left, right)


@numpy_compatible
def element_min(left, right) -> Tensor:
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
    tensor([0.5000, 0.2500, 0.1250, 0.0000])
    >>> element_min(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
    tensor([2., 2., 2., 2.])
    >>> element_min(to_tensor([5., 10., 15., 30.]), to_tensor([1., 2., 1., 2.]))
    tensor([1., 2., 1., 2.])
    """

    return torch.min(left, right)


@numpy_compatible
def element_divide(left, right) -> Tensor:
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
    tensor([2., 4., 8., inf])
    >>> element_divide(to_tensor([5., 10., 15., 30.]),to_tensor([2.]))
    tensor([ 2.5000,  5.0000,  7.5000, 15.0000])
    >>> element_divide(to_tensor([5., 10., 15., 30.]), to_tensor([1., 2., 1., 2.]))
    tensor([ 5.,  5., 15., 15.])
    """
    return torch.true_divide(left, right)


@numpy_compatible
def element_cosine_distance(v1, v2, axis=-1):
    """

    Args:
        v1 (ndarray, tensor):  has the shape [batch: embedded dimensions]
        v2 (ndarray, tensor): has the shape [batch: embedded dimensions]
        axis ():

    Returns:

    """

    x_normalized = l2_normalize(v1, axis=axis, keepdims=True)
    y_normalized = l2_normalize(v2, axis=axis, keepdims=True)

    cos = matmul(x_normalized, y_normalized, False, True)

    # cos1 = (v1 * v2).sum(dim=reduce_dim, keepdims=False) / (
    #         (v1 * v1).sum(dim=reduce_dim, keepdims=False).sqrt() * (v2 * v2).sum(dim=reduce_dim,
    #                                                                              keepdims=False).sqrt())
    return cos


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
        return torch.where(flag)
    else:
        return torch.where(flag.bool(), value_if_true, value_if_false)


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

    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)

    if x.element_size() == 0:
        return x
    if x.dtype == Dtype.bool:
        x = x.to(_float_dtype)
    if axis is None or isinstance(axis, (int, list, tuple)):
        if axis is None and keepdims == False:
            return torch.mean(x)
        else:
            return torch.mean(x, axis, keepdim=keepdims)
    else:
        return torch.mean(x)


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

    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if x.element_size() == 0:
        return x
    if x.dtype == Dtype.bool:
        x = x.to(_float_dtype)
    _xdtype = x.dtype
    scale = 1 if _xdtype == Dtype.float32 else builtins.max(numel(x), 1)
    if axis is None or isinstance(axis, (int, list, tuple)):
        if axis is None and keepdims == False:
            return torch.sum(x / scale) * scale
        else:
            return torch.sum(x / scale, axis, keepdim=keepdims) * scale
    else:
        return torch.sum(x / scale) * scale


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
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if x.element_size() == 0:
        return x
    if x.dtype == Dtype.bool:
        x = x.to(_float_dtype)
    if axis is None or isinstance(axis, (int, list, tuple)):
        if axis is None and keepdims == False:
            result = x.max()
        elif keepdims == False:
            result = x.max(axis)
        else:
            result = x.max(axis, keepdims)
        if is_tensor(result):
            return result
        elif isinstance(result, tuple):  # (values, indices)
            # RuntimeError: Please look up dimensions by name, got: name = None.
            return result[0]
    else:
        return torch.max(x)


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
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if x.element_size() == 0:
        return x
    if x.dtype == Dtype.bool:
        x = x.to(_float_dtype)
    if axis is None or isinstance(axis, (int, list, tuple)):

        if axis is None and keepdims == False:
            result = x.min()
        elif keepdims == False:
            result = x.min(axis)
        else:
            result = x.min(axis, keepdims)
        if is_tensor(result):
            return result
        elif isinstance(result, tuple):  # (values, indices)
            # RuntimeError: Please look up dimensions by name, got: name = None.
            return result[0]
    else:
        return torch.min(x)


@numpy_compatible
def reduce_std(x: Tensor, axis=None, keepdims=False, **kwargs):
    """Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
     The standard deviation is computed for the flattened array by default, otherwise over the specified axis.

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
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)

    return torch.std(x, dim=axis, keepdim=keepdims)


@numpy_compatible
def reduce_logsumexp(x: Tensor, axis=None, keepdims=False, **kwargs):
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
        >>> x =to_tensor([[0., 0., 0.], [0., 0., 0.]])
        >>> reduce_logsumexp(x)  # log(6)
        >>> reduce_logsumexp(x, 0)  # [log(2), log(2), log(2)]
        >>> reduce_logsumexp(x, 1)  # [log(3), log(3)]
        >>> reduce_logsumexp(x, 1, keepdims=True)  # [[log(3)], [log(3)]]
        >>> reduce_logsumexp(x, [0, 1])  # log(6)


    Args:
        x (Tensor): input tensor.
        axis (int, list, tuple): The dimensions to reduce. If `None` (the default), reduces all dimensions. Must be
        in the range `(-rank(input_tensor), rank(input_tensor))`.
        keepdims (bool): If true, retains reduced dimensions with length 1.

    Returns:
      The reduced tensor.

    """
    if x.element_size() == 0:
        return x
    if x.dtype == Dtype.bool:
        x = x.to(_float_dtype)
    if axis is None or isinstance(axis, (int, list, tuple)):

        return torch.logsumexp(x, dim=axis, keepdim=keepdims)

    else:
        return log(reduce_sum(exp(x), axis=axis, keepdims=keepdims))


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

    """
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if x.element_size() == 0:
        return x

    if axis is None:
        if ndim(x) == 1:
            axis = 0
        elif ndim(x) > 1:
            axis = tuple(list(range(ndim(x))))
        return torch.prod(x, dim=axis, keepdim=keepdims)
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = x.prod(dim=a, keepdim=keepdims)
            x = arr
        return x


@numpy_compatible
def reduce_any(x: Tensor, axis=None, keepdims=False, **kwargs):
    axis = kwargs.get('dim', axis)
    keepdims = kwargs.get('keepdim', keepdims)
    if x.element_size() == 0:
        return x
    x = x.gt(0)
    if isinstance(axis, int):
        return x.any(dim=axis, keepdim=keepdims)
    elif isinstance(axis, list):
        axis = sorted(axis)
        axis.reverse()
        for a in axis:
            arr, idx = x.any(dim=a, keepdim=keepdims)
            x = arr
        return x


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
            'axis' in kwargs or 'dim' in kwargs or 'keepdims' in kwargs or 'keepdim' in kwargs):
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return reduce_max(allargs[0], axis=axis, keepdims=keepdims)
    elif len(args) > 1 and is_tensor(args[0]) and all(
            [is_tensor(arg) or isinstance(arg, (np.ndarray, float, int)) for arg in args]):
        new_args = [to_tensor(a) for a in args]
        return torch.max(*new_args)
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
        return torch.min(allargs[0])
    elif len(allargs) > 1 and is_tensor(allargs[0]) and not is_tensor(allargs[1]) and (
            'axis' in kwargs or 'dim' in kwargs or 'keepdims' in kwargs or 'keepdim' in kwargs):
        axis = kwargs.get('axis', kwargs.get('dim', None))
        keepdims = kwargs.get('keepdims', kwargs.get('keepdim', False))
        return reduce_min(allargs[0], axis=axis, keepdims=keepdims)
    elif len(args) > 1 and is_tensor(args[0]) and all(
            [is_tensor(arg) or isinstance(arg, (np.ndarray, float, int)) for arg in args]):
        new_args = [to_tensor(a) for a in args]
        return torch.min(*new_args)
    else:
        raise NotImplementedError('Min({0},{1}) is not implemented yet '.format(*args, **kwargs))


############################
# activationoperation
###########################

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
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return torch.relu(x)


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
    return F.relu6(x)


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
    return F.leaky_relu(x, negative_slope=slope)


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
    return torch.clamp(F.leaky_relu(x, negative_slope=slope), -6, 6)


@numpy_compatible
def smooth_relu(x):
    """smooth_relu activation function


    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return torch.log(1 + torch.exp(x))


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
    return torch.where(x > 0, x, alpha * torch.expm1(x / alpha))


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

    return torch.cat([relu(x), relu(-x)], dim=axis)


@numpy_compatible
def elu(x, alpha=1):
    """ Exponential Linear Unit.
    It follows:

        f(x) =  alpha * (exp(x) - 1.) for x < 0
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
    return F.elu(x, alpha=alpha)


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
    return torch.prelu(x, weight=weight)


@numpy_compatible
def sigmoid(x):
    """softmax activation function

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return torch.sigmoid(x)


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
    return x * sigmoid(x)


@numpy_compatible
def selu(x):
    """
    selu activation function


    .. math::
            \text{SELU}(x) = \text{scale} * (\\max(0,x) + \\min(0, \\alpha * (\\exp(x) - 1)))

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
    return torch.selu(x)


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
    return 1.7159 * torch.tanh(2 / 3 * x)


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

    return torch.clamp(x * 0.2 + 0.5, 0., 1.)


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
    return torch.clamp(x, -1, 1)


@numpy_compatible
def soft_plus(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return F.softplus(x)


@numpy_compatible
def square_plus(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return (x + (x ** 2 + 4).sqrt()) / 2.0


@numpy_compatible
def logit(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return (x / (1 - x)).log()


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
        >>> log_log(to_tensor([-3.0, -1.0, 0.0, 2.0])).cpu()
        tensor([0.0486, 0.3078, 0.6321, 0.9994])

    References:
        "Complementary Log-Log and Probit: Activation Functions Implemented in Artificial Neural Networks"
        https://ieeexplore.ieee.org/document/4626755/


    """
    return 1 - torch.exp(-torch.exp(x))


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
    return x * (torch.tanh(F.softplus(x)))


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
    return x * hard_tanh(F.softplus(x))


@numpy_compatible
def softmax(x, axis=1):
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
    return torch.softmax(x, dim=axis)


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
    return nn.functional.gelu(x)


def gpt_gelu(x):
    """

    Args:
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.


    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def silu(x):
    return torch.nn.functional.silu(x)


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


def norm(x: Tensor, order=None, axis=1, keepdims=False):
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
    #     return torch.linalg.norm(x, ord=order,dim=axis, keepdim=keepdims)
    # else:
    return x.norm(p=order, dim=axis, keepdim=keepdims)


@numpy_compatible
def l2_normalize(x: Tensor, axis=1, keepdims=True, eps=epsilon()):
    """

    Args:
        eps ():
        keepdims ():
        axis ():
        x (Tensor): input tensor.

    Returns:
        (Tensor): output tensor and get same shape with x.

    Examples:
        >>> a=arange(9).float()-4.0
        >>> b=a.reshape((3, 3))
        >>> l2_normalize(a)
        >>> l2_normalize(b)
        >>> torch.nn.functional.normalize(a , p=2,dim=0)-l2_normalize(a)
        0.0
        >>> torch.nn.functional.normalize(b , p=2,dim=1)-l2_normalize(b)
        0.0


    """
    if ndim(x) == 1:
        axis = 0
    # if pt_version>version1_7:
    #     return x / (torch.linalg.norm(x,dim=axis, keepdim=keepdims) + eps)
    # else:
    return x / (x.norm(dim=axis, keepdim=keepdims) + eps)


@numpy_compatible
def spectral_norm(module, n_iterations=1, axis=1):
    return nn.utils.spectral_norm(module, n_power_iterations=n_iterations, dim=axis)


############################
# tensor shape operation
###########################


def broadcast_to(x: Tensor, shape: Union[(List, Tuple, torch.Size, TensorShape)] = None) -> Tensor:
    if shape is None:
        return x
    elif isinstance(shape, TensorShape):
        shape = shape.dims
    if len(shape) > 2 and int_shape(x)[-1] != shape[-1] and int_shape(x)[-1] == shape[1]:
        shape = to_list(shape)
        new_shape = shape[0:1] + shape[2:] + shape[1:2]
        x = torch.broadcast_to(x, new_shape)
        return x.transpose(-1, 1)
    else:
        return torch.broadcast_to(x, shape)


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
def reshape(x: Tensor, shape: Union[(List, Tuple, torch.Size, TensorShape)] = None) -> Tensor:
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

    return torch.reshape(x, shape)


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
    x = torch.transpose(x, dim0, dim1)
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
    x = torch.permute(x, *dims)
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
def expand_dims(x: Tensor, axis):
    """

    Args:
        x (Tensor): input tensor.
        axis ():

    Returns:

    """
    return x.unsqueeze(axis)


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
    >>> x = to_tensor(np.tile(np.array(np.reshape(range(8), (8, 1, 1)), dtype=np.float32), (1, 2, 3)))
    >>> x
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
    >>> arr=depth_to_space(x,block_size=2)
    >>> print(arr.shape)
    torch.Size([2, 4, 6])
    >>> arr
    tensor([[[0., 1., 0., 1., 0., 1.],
             [2., 3., 2., 3., 2., 3.],
             [0., 1., 0., 1., 0., 1.],
             [2., 3., 2., 3., 2., 3.]],
    <BLANKLINE>
            [[4., 5., 4., 5., 4., 5.],
             [6., 7., 6., 7., 6., 7.],
             [4., 5., 4., 5., 4., 5.],
             [6., 7., 6., 7., 6., 7.]]])

    """
    if ndim(x) not in (3, 4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    elif x.shape[-3] % (block_size * block_size) != 0:
        raise ValueError('Input tensort channel must be divisible by square of block_size')
    else:
        orig_ndim = ndim(x)
        if orig_ndim == 3:
            x = expand_dims(x, 0)
        x = F.pixel_shuffle(x, block_size)
        if orig_ndim == 3:
            return x[0]
        return x


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
    torch.Size([8, 2, 3])
    """
    if ndim(x) not in (3, 4):
        raise ValueError('Input tensort length of shape should be 3 or 4 ')
    elif x.shape[-2] % block_size != 0 or x.shape[-1] % block_size != 0:
        raise ValueError('Input tensort channel must be divisible by square of block_size')
    else:
        orig_ndim = ndim(x)
        if orig_ndim == 3:
            x = expand_dims(x, 0)
        orig_shape = list(int_shape(x))
        x = reshape(x, (
            orig_shape[0], orig_shape[1], orig_shape[2] // block_size, block_size, orig_shape[3] // block_size,
            block_size))
        x = permute(x, [0, 1, 3, 5, 2, 4])
        x = reshape(x, (orig_shape[0], orig_shape[1] * block_size * block_size, orig_shape[2] // block_size,
                        orig_shape[3] // block_size))
        if orig_ndim == 3:
            return x[0]
        return x


def pad(x: Tensor, paddings: Sequence[int], mode='constant', value=0):
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
        paddings ():
        x (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Examples::

        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 9, 7, 3])

    """
    valid_items = ['constant', 'reflect', 'replicate', 'circular', 'symmetric', 'zero']
    if mode not in valid_items:
        raise ValueError('{0} is not valid for mode.'.format(mode))
    if mode == 'zero':
        mode = 'constant'
        value = 0
    if mode == 'symmetric':
        mode = 'circular'
    return torch.nn.functional.pad(x, pad=paddings, mode=mode, value=value)


############################
# tensor generation
###########################

def ones(shape: Union[(List, Tuple, torch.Size, TensorShape)], dtype=None, requires_grad=False):
    """Instantiates an all-ones tensor and returns it.

    Args
        shape (Tuple of integers):  the  output shape
        dtype (String):  data type
        requires_grad (bool):  whether we need gradient

    Returns
        A tensor, filled with `1.0`.

    Example
        >>> ones((3,4))
        tensor([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]])

    {{np_implementation}}
    """
    if isinstance(shape, TensorShape):
        shape = shape.dims
    if dtype is None:
        dtype = _float_dtype
    return torch.ones(shape, dtype=dtype, requires_grad=requires_grad).to(get_session_value('device'))


@numpy_compatible
def ones_like(a, dtype=None, requires_grad=False):
    """Instantiates an all-ones variable of the same shape as another tensor.

    Args
        a (Tensor):  another tensor
        dtype (String):  data type
        requires_grad (bool):  whether we need gradient

    Returns
        A tensor, filled with `1.0` and shape is the same as another tensor.

    Example
        >>> ones_like(torch.randn((3,4)))
        array([[ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.]], dtype=float32)

    {{np_implementation}}
    """
    if dtype is None:
        dtype = a.dtype
    return torch.ones(a.shape, dtype=dtype, requires_grad=requires_grad).to(get_session_value('device'))


def zeros(shape: Union[(List, Tuple, torch.Size, TensorShape)], dtype=None, requires_grad=False):
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
    return torch.zeros(shape, dtype=dtype, requires_grad=requires_grad).to(get_session_value('device'))


@numpy_compatible
def zeros_like(a, dtype=None, requires_grad=False):
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
    return torch.zeros(a.shape, dtype=dtype, requires_grad=requires_grad).to(get_session_value('device'))


def eye(shape: Union[(List, Tuple, torch.Size, TensorShape)], dtype=None, requires_grad=None):
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
        return torch.eye(shape[0], shape[1], dtype=dtype, requires_grad=requires_grad).to(get_session_value('device'))
    else:
        raise ValueError('input tensor must have exactly two axe.')


@numpy_compatible
def eye_like(a, dtype=None, requires_grad=False):
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
        return torch.eye(a.shape[0], a.shape[1], dtype=dtype, requires_grad=requires_grad).to(
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

    onehot = torch.nn.functional.one_hot(label.long(), num_classes).to(_float_dtype)
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


def arange(*args, dtype=Dtype.int32, requires_grad=False):
    """

    Args:
        *args (int): the start, end, step
        dtype (dtype): dtype of the tensor
        requires_grad (bool): whether we need gradient

    Returns:

    """
    if len(args) == 1:
        return torch.arange(end=args[0], dtype=dtype, requires_grad=requires_grad).to(get_session_value('device'))
    elif len(args) == 2:
        return torch.arange(start=args[0], end=args[1], dtype=dtype, requires_grad=requires_grad).to(
            get_session_value('device'))
    elif len(args) == 3:
        return torch.arange(start=args[0], end=args[1], step=args[2], dtype=dtype, requires_grad=requires_grad).to(
            get_session_value('device'))
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
    xs = torch.linspace(0, int(x - 1), int(x), device=get_session_value('device'), dtype=_float_dtype,
                        requires_grad=requires_grad)
    ys = torch.linspace(0, int(y - 1), int(y), device=get_session_value('device'), dtype=_float_dtype,
                        requires_grad=requires_grad)
    if normalized_coordinates:
        xs = torch.linspace(0, 1, int(x), device=get_session_value('device'), dtype=_float_dtype,
                            requires_grad=requires_grad)
        ys = torch.linspace(0, 1, int(y), device=get_session_value('device'), dtype=_float_dtype,
                            requires_grad=requires_grad)
    grid_x, grid_y = torch.meshgrid([xs, ys])

    grid = torch.stack([grid_y, grid_x], -1).to(get_session_value('device'))
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
    return torch.flip(x, dims=axis)


############################
# tensor manipulation
###########################

def concate(x: List[Tensor], axis=1):
    """

    Args:
        x ():
        axis ():

    Returns:

    """
    return torch.cat(x, dim=axis)


def stack(x: List[Tensor], axis=1):
    """

    Args:
        x ():
        axis ():

    Returns:

    """
    return torch.stack(x, dim=axis)


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
      >>>
      >>> # Split `x` into 3 tensors along dimension 1
      >>> s0, s1, s2 = split(x, num_splits=3, axis=1)
      >>> int_shape(s0)
      array([ 5, 10], dtype=int32)
      >>>
      >>> # Split `x` into 3 tensors with sizes [4, 15, 11] along dimension 1
      >>> split0, split1, split2 = split(x, [4, 15, 11], 1)
      >>> int_shape(split0)
      array([5, 4], dtype=int32)
      >>> int_shape(split1)
      array([ 5, 15], dtype=int32)
      >>>int_shape(split2)
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

    return torch.chunk(x, dim=axis, chunks=num_splits)


@numpy_compatible
def repeat_elements(x: Tensor, multiples: int, axis=1):
    """Repeat elements of a tensor.

    Args:
        x (Tensor):the input tensor.
        multiples(Tensor or int):The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
        axis (int): The dimension along which to repeat values. By default, use the flattened input array, and return a flat output array.

    Returns (Tensor)::Repeated tensor which has the same shape as input, except along the given axis.

    """
    return torch.repeat_interleave(x, repeats=multiples, dim=axis)


@numpy_compatible
def gather(x: Tensor, gather_axis, indices):
    return torch.gather(input=x, dim=gather_axis, index=indices)


def index_select(x: Tensor, axis: int, indices: Tensor):
    """
    Args:
        x (tensor): input tensor
        axis(int): dimension
        indices(list): selected indices list
    Examples:
        >>> arr=arange(9).reshape(3, 3)
        >>> idx=to_tensor([[0, 1, 2], [2, 3, 4], [0, 2, 4]]).long()
        >>>gather(arr,0,idx)

    """
    num_class = int_shape(x)[axis]
    if reduce_sum(greater_equal(indices, num_class, dtype=_float_dtype)):
        raise ValueError('Number of class are {0}, indices should not out of the range.'.format(num_class))
    return torch.index_select(x, dim=axis, index=indices)


@numpy_compatible
def scatter_add(x: Tensor, indices: Tensor, updates: Tensor):
    return torch.scatter_add(x, dim=None, index=indices, src=updates)


@numpy_compatible
def scatter_sub(x: Tensor, indices: Tensor, updates: Tensor):
    return torch.scatter_sub(x, dim=None, index=indices, src=updates)


@numpy_compatible
def scatter_max(x: Tensor, indices: Tensor, updates: Tensor):
    return torch.ops.torch_scatter.scatter_max(x, indices, None, updates, None)


@numpy_compatible
def scatter_min(x: Tensor, indices: Tensor, updates: Tensor):
    return torch.ops.torch_scatter.scatter_min(x, indices, None, updates, None)


@numpy_compatible
def gram_matrix(x: Tensor):
    """

    Args:
        x (Tensor): input tensor.

    Returns:

    """
    a, b, c, d = x.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL
    features = features
    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.

    return G


############################
# random
###########################


def set_seed(seed: int) -> None:
    """Setup random state from a seed for `torch.random`, `random` and  `numpy` (if it can be imported).

    Args:
        seed (int): Random state seed

    """
    random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case,
        # it is silently ignored.
        torch.cuda.manual_seed_all(
            seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def shuffle(x: Tensor):
    """

    Args:
        x (Tensor): input tensor.

    Returns:

    """
    idxes = np.arange(len(x))
    np.random.shuffle(idxes)
    x[np.array(range(len(x)))] = x[idxes]
    return x


def random_choice(x: Tensor, n: int = 1):
    """Generates a random sample from a given 1-D array

    Args:
        x (Tensor): input tensor (1-D  tensor).
        n (int): how many items

    Returns:
        (Tensor) : single item ,the generated random samples

    """
    idxes = np.arange(len(x))
    np.random.shuffle(idxes)
    if is_tensor(x):
        idx = to_tensor(idxes)[:n].long()
        return x[idx]
    elif isinstance(x, (list, tuple)):
        return [x[idx] for idx in idxes[:n]]


def random_normal(shape, mean:Union[Tensor,float]=0.0, std:Union[Tensor,float]=1.0, dtype=None, device=None, seed=None):
    """Outputs random values from a normal distribution.

    In this case, we are setting both the global and operation-level seed to
    ensure this result is reproducible.  See `random.set_seed` for more
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
        `random.set_seed`
        for behavior.

    Returns:
      A tensor of the specified shape filled with random normal values.


    Example :
        >>> #that generates a new set of random values every time
        >>> random_normal([4],dtype='float32' ,mean=0, stddev=1,seed=5)
        <Tensor: shape=(4,), dtype=float32, numpy=..., dtype=float32>
        >>> #that outputs a reproducible result:
        >>> random_normal([2,2],dtype='float32' ,mean=0, stddev=1,seed=5)
        <Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[-1.3768897 , -0.01258316],
              [-0.169515   ,  1.0824056 ]], dtype=float32>


    """
    if seed is not None:
        set_seed(seed)
        if dtype is not None:
            dtype = str2dtype(dtype)
    if dtype is None:
        dtype = _float_dtype
    if device is None:
        device = _get_device()
    if std is None or std < 0.02:
        std = 0.02
    return torch.normal(mean=mean, std=std, size=shape, dtype=dtype, device=device)


@numpy_compatible
def random_normal_like(x, mean:Union[Tensor,float]=0.0, std:Union[Tensor,float]=1.0, dtype=None, device=None, seed=None):
    """Outputs random values from a normal distribution.

    In this case, we are setting both the global and operation-level seed to
    ensure this result is reproducible.  See `random.set_seed` for more
    information.

    Args:
      x: A 1-D integer Tensor or Python array. The shape of the output tensor.
      mean: A Tensor or Python value of type `dtype`, broadcastable with `stddev`.
        The mean of the normal distribution.
      std: A Tensor or Python value of type `dtype`, broadcastable with `mean`.
        The standard deviation of the normal distribution.
      dtype: The type of the output.
      seed: A Python integer. Used to create a random seed for the distribution.
        See
        `random.set_seed`
        for behavior.

    Returns:
      A tensor of the specified shape filled with random normal values.


    Example :
        >>> #that generates a new set of random values every time
        >>> random_normal([4],dtype='float32' ,mean=0, stddev=1,seed=5)
        <Tensor: shape=(4,), dtype=float32, numpy=..., dtype=float32>
        >>> #that outputs a reproducible result:
        >>> random_normal([2,2],dtype='float32' ,mean=0, stddev=1,seed=5)
        <Tensor: shape=(2, 2), dtype=float32, numpy=
        array([[-1.3768897 , -0.01258316],
              [-0.169515   ,  1.0824056 ]], dtype=float32>


    """
    if seed is not None:
        set_seed(seed)
    if dtype is not None:
        dtype = str2dtype(dtype)
    else:
        dtype = x.dtype
    if std is None or std < 0.02:
        std = 0.02
    if is_tensor(std):
        std = std.item()
    if is_tensor(mean):
        mean = mean.item()

    return torch.normal(mean=mean, std=std, size=x.shape, dtype=x.dtype, device=x.device)


def random_uniform(shape, min_value:Union[Tensor,float]=0.0, max_value:Union[Tensor,float]=None, dtype=None, device=None, seed=None):
    """Outputs random values from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `(min, max)`. The lower bound `minval` is included in the range, while
    the upper bound `maxval` is excluded.

    For floats, the default range is `(0, 1)`.  For ints, at least `maxval` must
    be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `max_value - min_value` is an exact power of two.  The bias is small for values of
    `max_value - min_value` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    Examples:

    >>> random.uniform(shape=[2])
    <Tensor: shape=(2,), dtype=float32, numpy=array([..., ...]), dtype=float32>
    >>> random.uniform(shape=[], minval=-1., maxval=0.)
    <Tensor: shape=(), dtype=float32, numpy=-...>
    >>> random.uniform(shape=[], minval=5, maxval=10, dtype=Dtype.int64)
    <Tensor: shape=(), dtype=int64, numpy=...>

    The `seed` argument produces a deterministic sequence of tensors across
    multiple calls. To repeat that sequence, use `random.set_seed`:

    >>> random.set_seed(5)
    >>> random.uniform(shape=[], maxval=3, dtype=Dtype.int32, seed=10)
    <Tensor: shape=(), dtype=int32, numpy=2>
    >>> random.uniform(shape=[], maxval=3, dtype=Dtype.int32, seed=10)
    <Tensor: shape=(), dtype=int32, numpy=0>
    >>> random.set_seed(5)
    >>> random.uniform(shape=[], maxval=3, dtype=Dtype.int32, seed=10)
    <Tensor: shape=(), dtype=int32, numpy=2>
    >>> random.uniform(shape=[], maxval=3, dtype=Dtype.int32, seed=10)
    <Tensor: shape=(), dtype=int32, numpy=0>

    Without `random.set_seed` but with a `seed` argument is specified, small
    changes to function graphs or previously executed operations will change the
    returned value. See `random.set_seed` for details.

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
      seed: A Python integer. Used in combination with `random.set_seed` to
        create a reproducible sequence of tensors across multiple calls.


    Returns:
      A tensor of the specified shape filled with random uniform values.

    Raises:
      ValueError: If `dtype` is integral and `maxval` is not specified.
    """
    if seed is not None:
        set_seed(seed)
        if dtype is not None:
            dtype = str2dtype(dtype)
    if dtype is None:
        dtype = _float_dtype
    t = zeros(shape=shape, dtype=dtype)
    t.uniform_(min_value, max_value)
    return t


@numpy_compatible
def random_uniform_like(x, min_value:Union[Tensor,float]=0.0, max_value:Union[Tensor,float]=1.0, dtype=None, device=None, seed=None):
    """Outputs random values from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `(min, max)`. The lower bound `minval` is included in the range, while
    the upper bound `maxval` is excluded.

    For floats, the default range is `(0, 1)`.  For ints, at least `maxval` must
    be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `max_value - min_value` is an exact power of two.  The bias is small for values of
    `max_value - min_value` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    Examples:

    >>> random.uniform(shape=[2])
    <Tensor: shape=(2,), dtype=float32, numpy=array([..., ...]), dtype=float32>
    >>> random.uniform(shape=[], min_value=-1., max_value=0.)
    <Tensor: shape=(), dtype=float32, numpy=-...>
    >>> random.uniform(shape=[], min_value=5, max_value=10, dtype=Dtype.int64)
    <Tensor: shape=(), dtype=int64, numpy=...>

    The `seed` argument produces a deterministic sequence of tensors across
    multiple calls. To repeat that sequence, use `random.set_seed`:

    >>> random.set_seed(5)
    >>> random.uniform(shape=[], max_value=3, dtype=Dtype.int32, seed=10)
    <Tensor: shape=(), dtype=int32, numpy=2>
    >>> random.uniform(shape=[], max_value=3, dtype=Dtype.int32, seed=10)
    <Tensor: shape=(), dtype=int32, numpy=0>
    >>> random.set_seed(5)
    >>> random.uniform(shape=[], max_value=3, dtype=Dtype.int32, seed=10)
    <Tensor: shape=(), dtype=int32, numpy=2>
    >>> random.uniform(shape=[], max_value=3, dtype=Dtype.int32, seed=10)
    <Tensor: shape=(), dtype=int32, numpy=0>

    Without `random.set_seed` but with a `seed` argument is specified, small
    changes to function graphs or previously executed operations will change the
    returned value. See `random.set_seed` for details.

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
      seed: A Python integer. Used in combination with `random.set_seed` to
        create a reproducible sequence of tensors across multiple calls.


    Returns:
      A tensor of the specified shape filled with random uniform values.

    Raises:
      ValueError: If `dtype` is integral and `maxval` is not specified.
    """
    if seed is not None:
        set_seed(seed)
    if dtype is not None:
        dtype = str2dtype(dtype)

    t = torch.zeros(int_shape(x), dtype=x.dtype)

    t.uniform_(min_value, max_value)
    return t


@numpy_compatible
def multinomial(x: Tensor, num_samples: int = 1):
    return torch.multinomial(x, num_samples)


def random_bernoulli(x: Tensor):
    return torch.bernoulli(x)


############################
## probability distribution
###########################


############################
## loss
###########################

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
        output = output
    else:
        output = sigmoid(output)
    bce = target * torch.log(output)
    bce += (1 - target) * torch.log(1 - output)
    return -bce


def cross_entropy(output, target, from_logits=False):
    if not from_logits:
        output = output.sigmoid()
    output = output.clamp(epsilon(), 1.0 - epsilon())
    target = target.clamp(epsilon(), 1.0 - epsilon())
    loss = -target * torch.log(output)  # (1.0 - target) * torch.log(1.0 - output)
    return loss


def binary_hinge(output, target, margin=1, pos_weight=1.0):
    """
    Implements Hinge loss.
    Args:
        output (torch.Tensor): of shape `Nx*` where * means any number
             of additional dimensions
        target (torch.Tensor): same shape as target
        margin (float): margin for y_pred after which loss becomes 0.
        pos_weight (float): weighting factor for positive class examples. Useful in case
            of class imbalance.
    """
    target_shifted = 2 * target - 1  # [0, 1] -> [-1, 1]
    hinge = (margin - output * target_shifted).relu()
    hinge *= target * pos_weight + (1 - target)
    return hinge  # reduction == mean


############################
## color space
###########################

def rgb2gray(rgb: Tensor, axis=-1):
    """Compute grayscale of an RGB image.

    Args:
        rgb (tensor):  rgb image : shape:(H,W,C), range [0,255]
        axis(int): the channel axis
    Returns:
        gray(tensor):  gray-scale image (shape: (H,W), range [0,255])

    Examples:
        >>> import cv2
        >>> img=cv2.cvtColor(cv2.imread('../../images/cat.jpg'),cv2.COLOR_BGR2RGB)
        >>> gray_tensor=to_numpy(rgb2gray(to_tensor(img.copy()).float()))
        >>> groundtruth_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        >>> abs(np.round(gray_tensor.astype(np.float32))-groundtruth_gray.astype(np.float32)).mean()<2
        True
        >>> print( groundtruth_gray.shape,groundtruth_gray)
        >>> print( gray_tensor.shape,gray_tensor.astype(np.uint8))
        >>> print( abs(np.round(gray_tensor.astype(np.float32))-groundtruth_gray.astype(np.float32)).mean())

    """
    rgb = rgb.copy().to(_float_dtype)
    if ndim(rgb) not in [3, 4]:
        raise ValueError('input rgb image ndim should equal 3 but get {0}'.format(ndim(rgb)))
    if ndim(rgb) == 3:
        r, g, b = split(rgb, 3, axis=axis)
        gray = clip((0.2125 * r + 0.7154 * g + 0.0721 * b).squeeze(axis), 0, 255)
        return gray
    elif ndim(rgb) == 4:
        shp = int_shape(rgb)
        if shp[-1] == 3:
            axis = -1
        elif shp[1] == 3:
            axis = 1
        r, g, b = split(rgb, 3, axis=axis)
        gray = clip((0.2125 * r + 0.7154 * g + 0.0721 * b).squeeze(axis), 0, 255)
        return gray


def rgb2hsv(rgb: Tensor):
    """Compute luminance of an RGB image.

    Args:
        rgb (tensor):  rgb image  =>shape: (H,W,C)
    Returns:
        gray(tensor):  gray-scale image => shape: (H,W)

    Examples:
         >>> import cv2
        >>> img=cv2.cvtColor(cv2.imread('../../images/cat.jpg'),cv2.COLOR_BGR2RGB)
        >>> hsv_tensor=to_numpy(rgb2hsv(to_tensor(img.copy()).float()))
        >>> groundtruth_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        >>> abs(np.round(hsv_tensor.astype(np.float32))-groundtruth_hsv.astype(np.float32)).mean()<2
        True
        >>> print( groundtruth_hsv.shape,groundtruth_hsv)
        >>> print( hsv_tensor.shape,hsv_tensor.astype(np.uint8))
        >>> print(abs(np.round(hsv_tensor.astype(np.float32))-groundtruth_hsv.astype(np.float32)).mean())

    """
    rgb = rgb.to(_float_dtype) / 255.0
    if ndim(rgb) not in [3, 4]:
        raise ValueError('input rgb image ndim should equal 3 but get {0}'.format(ndim(rgb)))
    # rgb=rgb[np.newaxis, ...]
    rgb = zeros_like(rgb.copy())
    axis = -1
    if ndim(rgb) == 4:
        axis = 1

    # -- V channel
    out_v = reduce_max(rgb.copy(), axis=axis, keepdims=True)

    # -- S channel
    delta = reduce_max(rgb.copy(), axis=axis, keepdims=True) - reduce_min(rgb.copy(), axis=axis, keepdims=True)
    delta_zeros = zeros_like(delta, dtype=delta.dtype)
    out_s = where(delta == 0, delta_zeros, delta / out_v)

    # -- H channel
    # red is max
    maxc_tmp = equal(out_v, rgb, dtype=_float_dtype)
    _, max_indices = rgb.copy().max(dim=axis)
    out_h = None
    if ndim(rgb) == 3:
        rc, gc, bc = split(maxc_tmp.copy(), 3, axis=axis)

        out_h = torch.cat([
            bc - gc,
            2.0 * delta + rc - bc,
            4.0 * delta + gc - rc, ], dim=axis)
        out_h = torch.gather(out_h, dim=axis, index=max_indices[:, :, None])


    elif ndim(rgb) == 4:
        rc, gc, bc = split(maxc_tmp.copy(), 3, axis=-3)
        out_h = torch.cat([
            bc - gc,
            2.0 * delta + rc - bc,
            4.0 * delta + gc - rc,
        ], dim=-3)
        out_h = torch.gather(out_h, dim=-3, index=max_indices[..., None, :, :])

    # out_h = out_h / delta
    out_h = (out_h / 6.0) % 1.0

    # -- output
    return torch.cat([out_h * 255.0, out_s * 255.0, out_v * 255.0], dim=axis)


def xyz2rgb(xyz: Tensor):
    """
    input xyz as pytorch tensor of size (batch_size,  h, w, 3) or (h, w,3)
    """
    if len(xyz.shape) == 4:
        if int_shape(xyz)[-1] == 3:
            xyz = xyz.permute(0, 3, 1, 2)
        elif int_shape(xyz)[1] == 3:
            pass
    elif len(xyz.shape) == 3:
        if int_shape(xyz)[-1] == 3:
            xyz = xyz.permute(2, 0, 1)
        elif int_shape(xyz)[0] == 3:
            pass
    xyz = xyz / 255.0
    transform_tensor = to_tensor([[3.2404542, - 1.5371385, - 0.4985314],
                                  [-0.9692660, 1.8760108, 0.0415560],
                                  [0.0556434, - 0.2040259, 1.0572252]], dtype=xyz.dtype, requires_grad=False).to(
        _get_device())

    transform_tensor.unsqueeze_(2).unsqueeze_(3)
    convolved = None
    if len(xyz.shape) == 4:
        convolved = F.conv2d(xyz, transform_tensor)
    else:
        convolved = F.conv2d(xyz.unsqueeze(0), transform_tensor).squeeze(0)
    # return convolved
    rgb = convolved * 255.0
    if len(rgb.shape) == 4:
        if int_shape(rgb)[-1] == 3:
            return rgb
        elif int_shape(rgb)[1] == 3:
            rgb = rgb.permute(0, 2, 3, 1)
            return rgb

    elif len(rgb.shape) == 3:
        if int_shape(rgb)[-1] == 3:
            return rgb
        elif int_shape(rgb)[0] == 3:
            rgb = rgb.permute(1, 2, 0)
            return rgb

    raise ValueError('image should channel-last')


def rgb2xyz(rgb: Tensor):
    """
    input rgb as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if len(rgb.shape) == 4:
        if int_shape(rgb)[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)
        elif int_shape(rgb)[1] == 3:
            pass
    elif len(rgb.shape) == 3:
        if int_shape(rgb)[-1] == 3:
            rgb = rgb.permute(2, 0, 1)
        elif int_shape(rgb)[0] == 3:
            pass
    rgb = rgb / 255.0
    rgb = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)

    transform_tensor = to_tensor([[0.4124564, 0.3575761, 0.1804375],
                                  [0.2126729, 0.7151522, 0.0721750],
                                  [0.0193339, 0.1191920, 0.9503041]], dtype=rgb.dtype, requires_grad=False).to(
        _get_device())

    transform_tensor.unsqueeze_(2).unsqueeze_(3)
    xyz = None
    if len(rgb.shape) == 4:
        xyz = F.conv2d(rgb, transform_tensor)
    else:
        xyz = F.conv2d(rgb.unsqueeze(0), transform_tensor).squeeze(0)
    xyz = xyz * 255.0
    if len(xyz.shape) == 4:
        if int_shape(xyz)[-1] == 3:
            return xyz
        elif int_shape(xyz)[1] == 3:
            xyz = xyz.permute(0, 2, 3, 1)
            return xyz

    elif len(xyz.shape) == 3:
        if int_shape(xyz)[-1] == 3:
            return xyz
        elif int_shape(xyz)[0] == 3:
            xyz = xyz.permute(1, 2, 0)
            return xyz

    raise ValueError('image should channel-last')


# LAB
# CIE-L*a*b*: A perceptually uniform color space,
# i.e. distances are meaningful. L* in [0..1] and a*, b* almost in [-1..1].
D65 = [0.95047, 1.00000, 1.08883]


def lab_f(t: Tensor):
    return where(t > 0.008856451679035631, cast(t.pow(1.0 / 3.0), cast_dtype=t.dtype).to(_get_device()),
                 cast(t * 7.787037037037035 + 0.13793103448275862, cast_dtype=t.dtype).to(_get_device()))


def lab_finv(t: Tensor):
    return where(t > 0.20689655172413793, cast(t.pow(3.0), cast_dtype=t.dtype).to(_get_device()),
                 cast(0.12841854934601665 * (t - 0.13793103448275862), cast_dtype=t.dtype).to(_get_device()))


def lab2xyz(lab: Tensor, wref=None):
    """
    input lab as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    l
    """
    if len(lab.shape) == 4:
        if int_shape(lab)[-1] == 3:
            lab = lab.permute(0, 3, 1, 2)
        elif int_shape(lab)[1] == 3:
            pass
        lab[:, 0, :, :] = lab[:, 0, :, :] / 100.0
        lab[:, 1:, :, :] = (lab[:, 1:, :, :] - 127) / 128
    elif len(lab.shape) == 3:
        if int_shape(lab)[-1] == 3:
            lab = lab.permute(2, 0, 1)
        elif int_shape(lab)[0] == 3:
            pass
        lab[0, :, :] = lab[0, :, :] / 100.0
        lab[1:, :, :] = (lab[1:, :, :] - 127) / 128
    if wref is None:
        wref = D65
    dim = 1 if len(lab.shape) == 4 else 0
    l, a, b = lab.chunk(3, dim=dim)

    l2 = (l + 0.16) / 1.16
    x = wref[0] * lab_finv(l2 + a / 5)
    y = wref[1] * lab_finv(l2)
    z = wref[2] * lab_finv(l2 - b / 2)
    xyz = torch.cat([x, y, z], dim=dim)
    xyz = xyz * 255.0
    if len(xyz.shape) == 4:
        if int_shape(xyz)[-1] == 3:
            return xyz
        elif int_shape(xyz)[1] == 3:
            xyz = xyz.permute(0, 2, 3, 1)
            return xyz
    elif len(xyz.shape) == 3:
        if int_shape(xyz)[-1] == 3:
            return xyz
        elif int_shape(xyz)[0] == 3:
            xyz = xyz.permute(1, 2, 0)
            return xyz

    raise ValueError('image should channel-last')


def xyz2lab(xyz: Tensor, wref=None):
    """
    input xyz as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if len(xyz.shape) == 4:
        if int_shape(xyz)[-1] == 3:
            xyz = xyz.permute(0, 3, 1, 2)
        elif int_shape(xyz)[1] == 3:
            pass
    elif len(xyz.shape) == 3:
        if int_shape(xyz)[-1] == 3:
            xyz = xyz.permute(2, 0, 1)
        elif int_shape(xyz)[0] == 3:
            pass
    xyz = xyz / 255.0
    if wref is None:
        wref = D65
    dim = 1 if len(xyz.shape) == 4 else 0
    x, y, z = xyz.chunk(3, dim=dim)

    fy = lab_f(y / wref[1])
    l = 1.16 * fy - 0.16
    a = 5.0 * (lab_f(x / wref[0]) - fy)
    b = 2.0 * (fy - lab_f(z / wref[2]))
    lab = torch.cat([clip(l, 0, 1) * 100, clip(a, -1, 1) * 128 + 127, clip(b, -1, 1) * 128 + 127], dim=dim)

    if len(lab.shape) == 4:
        if int_shape(lab)[-1] == 3:
            return lab
        elif int_shape(lab)[1] == 3:
            lab = lab.permute(0, 2, 3, 1)
            return lab

    elif len(lab.shape) == 3:
        if int_shape(lab)[-1] == 3:
            return lab
        elif int_shape(lab)[0] == 3:
            lab = lab.permute(1, 2, 0)
            return lab

    raise ValueError('image should channel-last')


def lab2rgb(lab: Tensor):
    """
    input lab as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if ndim(lab) == 4:
        channel_idx = 1
        if int_shape(lab)[1] == 3:
            channel_idx = 1
        elif int_shape(lab)[-1] == 3:
            channel_idx = -1
        rgb = xyz2rgb(lab2xyz(lab))
        if channel_idx == 1:
            rgb = rgb.permute(0, 3, 1, 2)
        return rgb
    else:
        rgb = xyz2rgb(lab2xyz(lab))
        return rgb


def rgb2lab(rgb: Tensor):
    """
    input rgb as pytorch tensor of size (batch_size, 3, h, w) or (3, h, w)
    """
    if ndim(rgb) == 4:
        channel_idx = 1
        if int_shape(rgb)[1] == 3:
            channel_idx = 1
        elif int_shape(rgb)[-1] == 3:
            channel_idx = -1
        xyz = xyz2lab(rgb2xyz(rgb))
        if channel_idx == 1:
            xyz = xyz.permute(0, 3, 1, 2)
        return xyz
    else:
        xyz = xyz2lab(rgb2xyz(rgb))
        return xyz


def gray2rgb(gray: Tensor):
    """Compute luminance of an RGB image.

    Args:
        gray(tensor):  gray-scale image ;shape: (H,W)
    Returns:
        rgb (tensor):  rgb image ; shape:(H,W,C)

    """
    gray = gray.copy().to(_float_dtype)
    if ndim(gray) == 3 and int_shape(gray)[-1] == 1:
        gray = gray[:, :, 0]
    if ndim(gray) != 2:
        raise ValueError('input gray image ndim should equal 2 but get {0}'.format(ndim(gray)))
    rgb = stack([gray, gray, gray], axis=-1)
    return rgb


############################
# bounding box
###########################

@numpy_compatible
def xywh2xyxy(boxes, image_size=None):
    """
    Args:
        boxes (tensor or ndarray):
            boxes  with xywh  (centerx,centery,width, height) format
            boxes shape should be [n,m] m>=4
        image_size (size): (height, width)
    Returns
        xyxy (x1,y1,x2,y2)
    """
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""

    if is_tensor(boxes):
        class_info = None
        if boxes.shape[-1] > 4:
            class_info = boxes[:, 4:]
            boxes = boxes[:, :4]
        x1y1 = clip(boxes[:, 0:2] - boxes[:, 2:4] / 2, 0)
        x2y2 = clip(x1y1 + boxes[:, 2:4], 0)
        if class_info is not None:
            boxes = concate([x1y1, x2y2, class_info], axis=-1)
        else:
            boxes = concate([x1y1, x2y2], axis=-1)
        return boxes

    else:
        raise TypeError('Argument xywh must be a list, tuple, numpy array or tensor.')


@numpy_compatible
def xyxy2xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""

    if is_tensor(boxes):
        if boxes.ndim == 1:
            boxes = expand_dims(boxes, 0)
        if boxes.shape[-1] > 4:
            return concate([(boxes[:, 2:4] + boxes[:, 0:2]) / 2,  # cx, cy
                            boxes[:, 2:4] - boxes[:, 0:2], boxes[:, 4:]], 1)  # w, h
        else:
            return concate([(boxes[:, 2:4] + boxes[:, 0:2]) / 2,  # cx, cy
                            boxes[:, 2:4] - boxes[:, 0:2]], 1)  # w, h
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


@numpy_compatible
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


@numpy_compatible
def bbox_iou(bboxes1, bboxes2):
    """

    Args:
        bboxes1 (Tensor): shape (n, 4)
        bboxes2 (Tensor): shape (n, 4)

    Returns:
         ious(Tensor): shape (n)

    Examples;
    >>> boxes1=to_tensor(np.array([[39, 63, 203, 112], [49, 75, 203, 125],[31, 69, 201, 125],[50, 72, 197, 121],[35, 51, 196, 110]]))
    >>> boxes2=to_tensor(np.array([[54, 66, 198, 114], [42, 78, 186, 126], [18, 63, 235, 135],[54, 72, 198, 120],[36, 60, 180, 108]]))
    >>> print(bbox_iou(boxes1,boxes2).cpu())
    tensor([0.7958, 0.7878, 0.6093, 0.9466, 0.7277])
    >>> iou_loss=(1-bbox_iou(boxes1,boxes2)).sum()/(boxes1.shape[0])
    >>> print(iou_loss.cpu())
    tensor(0.2266)
    """

    bboxes1 = bboxes1.to(_float_dtype)
    bboxes2 = bboxes2.to(_float_dtype)
    x1, y1, x2, y2 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    x1g, y1g, x2g, y2g = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    x2 = torch.maximum(x1, x2)
    y2 = torch.maximum(y1, y2)

    xkis1 = torch.maximum(x1, x1g)
    ykis1 = torch.maximum(y1, y1g)
    xkis2 = torch.minimum(x2, x2g)
    ykis2 = torch.minimum(y2, y2g)

    intsctk = zeros(x1.size()).to(_float_dtype)
    mask = ((ykis2 > ykis1) * (xkis2 > xkis1)).bool()

    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    return iouk


@numpy_compatible
def bbox_diou(bboxes1, bboxes2):
    """

    Args:
        bboxes1 (Tensor): shape (n, 4)
        bboxes2 (Tensor): shape (n, 4)

    Returns:
         ious(Tensor): shape (n)

    Examples;
    >>> boxes1=to_tensor(np.array([[39, 63, 203, 112], [49, 75, 203, 125],[31, 69, 201, 125],[50, 72, 197, 121],[35, 51, 196, 110]]))
    >>> boxes2=to_tensor(np.array([[54, 66, 198, 114], [42, 78, 186, 126], [18, 63, 235, 135],[54, 72, 198, 120],[36, 60, 180, 108]]))
    >>> bbox_diou(boxes1,boxes2).cpu()
    tensor([0.7947, 0.7826, 0.6071, 0.9464, 0.7253])
    >>> iou_loss=(1-bbox_diou(boxes1,boxes2)).sum()/(boxes1.shape[0])
    >>> print(iou_loss.cpu())
    tensor(0.2288)

    """
    bboxes1 = bboxes1.to(_float_dtype)
    bboxes2 = bboxes2.to(_float_dtype)
    x1, y1, x2, y2 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    x1g, y1g, x2g, y2g = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = zeros(x1.size()).to(_float_dtype)
    mask = ((ykis2 > ykis1) * (xkis2 > xkis1)).bool()
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk = iouk - u
    return diouk


@numpy_compatible
def bbox_ciou(bboxes1, bboxes2):
    """

    Args:
        bboxes1 (Tensor): shape (n, 4)
        bboxes2 (Tensor): shape (n, 4)

    Returns:
         ious(Tensor): shape (n)

    Examples;
    >>> boxes1=to_tensor(np.array([[39, 63, 203, 112], [49, 75, 203, 125],[31, 69, 201, 125],[50, 72, 197, 121],[35, 51, 196, 110]]))
    >>> boxes2=to_tensor(np.array([[54, 66, 198, 114], [42, 78, 186, 126], [18, 63, 235, 135],[54, 72, 198, 120],[36, 60, 180, 108]]))
    >>> bbox_ciou(boxes1,boxes2).cpu()
    tensor([0.7947, 0.7826, 0.6071, 0.9464, 0.7253])
    >>> iou_loss=(1-bbox_ciou(boxes1,boxes2)).sum()/(boxes1.shape[0])
    >>> print(iou_loss.cpu())
    tensor(0.2288)

    """
    bboxes1 = bboxes1.to(_float_dtype)
    bboxes2 = bboxes2.to(_float_dtype)
    x1, y1, x2, y2 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    x1g, y1g, x2g, y2g = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = zeros(x1.size()).to(_float_dtype)
    mask = ((ykis2 > ykis1) * (xkis2 > xkis1)).bool()
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + ctx.epsilon
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + ctx.epsilon
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        S = 1 - iouk
        alpha = v / (S + v)
    ciouk = iouk - (u + alpha * v)
    return ciouk


@numpy_compatible
def bbox_giou(bboxes1, bboxes2):
    """

    Args:
        bboxes1 (Tensor): shape (n, 4)
        bboxes2 (Tensor): shape (n, 4)

    Returns:
         ious(Tensor): shape (n)

    Examples;
    >>> boxes1=to_tensor(np.array([[39, 63, 203, 112], [49, 75, 203, 125],[31, 69, 201, 125],[50, 72, 197, 121],[35, 51, 196, 110]]))
    >>> boxes2=to_tensor(np.array([[54, 66, 198, 114], [42, 78, 186, 126], [18, 63, 235, 135],[54, 72, 198, 120],[36, 60, 180, 108]]))
    >>> bbox_giou(boxes1,boxes2).cpu()
    tensor([0.7910, 0.7832, 0.6093, 0.9465, 0.7277])
    >>> iou_loss=(1-bbox_giou(boxes1,boxes2)).sum()/(boxes1.shape[0])
    >>> print(iou_loss.cpu())
    tensor(0.2285)





    """
    bboxes1 = bboxes1.to(_float_dtype)
    bboxes2 = bboxes2.to(_float_dtype)
    x1, y1, x2, y2 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    x1g, y1g, x2g, y2g = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = zeros(x1.size()).to(_float_dtype)
    mask = ((ykis2 > ykis1) * (xkis2 > xkis1)).bool()

    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    giouk = iouk - ((area_c - unionk) / area_c)
    return giouk


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    return visionop.nms(boxes, scores, iou_threshold)


############################
# summary
###########################

def torch_rot90_(x: Tensor):
    return x.transpose_(2, 3).flip(2)


def torch_rot90(x: Tensor):
    return x.transpose(2, 3).flip(2)


def torch_rot180(x: Tensor):
    return x.flip(2).flip(3)


def torch_rot270(x: Tensor):
    return x.transpose(2, 3).flip(3)


def torch_flipud(x: Tensor):
    """
    Flip image tensor vertically
    :param x:
    :return:
    """
    return x.flip(2)


def torch_fliplr(x: Tensor):
    """
    Flip image tensor horizontally
    :param x:
    :return:
    """
    return x.flip(3)


def pad_image_tensor(image_tensor: Tensor, pad_size: int = 32):
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    if (isinstance(pad_size, Sized) and isinstance(pad_size, Iterable) and len(pad_size) == 2):
        pad_height, pad_width = [int(val) for val in pad_size]
    elif isinstance(pad_size, int):
        pad_height = pad_width = pad_size
    else:
        raise ValueError(
            "Unsupported pad_size: {pad_size}, must be either tuple(pad_rows,pad_cols) or single int scalar.")

    if rows > pad_height:
        pad_rows = rows % pad_height
        pad_rows = pad_height - pad_rows if pad_rows > 0 else 0
    else:
        pad_rows = pad_height - rows

    if cols > pad_width:
        pad_cols = cols % pad_width
        pad_cols = pad_width - pad_cols if pad_cols > 0 else 0
    else:
        pad_cols = pad_width - cols

    if pad_rows == 0 and pad_cols == 0:
        return image_tensor, (0, 0, 0, 0)

    pad_top = pad_rows // 2
    pad_btm = pad_rows - pad_top

    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    pad = [pad_left, pad_right, pad_top, pad_btm]
    image_tensor = torch.nn.functional.pad(image_tensor, pad)
    return image_tensor, pad


#
# def unpad_image_tensor(image_tensor, pad):
#     pad_left, pad_right, pad_top, pad_btm = pad
#     rows, cols = image_tensor.size(2), image_tensor.size(3)
#     return image_tensor[..., pad_top: rows - pad_btm, pad_left: cols - pad_right]
#
#
# def unpad_xyxy_bboxes(bboxes_tensor: Tensor, pad, dim=-1):
#     pad_left, pad_right, pad_top, pad_btm = pad
#     pad = torch.tensor([pad_left, pad_top, pad_left, pad_top], dtype=bboxes_tensor.dtype).to(bboxes_tensor.device)
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
def angle_to_rotation_matrix(angle) -> Tensor:
    """
    Creates a rotation matrix out of angles in degrees
    Args:
        angle: (Tensor): tensor of angles in degrees, any shape.

    Returns:
        Tensor: tensor of *x2x2 rotation matrices.

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*, 2, 2)`

    Examples:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = angle_to_rotation_matrix(input)  # Nx3x2x2
    """
    ang_rad = angle * np.pi / 180
    cos_a = torch.cos(ang_rad)
    sin_a = torch.sin(ang_rad)
    return torch.stack([cos_a, sin_a, -sin_a, cos_a], dim=-1).view(*angle.shape, 2, 2)


def get_rotation_matrix2d(center: Tensor, angle, scale) -> Tensor:
    r"""Calculates an affine matrix of 2D rotation.

    The function calculates the following matrix:

    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}

    where

    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})

    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.

    Args:
        center (Tensor,tuple): center of the rotation in the source image.
        angle (Tensor,float): rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner).
        scale (Tensor,float): isotropic scale factor.

    Returns:
        Tensor: the affine matrix of 2D rotation.

    Shape:
        - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
        - Output: :math:`(B, 2, 3)`

    Examples:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1)
        >>> angle = 45. * torch.ones(1)
        >>> get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    """
    center = to_tensor(center)
    angle = to_tensor(angle)
    scale = to_tensor(scale)

    if len(center) == 2 and ndim(center) == 1:
        center = center.unsqueeze(0)
    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}".format(center.shape))

    # convert angle and apply scale
    scaled_rotation = angle_to_rotation_matrix(angle) * scale.view(-1, 1, 1)
    alpha = scaled_rotation[:, 0, 0]
    beta = scaled_rotation[:, 0, 1]

    # unpack the center to x, y coordinates
    x = center[..., 0]
    y = center[..., 1]

    # create output tensor
    batch_size = center.shape[0]
    M = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
    M[..., 0:2, 0:2] = scaled_rotation
    M[..., 0, 2] = (torch.tensor(1.) - alpha) * x - beta * y
    M[..., 1, 2] = beta * x + (torch.tensor(1.) - alpha) * y
    return M


def _compute_rotation_matrix(angle: Tensor, center: Tensor) -> Tensor:
    """Computes a pure affine rotation matrix."""
    scale = torch.ones_like(angle)
    matrix_tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix_tensor


#
#
# def _compute_translation_matrix(translation: Tensor) -> Tensor:
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
# def _compute_scaling_matrix(scale: Tensor, center: Tensor) -> Tensor:
#     """Computes affine matrix for scaling."""
#     angle_tensor = torch.zeros_like(scale)
#     matrix_tensor = get_rotation_matrix2d(center, angle_tensor, scale)
#     return matrix_tensor
#
#
# def _compute_shear_matrix(shear: Tensor) -> Tensor:
#     """Computes affine matrix for shearing."""
#     matrix_tensor = torch.eye(3, device=shear.device, dtype=shear.dtype)
#     matrix = matrix_tensor.repeat(shear.shape[0], 1, 1)
#
#     shx, shy = torch.chunk(shear, chunks=2, dim=-1)
#     matrix[..., 0, 1:2] += shx
#     matrix[..., 1, 0:1] += shy
#     return matrix
#

# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166

def normal_transform_pixel(height, width):
    """

    Args:
        height ():
        width ():

    Returns:

    """
    tr_mat = Tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])  # 1x3x3
    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)

    tr_mat = tr_mat.unsqueeze(0)

    return tr_mat


def dst_norm_to_dst_norm(dst_pix_trans_src_pix, dsize_src, dsize_dst):
    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst
    # the devices and types
    device = dst_pix_trans_src_pix.device
    dtype = dst_pix_trans_src_pix.dtype
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel(src_h, src_w).to(device).to(dtype)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix = normal_transform_pixel(dst_h, dst_w).to(device).to(dtype)
    # compute chain transformations
    dst_norm_trans_src_norm = torch.matmul(dst_norm_trans_dst_pix,
                                           torch.matmul(dst_pix_trans_src_pix, src_pix_trans_src_norm))
    return dst_norm_trans_src_norm


def transform_points(trans_01: Tensor, points_1: Tensor) -> Tensor:
    r"""Function that applies transformations to a set of points.
    Args:
        trans_01 (Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        Tensor: tensor of N-dimensional points.
    Shape:
        - Output: :math:`(B, N, D)`

    Examples:

        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """
    if not torch.is_tensor(trans_01) or not torch.is_tensor(points_1):
        raise TypeError("Input type is not a Tensor")
    if not trans_01.device == points_1.device:
        raise TypeError("Tensor must be in the same device")
    if not trans_01.shape[0] == points_1.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differe by one unit")
    # to homogeneous
    points_1_h = torch.nn.functional.pad(points_1, [0, 1], "constant", 1.0)
    # transform coordinates
    points_0_h = torch.matmul(trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    z_vec_tensor = points_0_h[..., -1:]
    mask_tensor = torch.abs(z_vec_tensor) > 1e-8
    scale_tensor = torch.ones_like(z_vec_tensor).masked_scatter_(mask_tensor,
                                                                 torch.tensor(1.0) / z_vec_tensor[mask_tensor])

    return scale_tensor * points_0_h[..., :-1]


def warp_grid(dst_homo_src: Tensor, dsize) -> Tensor:
    r"""Computes the grid to warp the coordinates grid by homography.

    Args:
        dsize ():
        dst_homo_src (Tensor): Homography or homographies (stacked) to
                          transform all points in the grid. Shape of the
                          homography has to be :math:`(N, 3, 3)`.

    Returns:
        Tensor: the transformed grid of shape :math:`(N, H, W, 2)`.
    """
    height, width = dsize
    grid = meshgrid(height, width, normalized_coordinates=True)

    batch_size = dst_homo_src.shape[0]
    device = dst_homo_src.device
    dtype = dst_homo_src.dtype
    # expand grid to match the input batch size
    grid_tensor = grid.expand(batch_size, -1, -1, -1)  # NxHxWx2
    if len(dst_homo_src.shape) == 3:  # local homography case
        dst_homo_src = dst_homo_src.view(batch_size, 1, 3, 3)  # NxHxWx3x3
    # perform the actual grid transformation,
    # the grid is copied to input device and cast to the same type
    flow_tensor = transform_points(dst_homo_src, grid_tensor.to(device).to(dtype))  # NxHxWx2
    return flow_tensor.view(batch_size, height, width, 2)  # NxHxWx2


def warp_affine(src: Tensor, M: Tensor, dsize: Tuple[int, int], mode: Optional[str] = 'bilinear',
                padding_mode: Optional[str] = 'zeros') -> Tensor:
    r"""Applies an affine transformation to a tensor.

    The function warp_affine transforms the source tensor using
    the specified matrix:

    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )

    Args:
        src (Tensor): input tensor of shape :math:`(B, C, H, W)`.
        M (Tensor): affine transformation of shape :math:`(B, 2, 3)`.
        dsize (Tuple[int, int]): size of the output image (height, width).
        mode (Optional[str]): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (Optional[str]): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.

    Returns:
        Tensor: the warped tensor.

    Shape:
        - Output: :math:`(B, C, H, W)`

    .. note::
       See a working example `here <https://github.com/arraiyopensource/
       kornia/blob/master/docs/source/warp_affine.ipynb>`__.
    """
    if not torch.is_tensor(src):
        raise TypeError("Input src type is not a Tensor. Got {}".format(type(src)))
    if not torch.is_tensor(M):
        raise TypeError("Input M type is not a Tensor. Got {}".format(type(M)))
    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}".format(src.shape))
    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}".format(src.shape))
    try:
        # we generate a 3x3 transformation matrix from 2x3 affine
        M_3x3_tensor = F.pad(M, [0, 0, 0, 1, 0, 0], mode="constant", value=0)
        M_3x3_tensor[:, 2, 2] += 1.0

        dst_norm_trans_dst_norm = dst_norm_to_dst_norm(M_3x3_tensor, (src.shape[-2:]), dsize)
        # launches the warper
        return F.grid_sample(src, warp_grid(torch.inverse(dst_norm_trans_dst_norm), dsize=dsize), mode=mode,
                             padding_mode=padding_mode)
    except Exception:
        PrintException()
        return None


def affine(tensor: Tensor, matrix: Tensor) -> Tensor:
    r"""Apply an affine transformation to the image.

    Args:
        tensor (Tensor): The image tensor to be warped.
        matrix (Tensor): The 2x3 affine transformation matrix.

    Returns:
        Tensor: The warped image.
    """
    # warping needs data in the shape of BCHW
    is_unbatched = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    warped_tensor = warp_affine(tensor, matrix, (height, width))

    # return to the original shape
    if is_unbatched:
        warped_tensor = torch.squeeze(warped_tensor, dim=0)

    return warped_tensor


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185

#
# def get_rotation_matrix2d(center, angle, scale):
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
#         center (Tensor): center of the rotation in the source image.
#         angle (Tensor): rotation angle in degrees. Positive values mean
#             counter-clockwise rotation (the coordinate origin is assumed to
#             be the top-left corner).
#         scale (Tensor): isotropic scale factor.
#
#     Returns:
#         Tensor: the affine matrix of 2D rotation.
#
#     Shape:
#         - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
#         - Output: :math:`(B, 2, 3)`
#
#     Example:
#         >>> center = torch.zeros(1, 2)
#         >>> scale = torch.ones(1)
#         >>> angle = 45. * torch.ones(1)
#         >>> M = tgm.get_rotation_matrix2d(center, angle, scale)
#         tensor([[[ 0.7071,  0.7071,  0.0000],
#                  [-0.7071,  0.7071,  0.0000]]])
#     """
#     if not torch.is_tensor(center):
#         raise TypeError("Input center type is not a torch.Tensor. Got {}"
#                         .format(type(center)))
#     if not torch.is_tensor(angle):
#         raise TypeError("Input angle type is not a torch.Tensor. Got {}"
#                         .format(type(angle)))
#     if not torch.is_tensor(scale):
#         raise TypeError("Input scale type is not a torch.Tensor. Got {}"
#                         .format(type(scale)))
#     if not (len(center.shape) == 2 and center.shape[1] == 2):
#         raise ValueError("Input center must be a Bx2 tensor. Got {}"
#                          .format(center.shape))
#     if not len(angle.shape) == 1:
#         raise ValueError("Input angle must be a B tensor. Got {}"
#                          .format(angle.shape))
#     if not len(scale.shape) == 1:
#         raise ValueError("Input scale must be a B tensor. Got {}"
#                          .format(scale.shape))
#     if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
#         raise ValueError("Inputs must have same batch size dimension. Got {}"
#                          .format(center.shape, angle.shape, scale.shape))
#     # convert angle and apply scale
#     angle_rad = torch.deg2rad(angle)
#     alpha = torch.cos(angle_rad) * scale
#     beta = torch.sin(angle_rad) * scale
#
#     # unpack the center to x, y coordinates
#     x, y = center[..., 0], center[..., 1]
#
#     # create output tensor
#     batch_size, _ = center.shape
#     M = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
#     M[..., 0, 0] = alpha
#     M[..., 0, 1] = beta
#     M[..., 0, 2] = (1. - alpha) * x - beta * y
#     M[..., 1, 0] = -beta
#     M[..., 1, 1] = alpha
#     M[..., 1, 2] = beta * x + (1. - alpha) * y
#     return M
#
# def _compute_rotation_matrix(angle: torch.Tensor,
#                              center: torch.Tensor) -> torch.Tensor:
#     """Computes a pure affine rotation matrix."""
#     scale: torch.Tensor = torch.ones_like(center)
#     matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
#     return matrix
#
# def rotate(tensor: Tensor, angle: Tensor) -> Tensor:
#     r"""Rotate the image anti-clockwise about the centre.
#
#     See :class:`~kornia.Rotate` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not a Tensor. Got {}".format(type(tensor)))
#     if not torch.is_tensor(angle):
#         raise TypeError("Input angle type is not a Tensor. Got {}".format(type(angle)))
#
#     if len(tensor.shape) not in (3, 4,):
#         raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
#                          "Got: {}".format(tensor.shape))
#
#     # compute the rotation matrix
#     # TODO: add broadcasting to get_rotation_matrix2d for center
#     angle = angle.expand(tensor.shape[0])
#     center = torch.tensor([(tensor.size(4) - 1) / 2, (tensor.size(3) - 1) / 2]).expand(tensor.shape[0], -1).to(
#         tensor.device)
#     rotation_matrix = _compute_rotation_matrix(angle, center)
#
#     # warp using the affine transform
#     return affine(tensor, rotation_matrix[..., :2, :3])
#
#
# def translate(tensor: Tensor, translation: Tensor) -> Tensor:
#     r"""Translate the tensor in pixel units.
#
#     See :class:`~kornia.Translate` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not a Tensor. Got {}".format(type(tensor)))
#     if not torch.is_tensor(translation):
#         raise TypeError("Input translation type is not a Tensor. Got {}".format(type(translation)))
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
# def scale(tensor: Tensor, scale_factor: Tensor) -> Tensor:
#     r"""Scales the input image.
#
#     See :class:`~kornia.Scale` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not a Tensor. Got {}".format(type(tensor)))
#     if not torch.is_tensor(scale_factor):
#         raise TypeError("Input scale_factor type is not a Tensor. Got {}".format(type(scale_factor)))
#
#     # compute the tensor center
#
#     # compute the rotation matrix
#     # TODO: add broadcasting to get_rotation_matrix2d for center
#     center = torch.tensor([(tensor.size(4) - 1) / 2, (tensor.size(3) - 1) / 2]).expand(tensor.shape[0], -1).to(
#         tensor.device)
#     scale_factor = scale_factor.expand(tensor.shape[0])
#     scaling_matrix = _compute_scaling_matrix(scale_factor, center)
#
#     # warp using the affine transform
#     return affine(tensor, scaling_matrix[..., :2, :3])
#
#
# def shear(tensor: Tensor, shear: Tensor) -> Tensor:
#     r"""Shear the tensor.
#
#     See :class:`~kornia.Shear` for details.
#     """
#     if not torch.is_tensor(tensor):
#         raise TypeError("Input tensor type is not a Tensor. Got {}".format(type(tensor)))
#     if not torch.is_tensor(shear):
#         raise TypeError("Input shear type is not a Tensor. Got {}".format(type(shear)))
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


_FUN_NAMES = [
    # source_fun, target_fun
    ('to_numpy', to_numpy),
    ('copy', copy),
    ('numel', numel),
    ('ndim', ndim),
    ('int_shape', int_shape),
    ('cast', cast),
    ('to', to),
    ('is_sparse', is_sparse),
    ('is_nan', is_nan),
    ('is_inf', is_inf),
    ('is_abnormal_number', is_abnormal_number),
    ('any_nan', any_nan),
    ('any_inf', any_inf),
    ('any_abnormal_number', any_abnormal_number),
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
    ('expand_dims', expand_dims),
    ('repeat_elements', repeat_elements),
    ('gather', gather),
    ('gram_matrix', gram_matrix),
    ('shuffle', shuffle),
    ('random_choice', random_choice),
    ('random_normal_like', random_normal_like)
]
for target_fun_name, source_fun in _FUN_NAMES:
    if not hasattr(Tensor, target_fun_name):
        setattr(Tensor, target_fun_name, source_fun)
    elif target_fun_name in ["to", "float", "int", "long", "sum", "mean"]:
        setattr(Tensor, target_fun_name, source_fun)
del _FUN_NAMES
