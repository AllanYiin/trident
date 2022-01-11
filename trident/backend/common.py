""" common define the session ,basic class and basic function without internal dependency """
import builtins
import collections
import copy
import json
import datetime
import functools
import importlib
import inspect
import linecache
import math
import numbers
import operator
import os
from time import sleep
import platform
import re
import shlex
import string
import struct
import webbrowser
import subprocess
import sys
import time
import traceback
import types
from enum import Enum
from pydoc import locate
from typing import Iterable, Generator, Union, Tuple, Any, overload, NewType, Dict

import numpy as np

__all__ = ['get_session', 'set_session', 'get_session_value', 'is_autocast_enabled', 'set_autocast_enabled', 'get_backend', 'get_plateform', 'get_image_backend', 'get_trident_dir',
           'epsilon', 'floatx', 'import_or_install','compile_and_install_module',
           'check_keys', 'make_sure', 'if_none', 'camel2snake', 'snake2camel', 'to_onehot', 'to_list', 'addindent', 'format_time',
           'get_time_suffix', 'get_file_modified_time', 'get_function', 'get_class', 'get_terminal_size', 'gcd', 'get_divisors', 'isprime',
           'next_prime', 'prev_prime', 'nearest_prime', 'PrintException', 'TensorShape', 'unpack_singleton', 'enforce_singleton',
           'OrderedDict', 'map_function_arguments', 'ClassfierType', 'PaddingMode', 'Signature', 'is_iter', 'get_string_actual_length',
           'Interpolation', 'is_numpy', 'find_minimal_edit_distance_key', 'jaccard_similarity', 'text_similarity', 'levenshtein', 'is_alphabet', 'is_punctuation',
           'remove_nonprintable',

           'GetImageMode', 'split_path', 'make_dir_if_need', 'sanitize_path', 'ShortcutMode', 'adaptive_format', 'num_cpus',
           'get_args_spec', 'get_gpu_memory_map', 'get_memory_profile', 'red_color', 'green_color', 'cyan_color', 'blue_color', 'orange_color',
           'gray_color', 'yellow_color','magenta_color','violet_color','open_browser','launchTensorBoard','launchMLFlow']

# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
import six
from trident import context
import warnings

from math import e , nan , inf , pi
__all__.extend(['e', 'pi', 'nan', 'inf'])

warnings.simplefilter("ignore")
_int = builtins.int
_float = builtins.float
_bool = builtins.bool

_SESSION = context._context()


def sanitize_path(path):
    """

    Args:
        path (str): a path of file or folder

    Returns:
        sanitized path

    """
    if isinstance(path, str):
        return os.path.normpath(path.strip()).replace('\\', '/')
    else:
        return path


def split_path(path: str):
    """split path into folder, filename and ext

    Args:
        path (str): a path of file or folder

    Returns:

    """
    if path is None or len(path) == 0:
        return '', '', ''
    path = sanitize_path(path)
    folder, filename = os.path.split(path)
    ext = ''
    if '.' in filename:
        filename, ext = os.path.splitext(filename)
        # handle double ext, like 'mode.pth.tar'
        filename, ext2 = os.path.splitext(filename)
        ext = ext2 + ext
    else:
        folder = os.path.join(folder, filename)
        filename = ''
    return folder, filename, ext


def make_dir_if_need(path):
    """Check the base folder in input path whether exist, if not , then create it.

    Args:
        path (str): a path of file or folder

    Returns:
        sanitized path

    """
    folder, filename, ext = split_path(path)
    if len(folder) > 0 and not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            PrintException()
            sys.stderr.write('folder:{0} is not valid path'.format(folder))
    return sanitize_path(path)


def if_none(a, b):
    "`b` if `a` is None else `a`"
    return a if a is not None else b


def get_trident_dir():
    """Method for access trident_dir attribute in session

    Returns:
        trident_dir in _SESSION
    Example
        >>> print(get_trident_dir())
        '~/.trident'

    """
    return context._context().trident_dir


def get_plateform():
    """

    Returns:
        check current system os plateform.

    """
    plateform_str = platform.system().lower()
    if 'darwin' in plateform_str:
        return 'mac'
    elif 'linux' in plateform_str:
        return 'linux'
    elif 'win' in plateform_str:
        return 'windows'
    else:
        return plateform_str


def get_session():
    """

    Returns:
        the trident _SESSION

    """
    return _SESSION


def get_session_value(key):
    """

    Returns:
        the trident _SESSION

    """
    if hasattr(_SESSION, key):
        return getattr(_SESSION, key)
    else:
        return None


def set_session(key, value):
    """
    Assign new value to specified attribute in  _SESSION
    Args:
        key (str): attribute key
        value (obj): new value

    Returns:

    """
    setattr(_SESSION, key, value)
    return _SESSION


def get_backend():
    return context._context().get_backend()


def get_image_backend():
    return context._context().image_backend


def get_device():
    return context._context().device


def is_autocast_enabled():
    return get_session_value('is_autocast_enabled')


def set_autocast_enabled(enabled: bool):
    set_session('is_autocast_enabled', enabled)


def _is_c_contiguous(data):
    while isinstance(data, list):
        data = data[0]
    return data.flags.c_contiguous


def epsilon():
    """Method for access epsilon attribute in session

    Returns: a float

    Example
        >>> print(epsilon())
        1e-08

    """
    return _SESSION.epsilon


def floatx():
    """Returns the default float type, as a string.
    "e.g. 'float16', 'float32', 'float64').

 Returns
        String, the current default float type.

    # Example
    >>> print(floatx())
    float32

    """
    return _SESSION.floatx


def camel2snake(string1):
    """ Convert string from camelCase style to snake_case style

    Args:
        string1 (str): camelCase style string

    Returns:
        snake_case style  string

    """
    if string1 is None:
        return None
    else:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string1)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake2camel(string1):
    """Convert string from snake_case style to camelCase style

    Args:
        string1 (str):snake_case style  string

    Returns:
        camelCase style string

    """
    if string1 is None:
        return None
    else:
        return ''.join(x.capitalize() or '_' for x in string1.split('_'))


def adaptive_format(num: numbers.Number, prev_value: Union[numbers.Number, Iterable] = None, value_type=None, name=None):
    valid_value_type = ['loss', 'metric']
    percentage_name = ['accuracy', 'rate', 'ratio', 'iou', 'recall', 'rmse', 'similarity', 'fitness', 'utility']

    if num is None:
        return 'none'
    elif name is not None and any([name.lower().endswith(n) for n in percentage_name]):
        return '{0:.3%}'.format(num)

    is_current_num_integer = math.modf(num)[0] == 0
    digitpart_list = [math.modf(v)[0] for v in prev_value] if isinstance(prev_value, Iterable) else []
    is_all_history_integer = all([math.modf(v)[0] == 0 for v in prev_value]) if isinstance(prev_value, Iterable) \
        else math.modf(num)[0] == 0 if isinstance(prev_value, numbers.Number) else False

    if isinstance(num, numbers.Integral) or is_current_num_integer and is_all_history_integer:
        if not isinstance(num, numbers.Integral):
            num = int(num)
        return '{0:,}'.format(num)
    elif isinstance(prev_value, Iterable) and len(prev_value) > 1:
        if value_type != 'loss' and all([1.2 >= s >= 0.001 or -0.001 >= s >= -1.2 or s == 0 for s in prev_value]):
            return '{0:.3%}'.format(num)
        elif len(prev_value) > 0:
            digit = int(np.array([builtins.abs(builtins.min(math.log10(builtins.abs(s)), 0)) + 3 if s != 0 else 0 for s in prev_value]).mean())
            if digit > 8:
                return '{0:{1}}'.format(num, '.3e')
            elif digit < 4:
                return '{0:{1}}'.format(num, '.3f')
            elif digit <= 5:
                return '{0:{1}}'.format(num, '.4f')
            else:
                return '{0:.{1}%}'.format(num, digit)
    elif name is not None and len(name) >= 3 and any([name.lower() in s.lower() for s in percentage_name]):
        return '{0:.3%}'.format(num)
    elif name is not None and len(name) >= 3 and (name.endswith('s')):
        return '{0:{1}}'.format(num, '.3f')
    else:
        format_string = ',.3f'
        if value_type == 'metric':
            if math.modf(num)[0] == 0:
                num = int(num)
                return '{0:,}'.format(num)
            elif 1.5 >= num >= 0.001 or -0.001 >= num >= -1.5 or num == 0:
                return '{0:.3%}'.format(num)
            elif 10000>num  or num >= -10000:
                return '{0:{1}}'.format(num, '.3f')
            else:
                return '{0:{1}}'.format(num, '.3e')
        elif value_type == 'loss':
            if 10000 >= num >= 0.001 or -0.001 >= num >= -10000 or num == 0:
                return '{0:{1}}'.format(num, '.3f')
            else:
                return '{0:{1}}'.format(num, '.3e')

        return '{0:{1}}'.format(num, format_string)


def get_string_actual_length(input_string: str):
    """Get the string acutal length (considering Chinese double byte)

    Args:
        input_string ():

    Returns:

    Examples:
        >>> get_string_actual_length('你好')
        4
        >>> get_string_actual_length('深度學習deep learning')
        21

    """

    return builtins.sum(
        [len(input_string[i].encode("UTF-8")) if len(input_string[i].encode("UTF-8")) == 3 else len(input_string[i].encode("UTF-8")) for i in range(len(input_string))])


def PrintException():
    """
        Print exception with the line_no.

    """
    exc_type, exc_obj, tb = sys.exc_info()
    traceback.print_exception(*sys.exc_info())
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}\n'.format(filename, lineno, line.strip(), exc_obj))
    traceback.print_exc(limit=None, file=sys.stderr)
    # traceback.print_tb(tb, limit=1, file=sys.stdout)
    # traceback.print_exception(exc_type, exc_obj, tb, limit=2, file=sys.stdout)


def make_sure(bool_val, error_msg, *args):
    if not bool_val:
        raise ValueError("make_sure failure: " + error_msg % args)


class DeviceType(object):
    _Type = NewType('_Type', int)
    CPU = _Type(0)  # type: _Type
    CUDA = _Type(1)  # type: _Type

    type: str  # THPDevice_type
    index: _int  # THPDevice_index

#
# class dtype:
#     backend = get_backend()
#     if backend == 'pytorch':
#         import torch
#         # type definition
#         bool = torch.bool
#         int8 = torch.int8
#         byte = torch.int8
#         int16 = torch.int16
#         short = torch.int16
#         int32 = torch.int32
#         intc = torch.int32
#         int64 = torch.int64
#         intp = torch.int64
#
#         uint8 = torch.uint8
#         ubyte = torch.uint8
#         float16 = torch.float16
#         half = torch.float16
#         float32 = torch.float32
#         single = torch.float32
#         float64 = torch.float64
#         double = torch.float64
#         long = torch.int64
#         float = torch.float32
#         complex64 = torch.complex64
#         complex128 = torch.complex128
#         cfloat = torch.cfloat
#     elif backend == 'tensorflow':
#         import tensorflow as tf
#         bool = tf.bool
#         int8 = tf.int8
#         byte = tf.int8
#         int16 = tf.int16
#         short = tf.int16
#         int32 = tf.int32
#         intc = tf.int32
#         int64 = tf.int64
#         intp = tf.int64
#
#         uint8 = tf.uint8
#         ubyte = tf.uint8
#         float16 = tf.float16
#         half = tf.float16
#         float32 = tf.float32
#         single = tf.float32
#         float64 = tf.float64
#         double = tf.float64
#         long = tf.int64
#         float = tf.float32
#         complex64 = tf.complex64
#         complex128 = tf.complex128
#         cfloat = tf.complex64
#     elif backend == 'onnx':
#         from onnx import onnx_pb
#         bool = onnx_pb.TensorProto.BOOL
#         int8 = onnx_pb.TensorProto.INT8
#         byte = onnx_pb.TensorProto.INT8
#         int16 = onnx_pb.TensorProto.INT16
#         short = onnx_pb.TensorProto.INT16
#         int32 = onnx_pb.TensorProto.INT32
#         intc = onnx_pb.TensorProto.INT32
#         int64 = onnx_pb.TensorProto.INT64
#         intp = onnx_pb.TensorProto.INT64
#
#         uint8 = onnx_pb.TensorProto.UINT8
#         ubyte = onnx_pb.TensorProto.UINT8
#         float16 = onnx_pb.TensorProto.FLOAT1
#         half = onnx_pb.TensorProto.FLOAT1
#         float32 = onnx_pb.TensorProto.FLOAT
#         single = onnx_pb.TensorProto.FLOAT
#         float64 = onnx_pb.TensorProto.DOUBLE
#         double = onnx_pb.TensorProto.DOUBLE
#         long = onnx_pb.TensorProto.INT64
#         float = onnx_pb.TensorProto.FLOAT
#     elif backend == 'numpy':
#         bool = np.bool
#
#         int8 = np.int8
#         byte = np.int8
#         int16 = np.int16
#         short = np.int16
#         int32 = np.int32
#         intc = np.int32
#         int64 = np.int64
#         intp = np.int64
#
#         uint8 = np.uint8
#         ubyte = np.uint8
#         float16 = np.float16
#         half = np.float16
#         float32 = np.float32
#         single = np.float32
#         float64 = np.float64
#         double = np.float64
#         long = np.int64
#         float = np.float32
#         complex64 = np.complex64
#         complex128 = np.complex128
#         cfloat = np.complex64
#
#     def __repr__(self):
#         return 'dtype'


# class Device(object):
#     '''
#     Describes device type and device id
#     syntax: device_type:device_id(option
#     al)
#     example: 'CPU', 'CUDA', 'CUDA:1'
#     '''
#
#     def __init__(self, device):  # type: (Text) -> None
#         options = device.split(':')
#         self.type = getattr(DeviceType, options[0])
#         self.device_id = 0
#         if len(options) > 1:
#             self.device_id = int(options[1])
#
class device:
    type: str  # THPDevice_type
    index: int  # THPDevice_index

    # THPDevice_pynew
    @overload
    def __init__(self, device: Union[int, str]) -> None: ...

    @overload
    def __init__(self, type: str, index: int) -> None: ...

    def __reduce__(self) -> Tuple[Any, ...]: ...  # THPDevice_reduce


class TensorShape(object):
    # TODO: __reduce__
    """

    Examples:
        >>> a=TensorShape([2,128,64,64])
        >>> print(a)
        (2, 128, 64, 64)
        >>> print(a[2])
        64
        >>> print(a[1:3])
        (128, 64)
        >>> b=TensorShape([2,128, 64, 64])
        >>> b.is_compatible_with(a[1:3])
        True
        >>> c=TensorShape([None,128, 64, 64])
        >>> b.is_compatible_with(c)
        True
        >>> c.is_compatible_with(b)
        True
    """

    def __init__(self, dims):
        """Creates a new TensorShape with the given dimensions.
        Args:
          dims: A list of Dimensions, or None if the shape is unspecified.
        Raises:
          TypeError: If dims cannot be converted to a list of dimensions.
        """

        if dims is None:
            self._dims = None
        elif 'tensor' in dims.__class__.__name__.lower():
            if hasattr(dims, 'cpu'):
                dims.cpu()
            if hasattr(dims, 'detach'):
                dims.detach().numpy()
            if hasattr(dims, 'numpy'):
                dims = [d for d in dims]
            self._dims = [d for d in dims]
        elif isinstance(dims, (tuple, list)):  # Most common case.
            self._dims = [d if isinstance(d, numbers.Integral) else None if d is None else d.item() if hasattr(d, "item") else d.numpy() for d in dims]
        elif isinstance(dims, TensorShape):
            self._dims = dims.dims
        else:
            try:
                dims_iter = iter(dims)
            except TypeError:
                # Treat as a singleton dimension
                self._dims = to_list(dims)
            else:
                self._dims = []
                for d in dims_iter:
                    try:
                        self._dims.append(d)
                    except TypeError as e:
                        six.raise_from(
                            TypeError(
                                "Failed to convert '{0!r}' to a shape: '{1!r}'"
                                "could not be converted to a dimension. A shape should "
                                "either be single dimension (e.g. 10), or an iterable of "
                                "dimensions (e.g. [1, 10, None])."
                                    .format(dims, d)), e)

    def __repr__(self):
        if self._dims is not None:
            return "TensorShape(%r)" % [dim for dim in self._dims]
        else:
            return "TensorShape(None)"

    def __str__(self):
        if self._dims is None:
            return "<unknown>"
        elif self.rank == 1:
            return "(%s,)" % self.dims[0]
        else:
            return "(%s)" % ", ".join(str(d) for d in self.dims)

    @property
    def rank(self):
        """Returns the rank of this shape, or None if it is unspecified."""
        if self._dims is not None:
            return len(self)
        return None

    @property
    def dims(self):
        """Deprecated.  Returns list of dimensions for this shape.
        Suggest `TensorShape.as_list` instead.
        Returns:
          A list containing `tf.compat.v1.Dimension`s, or None if the shape is
          unspecified.
        """
        return self._dims

    @property
    def ndims(self):
        """Deprecated accessor for `rank`."""
        return self.rank

    def tolist(self):
        """Returns a list of integers or `None` for each dimension.
        Returns:
          A list of integers or `None` for each dimension.
        Raises:
          ValueError: If `self` is an unknown shape with an unknown rank.
        """
        if self._dims is None:
            raise ValueError("as_list() is not defined on an unknown TensorShape.")
        return [dim for dim in self.dims]

    def is_fully_defined(self):
        """Returns True iff `self` is fully defined in every dimension."""
        return (self._dims is not None and all(dim is not None for dim in self.dims))

    def numel(self):
        """Returns the total number of elements, or none for incomplete shapes."""
        if self.is_fully_defined():
            return functools.reduce(operator.mul, self.tolist(), 1)
        else:
            return None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        """Returns the rank of this shape, or raises ValueError if unspecified."""
        if self._dims is None:
            raise ValueError("Cannot take the length of shape with unknown rank.")
        return len(self._dims)

    def __iter__(self):
        """Returns `self.dims` if the rank is known, otherwise raises ValueError."""
        if self._dims is None:
            raise ValueError("Cannot iterate over a shape with unknown rank.")
        else:
            return iter(d for d in self._dims)

    def __getitem__(self, key):
        """Returns the value of a dimension or a shape, depending on the key.
        Args:
          key: If `key` is an integer, returns the dimension at that index;
            otherwise if `key` is a slice, returns a TensorShape whose dimensions
            are those selected by the slice from `self`.
        Returns:
          An integer if `key` is an integer, or a `TensorShape` if `key` is a
          slice.
        Raises:
          ValueError: If `key` is a slice and `self` is completely unknown and
            the step is set.
        """
        if self._dims is not None:
            if isinstance(key, slice):
                return TensorShape(self._dims[key])
            else:
                return self._dims[key]
        else:
            if isinstance(key, slice):
                start = key.start if key.start is not None else 0
                stop = key.stop

                if key.step is not None:
                    # TODO(mrry): Handle these maybe.
                    raise ValueError("Steps are not yet handled")
                if stop is None:
                    # NOTE(mrry): This implies that TensorShape(None) is compatible with
                    # TensorShape(None)[1:], which is obviously not true. It would be
                    # possible to track the number of dimensions symbolically,
                    # and perhaps we should do that.
                    return TensorShape(None)
                elif start < 0 or stop < 0:
                    # TODO(mrry): Handle this better, as it will be useful for handling
                    # suffixes of otherwise unknown shapes.
                    return TensorShape(None)
                else:
                    return TensorShape(None)
            else:
                return None

    def __eq__(self, other):
        """Returns True if `self` is equivalent to `other`."""
        try:
            other = as_shape(other)
        except TypeError:
            return NotImplemented
        return self.dims == other.dims

    def __ne__(self, other):
        """Returns True if `self` is known to be different from `other`."""
        try:
            other = as_shape(other)
        except TypeError:
            return NotImplemented
        if self.rank is None or other.rank is None:
            raise ValueError("The inequality of unknown TensorShapes is undefined.")
        if self.rank != other.rank:
            return True
        return self.dims != other.dims

    def is_compatible_with(self, other):
        """Returns True iff `self` is compatible with `other`.
        Two possibly-partially-defined shapes are compatible if there
        exists a fully-defined shape that both shapes can represent. Thus,
        compatibility allows the shape inference code to reason about
        partially-defined shapes. For example:
        * TensorShape(None) is compatible with all shapes.
        * TensorShape([None, None]) is compatible with all two-dimensional
          shapes, such as TensorShape([32, 784]), and also TensorShape(None). It is
          not compatible with, for example, TensorShape([None]) or
          TensorShape([None, None, None]).
        * TensorShape([32, None]) is compatible with all two-dimensional shapes
          with size 32 in the 0th dimension, and also TensorShape([None, None])
          and TensorShape(None). It is not compatible with, for example,
          TensorShape([32]), TensorShape([32, None, 1]) or TensorShape([64, None]).
        * TensorShape([32, 784]) is compatible with itself, and also
          TensorShape([32, None]), TensorShape([None, 784]), TensorShape([None,
          None]) and TensorShape(None). It is not compatible with, for example,
          TensorShape([32, 1, 784]) or TensorShape([None]).
        The compatibility relation is reflexive and symmetric, but not
        transitive. For example, TensorShape([32, 784]) is compatible with
        TensorShape(None), and TensorShape(None) is compatible with
        TensorShape([4, 4]), but TensorShape([32, 784]) is not compatible with
        TensorShape([4, 4]).
        Args:
          other: Another TensorShape.
        Returns:
          True iff `self` is compatible with `other`.
        """
        other = TensorShape(other)
        if self.dims is not None and other.dims is not None:
            if self.rank != other.rank:
                return False
            for x_dim, y_dim in zip(self.dims, other.dims):
                if x_dim != y_dim and (x_dim is not None and y_dim is not None):
                    return False
        return True

    def get_dummy_tensor(self,batch_size=2):
        shape = [d for d in self._dims]
        if shape[0] is None:
            shape[0] = batch_size
        else:
            shape = [batch_size, ] + shape
        return np.clip(np.abs(np.random.standard_normal(shape)), 0, 1)


def as_shape(shape):
    """Converts the given object to a TensorShape."""
    if isinstance(shape, TensorShape):
        return shape
    else:
        return TensorShape(shape)


class OrderedDict(collections.OrderedDict):
    """ more easy-to-use OrderedDict"""

    def __init__(self, *args, **kwds):
        super(OrderedDict, self).__init__(*args, **kwds)

    @property
    def key_list(self):
        """
        Returns:
            list of keys

        """
        return list(super().keys())

    @property
    def value_list(self):
        """
        Returns:
            list of values

        """
        return list(super().values())

    @property
    def item_list(self):
        """
        Returns:
            list of items

        """
        return list(super().items())

    def __repr__(self):
        return '{ ' + (', '.join(['{0}: {1}'.format(k, v if v is not None else 'none') for k, v in self.item_list])) + ' }'


class Signature(object):
    def __init__(self, inputs=None, outputs=None, name=None):
        super().__init__()
        self.name = name
        self.inputs = OrderedDict() if inputs is None else inputs
        self.outputs = OrderedDict() if outputs is None else outputs

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a 'jsonable' dictionary.

        Input and output schema are represented as json strings. This is so that the
        representation is compact when embedded in a MLmofel yaml file.

        :return: dictionary representation with input and output shcema represented as json strings.
        """

        return {
            "inputs": json.dumps([x.to_dict() for x in self.inputs.value_list]) if self.inputs is not None else None,
            "outputs": json.dumps([x.to_dict() for x in self.outputs.value_list]) if self.outputs is not None else None,
        }

    # @classmethod
    # def get_signature(cls, fn:callable):
    #
    #     if "Layer"==fn.__class__.__base__:
    #
    #     if inspect.isfunction(fn)

    def maybe_not_complete(self):
        if len(self.inputs) < 1 or len(self.outputs) < 1:
            return True
        completeness = 0
        if self.inputs.value_list[0] is not None and self.inputs.value_list[0].__class__.__name__ == 'TensorSpec':
            if isinstance(self.inputs.value_list[0].shape, TensorShape):
                completeness += 1
        if self.outputs.value_list[0] is not None and self.outputs.value_list[0].__class__.__name__ == 'TensorSpec':
            if isinstance(self.outputs.value_list[0].shape, TensorShape):
                completeness += 1
        if completeness == 2:
            return False
        else:
            return True

    def _get_kvsting(self, k, v):
        if v is None:
            return '{0}'.format(k)
        elif isinstance(v, (list, tuple)):
            return '{0}: Tensor[{1}]'.format(k, v)
        elif v.__class__.__name__ == "TensorSpec" and (v.dtype is int or v.dtype is float or v.dtype is str or v.dtype is bool):
            return '{0}: {1} '.format(k, v.dtype.__name__) + ('={0}'.format(v.default) if v.optional else '')
        elif v.__class__.__name__ == "TensorSpec" and v.object_type is None:
            return '{0}: Tensor[{1}] '.format(k, v._shape_tuple)
        elif v.__class__.__name__ == "TensorSpec":
            return '{0}: Tensor[{1}] ({2})'.format(k, v._shape_tuple, v.object_type)
        else:
            return '{0}:{1}'.format(k, v)

    def __len__(self):
        return len(self.inputs) + len(self.outputs)

    def __repr__(self):
        # ElementTimes(x: Tensor[13]) -> Tensor[13]
        input_str = ', '.join([self._get_kvsting(k, v) for k, v in self.inputs.item_list]) if len(self.inputs.item_list) > 0 else ''
        output_str = ', '.join([self._get_kvsting(k, v) for k, v in self.outputs.item_list]) if len(self.outputs.item_list) > 0 else ''
        return '{0}( {1}) -> {2} '.format(self.name, input_str, output_str)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compile_and_install_module(module_name: str, source_code: str) -> types.ModuleType:
    """Compile source code and install it as a module.

    End result is that `import <module_name>` and `from <module_name> import ...` should work.
    """
    module = types.ModuleType(module_name, "Module created from source code")

    # Execute source in context of empty/fake module
    exec(source_code, module.__dict__)

    # Insert fake module into sys.modules. It's now a real module
    sys.modules[module_name] = module

    # Imports should work now
    return import_module(module_name)

def import_or_install(package_name: str, install_package_name: str = None) -> None:
    """Import [package_name] if possibile, or install [install_package_name] and then import it

    Args:
        install_package_name (str): install package_name
        package_name (str): package_name

    Returns:
        None

    Examples:
        >>> import_or_install('onnxruntime')

    """
    try:
        # If Module it is already installed, try to Import it
        importlib.import_module(package_name)
    except ImportError:
        if install_package_name is None:
            install_package_name = package_name
        if os.system('PIP --version') ==0:
            # No error from running PIP in the Command Window, therefor PIP.exe is in the %PATH%
            os.system('PIP install {0}  --upgrade'.format(install_package_name))
            importlib.import_module(package_name)
        else:
            # Error, PIP.exe is NOT in the Path!! So I'll try to find it.
            pip_location_attempt_1 = sys.executable.replace("python.exe", "") + "pip.exe"
            pip_location_attempt_2 = sys.executable.replace("python.exe", "") + "scripts\pip.exe"
            if os.path.exists(pip_location_attempt_1):
                # The Attempt #1 File exists!!!
                os.system(pip_location_attempt_1 + " install " + install_package_name+'  --upgrade')
                importlib.import_module(package_name)
            elif os.path.exists(pip_location_attempt_2):
                # The Attempt #2 File exists!!!
                os.system(pip_location_attempt_2 + " install " + install_package_name+'  --upgrade')
                importlib.import_module(package_name)
            else:
                # Neither Attempts found the PIP.exe file, So i Fail...
                print('Neither Attempts found the PIP.exe file, So i Fail...')
                exit()


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    This allows them to only be loaded when they are used.
    Code copied from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        super(LazyLoader, self).__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Lazy module loader cannot find module "
                f"named `{self.__name__}`. "
                f"This might be because textflint does not automatically "
                f"install some optional dependencies. "
                f"Please run `pip install {self.__name__}` "
                f"to install the package.") from e
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        # LazyLoader, lookups are efficient
        # (__getattr__ is only called on lookups
        # that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


def to_onehot(label, classes):
    onehot = np.zeros(classes)
    onehot[label] = 1
    return onehot


def is_iter(x):
    "Test whether `x 'can be used in a `for` loop"
    # Rank 0 tensors in PyTorch are not really iterable
    return isinstance(x, (Iterable, Generator)) and getattr(x, 'ndim', 1)


def to_list(x):
    """
     Convert anything to a list.
     if input is a tensor or a ndarray, to_list only unfold the first dimention , its different to the numpy behavior.

    Args:
        x ():

    Returns:
        a list

    Examples:
        >>> np.arange(16).reshape((4,2,2)).tolist()
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]]
        >>> to_list(np.arange(16).reshape((4,2,2)))
        [array([[0, 1],
               [2, 3]]), array([[4, 5],
               [6, 7]]), array([[ 8,  9],
               [10, 11]]), array([[12, 13],
               [14, 15]])]
        >>> to_list((2,5,(3,8)))
        [2, 5, (3, 8)]
        >>> to_list(range(8))
        [0, 1, 2, 3, 4, 5, 6, 7]
        >>> to_list(5)
        [5]
        >>> to_list({'x':3,'y':5})
        [('x', 3), ('y', 5)]

    """
    if x is None:
        return None
    elif isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return [x[i] for i in range(len(x))]
    elif isinstance(x, np.ndarray):
        return [x[i] for i in range(len(x))]
    elif 'tensor' in x.__class__.__name__.lower():
        return [x[i] for i in range(len(x))]
    elif is_iter(x):
        return list(x)
    elif hasattr(x, 'tolist') and callable(x.tolist):
        return x.tolist()
    elif isinstance(x, (int, float)):
        return [x]
    elif isinstance(x, dict):
        return list(x.items())
    elif isinstance(x, types.GeneratorType):
        return list(x)
    elif inspect.isgenerator(x):
        return list(x)
    elif isinstance(x, collections.Iterable):
        return list(x)
    else:
        try:
            return list(x)
        except:
            return x.tolist()


def is_numpy(x):
    return isinstance(x, np.ndarray)


def is_alphabet(x: str):
    return all([s in string.ascii_lowercase for s in x.lower()])


def is_punctuation(x: str):
    return all([s in string.punctuation for s in x.lower()])


def remove_nonprintable(x: str):
    import itertools
    # Use characters of control category

    nonprintable = itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))
    # Use translate to remove all non-printable characters
    return x.translate({character: None for character in nonprintable if chr(character) not in '\n\r\t'})


def unpack_singleton(x):
    """
    Gets the first element if the iterable has only one value. Otherwise return the iterable.But would not split a
    tensor.

    Args:
        x (iterable, except tensor and array):

    Returns:
        The same iterable or the first element.

    Examples
        >>> unpack_singleton(10, )
        10
        >>> unpack_singleton([0] )
        0
        >>> unpack_singleton(np.ones((2,5), dtype=np.int32))
        array([[1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1]])
        >>> unpack_singleton({'x':3,'y':5})
        {'x': 3, 'y': 5}

    """
    if x is None:
        return None
    elif 'tensor' in x.__class__.__name__.lower() or isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (tuple, list)) and len(x) == 1:
        return x[0]
    return x


def enforce_singleton(x):
    """
    Enforce only first element can pass if input is a a tuple or a list. It always use for singleton check if the
    function only accept one input.
    Args:
        x ():

    Returns:
        first element

    """
    if 'tensor' in x.__class__.__name__.lower() or isinstance(x, np.ndarray):
        return x
    elif hasattr(x, '__len__') and len(x) > 0:
        return x[0]
    return x


def check_for_unexpected_keys(name, input_dict, expected_values):
    unknown = set(input_dict.keys()).difference(expected_values)
    if unknown:
        raise ValueError(
            'Unknown entries in {} dictionary: {}. Only expected following keys: {}'.format(name, list(unknown),
                                                                                            expected_values))


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('\n\t'.join(missing_keys))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('\n\t'.join(unused_pretrained_keys))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    if len(missing_keys) >= len(unused_pretrained_keys) > 0:
        print('Try to mapping missing_keys to unused_pretrained_keys:')
        tmp_unused_pretrained_keys = list(unused_pretrained_keys)
        fix_dict = find_minimal_edit_distance_key(missing_keys, unused_pretrained_keys)
        for k, v in fix_dict.items():
            print('{0}=>{1}'.format(k, v))
        print('Is mapping results accetable?')
        ans = input('(Y/N) << ').lower()
        if ans in ['yes', 'y']:
            for k, v in fix_dict.items():
                pretrained_state_dict[k] = pretrained_state_dict[v]

    # print('\n\t'.join(used_pretrained_keys))

    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def levenshtein(seq1, seq2):
    """ calculate levenshtein edit distance

    Args:
        seq1 (list(string):
        seq2 (list(string)):

    Returns:

    Examples
        >>> seq1=list('block5a.3.norm.num_batches_tracked')
        >>> seq2=list('block5a.sequential_2.norm.num_batches_tracked')
        >>> levenshtein(seq1,seq2)
        12.0

    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    intersects = s1.intersection(s2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def text_similarity(list1, list2):
    # if len(list1)>0:
    #     first=list1[0]
    #     last=list1[-1]
    tmp_list1 = list1.copy()
    tmp_list2 = list2.copy()
    overlap = 0
    forward_idx = 0
    backward_idx = -1
    for i in range(len(list1)):
        if list1[i] == list2[i] and i == 0:
            overlap += 2
            forward_idx = i
            tmp_list1[i] = ''
            tmp_list2[i] = ''
        elif list1[i] == list2[i]:
            overlap += 1
            forward_idx = i
            tmp_list1[i] = ''
            tmp_list2[i] = ''
        else:
            break
    if forward_idx < len(list1):
        for i in range(len(list1) - forward_idx):
            backward_idx = -1 - i
            if list1[backward_idx] == list2[backward_idx] and backward_idx == -1:
                overlap += 2
                tmp_list1[backward_idx] = ''
                tmp_list2[backward_idx] = ''
            elif list1[backward_idx] == list2[backward_idx]:
                overlap += 1
                tmp_list1[backward_idx] = ''
                tmp_list2[backward_idx] = ''
            else:
                break
    tmp_list1 = list('.'.join([s for s in tmp_list1 if s != '']))
    tmp_list2 = list('.'.join([s for s in tmp_list2 if s != '']))
    edit_distance = levenshtein(tmp_list1, tmp_list2)
    max_edit_length = builtins.max(len(tmp_list1), len(tmp_list2))
    score = overlap + (builtins.max(max_edit_length - edit_distance, 0) / max_edit_length)
    return score / float(builtins.max(len(list1), 1))


def find_minimal_edit_distance_key(keys, lookup_keys):
    def only_keep_number(input_string):
        input_string = input_string.split('_')[-1].split('-')[-1]
        return ''.join([s for s in list(input_string) if s in string.digits])

    candidates_keys = None
    final_mapping = OrderedDict()
    section_keys = [key.split('.') for key in keys]
    is_allkey_same_section = len(list(set([len(k) for k in section_keys]))) == 1
    section_keys_annotation = []
    section_dict = []
    section_lookup_keys = [key.split('.') for key in lookup_keys]
    is_alllookupkey_same_section = len(list(set([len(k) for k in section_lookup_keys]))) == 1
    avaiable_section = list(range(len(section_lookup_keys[0])))

    if is_allkey_same_section and is_alllookupkey_same_section:
        for k in range(len(section_keys[0])):
            section_dict.append(OrderedDict())
            subdata = [item[k] for item in section_keys]
            number_parts = [int(only_keep_number(item)) for item in subdata if len(only_keep_number(item))]
            section_distinct = len(set(subdata))
            section_keys_annotation.append((len(set(subdata)), number_parts))
            for n in avaiable_section:
                sublookup = [item[n] for item in section_lookup_keys]
                lookup_number_parts = OrderedDict([(item, int(only_keep_number(item))) for item in sublookup if len(only_keep_number(item))])
                lookup_section_distinct = len(set(sublookup))

                if section_distinct == lookup_section_distinct:
                    if len(lookup_number_parts) == len(set(number_parts)) == section_distinct and len(set(lookup_number_parts.value_list)) == lookup_section_distinct:
                        if len(set(number_parts)) == len(set(lookup_number_parts.value_list)) == 1:
                            section_dict[k][str(subdata[0])] = sublookup[0]
                            avaiable_section.remove(n)
                            break
                        elif len(set(number_parts)) == len(set(lookup_number_parts.value_list)) == section_distinct:
                            sorted_number_parts = list(sorted(set(number_parts)))
                            sorted_lookup_number_parts = list(sorted(lookup_number_parts.value_list))

                            for item in subdata:
                                idx = sorted_number_parts.index(int(only_keep_number(item)))
                                section_dict[k][str(item)] = lookup_number_parts.key_list[lookup_number_parts.value_list.index(sorted_lookup_number_parts[idx])]
                            avaiable_section.remove(n)
                            break
                    elif len(lookup_number_parts) == len(set(number_parts)) == 0:
                        tmp_lookup_subdata = copy.deepcopy(sublookup)
                        for item in subdata:
                            candidates_keys = [text_similarity(list(item), key.split('.')) for key in tmp_lookup_subdata]
                            candidates_keys = np.array(candidates_keys)
                            candidate_idx = np.argmax(candidates_keys)
                            section_dict[k][str(item)] = tmp_lookup_subdata[candidate_idx]
                            tmp_lookup_subdata.remove(tmp_lookup_subdata[candidate_idx])
                        avaiable_section.remove(n)
                        break
                    else:
                        section_dict[k][str(item)] = sublookup[subdata.index(item)]
                        avaiable_section.remove(n)
                        break

        for section in section_keys:
            conver_key = []
            for i in range(len(section)):
                if section[i] in section_dict[i]:
                    conver_key.append(section_dict[i][section[i]])
            conver_key = '.'.join(conver_key)
            if conver_key in lookup_keys:
                final_mapping['.'.join(section)] = conver_key
            else:
                raise ValueError('{0} not exist!'.format(conver_key))
        return final_mapping


def addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def format_time(seconds):
    """ Format the seconds into human readable style

    Args:
        seconds (int,long): total seconds

    Returns:
         human readable style

    """
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_time_suffix():
    """

    Returns:
        timestamp string , usually use when save a file.

    """
    prefix = str(datetime.datetime.fromtimestamp(time.time())).replace(' ', '').replace(':', '').replace('-', '').replace('.', '')
    return prefix


def get_file_modified_time(file_path):
    """
    Try to get the date that a file was modified, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    t = None
    try:
        t = os.path.getmtime(file_path)
    except:
        stat = os.stat(file_path)
        try:
            t = stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            t = stat.st_mtime
    if t is not None:
        return datetime.datetime.fromtimestamp(t)
    else:
        return None


def get_function(fn_name, module_paths=None):
    """
    Returns the function based on function name.

    Args:
        fn_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            ``class_name``. The first module in the list that contains the class
            is used.

    Returns:
        The target function.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.

    """
    if callable(fn_name):
        return fn_name
    fn = None
    if (fn_name is not None) and (module_paths is not None):
        for module_path in module_paths:
            fn = locate('.'.join([module_path, fn_name]))
            if fn is not None:
                break

    if fn is None:
        fn = locate(fn_name)
        if fn is not None:
            return fn
        else:
            return None
    else:
        return fn  # type: ignore


def get_class(class_name, module_paths=None):
    """
    Returns the class based on class name.

    Args:
        class_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            ``class_name``. The first module in the list that contains the class
            is used.

    Returns:
        The target class.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.

    """
    class_ = None
    if (class_name is not None) and (module_paths is not None):
        for module_path in module_paths:
            class_ = locate('.'.join([module_path, class_name]))
            if class_ is not None:
                break

    if class_ is None:
        class_ = locate(class_name)
        raise ValueError("Class not found in {}: {}".format(module_paths, class_name))
    return class_  # type: ignore


def get_terminal_size():
    """ getTerminalSize()
     - get width and height of console
     - works on linux,os x,windows,cygwin(windows)
     originally retrieved from:
     http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
    """

    def _get_terminal_size_windows():
        try:
            from ctypes import windll, create_string_buffer
            # stdin handle is -10
            # stdout handle is -11
            # stderr handle is -12
            h = windll.kernel32.GetStdHandle(-12)
            csbi = create_string_buffer(22)
            res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
            if res and res != 0:
                (bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh",
                                                                                                      csbi.raw)
                sizex = right - left + 1
                sizey = bottom - top + 1
                return sizex, sizey
            else:
                import tkinter as tk
                root = tk.Tk()
                sizex, sizey = root.winfo_screenwidth() // 8, root.winfo_screenheight() // 8
                root.destroy()
                return sizex, sizey
        except Exception as e:
            print(e)
            pass

    def _get_terminal_size_tput():
        # get terminal width
        # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width-height-of-a-terminal-window
        try:
            cols = int(subprocess.check_call(shlex.split('tput cols')))
            rows = int(subprocess.check_call(shlex.split('tput lines')))
            return (cols, rows)
        except:
            pass

    def _get_terminal_size_linux():
        def ioctl_GWINSZ(fd):
            try:
                import fcntl
                import termios
                cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
                return cr
            except:
                pass

        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = os.open(os.ctermid(), os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                os.close(fd)
            except:
                pass
        if not cr:
            try:
                cr = (os.environ['LINES'], os.environ['COLUMNS'])
            except:
                return None
        return int(cr[1]), int(cr[0])

    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()  # needed for window's python in cygwin's xterm!
    if current_os in ['Linux', 'Darwin'] or current_os.startswith('CYGWIN'):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)  # default value
    return tuple_xy


def gcd(x, y):
    gcds = []
    gcd = 1
    if x % y == 0:
        gcds.append(int(y))
    for k in range(int(y // 2) + 1, 0, -1):
        if x % k == 0 and y % k == 0:
            gcd = k
            gcds.append(int(k))
    return gcds


def get_divisors(n):
    return [d for d in range(2, n // 2 + 1) if n % d == 0]


def isprime(n):
    if n >= 9:
        divisors = [d for d in range(2, int(math.sqrt(n))) if n % d == 0]
        return all(n % od != 0 for od in divisors if od != n)
    elif n in [1, 2, 3, 5, 7]:
        return True
    else:
        return False


def next_prime(n):
    pos = n + 1
    while True:
        if isprime(pos):
            return pos
        pos += 1


def prev_prime(n):
    pos = n - 1
    while True:
        if isprime(pos):
            return pos
        pos -= 1


def nearest_prime(n):
    nextp = next_prime(n)
    prevp = prev_prime(n)
    if abs(nextp - n) < abs(prevp - n):
        return nextp
    else:
        return nextp


def num_cpus():
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()



def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.free,memory.total,temperature.gpu',
            '--format=csv,nounits'
        ], encoding='utf-8')
    gpu_memory = [x.split(',') for x in result.strip().split('\n')]

    header=[h.strip() for h in gpu_memory[0]]
    memory_map_list=[]
    for n in range(1,len(gpu_memory)):
        memory_map = OrderedDict()
        for i,(k,v) in enumerate(zip(header,gpu_memory[n])):
            if i==0:
                memory_map[k]=datetime.datetime.strptime(v, "%Y/%m/%d %H:%M:%S.%f")
            elif i == 1:
                memory_map[k] =int(v)
            elif i == 3:
                memory_map[k] =float(v)
            elif i > 3:
                memory_map[k] = float(v)
            else:
                memory_map[k] = v
        memory_map['memory usage'] =memory_map[memory_map.key_list[-4]]/memory_map[memory_map.key_list[-2]]
        memory_map_list.append(memory_map)
    return memory_map_list

def get_memory_profile(mode):
    """
    'all' means return memory for all gpus
    'min_max' means return memory for max and min
    :param mode:
    :return:
    """
    memory_map = get_gpu_memory_map()

    if mode == 'min_max':
        min_mem = 1000000
        min_k = None
        max_mem = 0
        max_k = None
        for k, v in memory_map:
            if v > max_mem:
                max_mem = v
                max_k = k
            if v < min_mem:
                min_mem = v
                min_k = k

        memory_map = {min_k: min_mem, max_k: max_mem}

    return memory_map


def map_function_arguments(params, params_dict, *args, **kwargs):
    """
    Helper to determine the argument map for use with various call operations.
    Returns a dictionary from parameters to whatever arguments are passed.
    Accepted are both positional and keyword arguments.
    This mimics Python's argument interpretation, except that keyword arguments are not optional.
    This does not require the arguments to be Variables or Functions. It is also called by train_minibatch() and

    """
    # start with positional arguments
    arg_map = dict(zip(params, args))

    # now look up keyword arguments
    if len(kwargs) != 0:
        for name, arg in kwargs.items():  # keyword args are matched by name
            if name not in params_dict:
                raise TypeError("got an unexpected keyword argument '%s'" % name)
            param = params_dict[name]
            if param in arg_map:
                raise SyntaxError("got multiple values for argument '%s'" % name)
            arg_map[param] = arg  # add kw argument to dict
    assert len(arg_map) == len(params)

    return arg_map


class ClassfierType(Enum):
    dense = 'dense'
    global_avgpool = 'global_avgpool'
    centroid = 'centroid'


class PaddingMode(Enum):
    zero = 'constant'
    reflection = 'reflect'
    replicate = 'replicate'
    circular = 'circular'


class Color(Enum):
    rgb = 'rgb'
    bgr = 'bgr'
    gray = 'gray'
    rgba = 'rgba'


class ShortcutMode(Enum):
    add = 'add'
    dot = 'dot'
    concate = 'concate'


class Interpolation(Enum):
    Nearest = 'Nearest'
    Bilinear = 'Bilinear'
    Bicubic = 'Bicubic'


class GetImageMode(Enum):
    path = 'path'
    raw = 'raw'
    expect = 'expect'
    processed = 'processed'


class _empty:
    """Marker object for Signature.empty and Parameter.empty."""


def red_color(text, bolder=False):
    if bolder:
        return '\033[1;31m{0}\033[0;0m'.format(text)
    else:
        return '\033[31m{0}\033[0;0m'.format(text)


def green_color(text, bolder=False):
    if bolder:
        return '\033[1;32m{0}\033[0;0m'.format(text)
    else:
        return '\033[32m{0}\033[0;0m'.format(text)


def blue_color(text, bolder=False):
    if bolder:
        return '\033[1;34m{0}\033[0m'.format(text)
    else:
        return '\033[34m{0}\033[0;0m'.format(text)


def cyan_color(text, bolder=False):
    if bolder:
        return '\033[1;36m{0}\033[0m'.format(text)
    else:
        return '\033[36m{0}\033[0;0m'.format(text)


def yellow_color(text, bolder=False):
    if bolder:
        return '\033[1;93m{0}\033[0m'.format(text)
    else:
        return '\033[93m{0}\033[0;0m'.format(text)


def orange_color(text, bolder=False):
    if bolder:
        return u'\033[1;33m%s\033[0m' % text
    else:
        return '\033[33m {0}\033[0;0m'.format(text)


def gray_color(text, bolder=False):
    if bolder:
        return u'\033[1;337m%s\033[0m' % text
    else:
        return '\033[37m {0}\033[0;0m'.format(text)

def violet_color(text, bolder=False):
    if bolder:
        return u'\033[1;35m%s\033[0m' % text
    else:
        return '\033[35m {0}\033[0;0m'.format(text)


def magenta_color(text, bolder=False):
    if bolder:
        return u'\033[1;35m%s\033[0m' % text
    else:
        return '\033[35m {0}\033[0;0m'.format(text)



def get_args_spec(fn):
    full_arg_spec = inspect.getfullargspec(fn)
    arg_spec = inspect.ArgSpec(args=full_arg_spec.args, varargs=full_arg_spec.varargs, keywords=full_arg_spec.varkw,
                               defaults=full_arg_spec.defaults)
    return arg_spec


def format_arg_spec(v, is_output=False):
    s = v.name + ': ' if not is_output and v.name else ''  # (suppress output names, since they duplicate the
    # function name)
    return s + str(v._type)


def is_instance(instance, check_class):
    if not inspect.isclass(instance) and inspect.isclass(check_class):
        mro_list = [b.__module__ + '.' + b.__qualname__ for b in instance.__class__.__mro__]
        return check_class.__module__ + '.' + check_class.__qualname__ in mro_list
    elif  isinstance(check_class, str):
        mro_list = [b.__module__ + '.' + b.__qualname__ for b in instance.__class__.__mro__]
        mro_list2 = [b.__qualname__ for b in instance.__class__.__mro__]
        return check_class in mro_list or check_class in mro_list2
    else:
        if not inspect.isclass(check_class):
            print(red_color('Input check_class {0} should a class, but {1}'.format(check_class,type(check_class))))
        return False


def open_browser(url,delay=0):
    sleep(delay)
    webbrowser.open_new(url)

def launchTensorBoard():
    make_dir_if_need(os.path.join(_SESSION.working_directory, 'Logs'))
    print('tensorboard --logdir {0} --port {1}'.format(os.path.join(_SESSION.working_directory, 'Logs'),_SESSION.tensorboard_port))
    os.system('tensorboard --logdir {0} --port {1}'.format(os.path.join(_SESSION.working_directory, 'Logs'),_SESSION.tensorboard_port))
    return


def launchMLFlow():
    make_dir_if_need(os.path.join(_SESSION.working_directory, 'Logs'))
    print('mlflow ui')
    os.system('mlflow ui')
    return