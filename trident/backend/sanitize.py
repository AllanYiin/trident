
import numbers
import collections
import numpy as np



def is_string(s):
    """
    Tests whether ``s`` is a string in a way that works on Python 2 and 3.
    """
    return isinstance(s, ("".__class__, u"".__class__))

def is_byte_buffer(s):
    """
    Tests whether ``s`` is a byte buffer (not a string) in a way that
    works on Python 2 and 3.
    """
    return (isinstance(s, bytearray) or
        (isinstance(s, type(b'')) and not isinstance(b'', str)))


def _as_tuple(x):
    """
    Convert an argument to a tuple.

    Args:
        x: if scalar, it returns ``(x,)``. If iterable, it converts it to
        tuple.

    Returns:
        Tuple of ``x``.
    """
    if np.isscalar(x):
        x = (x,)
    return tuple(x)


def sanitize_precision(precision):
    """
    Converts precision to NumPy precision

    Args:
        precision (str or `np.float16` or `np.float32` or `np.float64`): precision, if string
         it can be one of 'float' 'float32, 'double', 'float64' or 'float16'

    Returns:
        NumPy precision
    """
    if precision in [cntk_py.DataType_Float, 'float', 'float32', np.float32]:
        return np.float32
    elif precision in [cntk_py.DataType_Double, 'double', 'float64', np.float64]:
        return np.float64
    elif precision in [cntk_py.DataType_Float16, 'float16', np.float16]:
        return np.float16
    elif precision in [cntk_py.DataType_Int8, 'int8_t', 'int8', np.int8]:
        return np.int8
    elif precision in [cntk_py.DataType_Int16, 'int16_t', 'int16', np.int16]:
        return np.int16
    elif precision in [cntk_py.DataType_Unknown]:
        return None
    else:
        raise ValueError('precision value: "%s" is not supported' % precision)


def sanitize_shape(shape):
    """
    If shape is scalar, it creates a tuple out of it.
    """
    return _as_tuple(shape)


def sanitize_input(arg, fallback_dtype=np.float32, reshape=None):
    """sanitize_input(arg, fallback_dtype=np.float32, reshape=None)
    Convert to :class:`~cntk.variables.Variable` so that it can be passed
    as Variable to the CNTK operators.

      * If ``arg`` is a NumPy array and its type is not among (`np.float32`,
        `np.float64`, `np.float16`), it sets it to `np.float32`.
      * If ``arg`` is an op, it is assumed that it has only one output, which
        will be returned.

    Args:
        arg (number, NumPy array, :class:`~cntk.variables.Variable`, or :class:`~cntk.ops.functions.Function`): input
        fallback_dtype (NumPy dtype): fallback dtype in case ``arg`` is a list

    Returns:
      Leaves Constant, Parameter, and Variable as is. Returns Constant, if
      ``arg`` is a number or NumPy array. Variable otherwise.
    """

    from cntk.ops.functions import UserFunction
    from cntk.variables import Constant, Variable, Parameter
    from cntk.ops.functions import Function
    from cntk.ops import constant
    from ..core import asarray

    # is it a Variable or a Function?
    if isinstance(arg,
                  (Constant, cntk_py.Constant,
                   Variable, cntk_py.Variable,
                   Parameter, cntk_py.Parameter,
                   Function, cntk_py.Function)):
        return arg

    if isinstance(arg, Variable._Type):
        raise ValueError("Input is a type object (" + str(arg) + "). Did you mean to pass 'input_variable(**" + str(arg) + ")'?")

    # maybe a Python list that we can interpret as a NumPy array?
    if isinstance(arg, list) and not arg:
        raise ValueError('input is empty')

    if not isinstance(arg, np.ndarray) or arg.dtype != fallback_dtype:
        # TODO: check whether Values can be ingested directly
        arg = asarray(arg, fallback_dtype)

    if reshape:
        arg = np.reshape(arg, reshape)

    return constant(value=arg)

def sanitize_2d_number(x):
    """
    Convert ``x`` to a tuple.

    Args:
        x: a scalar number or a tuple of length 2 that contains the 2D values.

    Returns:
        A tuple of length two.
    """
    if isinstance(x, numbers.Number):
        return (x,x)
    elif (isinstance(x, tuple) and (len(x) == 2) and
          isinstance(x[0], numbers.Number) and isinstance(x[1], numbers.Number)):
        return x

    raise ValueError('Input argument must be a number or a tuple of two numbers.')

def sanitize_range(x):
    """
    Convert ``x`` to a tuple such as the first element is less than or equal to the
    second element.

    Args:
        x: a scalar number or a tuple of length 2 that contains the range values.

    Returns:
        A tuple of length two where the first element is less than or equal to the
        second element.
    """
    x =  sanitize_2d_number(x)
    if x[0] <= x[1]:
        return x

    raise ValueError('Input argument must be a number or a tuple of two numbers such as the first number is smaller than or equal to the second number.')






def data_type_to_dtype(data_type):
    if data_type == cntk_py.DataType_Float:
        return np.float32
    elif data_type == cntk_py.DataType_Double:
        return np.float64
    elif data_type == cntk_py.DataType_Float16:
        return np.float16
    elif data_type == cntk_py.DataType_Int8:
        return np.int8
    elif data_type == cntk_py.DataType_Int16:
        return np.int16
    elif data_type == cntk_py.DataType_Unknown:
        return object
    else:
        raise ValueError('data_type %s is not supported'%data_type)


def sanitize_dtype_numpy(dtype):
    is_type = isinstance(dtype, type) or isinstance(dtype, np.dtype)
    is_str = is_string(dtype)
    if is_type and dtype in (int, np.float32) or \
            hasattr(dtype, 'kind') and dtype.kind in 'iu' \
            or is_str and dtype in ('float', 'float32'):
        return np.float32
    elif is_type and dtype in (float, np.float64) or \
            is_str and dtype in ('double', 'float64'):
        # The Python type 'float' is a np.float64
        return np.float64
    elif is_type and dtype in (float, np.float16) or \
            is_str and dtype in ('float16'):
        return np.float16
    elif is_type and dtype in (int, np.int8) or \
            is_str and dtype in ('int8'):
        return np.int8
    elif is_type and dtype in (int, np.int16) or \
            is_str and dtype in ('int16'):
        return np.int16
    else:
        raise ValueError('data type "%s" is not supported' % dtype)


def sanitize_dtype_cntk(dtype):
    if isinstance(dtype, int) and dtype in (cntk_py.DataType_Float, cntk_py.DataType_Double, cntk_py.DataType_Float16, cntk_py.DataType_Int8, cntk_py.DataType_Int16, cntk_py.DataType_Unknown):
        return dtype
    if dtype is None:
        return cntk_py.DataType_Unknown

    dtype = sanitize_dtype_numpy(dtype)
    if dtype == np.float32:
        return cntk_py.DataType_Float
    elif dtype == np.float64:
        return cntk_py.DataType_Double
    elif dtype == np.float16:
        return cntk_py.DataType_Float16
    elif dtype == np.int8:
        return cntk_py.DataType_Int8
    elif dtype == np.int16:
        return cntk_py.DataType_Int16
    elif dtype == object:
        return cntk_py.DataType_Unknown
    else:
        raise ValueError('data type "%s" is not supported' % dtype)


def sanitize_axis(axis):
    '''
    Sanitizes the axis.

    Args:
        axis (:class:`~cntk.axis.Axis` or int or None): the axis to be used.

          * :class:`~cntk.axis.Axis`: use axis instance directly (will convert
            row- to col-major in case of static axis).
          * int: if positive, use it as static axis. If negative, count from
            last to first axis
          * None: denote all available axes
    '''
    if axis is None:
        return Axis.all_static_axes()
    elif isinstance(axis, numbers.Integral):
        return Axis(-axis - 1)
    elif axis.is_static_axis and (axis.static_axis_index() != Axis.new_leading_axis().static_axis_index()):
        return Axis(-1 - axis.static_axis_index())
    else:
        return axis

def sanitize_axis_list(axes):
    '''
    Sanitizes a list of axes.

    Args:
        axes (list of :class:`~cntk.axis.Axis` or int or None): the axes to be used.

          * :class:`~cntk.axis.Axis`: use axis instance directly (will convert
            row- to col-major in case of static axis).
          * int: if positive, use it as static axis. If negative, count from
            last to first axis
          * None: denote all available axes
    '''
    if not type(axes) in (list, tuple):
        axes = [axes]
    retAxes = []
    for ax in axes:
        retAxes.append(sanitize_axis(ax))
    return retAxes

def sanitize_multi_axis_reduction_list(axes):
    '''
    Sanitizes a list of axes for multi-axis reduction which can not contain sequence axis.

    Args:
        axes (list of :class:`~cntk.axis.Axis` or int or None): the axes to be used.

          * :class:`~cntk.axis.Axis`: use axis instance directly (will convert
            row- to col-major in case of static axis).
          * int: if positive, use it as static axis. If negative, count from
            last to first axis
          * None: denote all available axes
    '''
    if not type(axes) in (list, tuple):
        axes = [axes]
    retAxes = []
    for ax in axes:
        if (isinstance(ax, Axis)) and (ax.is_sequence_axis):
            raise ValueError('Reduction operation over multiple axes can not contain sequence axis: %s' % ax)
        retAxes.append(sanitize_axis(ax))
    return retAxes

def sanitize_dynamic_axes(axes):
    if not type(axes) in (list, tuple):
        axes = [axes]
    for ax in axes:
        if not isinstance(ax, cntk_py.Axis):
            raise TypeError('type Axis expected, got %s instead' % type(ax))
    axes = tuple(reversed(axes))
    return axes


def sanitize_variable_value_dict(var_value_dict):
    if len(var_value_dict) > 1:
        return var_value_dict
    else:
        return list(var_value_dict.values())[0]

def _sanitize_common_conv_args(strides, auto_padding):
    strides = sanitize_shape(strides)

    # Reverse the 'auto_padding' argument to account for the col-major tensor
    # layout in core C++ implementation
    auto_padding = list(reversed(auto_padding))

    return strides, auto_padding

def sanitize_pooling_args(pooling_window_shape, strides, auto_padding):
    pooling_window_shape = sanitize_shape(pooling_window_shape)
    strides, auto_padding = _sanitize_common_conv_args(strides, auto_padding)
    return pooling_window_shape, strides, auto_padding

def sanitize_convolution_args(strides, sharing, auto_padding):
    strides, auto_padding = _sanitize_common_conv_args(strides, auto_padding)

    # Reverse the 'sharing' argument to account for the col-major tensor layout
    # in core C++ implementation
    sharing = list(reversed(sharing))

    return strides, sharing, auto_padding

def sanitize_Function_attributes(attributes):
    # Reverse the 'sharing' and 'auto_padding' attributes to account for the
    # col-major tensor layout in core C++ implementation
    if 'sharing' in attributes:
        attributes['sharing'] = list(reversed(attributes['sharing']))

    if 'autoPadding' in attributes:
        attributes['autoPadding'] = list(reversed(attributes['autoPadding']))

    return attributes

def sanitize_permutation(perm):
    # Find the permutation such that when it is applied to the reverse
    # of an input gives the reverse of perm applied to the input
    # Example:
    # input is [a, b, c, d], perm is [3, 0, 2, 1], perm of input is [d, a, c, b]
    # we are looking for [2, 1, 3, 0] because when we apply it to [d, c, b, a]
    # the result is [b, c, a, d] which is the revese of [d, a, c, b]

    n = len(perm)
    # first make sure the range of each element is valid
    if not all(-n <= i < n for i in perm):
        raise ValueError('invalid permutation element: elements must be from {-len(perm), ..., len(perm)-1}')
    # next take care of negative indices
    positive_perm = [perm[i] if perm[i]>=0 else n+perm[i] for i in range(n)]
    # check for duplicates
    if n != len(set(positive_perm)):
        raise ValueError('duplicate item in permutation')
    return [n-i-1 for i in reversed(positive_perm)]



# Workaround for Python 2.7 not having functools.lru_cache
def bounded_cache(maxsize):
    def memoize(func):
        class memodict(dict):
            def __init__(self, f):
                self.f = f
            def __call__(self, *args):
                return self[args]
            def __missing__(self, key):
                if len(self) >= maxsize:
                    self.clear()
                self[key] = ret = self.f(*key)
                return ret
        return memodict(func)
    return memoize


# The following is a convenience function that we call internally in cases
# such as when a user wants to inspect sparse data coming from a reader
# The conversion happens by calling forward on the network defined below.
# We memoize the last maxsize networks, because network building is slow.
# Adjust the maxsize below if you have more than maxsize many different
# shapes of sparse inputs.
@bounded_cache(maxsize=32)
def _sparse_to_dense_network_cache(input_shape, is_sequence, device):
    if is_sequence:
        temp_input = C.sequence.input_variable(input_shape, is_sparse=True)
    else:
        temp_input = C.input_variable(input_shape, is_sparse=True)

    eye_shape = input_shape[-1]
    return C.times(temp_input, np.eye(eye_shape))