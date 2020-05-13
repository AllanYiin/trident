import collections
import datetime
import inspect
import itertools
import json
import linecache
import math
import os
import platform
import re
import shlex
import struct
import subprocess
import sys
import threading
import time
import traceback
import types
from enum import Enum, unique
from inspect import signature
from pydoc import locate
from typing import List, Set, Tuple, Dict

import numpy as np
import six

__all__ = ['get_session', 'get_trident_dir', 'get_signature', 'epsilon', 'set_epsilon', 'floatx', 'set_floatx',
           'check_keys', 'if_else', 'camel2snake', 'snake2camel', 'to_onehot', 'to_list', 'addindent', 'format_time',
           'get_time_suffix', 'get_function', 'get_class', 'get_terminal_size', 'gcd', 'get_divisors', 'isprime',
           'next_prime', 'prev_prime', 'nearest_prime', 'PrintException', 'unpack_singleton', 'enforce_singleton',
           'OrderedDict', 'get_python_function_arguments', 'map_function_arguments', 'ClassfierType', 'PaddingMode',
           'DataRole',

           'ExpectDataType', 'GetImageMode', 'split_path', 'make_dir_if_need', 'sanitize_path', 'ShortcutMode',
           'DataSpec', 'get_args_spec', 'get_gpu_memory_map']


def sanitize_path(path):
    if isinstance(path, str):
        return os.path.normpath(path.strip()).replace('\\', '/')
    else:
        return path


_trident_dir = ''
if 'TRIDENT_HOME' in os.environ:
    _trident_dir = os.environ.get('TRIDENT_HOME')
else:
    _trident_base_dir = os.path.expanduser('~')
    if not os.access(_trident_base_dir, os.W_OK):
        _trident_dir = '/tmp/.trident'
    else:
        _trident_dir = os.path.expanduser('~/.trident')

_trident_dir = sanitize_path(_trident_dir)
if not os.path.exists(_trident_dir):
    try:
        os.makedirs(_trident_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

# Attempt to read Trident config file.


_SESSION = threading.local()

_SESSION.trident_dir = _trident_dir
_SESSION.backend = 'pytorch'
_SESSION.image_backend = 'pillow'
_SESSION.epoch_equivalent = 200
_SESSION.floatx = 'float32'
_SESSION.epsilon = 1e-8
_SESSION.numpy_print_format = '{0:.4e}'

_SESSION.device = None

_config_path = os.path.expanduser('~/.trident/trident.json')
if os.path.exists(_config_path):
    _config = {}
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        pass
    for k, v in _config.items():
        try:
            if k == 'floatx':
                assert v in {'float16', 'float32', 'float64'}
            if k != 'trident_dir':
                _SESSION.__setattr__(k, v)
        except:
            exc_type, exc_obj, tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_obj, tb, limit=2, file=sys.stdout)

from enum import Enum, unique


@unique
class Backend(Enum):
    pytorch = 'pytorch'
    tensorflow = 'tensorflow'


@unique
class IntevalUnit(Enum):
    batch = 'batch'
    epoch = 'epoch'
    once = 'once'


np.set_printoptions(formatter={'float_kind': lambda x: _SESSION.numpy_print_format.format(x)})


def get_plateform():
    return platform.system()


def _is_c_contiguous(data):
    while isinstance(data, list):
        data = data[0]
    return data.flags.c_contiguous


def get_session():
    return _SESSION


def set_session(key, value):
    setattr(_SESSION, key, value)
    return _SESSION


def get_trident_dir():
    return _SESSION.trident_dir


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()

    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
    traceback.print_tb(tb, limit=1, file=sys.stdout)
    traceback.print_exception(exc_type, exc_obj, tb, limit=2, file=sys.stdout)


def get_signature(fn, skip_default=False):
    signature = OrderedDict()
    parameters = list(inspect.signature(fn).parameters.values())
    for k in parameters:
        if skip_default == False or (skip_default == True and k.default == inspect._empty):
            if k.name not in ['args', 'kwargs']:
                signature[k.name] = k.default if k.default != inspect._empty else None
    return signature


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    Returns: a float

    # Example

    >>> print(epsilon())
    1e-08
    """
    return _SESSION.epsilon


def set_epsilon(e):
    _SESSION.epsilon = float(e)


def floatx():
    """Returns the default float type, as a string.
    "e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    >>> print(floatx())
    float32

    """
    return _SESSION.floatx


def set_floatx(floatx):
    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _SESSION.floatx = str(floatx)


def camel2snake(name):
    if name is None:
        return None
    else:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake2camel(name):
    if name is None:
        return None
    else:
        return ''.join(x.capitalize() or '_' for x in name.split('_'))


def to_onehot(label, classes):
    onehot = np.zeros(classes)
    onehot[label] = 1
    return onehot


def to_list(x):
    '''
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

    '''
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


def if_else(a, b):
    '''
    Syntax suggar for None replacement

    Args:
        a (obj):
        b (obj):

    Returns:
        None replacement

    '''
    if a is None:
        return b
    else:
        return a


def unpack_singleton(x):
    '''
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

    '''
    if x is None:
        return None
    elif 'tensor' in x.__class__.__name__.lower() or isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (tuple, list)) and len(x) == 1:
        return x[0]
    return x


def enforce_singleton(x):
    '''
    Enforce only first element can pass if input is a a tuple or a list. It always use for singleton check if the
    function only accept one input.
    Args:
        x ():

    Returns:
        first element

    '''
    if 'tensor' in x.__class__.__name__.lower() or isinstance(x, np.ndarray):
        return x
    elif hasattr(x, '__len__'):
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
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


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
    prefix = str(datetime.datetime.fromtimestamp(time.time())).replace(' ', '').replace(':', '').replace('-',
                                                                                                         '').replace(
        '.', '')
    return prefix


def get_function(fn_name, module_paths=None):
    '''
    Returns the class based on class name.
    Args:
        fn_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            ``class_name``. The first module in the list that contains the class
            is used.

    Returns:
        The target class.
    Raises:
            ValueError: If class is not found based on :attr:`class_name` and
                :attr:`module_paths`.

    '''
    if callable(fn_name):
        return fn_name
    fn = locate(fn_name)
    if (fn is None) and (module_paths is not None):
        for module_path in module_paths:
            fn = locate('.'.join([module_path, fn_name]))
            if fn is not None:
                break
    if fn is None:
        raise ValueError("Method not found in {}: {}".format(module_paths, fn_name))
    return fn  # type: ignore


def get_class(class_name, module_paths=None):
    '''
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
    '''
    class_ = locate(class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            class_ = locate('.'.join([module_path, class_name]))
            if class_ is not None:
                break
    if class_ is None:
        raise ValueError("Class not found in {}: {}".format(module_paths, class_name))
    return class_  # type: ignore


def get_terminal_size():
    """ getTerminalSize()
     - get width and height of console
     - works on linux,os x,windows,cygwin(windows)
     originally retrieved from:
     http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
    """
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


def get_python_function_arguments(f):
    '''
    Helper to get the parameter names and annotations of a Python function.
    '''
    # Note that we only return non-optional arguments (we assume that any optional args are not specified).
    # This allows to, e.g., accept max(a, b, *more, name='') as a binary function
    import sys
    param_specs = inspect.getfullargspec(f)
    annotations = param_specs.annotations
    arg_names = param_specs.args
    defaults = param_specs.defaults  # "if this tuple has n elements, they correspond to the last n elements listed
    # in args"
    if defaults:
        arg_names = arg_names[:-len(defaults)]
    return (arg_names, annotations)


def map_function_arguments(params, params_dict, *args, **kwargs):
    '''
    Helper to determine the argument map for use with various call operations.
    Returns a dictionary from parameters to whatever arguments are passed.
    Accepted are both positional and keyword arguments.
    This mimics Python's argument interpretation, except that keyword arguments are not optional.
    This does not require the arguments to be Variables or Functions. It is also called by train_minibatch() and

    '''
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


# def map_if_possible(obj_to_map, *args, **kwargs):
#     if inspect.isfunction(obj_to_map):


def split_path(path):
    if path is None or len(path) == 0:
        return '', '', ''
    path = sanitize_path(path)
    folder, filename = os.path.split(path)
    ext = ''
    if '.' in filename:
        filename, ext = os.path.splitext(filename)
        filename, ext2 = os.path.splitext(filename)
        ext = ext2 + ext
    else:
        folder = os.path.join(folder, filename)
        filename = ''
    return folder, filename, ext


def make_dir_if_need(path):
    folder, filename, ext = split_path(path)
    if len(folder) > 0 and not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            PrintException()
            sys.stderr.write('folder:{0} is not valid path'.format(folder))
    return sanitize_path(path)


class OrderedDict(collections.OrderedDict):
    def __init__(self, *args, **kwds):
        super(OrderedDict, self).__init__(*args, **kwds)

    @property
    def key_list(self):
        return list(super().keys())

    def keys(self):
        return super().keys()

    @property
    def value_list(self):
        return list(super().values())

    def values(self):
        return super().values()

    @property
    def item_list(self):
        return list(super().items())

    def items(self):
        return super().items()

    def __repr__(self):
        return '{ ' + (', '.join(['{0}: {1}'.format(k, v) for k, v in self.item_list])) + ' }'


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


class DataRole(Enum):
    input = 'input'
    target = 'target'
    mask = 'mask'


class ExpectDataType(Enum):
    array_data = 'array_data'
    gray = 'gray'
    rgb = 'rgb'
    rgba = 'rgba'
    label_mask = 'label_mask'
    color_mask = 'color_mask'
    binary_mask = 'binary_mask'
    alpha_mask = 'alpha_mask'
    multi_channel = 'multi_channel'
    absolute_bbox = 'absolute_bbox'
    relative_bbox = 'relative_bbox'
    random_noise = 'random_noise'
    classification_label = 'classification_label'


class GetImageMode(Enum):
    path = 'path'
    raw = 'raw'
    expect = 'expect'
    processed = 'processed'


class _empty:
    """Marker object for Signature.empty and Parameter.empty."""


class DataSpec:
    def __init__(self, name, symbol=None, kind='array', shape=None, annotation=_empty):
        self._name = name
        self._kind = kind
        self._shape = shape
        self._symbol = symbol
        self._annotation = annotation

    @property
    def name(self):
        return self._name

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value

    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, value):
        self._symbol = value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if value is None:
            self._shape = None
        elif isinstance(value, tuple):
            self._shape = value
        elif isinstance(value, [int, float]):
            self.shape = (int(value),)
        else:
            try:
                self.shape = tuple(to_list(value))
            except:
                PrintException()

    @property
    def annotation(self):
        return self._annotation

    @annotation.setter
    def annotation(self, value):
        self._annotation = value

    # x:Image[(3,128,128)]
    def __str__(self):
        strs = []
        if self.symbol is not None:
            strs.append('{0}: '.format(self.symbol))
        else:
            strs.append('{0}: '.format(self.name))
        strs.append('{0}[{1}]'.format(snake2camel(self.kind), self._shape))

        return ' '.join(strs)

    def __repr__(self):
        return self.__str__()


def get_args_spec(fn):
    full_arg_spec = inspect.getfullargspec(fn)
    arg_spec = inspect.ArgSpec(args=full_arg_spec.args, varargs=full_arg_spec.varargs, keywords=full_arg_spec.varkw,
                               defaults=full_arg_spec.defaults)
    return arg_spec


def format_arg_spec(v, is_output=False):
    s = v.name + ': ' if not is_output and v.name else ''  # (suppress output names, since they duplicate the
    # function name)
    return s + str(v._type)


def update_signature(fn: callable, args: list):
    sig = signature(fn)
    sig = sig.replace(tuple(sig.parameters.values())[1:])


def get_gpu_memory_map():
    '''

    Returns:
        usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    '''
    pathes = [p for p in os.environ['path'].split(';') if 'NVIDIA' in p and 'Corporation' in p]
    nv_path = 'C:/Program Files/NVIDIA Corporation/'
    sp = subprocess.Popen(['{0}/NVSMI/nvidia-smi'.format(nv_path), '-q'], encoding='utf-8-sig', stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

    out_str = sp.communicate()
    out_list = out_str[0].split('\n')

    # Convert lines into a dictionary

    gpu_memory_map = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            gpu_memory_map[key] = val
        except:
            pass
    return gpu_memory_map
