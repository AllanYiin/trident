from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import threading
import os
import math
import shlex
import struct
import platform
import subprocess
import numpy as np
import re
from pydoc import locate
import datetime
import time

__all__ = ['get_session','get_trident_dir','epsilon','set_epsilon','floatx','set_floatx','camel2snake','snake2camel','addindent','format_time','get_time_prefix','get_function','get_class','get_terminal_size','gcd','get_divisors','isprime','next_prime','prev_prime','nearest_prime']

if 'TRIDENT_HOME' in os.environ:
    _trident_dir = os.environ.get('TRIDENT_HOME')
else:
    _trident_base_dir = os.path.expanduser('~')
    if not os.access(_trident_base_dir, os.W_OK):
        _trident_base_dir = '/tmp'
    _trident_dir = os.path.join(_trident_base_dir, '.trident')


_SESSION = threading.local()

_SESSION.trident_dir=_trident_dir

def get_session():
    return _SESSION

def get_trident_dir():
    return _trident_dir

_SESSION.floatx='float32'
_SESSION.epsilon= 1e-8
# the type of float to use throughout the session.

_SESSION.backend='pytorch'
_SESSION.image_data_format='channels_first'
_SESSION.image_channel_order='bgr'


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> trident.get_backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON


def set_epsilon(e):

    global _EPSILON
    _EPSILON = float(e)


def floatx():
    """Returns the default float type, as a string.
    (e.g. 'float16', 'float32', 'float64').

    # Returns
        String, the current default float type.

    # Example
    ```python
        >>> trident.get_backend.floatx()
        'float32'
    ```
    """
    return _SESSION.floatx


def set_floatx(floatx):

    if floatx not in {'float16', 'float32', 'float64'}:
        raise ValueError('Unknown floatx type: ' + str(floatx))
    _SESSION.floatx = str(floatx)


def camel2snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake2camel(name):
    return ''.join(x.capitalize() or '_' for x in name.split('_'))


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
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

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


def get_time_prefix():
    prefix = str(datetime.datetime.fromtimestamp(time.time())).replace(' ', '').replace(':', '').replace('-','').replace( '.', '')
    return prefix

def get_function(fn_or_name,module_paths= None):
    if callable(fn_or_name):
        return fn_or_name
    fn = locate(fn_or_name)
    if (fn is None) and (module_paths is not None):
        for module_path in module_paths:
            fn = locate('.'.join([module_path, fn_or_name]))
            if fn is not None:
                break
    if fn is None:
        raise ValueError( "Method not found in {}: {}".format(module_paths, fn_or_name))
    return fn  # type: ignore

def get_class(class_name, module_paths = None) :
    r"""Returns the class based on class name.
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
    class_ = locate(class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            class_ = locate('.'.join([module_path, class_name]))
            if class_ is not None:
                break
    if class_ is None:
        raise ValueError(
            "Class not found in {}: {}".format(module_paths, class_name))
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
        print
        "default"
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
        if res:
            (bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh",
                                                                                                  csbi.raw)
            sizex = right - left + 1
            sizey = bottom - top + 1
            return sizex, sizey
    except:
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
    for k in range(int(y // 2), 0, -1):
        if x % k == 0 and y % k == 0:
            gcd = k
            gcds.append(int(k))
    return gcds


def get_divisors(n):
    return [d for d in range(2, n // 2) if n % d == 0]


def isprime(n):
    divisors = [d for d in range(2, int(math.sqrt(n))) if n % d == 0]
    return all(n % od != 0 for od in divisors if od != n)

def next_prime(n):
    pos=n+1
    while not isprime(pos):
        pos+=1
    return pos

def prev_prime(n):
    pos=n-1
    while not isprime(pos):
        pos-=1
    return pos

def nearest_prime(n):
    nextp=next_prime(n)
    prevp=prev_prime(n)
    if math.abs(nextp-n)<math.abs(prevp-n):
        return nextp
    else:
        return nextp











