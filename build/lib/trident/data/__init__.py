import os
import threading

_SESSION = threading.local()

if 'TRIDENT_HOME' in os.environ:
    _trident_dir = os.environ.get('TRIDENT_HOME')
else:
    _trident_base_dir = os.path.expanduser('~')
    if not os.access(_trident_base_dir, os.W_OK):
        _trident_base_dir = '/tmp'
    _trident_dir = os.path.join(_trident_base_dir, '.trident')

_SESSION.trident_dir=_trident_dir

from ..backend.load_backend import get_backend,get_image_backend
from .datasets_common import *
from .data_loaders import *
from .utils import *

