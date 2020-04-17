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
from .data_provider import *
from .image_common import *
from .text_common import *
from .image_reader import ImageReader,ImageThread
from .data_loaders import *
from .utils import *
from .preprocess_policy import *
from .augment_policy import *
from .mask_common import *
from .label_common import *
from .bbox_common  import *

# import pyximport; pyximport.install()
# from .cython_nms import *
# from .cython_bbox import *
from .bbox_common import *


