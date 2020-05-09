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
from trident.data.data_provider import *
from trident.data.image_common import *

from trident.data.image_reader import ImageReader,ImageThread
from trident.data.data_loaders import *
from trident.data.utils import *
from trident.data.preprocess_policy import *
from trident.data.augment_policy import *
from trident.data.mask_common import *
from trident.data.label_common import *
from trident.data.bbox_common  import *

# import pyximport; pyximport.install()
# from .cython_nms import *
# from .cython_bbox import *



