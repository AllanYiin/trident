from __future__ import absolute_import

import sys
from importlib import reload
from sys import stderr

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)



__version__ = '0.5.4'
stderr.write('trident {0}\n'.format(__version__))

from . import backend
from .backend import *
from . import models
from . import misc
from . import callbacks
from . import data
import threading



