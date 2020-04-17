from __future__ import absolute_import

import sys
from importlib import reload
from sys import stderr

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)



__version__ = '0.5.1'
stderr.write('trident {0}\n'.format(__version__))
from .backend import *
from trident import models
from trident import misc
from trident import callbacks
from trident import data

import threading



