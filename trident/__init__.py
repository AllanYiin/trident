"""trident api"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
from importlib import reload
from sys import stderr

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

__version__ = '0.7.1'
stderr.write('trident {0}\n'.format(__version__))

from trident import context
from trident.backend import *
import threading
import random
import numpy as np



