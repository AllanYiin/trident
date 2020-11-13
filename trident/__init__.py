"""trident api"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from importlib import reload
from sys import stderr

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)



__version__ = '0.6.1'
stderr.write('trident {0}\n'.format(__version__))


from trident.backend import *
import threading
import random



