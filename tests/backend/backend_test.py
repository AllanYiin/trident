import os
import pytest
from numpy.testing import assert_allclose
import numpy as np
import scipy.sparse as sparse
import warnings
from trident import get_backend as T
from trident.backend import set_backend

os.environ['TRIDENT_BACKEND']='cntk'



try:
    #os.environ['TRIDENT_BACKEND']='cntk'
    set_backend('cntk')
    from trident.backend import cntk_backend as CB
except ImportError:
    CB = None
    warnings.warn('Could not import the CNTK backend')

try:
    #os.environ['TRIDENT_BACKEND'] = 'pytorch'
    set_backend('pytorch')
    from trident.backend import pytorch_backend as PB
except ImportError:
    PB = None
    warnings.warn('Could not import the Pytorch backend')

try:
    #os.environ['TRIDENT_BACKEND'] = 'tensorflow'
    set_backend('tensorflow')
    from trident.backend import tensorflow_backend as TB
except ImportError:
    TB = None
    warnings.warn('Could not import the Tensorflow backend')


