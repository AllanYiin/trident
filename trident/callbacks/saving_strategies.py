from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import warnings

import numpy as np

from ..backend.common import *
from ..backend.load_backend import get_backend
from ..callbacks import *

if get_backend()=='pytorch':
    from ..backend.pytorch_ops import to_numpy,to_tensor
elif get_backend()=='tensorflow':
    from ..backend.tensorflow_ops import  to_numpy,to_tensor

