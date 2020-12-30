from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import warnings

import numpy as np

from trident.backend.common import *
from trident.backend.common import get_backend
from trident.callbacks.callback_base import *

if get_backend()=='pytorch':
    from trident.backend.pytorch_ops import to_numpy,to_tensor
elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import  to_numpy,to_tensor

