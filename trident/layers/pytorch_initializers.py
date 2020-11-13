from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trident.backend.common import get_session
from trident.backend.pytorch_ops import *

__all__ = []

_session=get_session()
_epsilon=_session.epsilon
