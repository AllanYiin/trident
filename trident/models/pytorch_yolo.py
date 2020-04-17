from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
import os
import uuid
from collections import *
from collections import deque
from copy import copy, deepcopy
from functools import partial
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch.nn import init
from torch.nn.parameter import Parameter

from ..backend.common import *
from ..backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential
from ..data.image_common import *
from ..data.utils import download_model_from_google_drive
from ..layers.pytorch_activations import get_activation, Identity
from ..layers.pytorch_blocks import *
from ..layers.pytorch_layers import *
from ..layers.pytorch_normalizations import get_normalization
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *

__all__ = []

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon=_session.epsilon
_trident_dir=_session.trident_dir
