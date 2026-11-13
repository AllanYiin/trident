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
from collections import abc
from torch.nn import init
from torch.nn.parameter import Parameter
from trident.models.pretrained_utils import _make_recovery_model_include_top
from trident import context
from trident.backend.common import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential, fix_layer, get_device,load
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity,Relu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *
from trident.data.vision_transforms import Resize,Normalize
__all__ = ['basic_block','bottleneck', 'ResNet','ResNet18','ResNet50','ResNet101','ResNet152','resnet','resnet18']

ctx = context._context()
_device =get_device()
_epsilon=ctx.epsilon
_trident_dir=ctx.trident_dir


dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass