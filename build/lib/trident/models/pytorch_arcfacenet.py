
from __future__ import division
from __future__ import print_function
import math
import os
import inspect
import numpy as np
from collections import *
from functools import partial
import uuid
from copy import copy, deepcopy
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torch._six import container_abcs
from itertools import repeat


from ..backend.common import *
from ..backend.pytorch_backend import to_numpy,to_tensor,Layer,Sequential,Input,summary
from ..layers.pytorch_layers import *
from ..layers.pytorch_activations import  get_activation,Identity,Relu,PRelu
from ..layers.pytorch_normalizations import get_normalization,BatchNorm2d,BatchNorm
from ..layers.pytorch_blocks import *
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *
from ..data.image_common import *
from ..data.utils import download_model_from_google_drive


__all__ = ['SEResNet_IR']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon=_session.epsilon
_trident_dir=_session.trident_dir
_backend = _session.backend

dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass






def BottleNeck_IR(num_filters, strides,keep_filter=True):
    if keep_filter:
        return ShortCut2d(MaxPool2d(1, strides=strides), Sequential(BatchNorm2d(),
                                                                    Conv2d_Block((3, 3), num_filters=num_filters,
                                                                                 strides=1, auto_pad=True,
                                                                                 use_bias=False, activation=PRelu(),
                                                                                 normalization='batch'),
                                                                    Conv2d_Block((3, 3), num_filters, strides=strides,
                                                                                 use_bias=False, activation=None,
                                                                                 normalization='batch')))
    else:
        return ShortCut2d(
            Conv2d_Block((1, 1), num_filters, strides=strides, use_bias=False, activation=None, normalization='batch'),
            Sequential(BatchNorm2d(),
                       Conv2d_Block((3, 3), num_filters=num_filters, strides=1, auto_pad=True, use_bias=False,
                                    activation=PRelu(), normalization='batch'),
                       Conv2d_Block((3, 3), num_filters, strides=strides, use_bias=False, activation=None,
                                    normalization='batch'), SqueezeExcite(num_filters, 16)))

def BottleNeck_IR_SE( num_filters, strides,keep_filter=True):
    if keep_filter:
        return ShortCut2d(MaxPool2d(1, strides=strides),
                          Sequential(BatchNorm2d(),
                                     Conv2d_Block((3, 3), num_filters=num_filters,strides=1, auto_pad=True,use_bias=False, activation=PRelu(),normalization='batch'),
                                     Conv2d_Block((3, 3), num_filters, strides=strides,use_bias=False, activation=None,  normalization='batch'),
                        SqueezeExcite(num_filters, 16)))
    else:
        return ShortCut2d(
            Conv2d_Block((1, 1), num_filters, strides=strides, use_bias=False, activation=None, normalization='batch'),
            Sequential(BatchNorm2d(),
                       Conv2d_Block((3, 3), num_filters=num_filters, strides=1, auto_pad=True, use_bias=False,activation=PRelu(), normalization='batch'),
                       Conv2d_Block((3, 3), num_filters, strides=strides, use_bias=False, activation=None,normalization='batch'),
                        SqueezeExcite(num_filters, 16)))

def get_block(Bottleneck, out_channel, num_units, strides=2,keep_filter=True):
    return Sequential([Bottleneck(out_channel, strides,keep_filter)] + [Bottleneck(out_channel, 1,keep_filter) for i in range(num_units - 1)])



def SEResNet_IR(num_layers=50,Bottleneck=BottleNeck_IR_SE,drop_ratio=0.4,feature_dim=128):
    blocks = Sequential([
        get_block(Bottleneck, out_channel=64, num_units=3),
        get_block(Bottleneck, out_channel=128, num_units=4),
        get_block(Bottleneck, out_channel=256, num_units=14),
        get_block(Bottleneck, out_channel=512, num_units=3)
    ])
    return Sequential(
    Conv2d_Block((3,3),64,strides=1,auto_pad=True,use_bias=False,activation=PRelu(64),normalization='batch'),
    blocks,
    BatchNorm2d(),
    Dropout(drop_ratio),
    Flatten(),
    Dense(feature_dim),
    BatchNorm())






