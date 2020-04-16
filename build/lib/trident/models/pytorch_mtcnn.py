from __future__ import absolute_import
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
from torch.nn.parameter import Parameter
from torch._six import container_abcs
from itertools import repeat


from ..backend.common import *
from ..backend.pytorch_backend import to_numpy,to_tensor,Layer,Sequential,Combine
from ..layers.pytorch_layers import *
from ..layers.pytorch_activations import  get_activation,Identity,PRelu
from ..layers.pytorch_normalizations import get_normalization
from ..layers.pytorch_blocks import *
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *
from ..data.image_common import *
from ..data.utils import download_file_from_google_drive

__all__ = ['P_net','R_net','O_net']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon=_session.epsilon
_trident_dir=_session.trident_dir


dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


p_net=Sequential(
    Conv2d((3,3),10,strides=1,auto_pad=False,name='conv1'),
    PRelu(),
    MaxPool2d((2,2),strides=2),
    Conv2d((3, 3), 16, strides=1, auto_pad=False,name='conv2'),
    PRelu(),
    Conv2d((3,3),32,strides=1,auto_pad=False,name='conv3'),
    PRelu(),
    Combine(
        Conv2d((1,1),1,strides=1,auto_pad=False,name='conv4_1'),
        Conv2d((1,1),4,strides=1,auto_pad=False,name='conv4_2'),
        Conv2d((1,1),10,strides=1,auto_pad=False,name='conv4_3')))
P_net=ImageDetectionModel(input_shape=(3,12,12),output=p_net)

r_net=Sequential(
    Conv2d((3,3),28,strides=1,auto_pad=False,name='conv1'),
    PRelu(),
    MaxPool2d((3,3),strides=2),
    Conv2d((3, 3), 48, strides=1, auto_pad=False,name='conv2'),
    PRelu(),
    Conv2d((3,3),64,strides=1,auto_pad=False,name='conv3'),
    PRelu(),
    Flatten(),
    Dense(128,activation=None,name='conv4'),
    PRelu(),
    Combine(
        Dense(1,activation=None,name='conv5_1'),
        Dense(4,activation=None,name='conv5_2'),
        Dense(10,activation=None,name='conv5_3')))
R_net=ImageDetectionModel(input_shape=(3,24,24),output=r_net)

o_net=Sequential(
    Conv2d((3,3),32,strides=1,auto_pad=False,name='conv1'),
    PRelu(),
    MaxPool2d((3,3),strides=2),
    Conv2d((3, 3), 64, strides=1, auto_pad=False,name='conv2'),
    PRelu(),
    MaxPool2d((3,3),strides=2),
    Conv2d((3,3),64,strides=1,auto_pad=False,name='conv3'),
    PRelu(),
    MaxPool2d((2, 2), strides=2),
    Conv2d((2, 2), 128, strides=1, auto_pad=False,name='conv4'),
    PRelu(),
    Flatten(),
    Dense(256,activation=None,name='conv5'),
    PRelu(),
    Combine(
        Dense(1,activation=None,name='conv6_1'),
        Dense(4,activation=None,name='conv6_2'),
        Dense(10,activation=None,name='conv6_3')))
O_net=ImageDetectionModel(input_shape=(3,48,48),output=o_net)