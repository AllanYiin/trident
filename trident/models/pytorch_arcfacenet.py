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

from trident.backend.common import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential, Input, summary
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity, Relu, PRelu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization, BatchNorm2d, BatchNorm
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *

__all__ = ['SEResNet_IR','BottleNeck_IR_SE','BottleNeck_IR']

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
    blocks = OrderedDict()
    blocks['res_layer'] = Sequential(BatchNorm2d(),
                                     Conv2d_Block((3, 3), num_filters=num_filters, strides=1, auto_pad=True, use_bias=False, activation=PRelu(num_filters)),
                                     Conv2d_Block((3, 3), num_filters, strides=strides, use_bias=False, activation=None,  normalization='batch'))
    if keep_filter:
        blocks['shortcut_layer']=MaxPool2d(1, strides=strides,name='shortcut_layer')
    else:
        blocks['shortcut_layer'] = Conv2d_Block((1, 1), num_filters, strides=strides, use_bias=False, activation=None, normalization='batch')
    return ShortCut2d(blocks,mode='add')

def BottleNeck_IR_SE( num_filters, strides,keep_filter=True):
    blocks = OrderedDict()
    blocks['res_layer'] = Sequential(BatchNorm2d(),
                                     Conv2d_Block((3, 3), num_filters=num_filters, strides=1, auto_pad=True, use_bias=False, activation=PRelu(num_filters)),
                                     Conv2d_Block((3, 3), num_filters, strides=strides, use_bias=False, activation=None, normalization='batch'),
                                     SqueezeExcite(num_filters//16,num_filters),name='res_layer')
    if keep_filter:
        blocks['shortcut_layer'] = MaxPool2d(1, strides=strides, name='shortcut_layer')

    else:
        blocks['shortcut_layer'] =Conv2d_Block((1, 1), num_filters, strides=strides, use_bias=False, activation=None, normalization='batch',name='shortcut_layer')
    return ShortCut2d(blocks,mode='add')


def get_block(Bottleneck, out_channel, num_units, strides=2,keep_filter=True):
    blocks=[Bottleneck(out_channel, strides,keep_filter)]
    for i in range(num_units - 1):
        blocks.append(Bottleneck(out_channel, 1,True))
    return blocks



def SEResNet_IR(num_layers=50,Bottleneck=BottleNeck_IR_SE,drop_ratio=0.4,feature_dim=128):
    blocks=OrderedDict()
    blocks['input_layer']=Conv2d_Block((3,3),64,strides=1,auto_pad=True,use_bias=False,activation=PRelu(64),normalization='batch',name='input_layer')
    blocks['body']=Sequential(
        get_block(Bottleneck, out_channel=64, num_units=3,keep_filter=True)+
        get_block(Bottleneck, out_channel=128, num_units=4,keep_filter=False)+
        get_block(Bottleneck, out_channel=256, num_units=14,keep_filter=False)+
        get_block(Bottleneck, out_channel=512, num_units=3,keep_filter=False)
    )
    blocks['output_layer']=Sequential(
        BatchNorm2d(),
        Dropout(drop_ratio),
        Flatten(),
        Dense(feature_dim),
        BatchNorm(),
        name='output_layer'
    )
    return Sequential(blocks).to(_device)






