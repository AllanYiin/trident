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
from ..backend.pytorch_backend import to_numpy,to_tensor,Layer,Sequential
from ..layers.pytorch_layers import *
from ..layers.pytorch_activations import  get_activation,Identity
from ..layers.pytorch_normalizations import get_normalization
from ..layers.pytorch_blocks import *
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *
from ..data.image_common import *
from ..data.utils import download_file_from_google_drive

__all__ = ['gan_builder']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon=_session.epsilon
_trident_dir=_session.trident_dir


def resnet_block(num_filters=64,strides=1,expansion = 4,name=''):
    shortcut = Identity()
    return ShortCut2d(Sequential(Conv2d_Block((3,3),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode=PaddingMode.reflection,normalization='instance',activation='leaky_relu',name=name + '_conv1'),
                                 Conv2d_Block((3,3),num_filters=num_filters,strides=1,auto_pad=True,padding_mode=PaddingMode.reflection,normalization='instance',activation=None,name=name + 'conv2')),
                      shortcut,activation='relu')


def efficient_block( expand_ratio=1 , filters_in=32, filters_out=16, kernel_size=3, strides=1, zero_pad=0, se_ratio=0,  drop_rate=0.2,is_shortcut=True,name='',**kwargs):
    expand_ratio=kwargs.get('expand_ratio',expand_ratio)
    is_shortcut=kwargs.get('id_skip',is_shortcut)
    filters_in = kwargs.get('filters_in', filters_in)
    filters_out = kwargs.get('filters_out', filters_out)
    kernel_size = kwargs.get('kernel_size', kernel_size)
    is_shortcut=filters_in==filters_out and strides==1 and kwargs.get('id_skip',is_shortcut)
    filters = filters_in * expand_ratio
    if expand_ratio ==1 and strides==1:

        bottleneck=Sequential(
            DepthwiseConv2d_Block((kernel_size,kernel_size),depth_multiplier=1,strides=strides,auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch',activation='swish',name=name + 'dwconv'),
            SqueezeExcite( se_filters= max(1, int(filters_in * se_ratio)),num_filters=filters_in,use_bias=True) if 0 < se_ratio <= 1 else Identity(),
            Conv2d_Block((1,1),num_filters=filters_out,strides=1,auto_pad=True,normalization='batch', activation=None,name=name + 'se'),
            Dropout(dropout_rate=drop_rate) if is_shortcut and drop_rate > 0 else Identity()
        )


        if is_shortcut:
            return ShortCut2d(Identity(),bottleneck)
        else:
            return bottleneck

    else:
        bottleneck=Sequential(Conv2d_Block((1, 1), num_filters=filters, strides=1, auto_pad=True, normalization='batch', activation='swish' ,name=name + 'expand_bn'),
            DepthwiseConv2d_Block((kernel_size, kernel_size), depth_multiplier=1, strides=strides, auto_pad=True,padding_mode=PaddingMode.zero, normalization='batch', activation='swish',name=name + 'dwconv'),
            SqueezeExcite(se_filters= max(1, int(filters_in * se_ratio)),num_filters=filters,use_bias=True) if 0 < se_ratio <= 1 else Identity(),
            Conv2d_Block((1, 1), num_filters=filters_out, strides=1, auto_pad=True,normalization='batch', activation=None,name=name + 'se'),
            Dropout(dropout_rate=drop_rate) if is_shortcut and drop_rate > 0 else Identity()
        )
        if is_shortcut:
            return ShortCut2d(Identity(),bottleneck)
        else:
            return bottleneck


def gan_builder(
        noise_shape=100,
        image_size=(3,256,256) ):

    noise_input=torch.tensor(data=np.random.normal(0, 1,size=noise_shape))

    def build_generator(noise_input):
        layers=[]
        layers.append(Conv2d_Block(()))

    def build_discriminator(input):


