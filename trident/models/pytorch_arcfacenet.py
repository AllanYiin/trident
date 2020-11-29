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
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential, summary, fix_layer, load
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity, Relu, PRelu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization, BatchNorm2d, BatchNorm
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *

__all__ = ['SEResNet_IR','load','BottleNeck_IR_SE','BottleNeck_IR','SEResNet_IR_50_512']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon=_session.epsilon
_trident_dir=_session.trident_dir
_backend = get_backend()

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



def SEResNet_IR(num_layers=50,Bottleneck=BottleNeck_IR_SE,drop_ratio=0.4,feature_dim=128,input_shape=(3,112,112)):
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
    facenet=Sequential(blocks).to(_device)
    facenet.name=camel2snake('SEResNet_IR')
    model=ImageClassificationModel(input_shape=input_shape,output=facenet)
    model.preprocess_flow=[resize((input_shape[1],input_shape[2]),keep_aspect=True),normalize(0,255),normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    #model.summary()
    return model


def SEResNet_IR_50_512(
             pretrained=True,
             freeze_features=False,
             input_shape=(3,112,112),
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 112, 112)
    seresnet = SEResNet_IR(num_layers=50,Bottleneck=BottleNeck_IR_SE,drop_ratio=0.4,feature_dim=512,input_shape=input_shape)
    if pretrained == True:
        download_model_from_google_drive('1aLYbFvtvsV2gQ16D_vwzrKdbCgij7IoZ', dirname, 'arcface_se_50_512.pth')
        recovery_model = load(os.path.join(dirname, 'arcface_se_50_512.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model.name = 'arcface_se_50_512'
        recovery_model.eval()
        recovery_model.to(_device)
        seresnet.model = recovery_model
    return seresnet




