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

from trident.backend.common import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential, fix_layer, load,get_device
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *
from trident.models.pretrained_utils import _make_recovery_model_include_top
from trident.data.vision_transforms import Resize,Normalize
__all__ = ['MobileNet','MobileNetV2']

_session = get_session()
_device =get_device()
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


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    Args
        v:
        divisor:
        min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def inverted_residual(in_filters,num_filters=64,strides=1,expansion = 4,name=''):
    mid_filters= int(round(in_filters * expansion))
    layers=[]
    if expansion!=1 :
        layers.append(Conv2d_Block((1,1),num_filters=mid_filters,strides=1,auto_pad=True,padding_mode='zero',normalization='batch',activation='relu6'))

    layers.append(DepthwiseConv2d_Block((3, 3), depth_multiplier=1, strides=strides, auto_pad=True,padding_mode='zero', normalization='batch', activation='relu6'))
    layers.append(Conv2d_Block((1, 1), num_filters=num_filters, strides=1, auto_pad=False, padding_mode='zero', normalization='batch', activation=None))
    if  strides == 1 and in_filters==num_filters:
        return ShortCut2d(Sequential(*layers), Identity(), activation=None)
    else:
        return Sequential(*layers)

def MobileNet( input_shape=(3, 224, 224), classes=1000, use_bias=False, width_mult=1.0,round_nearest=8, include_top=True, model_name='',
           **kwargs):
    input_filters = 32
    last_filters = 1280
    mobilenet=Sequential(name='mobilenet')
    inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
        ]
    input_filters = _make_divisible(input_filters * width_mult, round_nearest)
    last_filters = _make_divisible(last_filters * max(1.0, width_mult), round_nearest)
    features = []
    features.append(Conv2d_Block((3,3),num_filters=input_filters,strides=2,auto_pad=True,padding_mode='zero',normalization='batch',activation='relu6'))
    for t, c, n, s in inverted_residual_setting:
        output_filters = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            strides = s if i == 0 else 1
            features.append( inverted_residual(input_filters,num_filters=output_filters, strides=strides, expansion=t))
            input_filters = output_filters
    features.append(Conv2d_Block((1,1), last_filters,auto_pad=True,padding_mode='zero',normalization='batch',activation='relu6'))
    mobilenet.add_module('features',Sequential(*features,name='features'))

    if include_top:
        mobilenet.add_module('gap', GlobalAvgPool2d())
        mobilenet.add_module('drop', Dropout(0.2))
        mobilenet.add_module('fc',Dense((classes),activation=None))
        mobilenet.add_module('softmax', SoftMax(name='softmax'))
    model = ImageClassificationModel(input_shape=input_shape, output=mobilenet)
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels1.txt')):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels1.txt'), 'r',
                  encoding='utf-8-sig') as f:
            labels = [l.rstrip() for l in f]
            model.class_names = labels
    model.preprocess_flow = [Resize((224, 224), keep_aspect=True), Normalize(0, 255),
                             Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # model.summary()
    return model




def MobileNetV2(include_top=True,
             pretrained=True,
             freeze_features=True,
             input_shape=(3,224,224),
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    mob =MobileNet(input_shape=(3, 224, 224), classes=classes, use_bias=False, width_mult=1.0,round_nearest=8, include_top=include_top, model_name='mobilenet')
    if pretrained==True:
        download_model_from_google_drive('1ULenXTjOO5PdT3fHv6N8bPXEfoJAn5yL',dirname,'mobilenet_v2.pth')
        recovery_model=load(os.path.join(dirname,'mobilenet_v2.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        mob.model = recovery_model

    else:
        mob.model = _make_recovery_model_include_top(mob.model, include_top=include_top, classes=classes, freeze_features=True)

        mob.model.input_shape = input_shape
        mob.model.to(_device)
    return mob