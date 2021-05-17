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
from trident.models.pretrained_utils import _make_recovery_model_include_top

from trident.backend.common import *
from trident.backend.model import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential, fix_layer,load,get_device
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity, Relu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import ImageClassificationModel
from trident.data.vision_transforms import Resize,Normalize
__all__ = ['VGG19','VGG11','VGG13','VGG16']

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




cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_vgg_layers(cfg, num_classes=1000,input_shape=(3,224,224),include_top=True):
    layers = []
    in_channels = 3
    block=1
    conv=1
    vgg=Sequential()
    for v in cfg:
        if v == 'M':
            vgg.add_module('block{0}_pool'.format(block),MaxPool2d(kernel_size=2, strides=2,use_bias=True,name='block{0}_pool'.format(block)))
            block += 1
            conv = 1
        else:
            if len(vgg)==0:
                vgg.add_module('block{0}_conv{1}'.format(block,conv),Conv2d((3,3),v,auto_pad=True,activation=None,use_bias=True,name='block{0}_conv{1}'.format(block,conv)))
            else:
                vgg.add_module('block{0}_conv{1}'.format(block, conv), Conv2d((3, 3), v, auto_pad=True, activation=None, use_bias=True,name='block{0}_conv{1}'.format(block, conv)))

            vgg.add_module('block{0}_relu{1}'.format(block, conv),Relu(name='block{0}_relu{1}'.format(block, conv)))
            conv+=1
            in_channels = v
    if include_top:
        vgg.add_module('avgpool',AdaptiveAvgPool2d((7,7)))
        vgg.add_module('flattened', Flatten())
        vgg.add_module('fc1',Dense(4096,use_bias=True, activation='relu'))
        vgg.add_module('drop1', Dropout(0.5))
        vgg.add_module('fc2', Dense(4096, use_bias=True,activation='relu'))
        vgg.add_module('drop2', Dropout(0.5))
        vgg.add_module('fc3', Dense(num_classes,use_bias=True,activation='softmax'))


    model = ImageClassificationModel(input_shape=input_shape, output=vgg)
    model.model.to(_device)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels1.txt'), 'r',
              encoding='utf-8-sig') as f:
        labels = [l.rstrip() for l in f]
        model.class_names = labels
    model.preprocess_flow = [Resize((input_shape[2], input_shape[1]), keep_aspect=True), Normalize(0, 255),
                             Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # model.summary()

    return model


#vgg11 =make_vgg_layers(cfgs['A'], 1000)
def VGG11(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    vgg11 =make_vgg_layers(cfgs['A'], classes)
    vgg11.input_shape =input_shape
    if pretrained:
        download_model_from_google_drive('1PV9-AwgD1v-JxDRzduOjjGduIR7MDhPW',dirname,'vgg11.pth')
        recovery_model=load(os.path.join(dirname,'vgg11.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model.name='vgg11'
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        vgg11.model = recovery_model

    else:
        vgg11.model = _make_recovery_model_include_top(vgg11.model, include_top=include_top, classes=classes, freeze_features=False)

        vgg11.model.input_shape = input_shape
        vgg11.model.to(_device)
    return vgg11





#vgg13 =make_vgg_layers(cfgs['B'],  1000)
def VGG13(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    vgg13 =make_vgg_layers(cfgs['B'], classes)

    if pretrained==True:
        download_model_from_google_drive('1wx67gmQ8eHWXs2mhJmNl-t-cFNw7dJ7O',dirname,'vgg13.pth')
        recovery_model=load(os.path.join(dirname,'vgg13.pth'))
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        vgg13.model = recovery_model

    else:
        vgg13.model = _make_recovery_model_include_top(vgg13.model, include_top=include_top, classes=classes, freeze_features=False)

    vgg13.model.input_shape = input_shape
    vgg13.model.to(_device)
    return vgg13


#vgg16 =make_vgg_layers(cfgs['D'],  1000)
def VGG16(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    vgg16 =make_vgg_layers(cfgs['D'], classes)
    vgg16.input_shape =input_shape
    if pretrained==True:
        download_model_from_google_drive('1uXiH5MSy1rvxrHjW4uB9E2BHMM8b0Fwr',dirname,'vgg16.pth')
        recovery_model=load(os.path.join(dirname,'vgg16.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model.name = 'vgg16'
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        vgg16.model = recovery_model

    else:
        vgg16.model = _make_recovery_model_include_top(vgg16.model, include_top=include_top, classes=classes, freeze_features=False)

        vgg16.model.input_shape = input_shape
        vgg16.model.to(_device)
    return vgg16

#vgg19 =make_vgg_layers(cfgs['E'], 1000)
def VGG19(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    vgg19 =make_vgg_layers(cfgs['E'], classes)
    vgg19.input_shape =input_shape
    if pretrained==True:
        download_model_from_google_drive('1nqQJLYMzeiUX9hji39-rrBUG42YyjhYg',dirname,'vgg19.pth')
        recovery_model=load(os.path.join(dirname,'vgg19.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model.name = 'vgg19'
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        vgg19.model = recovery_model

    else:
        vgg19.model = _make_recovery_model_include_top(vgg19.model, include_top=include_top, classes=classes, freeze_features=False)

        vgg19.model.input_shape = input_shape
        vgg19.model.to(_device)
    return vgg19