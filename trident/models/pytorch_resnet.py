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

model_urls = {
    'resnet18': '156C4a0_nts8QbjCE8YWbA-QbvCnTrfb5',
    'resnet50': '1R_Ae0DiElUX6yiLqbnq93Iw6vIhaA929',
    'resnet101': '17moUOsGynsWALLHyv3yprHWbbDMrdiOP',
    'resnet152': '1BIaHb7_qunUVvt4TDAwonSKI2jYg4Ybj',
}

def basic_block(num_filters=64,base_width=64,strides=1,expansion = 4,conv_shortcut=False,use_bias=False,name=''):
    shortcut = Identity()
    if strides>1 or conv_shortcut is True:
        shortcut =Conv2d_Block((1,1),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias)

    return ShortCut2d(Sequential(Conv2d_Block((3,3),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation=Relu(inplace=True),use_bias=use_bias),
                                 Conv2d_Block((3,3),num_filters=num_filters,strides=1,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias)),
                      shortcut,activation=Relu(inplace=True))

def bottleneck(num_filters=64,strides=1,expansion = 4,conv_shortcut=True,use_bias=False,name=''):
    #width = int(num_filters * (base_width / 64.)) * 1#groups'
    shortcut = Identity()
    shortcut_name='Identity'
    if strides>1 or conv_shortcut is True:
        shortcut =Conv2d_Block((1,1),num_filters=num_filters*expansion,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias)
        shortcut_name = 'downsample'
    return ShortCut2d({'branch1':Sequential(Conv2d_Block((1,1),num_filters=num_filters ,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation=Relu(inplace=True),use_bias=use_bias),
                                 Conv2d_Block((3, 3), num_filters=num_filters , strides=1, auto_pad=True,padding_mode='zero',normalization='batch', activation=Relu(inplace=True),use_bias=use_bias),
                                 Conv2d_Block((1,1),num_filters=num_filters*expansion,strides=1,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias)),
                      shortcut_name:shortcut},activation=Relu(inplace=True))


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

def ResNet(block, layers, input_shape=(3, 224, 224), num_classes=1000, use_bias=False,  include_top=True, model_name='',
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Args
        block: a function that returns output tensor for the stacked residual blocks.
        layers: list of integer, the number of  repeat units in each blocks.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be`(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        include_top: whether to include the fully-connected layer at the top of the network.
        model_name: string, model name.

    Returns
        A Keras model instance.

    Raises
        ValueError: in case of invalid argument for `weights`,  or invalid input shape.

    """


    def _make_layer(block, num_filters, blocklayers, strides=1, dilate=False,use_bias=use_bias,layer_name=''):
        conv_shortcut=False
        if strides!=1 or block is bottleneck:
            conv_shortcut=True
        layers = []
        layers.append(block(num_filters=num_filters, strides=strides, expansion = 4, conv_shortcut=conv_shortcut,use_bias=use_bias, name=layer_name+'_0'))

        for k in range(1, blocklayers):
            layers.append(block(num_filters=num_filters,  strides=1, expansion = 4, conv_shortcut=False, use_bias=use_bias,name=layer_name+'_{0}'.format(k)))

        laters_block=Sequential(*layers)
        laters_block.name=layer_name
        return laters_block

    flow_list=[]
    resnet = Sequential()
    resnet.add_module('first_block',Conv2d_Block((7,7),64,strides=2,use_bias=use_bias,auto_pad=True,padding_mode='zero',normalization='batch',activation=Relu(inplace=True),name='first_block'))
    resnet.add_module('maxpool',(MaxPool2d((3,3),strides=2,auto_pad=True,padding_mode='zero')))
    resnet.add_module('layer1',(_make_layer(block, 64, layers[0],strides=1, dilate=None,use_bias=use_bias,layer_name='layer1' )))
    resnet.add_module('layer2',(_make_layer(block, 128, layers[1], strides=2, dilate=None,use_bias=use_bias,layer_name='layer2' )))
    resnet.add_module('layer3',(_make_layer(block, 256, layers[2], strides=2, dilate=None,use_bias=use_bias,layer_name='layer3' )))
    resnet.add_module('layer4' ,(_make_layer(block, 512, layers[3], strides=2, dilate=None,use_bias=use_bias,layer_name='layer4' )))
    if include_top:
        resnet.add_module('avg_pool', GlobalAvgPool2d(name='avg_pool'))
        resnet.add_module('fc',Dense(num_classes,activation=None,name='fc'))
        resnet.add_module('softmax', SoftMax(name='softmax'))

    resnet.name=model_name
    model=ImageClassificationModel(input_shape=input_shape,output=resnet)


    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'imagenet_labels1.txt'), 'r', encoding='utf-8-sig') as f:
        labels = [l.rstrip() for l in f]
        model.class_names=labels
    model.preprocess_flow=[Resize((input_shape[1],input_shape[2]),keep_aspect=True),Normalize(0,255),Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    #model.summary()
    return model

def resnet(block, layers, input_shape=(3, 224, 224), num_classes=1000, use_bias=False,  include_top=True, model_name='',
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Args
        block: a function that returns output tensor for the stacked residual blocks.
        layers: list of integer, the number of  repeat units in each blocks.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be`(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        include_top: whether to include the fully-connected layer at the top of the network.
        model_name: string, model name.

    Returns
        A Keras model instance.

    Raises
        ValueError: in case of invalid argument for `weights`,  or invalid input shape.

    """


    def _make_layer(block, num_filters, blocklayers, strides=1, dilate=False,use_bias=use_bias,layer_name=''):
        conv_shortcut=False
        if strides!=1 or block is bottleneck:
            conv_shortcut=True
        layers = []
        layers.append(block(num_filters=num_filters, strides=strides, expansion = 4, conv_shortcut=conv_shortcut,use_bias=use_bias, name=layer_name+'_0'))

        for k in range(1, blocklayers):
            layers.append(block(num_filters=num_filters,  strides=1, expansion = 4, conv_shortcut=False, use_bias=use_bias,name=layer_name+'_{0}'.format(k)))

        laters_block=Sequential(*layers)
        laters_block.name=layer_name
        return laters_block

    flow_list=[]
    resnet = Sequential()
    resnet.add_module('first_block',Conv2d_Block((7,7),64,strides=2,use_bias=use_bias,auto_pad=True,padding_mode='zero',normalization='batch',activation='relu',name='first_block'))
    resnet.add_module('maxpool',(MaxPool2d((3,3),strides=2,auto_pad=True,padding_mode='zero')))
    resnet.add_module('layer1',(_make_layer(block, 64, layers[0],strides=1, dilate=None,use_bias=use_bias,layer_name='layer1' )))
    resnet.add_module('layer2',(_make_layer(block, 128, layers[1], strides=2, dilate=None,use_bias=use_bias,layer_name='layer2' )))
    resnet.add_module('layer3',(_make_layer(block, 256, layers[2], strides=2, dilate=None,use_bias=use_bias,layer_name='layer3' )))
    resnet.add_module('layer4' ,(_make_layer(block, 512, layers[3], strides=2, dilate=None,use_bias=use_bias,layer_name='layer4' )))

    if include_top:
        resnet.add_module('avg_pool', GlobalAvgPool2d(name='avg_pool'))
        resnet.add_module('fc',Dense(num_classes,activation=None,name='fc'))
        resnet.add_module('softmax', SoftMax(name='softmax'))
    resnet.name=model_name
    return resnet

def resnet18(include_top=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    return resnet(basic_block, [2, 2, 2, 2], input_shape=input_shape, include_top=include_top,model_name='resnet18')

def ResNet18(include_top=True,
             pretrained=True,
             freeze_features=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    resnet18 = ResNet(basic_block, [2, 2, 2, 2], input_shape, use_bias=False,include_top=include_top,model_name='resnet18')

    if pretrained == True:
        download_model_from_google_drive(model_urls['resnet18'], dirname, 'resnet18.pth')
        recovery_model = load(os.path.join(dirname, 'resnet18.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        resnet18.model = recovery_model
    else:
        resnet18.model = _make_recovery_model_include_top(resnet18.model, include_top=include_top, classes=classes, freeze_features=True)

    resnet18.model.input_shape = input_shape
    resnet18.model.to(_device)
    return resnet18

def ResNet50(include_top=True,
             pretrained=True,
             freeze_features=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    resnet50 =ResNet(bottleneck, [3, 4, 6, 3], input_shape,num_classes=classes,include_top=include_top, model_name='resnet50')
    if pretrained:
        download_model_from_google_drive(model_urls['resnet50'],dirname,'resnet50.pth')
        recovery_model=load(os.path.join(dirname,'resnet50.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        resnet50.model.load_state_dict(recovery_model.state_dict())
        resnet50.model = _make_recovery_model_include_top(resnet50.model, include_top=include_top, classes=classes,freeze_features=freeze_features)
        #resnet50.model = recovery_model
    else:
        resnet50.model = _make_recovery_model_include_top(resnet50.model, include_top=include_top, classes=classes, freeze_features=True)

    resnet50.model.input_shape = input_shape
    resnet50.model.to(_device)

    return resnet50

def ResNet101(include_top=True,
             pretrained=True,
            freeze_features=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    resnet101 =ResNet(bottleneck, [3, 4, 23, 3], input_shape,num_classes=classes,include_top=include_top, model_name='resnet101')
    if pretrained==True:
        download_model_from_google_drive(model_urls['resnet101'],dirname,'resnet101.pth')
        recovery_model=load(os.path.join(dirname,'resnet101.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        resnet101.model.load_state_dict(recovery_model.state_dict())
        resnet101.model = _make_recovery_model_include_top(resnet101.model, include_top=include_top, classes=classes,
                                                          freeze_features=freeze_features)
        # resnet50.model = recovery_model
    else:
        resnet101.model = _make_recovery_model_include_top(resnet101.model, include_top=include_top, classes=classes, freeze_features=True)

    resnet101.model.input_shape = input_shape
    resnet101.model.to(_device)
    return resnet101


def ResNet152(include_top=True,
             pretrained=True,
            freeze_features=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    resnet152 =ResNet(bottleneck, [3, 8, 36, 3], input_shape,num_classes=classes,include_top=include_top, model_name='resnet152')
    if pretrained==True:
        download_model_from_google_drive(model_urls['resnet152'],dirname,'resnet152.pth')
        recovery_model=load(os.path.join(dirname,'resnet152.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
        resnet152.model.load_state_dict(recovery_model.state_dict())
        resnet152.model = _make_recovery_model_include_top(resnet152.model, include_top=include_top, classes=classes,
                                                           freeze_features=freeze_features)
    else:
        resnet152.model = _make_recovery_model_include_top(resnet152.model, include_top=include_top, classes=classes, freeze_features=True)

    resnet152.model.input_shape = input_shape
    resnet152.model.to(_device)
    return resnet152


#
#
# resnet34=ResNet(basic_block, [3, 4, 6, 3], (3, 224, 224))
# resnet50=ResNet(bottleneck, [3, 4, 6, 3], (3, 224, 224))
# resnet101=ResNet(bottleneck, [3, 4, 23, 3], (3, 224, 224))
# resnet152=ResNet(bottleneck, [3, 8, 36, 3], (3, 224, 224))