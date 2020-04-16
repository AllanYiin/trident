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
__all__ = ['basic_block','bottleneck','ResNet']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon=_session.epsilon

def basic_block(num_filters=64,base_width=64,strides=1,expansion = 4,conv_shortcut=False,name=''):
    shortcut = Identity()
    if strides>1 or conv_shortcut is True:
        shortcut =Conv2d_Block((1,1),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch',activation=None,name=name + '_0_conv')

    return ShortCut2d(shortcut,
                      Sequential(Conv2d_Block((3,3),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch',activation='relu',name=name + '_1_conv'),
                                 Conv2d_Block((3,3),num_filters=num_filters,strides=1,auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch',activation=None,name=name + '_2_conv')),
                      activation='relu')

def bottleneck(num_filters=64,base_width=64,strides=1,expansion = 4,conv_shortcut=True,name=''):
    width = int(num_filters * (base_width / 64.)) * 1#groups'
    shortcut = Identity()

    if strides>1 or conv_shortcut is True:
        shortcut =Conv2d_Block((1,1),num_filters=num_filters*expansion,strides=strides,auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch',activation=None,name=name + '_0_conv')

    return ShortCut2d(shortcut,
                      Sequential(Conv2d_Block((1,1),num_filters=width ,strides=strides,auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch',activation='relu',name=name + '_1_conv'),
                                 Conv2d_Block((3, 3), num_filters=width , strides=1, auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch', activation='relu',name=name + '_2_conv'),
                                 Conv2d_Block((1,1),num_filters=num_filters*expansion,strides=1,auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch',activation=None,name=name + '_3_conv')),
                      activation='relu')


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

def ResNet( block, layers, input_shape=(3,224,224),num_classes=1000, use_bias=True,zero_init_residual=False,
                  width_per_group=64, replace_stride_with_dilation=None,
                 **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """


    # if not (weights in {'imagenet', None} or os.path.exists(weights)):
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization), `imagenet` '
    #                      '(pre-training on ImageNet), '
    #                      'or the path to the weights file to be loaded.')
    #
    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
    #                      ' as true, `classes` should be 1000')

    def _make_layer(block, num_filters, blocklayers, strides=1, dilate=False,layer_name=''):
        conv_shortcut=False
        if strides!=1:
            conv_shortcut=True

        layers = []
        layers.append(block(num_filters=num_filters, base_width=64, strides=strides, expansion = 4, conv_shortcut=conv_shortcut, name=layer_name+'_0'))

        for k in range(1, blocklayers):
            layers.append(block(num_filters=num_filters, base_width=64, strides=1, expansion = 4, conv_shortcut=False, name=layer_name+'_{0}'.format(k)))

        return layers

    flow_list=[]
    flow_list.append(Conv2d_Block((7,7),64,strides=2,use_bias=use_bias,auto_pad=True,padding_mode=PaddingMode.zero,normalization='batch',activation='relu'))
    flow_list.append(MaxPool2d((3,3),strides=2,auto_pad=True,padding_mode=PaddingMode.zero))
    flow_list.extend(_make_layer(block, 64, layers[0],strides=1, dilate=None,layer_name='layer1' ))
    flow_list.extend(_make_layer(block, 128, layers[1], strides=2, dilate=None,layer_name='layer2' ))
    flow_list.extend(_make_layer(block, 256, layers[2], strides=2, dilate=None,layer_name='layer3' ))
    flow_list.extend(_make_layer(block, 512, layers[3], strides=2, dilate=None,layer_name='layer4' ))
    flow_list.append(GlobalAvgPool2d(name='avg_pool'))
    flow_list.append(Dense(num_classes, activation='softmax'))

    model=Model(input_shape=input_shape,output=Sequential(*flow_list))
    model.summary()
    return model