bsolute_import
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
from ..backend.pytorch_backend import to_numpy,to_tensor,Layer,Sequential,Input
from ..layers.pytorch_layers import *
from ..layers.pytorch_activations import  get_activation,Identity,Relu
from ..layers.pytorch_normalizations import get_normalization,BatchNorm2d
from ..layers.pytorch_blocks import *
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *
from ..data.image_common import *
from ..data.utils import download_file_from_google_drive

__all__ = ['DenseNet']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon=_session.epsilon
_trident_dir=_session.trident_dir

current_filters=32

def conv_block(growth_rate, name):
    return ConcateBlock(Identity,
                 Sequential(
                     BatchNorm2d(name=name + '_0_bn'),
                     Relu(name=name + '_0_relu'),
                     Conv2d_Block((1,1),4 * growth_rate,strides=1,use_bias=False, name=name + '_1_conv'),
                    Conv2d((3,3),growth_rate,strides=1,use_bias=False, name=name + '_2_conv')))

def dense_block(blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    items=OrderedDict()
    for i in range(blocks):
        items[name + '_block' + str(i + 1)]=conv_block(32, name=name + '_block' + str(i + 1))
    return Sequential(items)


def transition_block(reduction, name):
    return Sequential(
            BatchNorm2d(name=name + '_0_bn'),
            Relu(name=name + '_0_relu'),
           Conv2d((3, 3), current_filters*reduction, strides=1, use_bias=False, name=name + '_conv'),
       AvgPool2d((2,2),2, name=name + '_pool'))




def DenseNet(blocks,
             include_top=True,
             pretrained=True,
             input_shape=(3,224,224),
             num_classes=1000,
             **kwargs):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
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
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
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


    densenet=Sequential()
    densenet.add_module('input',Input(input_shape=input_shape))
    densenet.add_module('conv1/conv',Conv2d_Block((7,7),64,strides=2,use_bias=False,auto_pad=True,padding_mode=PaddingMode.zero,activation='relu',normalization='batch', name='conv1/conv'))
    densenet.add_module('maxpool', (MaxPool2d((3, 3), strides=2, auto_pad=True, padding_mode=PaddingMode.zero)))
    densenet.add_module('conv2', dense_block(blocks[0], name='conv2'))
    densenet.add_module('pool2', transition_block(0.5, name='pool2'))
    densenet.add_module('conv3', dense_block(blocks[1], name='conv3'))
    densenet.add_module('pool3', transition_block(0.5, name='pool3'))
    densenet.add_module('conv4', dense_block(blocks[2], name='conv4'))
    densenet.add_module('pool4', transition_block(0.5, name='pool4'))
    densenet.add_module('conv5', dense_block(blocks[3], name='conv5'))
    densenet.add_module('bn',BatchNorm2d(name='bn'))
    densenet.add_module('relu', Relu(name='relu'))
    densenet.add_module('avg_pool', GlobalAvgPool2d(name='avg_pool'))
    if include_top:
        densenet.add_module('fc', Dense(num_classes, activation='softmax', name='fc'))
    return densenet