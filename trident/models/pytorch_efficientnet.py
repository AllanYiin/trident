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
import builtins
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch.nn import init
from torch.nn.parameter import Parameter

from trident.backend.common import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity, Relu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *

__all__ = ['efficient_block', 'EfficientNet', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
           'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon = _session.epsilon
_trident_dir = _session.trident_dir

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16, 'expand_ratio': 1, 'id_skip': True,
     'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24, 'expand_ratio': 6, 'id_skip': True,
     'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40, 'expand_ratio': 6, 'id_skip': True,
     'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80, 'expand_ratio': 6, 'id_skip': True,
     'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112, 'expand_ratio': 6, 'id_skip': True,
     'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192, 'expand_ratio': 6, 'id_skip': True,
     'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320, 'expand_ratio': 6, 'id_skip': True,
     'strides': 1, 'se_ratio': 0.25}]


def efficient_block(expand_ratio=1, filters_in=32, filters_out=16, kernel_size=3, strides=1, zero_pad=0, se_ratio=0,
                    drop_rate=0.2, is_shortcut=True, name='', **kwargs):
    expand_ratio = kwargs.get('expand_ratio', expand_ratio)
    is_shortcut = kwargs.get('id_skip', is_shortcut)
    filters_in = kwargs.get('filters_in', filters_in)
    filters_out = kwargs.get('filters_out', filters_out)
    kernel_size = kwargs.get('kernel_size', kernel_size)
    is_shortcut = filters_in == filters_out and strides == 1 and kwargs.get('id_skip', is_shortcut)
    filters = filters_in * expand_ratio
    if expand_ratio == 1 and strides == 1:

        bottleneck = Sequential(
            DepthwiseConv2d_Block((kernel_size, kernel_size), depth_multiplier=1, strides=strides, auto_pad=True,
                                  padding_mode='zero', normalization='batch', activation='swish', name=name + 'dwconv'),
            SqueezeExcite(se_filters=max(1, int(filters_in * se_ratio)), num_filters=filters_in,
                          use_bias=True) if 0 < se_ratio <= 1 else Identity(),
            Conv2d_Block((1, 1), num_filters=filters_out, strides=1, auto_pad=True, normalization='batch',
                         activation=None, name=name + 'se'),
            Dropout(dropout_rate=drop_rate) if is_shortcut and drop_rate > 0 else Identity())

        if is_shortcut:
            return ShortCut2d(Identity(), bottleneck)
        else:
            return bottleneck

    else:
        bottleneck = Sequential(
            Conv2d_Block((1, 1), num_filters=filters, strides=1, auto_pad=True, normalization='batch',
                         activation='swish', name=name + 'expand_bn'),
            DepthwiseConv2d_Block((kernel_size, kernel_size), depth_multiplier=1, strides=strides, auto_pad=True,
                                  padding_mode='zero', normalization='batch', activation='swish', name=name + 'dwconv'),
            SqueezeExcite(se_filters=builtins.max(1, int(filters_in * se_ratio)), num_filters=filters,
                          use_bias=True) if 0 < se_ratio <= 1 else Identity(),
            Conv2d_Block((1, 1), num_filters=filters_out, strides=1, auto_pad=True, normalization='batch',
                         activation=None, name=name + 'se'),
            Dropout(dropout_rate=drop_rate) if is_shortcut and drop_rate > 0 else Identity())
        if is_shortcut:
            return ShortCut2d(Identity(), bottleneck)
        else:
            return bottleneck


dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {  # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2), 'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5), 'efficientnet-b7': (2.0, 3.1, 600, 0.5), }
    return params_dict[model_name]


def EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate=0.2, drop_connect_rate=0.2,
                 depth_divisor=8, model_name='efficientnet', include_top=True, num_classes=1000, **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in your Keras config at `~/.keras/keras.json`.
    Args
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.

        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.

        num-classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns
        A Efficientnet model instance.


    """
    default_block_args = deepcopy(DEFAULT_BLOCKS_ARGS)

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = builtins.max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    flow_list = []
    efficientnet = Sequential(name=model_name)
    efficientnet.add_module('stem', Conv2d_Block((3, 3), round_filters(32), strides=2, use_bias=False, auto_pad=True,
                                                 padding_mode='zero', normalization='batch', activation='swish',
                                                 name='stem'))
    b = 0
    blocks = float(builtins.sum(args['repeats'] for args in default_block_args))
    for (i, args) in enumerate(default_block_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        # args['filters_in'] = round_filters(args['filters_in'])
        # args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            efficientnet.add_module('block{}{}'.format(i + 1, chr(j + 97)),
                                    efficient_block(expand_ratio=args['expand_ratio'],
                                                    filters_in=round_filters(args['filters_in']),
                                                    filters_out=round_filters(args['filters_out']),
                                                    kernel_size=args['kernel_size'], strides=args['strides'],
                                                    zero_pad=0, se_ratio=args['se_ratio'],
                                                    drop_connect_rate=drop_connect_rate * b / blocks,
                                                    name='block{}{}_'.format(i + 1, chr(j + 97)))),
            b += 1
    efficientnet.add_module('top_conv',
                            Conv2d_Block((1, 1), round_filters(1280), strides=1, use_bias=False, auto_pad=True,
                                         padding_mode='zero', normalization='batch', activation='swish',
                                         name='top_conv'))
    efficientnet.add_module('avg_pool', GlobalAvgPool2d(name='avg_pool'))
    if include_top:
        if dropout_rate > 0:
            efficientnet.add_module('top_dropout', Dropout(dropout_rate, name='top_dropout'))
        efficientnet.add_module('fc', Dense(num_classes, activation=None, name='fc'))
        efficientnet.add_module('softmax', SoftMax(name='softmax'))
    if isinstance(default_size, int):
        default_size = default_size,
    if len(default_size) == 1:
        default_size = (3, default_size[0], default_size[0])
    model = ImageClassificationModel(input_shape=default_size, output=efficientnet)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels1.txt'), 'r',
              encoding='utf-8-sig') as f:
        labels = [l.rstrip() for l in f]
        model.class_names = labels
    model.preprocess_flow = [resize((default_size[2], default_size[1]), keep_aspect=True), normalize(0, 255),
                             normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    # model.summary()
    return model


def EfficientNetB0(include_top=True, pretrained=True, input_shape=(3, 224, 224), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 224, 224)
    effb0 = EfficientNet(1.0, 1.0, input_shape, 0.2, model_name='efficientnet-b0', include_top=include_top,
                         num_classes=classes)
    if pretrained == True:
        download_model_from_google_drive('1bxnoDerzoNfiZZLft4ocD3DAgx4v6aTN', dirname, 'efficientnet-b0.pth')
        recovery_model = torch.load(os.path.join(dirname, 'efficientnet-b0.pth'))
        recovery_model.input_shape = input_shape
        if include_top == False:
            recovery_model.__delitem__(-1)
            recovery_model.__delitem__(-1)
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape = recovery_model.fc.input_shape
                recovery_model.fc = new_fc
        recovery_model.to(_device)
        effb0._model = recovery_model

    return effb0


def EfficientNetB1(include_top=True, pretrained=True, input_shape=(3, 240, 240), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 240, 240)
    effb1 = EfficientNet(1.0, 1.1, 240, 0.2, model_name='efficientnet-b1', include_top=include_top, num_classes=classes)
    if pretrained == True:
        download_model_from_google_drive('1F3BtnAjmDz4G9RS9Q0hqU_K7WWXCni1G', dirname, 'efficientnet-b1.pth')
        recovery_model = torch.load(os.path.join(dirname, 'efficientnet-b1.pth'))
        recovery_model.input_shape = input_shape
        recovery_model.eval()
        recovery_model.to(_device)
        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape = recovery_model.fc.input_shape
                recovery_model.fc = new_fc
        effb1.model = recovery_model
    return effb1


def EfficientNetB2(include_top=True, pretrained=True, input_shape=(3, 260, 260), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 260, 260)
    effb2 = EfficientNet(1.1, 1.2, 260, 0.3, model_name='efficientnet-b2', include_top=include_top, num_classes=classes)
    if pretrained == True:
        download_model_from_google_drive('1PjqhB7WJasF_hqOwYtSBNSXSGBY-cRLU', dirname, 'efficientnet-b2.pth')
        recovery_model = torch.load(os.path.join(dirname, 'efficientnet-b2.pth'))
        recovery_model.input_shape = input_shape
        recovery_model.to(_device)
        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape = recovery_model.fc.input_shape
                recovery_model.fc = new_fc
        effb2.model = recovery_model
    return effb2


def EfficientNetB3(include_top=True, pretrained=True, input_shape=(3, 300, 300), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 300, 300)
    effb3 = EfficientNet(1.2, 1.4, 300, 0.3, model_name='efficientnet-b3', include_top=include_top, num_classes=classes)
    if pretrained == True:
        download_model_from_google_drive('11tMxdYdFfaEREwnESO4cwjtcoEB42zB_', dirname, 'efficientnet-b3.pth')
        recovery_model = torch.load(os.path.join(dirname, 'efficientnet-b3.pth'))
        recovery_model.input_shape = input_shape
        recovery_model.to(_device)
        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape = recovery_model.fc.input_shape
                recovery_model.fc = new_fc
        effb3.model = recovery_model
    return effb3


def EfficientNetB4(include_top=True, pretrained=True, input_shape=(3, 380, 380), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 380, 380)
    effb4 = EfficientNet(1.4, 1.8, 380, 0.4, model_name='efficientnet-b4', include_top=include_top, num_classes=classes)
    if pretrained == True:
        download_model_from_google_drive('1X4ZOBR_ETRHZJeffJHvCmWTTy9_aW8SP', dirname, 'efficientnet-b4.pth')
        recovery_model = torch.load(sanitize_path(os.path.join(dirname, 'efficientnet-b4.pth')))
        recovery_model.input_shape = input_shape
        recovery_model.to(_device)
        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape = recovery_model.fc.input_shape
                recovery_model.fc = new_fc
        effb4.model = recovery_model
    return effb4


def EfficientNetB5(include_top=True, pretrained=True, input_shape=(3, 456, 456), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 456, 456)
    effb5 = EfficientNet(1.6, 2.2, 456, 0.4, model_name='efficientnet-b5', include_top=include_top, num_classes=classes)
    if pretrained == True:
        download_model_from_google_drive('17iTD12G9oW3jYAui84MKtdY4gjd9vpgG', dirname, 'efficientnet-b5.pth')
        recovery_model = torch.load(os.path.join(dirname, 'efficientnet-b5.pth'))
        recovery_model.input_shape = input_shape
        recovery_model.to(_device)
        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape = recovery_model.fc.input_shape
                recovery_model.fc = new_fc
        effb5.model = recovery_model
    return effb5


def EfficientNetB6(include_top=True, pretrained=True, input_shape=(3, 528, 528), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 528, 528)
    effb6 = EfficientNet(1.8, 2.6, 528, 0.5, model_name='efficientnet-b6', include_top=include_top, num_classes=classes)
    if pretrained == True:
        download_model_from_google_drive('1XJrKmcmMObN_nnjP2Z-YH_BQ3img58qF', dirname, 'efficientnet-b6.pth')
        recovery_model = torch.load(os.path.join(dirname, 'efficientnet-b6.pth'))
        recovery_model.input_shape = input_shape
        recovery_model.to(_device)
        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape = recovery_model.fc.input_shape
                recovery_model.fc = new_fc
        effb6.model = recovery_model
    return effb6


def EfficientNetB7(include_top=True, pretrained=True, input_shape=(3, 600, 600), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 600, 600)
    effb7 = EfficientNet(2.0, 3.1, 600, 0.5, model_name='efficientnet-b7', include_top=include_top, num_classes=classes)
    if pretrained == True:
        download_model_from_google_drive('1M2DfvsNPRCWSo_CeXnUCQOR46rvOrhLl', dirname, 'efficientnet-b7.pth')
        recovery_model = torch.load(os.path.join(dirname, 'efficientnet-b7.pth'))
        recovery_model.input_shape = input_shape
        # recovery_model.to(_device)
        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape = recovery_model.fc.input_shape
                recovery_model.fc = new_fc
        effb7.model = recovery_model
    return effb7
