
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

from ..backend.common import *
from ..backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential, Input, summary
from ..data.image_common import *
from ..data.utils import download_model_from_google_drive
from ..layers.pytorch_activations import get_activation, Identity, Relu
from ..layers.pytorch_blocks import *
from ..layers.pytorch_layers import *
from ..layers.pytorch_normalizations import get_normalization, BatchNorm2d
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *

__all__ = ['DeeplabV3_plus','DeeplabV3']

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




def DeepLabHead(classes=20, atrous_rates=(6, 12, 18,24),num_filters=256):
    return Sequential(
        ASPP(atrous_rates,num_filters=num_filters),
        Conv2d_Block((3,3),num_filters,auto_pad=True,use_bias=False,activation='relu',normalization='batch'),
        Conv2d((1,1),num_filters=classes,strides=1,auto_pad=True,activation='sigmoid',name='classifier')
        )



def ASPPPooling(num_filters,size):
    return Sequential(AdaptiveAvgPool2d((1,1)),
                      Conv2d((1,1),num_filters,strides=1,use_bias=False,activation=None),
                      Upsampling2d(size=(size[-2], size[-1]),mode='bilinear', align_corners=False))



class ASPP(Layer):
    def __init__(self, atrous_rates,num_filters=256):
        super(ASPP, self).__init__()
        self.num_filters=num_filters
        self.convs = ShortCut2d( mode='concate')
        self.convs.add_module('conv1',Conv2d_Block((1,1),num_filters=num_filters,strides=1,use_bias=False,activation=None,normalization='batch'))

        for i in range(len(atrous_rates)):
            self.convs.add_module('aspp_dilation{0}'.format(i),Conv2d_Block((3,3),num_filters=num_filters,strides=1,use_bias=False,activation=None,normalization='batch',dilation=atrous_rates[i]))

        self.project =Conv2d_Block( (1,1),num_filters,strides=1,use_bias=False, bias=False,activation='relu',normalization='batch',dilation=1,dropout_rate=0.5)

    def build(self, input_shape):
        if self._built == False :
            self.convs.add_module('aspp_pooling',ASPPPooling(self.num_filters, to_list(input_shape[-2:])))
            self.to(self.device)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x=self.convs(x)
        x=self.project(x)
        return x



def DeeplabV3(backbond,
             input_shape=(3,224,224),
             classes=20,
             **kwargs):
    input_shape=tuple(input_shape)
    deeplab=Sequential(name='deeplabv3')

    deeplab.add_module('backbond',backbond)
    deeplab.add_module('classifier', DeepLabHead(classes=classes,num_filters=128))
    deeplab.add_module('upsample', Upsampling2d(scale_factor=16, mode='bilinear', align_corners=False))
    model = ImageSegmentationModel(input_shape=input_shape, output=deeplab)
    return model



class _DeeplabV3_plus(Layer):
    def __init__(self, backbond, input_shape=(3,224,224), atrous_rates=(6, 12, 18, 24), num_filters=256, classes=20):
        super(_DeeplabV3_plus, self).__init__()
        moduals=list(backbond.children())
        low_level_idx=-1
        high_level_idx=-1
        for i in range(len(moduals)):
            if low_level_idx<0 and moduals[i].output_shape[-1]==backbond.input_shape[-1]//8:
                low_level_idx=i

            if high_level_idx<0 and moduals[i].output_shape[-1]==backbond.input_shape[-1]//32:
                high_level_idx=i
                break
        self.num_filters=num_filters
        self.classes=classes
        self.atrous_rates=atrous_rates
        self.backbond1=Sequential(backbond[:low_level_idx])
        self.backbond2 = Sequential(backbond[low_level_idx:high_level_idx])
        self.aspp=ASPP(atrous_rates=self.atrous_rates,num_filters=self.num_filters)
        self.low_level_conv=Conv2d_Block((1,1),num_filters=int(48*self.num_filters/256),strides=1,use_bias=False,activation='leaky_relu',normalization='batch')
        self.decoder=Sequential(
            DepthwiseConv2d_Block((3,3),depth_multiplier=0.5,strides=1,use_bias=False,activation='leaky_relu',normalization='batch',dropout_rate=0.5),
            DepthwiseConv2d_Block((3,3),depth_multiplier=1,strides=1,use_bias=False,activation='leaky_relu',normalization='batch',dropout_rate=0.1),
            Conv2d((1, 1), num_filters=self.classes, strides=1, use_bias=False, activation='sigmoid'),

        )

    def forward(self, *x):
        x = enforce_singleton(x)
        low_level_feature=self.backbond1(x)
        high_level_feature = self.backbond2(low_level_feature)
        x=self.aspp(high_level_feature)
        x=F.interpolate(x, None, (4.0,4.0),mode='bilinear', align_corners=True)
        low_level_feature=self.low_level_conv(low_level_feature)
        x=torch.cat([x,low_level_feature],dim=1)
        x=self.decoder(x)
        x = F.interpolate(x, None, (4.0, 4.0), mode='bilinear', align_corners=True)
        return x



def DeeplabV3_plus(backbond,
             input_shape=(3,224,224),
             atrous_rates = (6, 12, 18, 24),
             num_filters = 256,
             classes=20,
             **kwargs):
    deeplab=_DeeplabV3_plus(backbond=backbond,input_shape=input_shape,atrous_rates=atrous_rates,num_filters=num_filters,classes=classes)
    deeplab.name='DeeplabV3_plus'
    model = ImageSegmentationModel(input_shape=input_shape, output=deeplab)
    return model
