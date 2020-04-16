
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
from ..backend.pytorch_backend import to_numpy,to_tensor,Layer,Sequential,Input,summary
from ..layers.pytorch_layers import *
from ..layers.pytorch_activations import  get_activation,Identity,Relu
from ..layers.pytorch_normalizations import get_normalization,BatchNorm2d
from ..layers.pytorch_blocks import *
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *
from ..data.image_common import *
from ..data.utils import download_model_from_google_drive


__all__ = []

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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




def DeepLabHead(classes=20):
    return Sequential(
        ASPP([12, 24, 36]),
        Conv2d_Block((3,3),256,auto_pad=True,use_bias=False,activation='relu',normalization='batch'),
        Conv2d((1,1),num_filters=classes,strides=1,auto_pad=True,name='classifier')
        )



def ASPPPooling(num_filters,size):
    return Sequential(AdaptiveAvgPool2d(1),
                      Conv2d_Block((1,1),num_filters,strides=1,use_bias=False,activation='relu',normalization='batch'),
                      Upsampling2d(size=size,mode='bilinear', align_corners=False))



class ASPP(Layer):
    def __init__(self, atrous_rates):
        super(ASPP, self).__init__()
        num_filters = 256
        self.convs = ShortCut2d( mode=ShortcutMode.concate)
        self.convs.add_module(Conv2d_Block((1,1),num_filters=256,strides=1,use_bias=False,activation='relu',normalization='batch'))


        rate1, rate2, rate3 = tuple(atrous_rates)

        self.convs.add_module('aspp_dilation1',Conv2d_Block((1,1),num_filters=256,strides=1,use_bias=False,activation='relu',normalization='batch',dilation=rate1))
        self.convs.add_module('aspp_dilation2',Conv2d_Block((1,1),num_filters=256,strides=1,use_bias=False,activation='relu',normalization='batch',dilation=rate2))
        self.convs.add_module('aspp_dilation3',Conv2d_Block((1,1),num_filters=256,strides=1,use_bias=False,activation='relu',normalization='batch',dilation=rate3))

        self.project =Conv2d_Block( (1,1),256,strides=1,use_bias=False, bias=False,activation='relu',normalization='batch',dilation=1,dropout_rate=0.5)

    def build(self, input_shape):
        if self._built == False :
            self.convs.add_module('aspp_pooling',ASPPPooling(256, input_shape[2:]))
            self.to(self.device)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x=self.convs(x)
        x=self.project(x)
        return x


def DeeplabV3(backbond='MobileNetV2',
             pretrained=True,
             input_shape=(3,224,224),
             classes=10,
             **kwargs):
    input_shape=tuple(input_shape)
    deeplab=Sequential(name='deeplabv3')

    backbond_net=get_function(backbond, ['trident.models'])(include_top=False,pretrained=pretrained,input_shape=input_shape)
    deeplab.add_module('backbond',backbond_net.model)
    deeplab.add_module('classifier', DeepLabHead(classes=classes))
    model = ImageSegmentationModel(input_shape=input_shape, output=deeplab)
    model.preprocess_flow =backbond_net.preprocess_flow
    return model


