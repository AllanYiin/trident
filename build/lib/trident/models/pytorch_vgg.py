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
from ..layers.pytorch_activations import  get_activation,Identity,Relu
from ..layers.pytorch_normalizations import get_normalization
from ..layers.pytorch_blocks import *
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *
from ..data.image_common import *
from ..data.utils import download_file_from_google_drive

__all__ = ['vgg19','VGG19']

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




cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_vgg_layers(cfg, num_classes=1000):
    layers = []
    in_channels = 3
    block=1
    conv=1
    vgg=Sequential()
    for v in cfg:
        if v == 'M':
            vgg.add_module('block{0}_pool'.format(block,conv),MaxPool2d(kernel_size=2, strides=2))
            block += 1
            conv = 1
        else:
            vgg.add_module('block{0}_conv{1}'.format(block,conv),Conv2d((3,3),v,auto_pad=True,activation=None))
            vgg.add_module('block{0}_relu{1}'.format(block, conv),Relu())
            conv+=1
            in_channels = v
    vgg.add_module('avgpool',GlobalAvgPool2d())
    vgg.add_module('classifier',Sequential(Flatten(),
               Dense(4096, activation='relu'),
               Dropout(), Dense(4096, activation='relu'),
               Dropout(),
        Dense(num_classes)))
    return vgg

vgg19 =make_vgg_layers('E', False)
def VGG19(include_top=True,
             pretrained=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    vgg19 =make_vgg_layers('E', False)
    # if pretrained==True:
    #     download_file_from_google_drive(model_urls['resnet50'],dirname,'resnet50.pth')
    #     recovery_model=torch.load(os.path.join(dirname,'resnet50.pth'))
    #     recovery_model.to(_device)
    #     if include_top==False:
    #         recovery_model.__delitem__(-1)
    #     else:
    #         if classes!=1000:
    #             recovery_model.fc=Dense(classes,activation='softmax',name='fc')
    #
    #     resnet50.model=recovery_model
    return vgg19