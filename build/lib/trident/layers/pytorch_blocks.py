from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
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


from . import  pytorch_layers as layer
from .pytorch_activations import  get_activation
from .pytorch_normalizations import get_normalization

__all__ = ['Conv2d_Block','GcdConv2d_Block','GcdConv2d_Block_1','Classifier1d','ShortCut2d']
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class Conv2d_Block(nn.Module):
    def __init__(self, kernel_size=(3,3), num_filters=32, strides=1, auto_pad=True,activation='relu6',
                 normalization='instance', init=None, use_bias=False, init_bias=0, dilation=1, groups=1,add_noise=False,dropout_rate=0,
                 weights_contraint=None):
        super(Conv2d_Block, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.auto_pad = auto_pad
        self.padding=0
        if self.auto_pad == False:
            self.padding = 0
        else:
            self.padding= tuple([n-2 for n in  list(self.kernel_size)]) if hasattr(self.kernel_size,'__len__') else self.kernel_size-2



        self.init = init
        self.use_bias = use_bias
        self.init_bias = init_bias
        self.dilation = dilation
        self.groups = groups
        self.weights_contraint = weights_contraint
        self.add_noise = False,
        self.dropout_rate=dropout_rate
        self.conv =None
        self.norm = get_normalization(normalization)
        self.activation=get_activation(activation)
        self.droupout = None
    def forward(self, x):
        self.input_shape = x.size()
        self.input_filter= x.size(1)
        #dynamic generation
        if self.conv is None:
            self.conv = nn.Conv2d(kernel_size=self.kernel_size,
                               in_channels=x.size(1),out_channels=self.num_filters,
                           dilation=self.dilation, groups=self.groups)
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        x = self.conv(x)
        if self.norm!=None:
            x = self.norm()(x)
        if self.activation != None:
            x = self.activation(x)
        if self.add_noise==True:
            noise = 0.005 * torch.randn_like(x, dtype=torch.float32)
            x=x+noise
        if self.dropout_rate > 0:
            x = nn.Dropout2d(self.dropout_rate)(x)
        if torch.isnan(x).any() :
            print(self._get_name()+'  nan detected!!')
        return x

        return x

    def extra_repr(self):
        s = ('kernel_size={kernel_size}, {num_filters}, strides={strides},activation={activation}')

        return s.format(**self.__dict__)



class GcdConv2d_Block(nn.Module):
    def __init__(self, kernel_size=(3,3), num_filters=32, strides=1, auto_pad=True,divisor_rank=0,activation='relu6',normalization=None,init=None, use_bias=False, init_bias=0, dilation=1, groups=1,add_noise=False,dropout_rate=0,
                 weights_contraint=None):
        super(GcdConv2d_Block, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = _pair(strides)
        self.auto_pad = auto_pad

        self.init = init
        self.use_bias = use_bias
        self.init_bias = init_bias
        self.dilation = dilation
        self.groups = groups
        self.weights_contraint = weights_contraint
        self.add_noise = False,
        self.dropout_rate=dropout_rate
        self.conv =None
        self.droupout=None
        self.divisor_rank=divisor_rank

        self.activation = get_activation(activation)
        self.norm = get_normalization(normalization)
    def forward(self, x):
        self.input_shape = x.size()
        self.input_filter = x.size(1)

        #dynamic generation
        if self.conv is None:
            self.conv = layer.GcdConv2d(self.kernel_size, input_filters=x.size(1), num_filters=self.num_filters, strides=self.strides,
                           auto_pad=self.auto_pad, activation=None, init=None, use_bias=self.use_bias, init_bias=0,divisor_rank=self.divisor_rank,
                           dilation=self.dilation).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        x = self.conv(x)
        if self.activation is not None:
            x = self.activation()(x)
        if self.add_noise==True:
            noise = 0.005 * torch.randn_like(x, dtype=torch.float32)
            x=x+noise
        if self.dropout_rate > 0:
            x = nn.Dropout2d(self.dropout_rate)(x)
        if torch.isnan(x).any():
            print(self._get_name() + '  nan detected!!')
        return x

    def extra_repr(self):
        s = ('{input_filters}, {num_filters}, kernel_size={kernel_size}'
             ', stride={stride}')

        return s.format(**self.__dict__)

class GcdConv2d_Block_1(nn.Module):
    def __init__(self, kernel_size=(3,3), num_filters=32, strides=1, auto_pad=True,divisor_rank=0,activation='relu6', normalization=None, self_norm=True,is_shuffle=False,init=None, use_bias=False, init_bias=0, dilation=1, groups=1,add_noise=False,dropout_rate=0,
                 weights_contraint=None):
        super(GcdConv2d_Block_1, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = _pair(strides)
        self.auto_pad = auto_pad

        self.init = init
        self.use_bias = use_bias
        self.init_bias = init_bias
        self.dilation = dilation
        self.groups = groups
        self.weights_contraint = weights_contraint
        self.add_noise = False,
        self.dropout_rate=dropout_rate
        self.activation = get_activation(activation)
        self.self_norm=self_norm
        self.is_shuffle=is_shuffle
        self.norm = get_normalization(normalization)
        self.normalization =normalization
        self.conv =None
        self.droupout=None
        self.divisor_rank=divisor_rank

    def forward(self, x):
        # if not hasattr(x,'shape') and not hasattr(x,'size') and callable(x):
        self.input_shape = x.size()
        self.input_filter = x.size(1)

        #dynamic generation
        if self.conv is None:
            self.input_shape = x.size()
            self.conv = layer.GcdConv2d_1(self.kernel_size, input_filters=x.size(1), num_filters=self.num_filters, strides=self.strides,
                           auto_pad=self.auto_pad, activation=None, init=None, use_bias=self.use_bias, init_bias=0,divisor_rank=self.divisor_rank,
                            self_norm=self.self_norm,is_shuffle=self.is_shuffle,dilation=self.dilation).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


        x = self.conv(x)
        if torch.isnan(x).any():
            print(self._get_name() + 'self.conv(x)     nan detected!!')
        if self.normalization is not None:
            x =self.norm(x)
        if torch.isnan(x).any():
            print(self._get_name() + 'self.normalization     nan detected!!')
        if self.activation is not None:
            x = self.activation(x)
        if torch.isnan(x).any():
            print(self._get_name() + 'self.activation    nan detected!!')
        if self.add_noise==True:
            noise = 0.005 * torch.randn_like(x, dtype=torch.float32)
            x=x+noise
        if self.dropout_rate > 0:
            x = nn.Dropout2d(self.dropout_rate)(x)
        if torch.isnan(x).any():
            print(self._get_name() + '  nan detected!!')
        return x

    def extra_repr(self):
        s = ('kernel_size={kernel_size}, {num_filters}, strides={strides},activation={activation} ')

        return s.format(**self.__dict__)


class Classifier1d(nn.Module):
    def __init__(self, num_classes=10, is_multiselect=False, classifier_type='dense'):
        super(Classifier1d, self).__init__()
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.is_multiselect = is_multiselect
        self.dense=None
        self.global_avgpool=None
        self.conv1x1=None
    def forward(self, x):
        self.input_shape=x.size()
        if self.classifier_type == 'dense' :
            x = x.view(x.size(0),x.size(1), -1)
            x =torch.mean(x, -1, False)
            if self.dense is None:
                self.dense = nn.Linear(x.size(1), self.num_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            x=self.dense(x)

        elif self.classifier_type == 'global_avgpool':
            if len(self.input_shape)!=4:
                raise  ValueError("GlobalAvgPool2d only accept BCHW shape")
            if x.size(1)!=self.num_classes:
                if self.conv1x1 is None:
                    self.conv1x1=layer.Conv2d((1,1), input_filters=x.size(1), num_filters=self.num_classes, strides=1,
                           padding=0, activation=None, use_bias=False).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                x=self.conv1x1(x)
            if self.global_avgpool is None:
                self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            x=self.global_avgpool(x)
            x = x.view(x.size(0), x.size(1))
        x = torch.sigmoid(x)
        return torch.softmax(x,dim=1)

    def extra_repr(self):
        s = ('{num_classes}, classifier_type={classifier_type}')
        return s.format(**self.__dict__)


class ShortCut2d(nn.Module):
    def __init__(self, layer_defs:OrderedDict):
        """

        Parameters
        ----------
        layer_defs : object
        """
        super(ShortCut2d, self).__init__()
        self.layer_defs = layer_defs
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        idx = 0
        branches = OrderedDict()
        for k, v in self.layer_defs.items():
            if hasattr(v, '__iter__'):
                out = x
                for f in v:
                    if callable(f):
                        f.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                        out = f.forward(out)
                branches[idx] = out
                idx += 1
        branch_aggregate=None
        for k, v in branches.items():
            if branch_aggregate is None:
                branch_aggregate=v
            else :
                branch_aggregate=torch.add(branch_aggregate, v)
        if x.size()==branch_aggregate.size():
            x = torch.add(x, branch_aggregate)
        return branch_aggregate




    def extra_repr(self):
        s = ('{num_classes}, classifier_type={classifier_type}')
        return s.format(**self.__dict__)


