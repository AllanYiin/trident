from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from collections import *
import uuid
from copy import copy, deepcopy
from collections import deque
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F  # import torch functions
from torch._six import container_abcs
from itertools import repeat
from backend.pytorch_backend import Module, Sequential, get_input_shape, get_out_shape
from .pytorch_layers import *
from .pytorch_normalizations import *
from .pytorch_activations import *



class Conv2d_Block(nn.Module):
    def __init__(self, kernel_size=1, in_filters=None, num_filters=None, strides=1, padding=0, activation='relu6',
                 normalization='instance', init=None, use_bias=False, init_bias=0, dilation=1, groups=1,
                 weights_contraint=None):
        super(Conv2d_Block, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.normalization = normalization
        self.init = init
        self.use_bias = use_bias
        self.init_bias = init_bias
        self.dilation = dilation
        self.groups = groups
        self.weights_contraint = weights_contraint

        self.conv = Conv2d(kernel_size, input_filters=in_filters, num_filters=num_filters, strides=strides,
                           padding=padding, activation=None, init=None, use_bias=use_bias, init_bias=0,
                           dilation=dilation, groups=groups)
        self.norm = nn.InstanceNorm2d(num_features=num_filters)

        if  self.activation !=None:
            self.activation = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        if self.norm!=None:
            x = self.norm(x)
        if self.activation != None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = ('{input_filters}, {num_filters}, kernel_size={kernel_size}'
             ', stride={stride}')

        return s.format(**self.__dict__)

    def calculate_output_shape(self, input_shape):
        test_tensor = torch.Tensor(np.ones(input_shape, dtype=np.float32))
        test_tensor = self.forward(test_tensor)
        self.calculated_output_shape = test_tensor.shape
        return test_tensor.shape


class Classifier1d(nn.Module):
    def __init__(self, num_classes=10, is_multiselect=False, classifier_type='dense'):
        super(Classifier1d, self).__init__()
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.is_multiselect = is_multiselect
        self.dense=None
        self.global_avgpool=None
        if self.classifier_type=='dense':
            self.dense=nn.Linear(num_classes)

    def forward(self, x):
        x
        return x

    def extra_repr(self):
        s = ('{input_filters}, {num_filters}, kernel_size={kernel_size}'
             ', stride={stride}')

        return s.format(**self.__dict__)

    def calculate_output_shape(self, input_shape):
        test_tensor = torch.Tensor(np.ones(input_shape, dtype=np.float32))
        test_tensor = self.forward(test_tensor)
        self.calculated_output_shape = test_tensor.shape
        return test_tensor.shape

