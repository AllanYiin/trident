from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numbers
import inspect
import math
import uuid
from collections import *
from collections import deque
from copy import copy, deepcopy
from functools import partial
from itertools import repeat, chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc
from torch.nn import init

from trident.layers.pytorch_activations import get_activation, Identity
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization, SpectralNorm
from trident.layers.pytorch_pooling import *
from trident.backend.common import *
from trident.backend.pytorch_backend import Layer, Sequential, ModuleList
from trident.backend.pytorch_ops import *

__all__ = ['FullConnect_Block','Conv2d_Block', 'Conv1d_Block', 'DepthwiseConv2d_Block', 'SeparableConv2d_Block', 'TemporalConv1d_Block',
           'TransConv2d_Block', 'Classifier1d', 'ShortCut2d', 'ShortCut', 'Hourglass', 'ConcateBlock', 'SqueezeExcite', 'For']

_session = get_session()

_epsilon = _session.epsilon


def _ntuple(n):
    def parse(x):
        if isinstance(x, abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class FullConnect_Block(Layer):
    def __init__(self, num_filters=None,
                 activation=None, normalization=None, use_spectral=False, use_bias=False,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None,
                 keep_output=False, sequence_rank='cna', **kwargs):
        super(FullConnect_Block, self).__init__(name=name, keep_output=keep_output)

        if sequence_rank in ['fna', 'naf', 'afn']:
            self.sequence_rank = sequence_rank
        else:
            self.sequence_rank = 'fna'

        self.num_filters = num_filters



        self.use_bias = use_bias

        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate
        self.droupout = None
        self.depth_multiplier = depth_multiplier
        self.keep_output = keep_output

        norm = get_normalization(normalization)
        fc = Dense(num_filters=self.num_filters, activation=None, use_bias=self.use_bias, depth_multiplier=self.depth_multiplier).to(self.device)
        self.use_spectral = use_spectral
        if isinstance(norm, SpectralNorm):
            self.use_spectral = True
            norm = None
            fc = nn.utils.spectral_norm(fc)
        if (hasattr(self, 'sequence_rank') and self.sequence_rank == 'fna') or not hasattr(self, 'sequence_rank'):
            self.add_module('fc', fc)
            self.add_module('norm', norm)
            self.add_module('activation',  get_activation(activation,only_layer=True))

        elif self.sequence_rank == 'naf':
            self.add_module('norm', norm)
            self.add_module('activation',  get_activation(activation,only_layer=True))
            self.add_module('fc', fc)

        elif self.sequence_rank == 'afn':
            self.add_module('activation',  get_activation(activation,only_layer=True))
            self.add_module('fc', fc)
            self.add_module('norm', norm)
        self._name = name

    def build(self, input_shape: TensorShape):
        if not self._built:
            # if self.norm is not None:
            #     self.norm.input_shape = self.conv.output_shape
            self.to(self.device)
            self._built = True

    def forward(self, x, **kwargs):

        if not hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'fna')
        if self.add_noise == True and self.training == True:
            noise = self.noise_intensity * torch.randn_like(x, dtype=x.dtype)
            x = x + noise
        for child in list(self.children())[:3]:
            if child is not None:
                x = child(x)
        if self.training and self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)

class Conv1d_Block(Layer):
    def __init__(self, kernel_size=3, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1, add_noise=False, noise_intensity=0.005,
                 dropout_rate=0, name=None, depth_multiplier=None, keep_output=False, sequence_rank='cna', **kwargs):
        super(Conv1d_Block, self).__init__(name=name, keep_output=keep_output)
        if sequence_rank in ['cna', 'nac']:
            self.sequence_rank = sequence_rank
        else:
            self.sequence_rank = 'cna'
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.auto_pad = auto_pad
        self.padding = 0
        self.padding_mode = padding_mode

        # if self.auto_pad == False:
        #     self.padding = 0
        # else:
        #     self.padding= tuple([n-2 for n in  list(self.kernel_size)]) if hasattr(self.kernel_size,'__len__') else
        #     self.kernel_size-2

        self.use_bias = use_bias
        self.dilation = dilation
        self.groups = groups
        self.depth_multiplier = depth_multiplier
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate

        norm = get_normalization(normalization)
        conv = Conv1d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                      auto_pad=self.auto_pad, padding_mode=self.padding_mode, activation=None,
                      use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name,
                      depth_multiplier=self.depth_multiplier).to(self.device)
        self.use_spectral = use_spectral
        if isinstance(norm, SpectralNorm):
            self.use_spectral = True
            norm = None
            conv = nn.utils.spectral_norm(conv)
        if (hasattr(self, 'sequence_rank') and self.sequence_rank == 'cna') or not hasattr(self, 'sequence_rank'):
            self.add_module('conv', conv)
            self.add_module('norm', norm)
            self.add_module('activation', get_activation(activation, only_layer=True))

        elif self.sequence_rank == 'nac':
            self.add_module('norm', norm)
            self.add_module('activation', get_activation(activation, only_layer=True))
            self.add_module('conv', conv)

        elif self.sequence_rank == 'acn':
            self.add_module('activation', get_activation(activation, only_layer=True))
            self.add_module('conv', conv)
            self.add_module('norm', norm)

    def build(self, input_shape: TensorShape):
        if self._built == False:
            if self.use_spectral:
                self.conv = nn.utils.spectral_norm(self.conv)
                if self.norm is SpectralNorm:
                    self.norm = None
            self._built = True

    def forward(self, x, **kwargs):

        if hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'cna')
        if self.add_noise == True and self.training == True:
            noise = self.noise_intensity * torch.randn_like(x, dtype=x.dtype)
            x = x + noise
        if self.sequence_rank == 'cna':
            x = self.conv(x)
            if self.norm is not None:
                x = self.norm(x)
            if self.activation is not None:
                x = self.activation(x)
        elif self.sequence_rank == 'nac':
            if self.norm is not None:
                x = self.norm(x)
            if self.activation is not None:
                x = self.activation(x)
            x = self.conv(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


class Conv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None,
                 keep_output=False, sequence_rank='cna', **kwargs):
        super(Conv2d_Block, self).__init__(name=name, keep_output=keep_output)

        if sequence_rank in ['cna', 'nac', 'acn']:
            self.sequence_rank = sequence_rank
        else:
            self.sequence_rank = 'cna'
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides

        self.padding_mode = padding_mode
        padding = kwargs.get('padding', None)
        if 'padding' in kwargs:
            kwargs.pop('padding')
        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int) and padding > 0:
            padding = _pair(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            auto_pad = False
            pass
        self.auto_pad = auto_pad
        self.padding = padding
        # if self.auto_pad == False:
        #     self.padding = 0
        # else:
        #     self.padding= tuple([n-2 for n in  list(self.kernel_size)]) if hasattr(self.kernel_size,'__len__') else
        #     self.kernel_size-2

        self.use_bias = use_bias
        self.dilation = dilation
        self.groups = groups

        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate
        self.droupout = None
        self.depth_multiplier = depth_multiplier
        self.keep_output = keep_output

        norm = get_normalization(normalization)
        conv = Conv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                      auto_pad=self.auto_pad, padding_mode=self.padding_mode, activation=None,
                      use_bias=self.use_bias, dilation=self.dilation, groups=self.groups,
                      depth_multiplier=self.depth_multiplier, padding=self.padding, **kwargs).to(self.device)
        self.use_spectral = use_spectral
        if isinstance(norm, SpectralNorm):
            self.use_spectral = True
            norm = None
            conv = nn.utils.spectral_norm(conv)
        if (hasattr(self, 'sequence_rank') and self.sequence_rank == 'cna') or not hasattr(self, 'sequence_rank'):
            self.add_module('conv', conv)
            self.add_module('norm', norm)
            self.add_module('activation',  get_activation(activation,only_layer=True))

        elif self.sequence_rank == 'nac':
            self.add_module('norm', norm)
            self.add_module('activation',  get_activation(activation,only_layer=True))
            self.add_module('conv', conv)

        elif self.sequence_rank == 'acn':
            self.add_module('activation',  get_activation(activation,only_layer=True))
            self.add_module('conv', conv)
            self.add_module('norm', norm)
        self._name = name

    def build(self, input_shape: TensorShape):
        if not self._built:

            self.conv.input_shape = input_shape
            if self.use_spectral:
                self.conv = nn.utils.spectral_norm(self.conv)
                if self.norm is SpectralNorm:
                    self.norm = None

            # if self.norm is not None:
            #     self.norm.input_shape = self.conv.output_shape
            self.to(self.device)
            self._built = True

    def forward(self, x, **kwargs):

        if not hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'cna')
        if self.add_noise == True and self.training == True:
            noise = self.noise_intensity * torch.randn_like(x, dtype=x.dtype)
            x = x + noise
        for child in list(self.children())[:3]:
            if child is not None:
                x = child(x)
        if self.training and self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


class TransConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None,
                 keep_output=False, sequence_rank='cna', **kwargs):
        super(TransConv2d_Block, self).__init__(name=name, keep_output=keep_output)
        if not hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'cna')
        if sequence_rank in ['cna', 'nac']:
            self.sequence_rank = sequence_rank
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.auto_pad = auto_pad
        self.padding = 0
        self.padding_mode = padding_mode
        self.use_bias = use_bias
        self.dilation = dilation
        self.groups = groups
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate
        self.use_spectral = use_spectral
        self.depth_multiplier = depth_multiplier
        conv = TransConv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                                auto_pad=self.auto_pad, padding_mode=self.padding_mode, activation=None,
                                use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self.name,
                                depth_multiplier=self.depth_multiplier).to(self.device)
        norm = get_normalization(normalization)

        if isinstance(norm, SpectralNorm):
            self.use_spectral = True
            norm = None
            conv = nn.utils.spectral_norm(conv)
        if (hasattr(self, 'sequence_rank') and self.sequence_rank == 'cna') or not hasattr(self, 'sequence_rank'):
            self.add_module('conv', conv)
            self.add_module('norm', norm)
            self.add_module('activation', get_activation(activation, only_layer=True))

        elif self.sequence_rank == 'nac':
            self.add_module('norm', norm)
            self.add_module('activation', get_activation(activation, only_layer=True))
            self.add_module('conv', conv)

        elif self.sequence_rank == 'acn':
            self.add_module('activation', get_activation(activation, only_layer=True))
            self.add_module('conv', conv)
            self.add_module('norm', norm)
        self.droupout = None

        self.keep_output = keep_output
        self._name = name

    def build(self, input_shape: TensorShape):
        if self._built == False or self.conv is None:
            self.num_filters = self.input_filters * self.depth_multiplier if self.num_filters is None else self.num_filters
            self.conv.input_shape = input_shape
            if self.use_spectral:
                self.conv = nn.utils.spectral_norm(self.conv)
                if self.norm is SpectralNorm:
                    self.norm = None

            self.to(self.device)
            self._built = True

    def forward(self, x, **kwargs):

        if not hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'cna')
        if self.add_noise == True and self.training == True:
            noise = self.noise_intensity * torch.randn_like(x, dtype=x.dtype)
            x = x + noise
        for child in list(self.children())[:3]:
            if child is not None:
                x = child(x)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, num_filters={num_filters}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()

        return s.format(**self.__dict__)


class DepthwiseConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), depth_multiplier=1, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, add_noise=False,
                 noise_intensity=0.005, dropout_rate=0, name=None, keep_output=False, sequence_rank='cna', **kwargs):
        super(DepthwiseConv2d_Block, self).__init__(name=name, keep_output=keep_output)
        if not hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'cna')
        if sequence_rank in ['cna', 'nac']:
            self.sequence_rank = sequence_rank
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier

        self.strides = strides
        self.auto_pad = auto_pad
        self.padding = 0
        self.padding_mode = padding_mode
        # if self.auto_pad == False:
        #     self.padding = 0
        # else:
        #     self.padding= tuple([n-2 for n in  list(self.kernel_size)]) if hasattr(self.kernel_size,'__len__') else
        #     self.kernel_size-2

        self.use_bias = use_bias
        self.dilation = dilation

        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate

        conv = DepthwiseConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier,
                                    strides=self.strides, auto_pad=self.auto_pad, padding_mode=self.padding_mode,
                                    activation=None, use_bias=self.use_bias, dilation=self.dilation, name=self._name).to(self.device)
        norm = get_normalization(normalization)
        self.use_spectral = use_spectral
        if isinstance(norm, SpectralNorm):
            self.use_spectral = True
            norm = None
            conv = nn.utils.spectral_norm(conv)
        if (hasattr(self, 'sequence_rank') and self.sequence_rank == 'cna') or not hasattr(self, 'sequence_rank'):
            self.add_module('conv', conv)
            self.add_module('norm', norm)
            self.add_module('activation', get_activation(activation, only_layer=True))

        elif self.sequence_rank == 'nac':
            self.add_module('norm', norm)
            self.add_module('activation', get_activation(activation, only_layer=True))
            self.add_module('conv', conv)

        elif self.sequence_rank == 'acn':
            self.add_module('activation', get_activation(activation, only_layer=True))
            self.add_module('conv', conv)
            self.add_module('norm', norm)

        self.droupout = None
        self.keep_output = keep_output
        self._name = name

    def build(self, input_shape: TensorShape):
        if self._built == False or self.conv is None:

            self.conv.input_shape = input_shape
            if self.use_spectral:
                self.conv = nn.utils.spectral_norm(self.conv)
                if self.norm is SpectralNorm:
                    self.norm = None

            self.to(self.device)
            self._built = True

    def forward(self, x, **kwargs):

        if not hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'cna')
        if self.add_noise == True and self.training == True:
            noise = self.noise_intensity * torch.randn_like(x, dtype=x.dtype)
            x = x + noise
        for child in list(self.children())[:3]:
            if child is not None:
                x = child(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, depth_multiplier={depth_multiplier}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


class SeparableConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), depth_multiplier=1, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, keep_output=False, sequence_rank='cna', **kwargs):
        super(SeparableConv2d_Block, self).__init__(name=name, keep_output=keep_output)

        if sequence_rank in ['cna', 'nac']:
            self.sequence_rank = sequence_rank
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        self.num_filters = kwargs.get('num_filters')
        self.strides = strides
        self.auto_pad = auto_pad
        self.padding = 0
        self.padding_mode = padding_mode
        # if self.auto_pad == False:
        #     self.padding = 0
        # else:
        #     self.padding= tuple([n-2 for n in  list(self.kernel_size)]) if hasattr(self.kernel_size,'__len__') else
        #     self.kernel_size-2

        self.use_bias = use_bias
        self.dilation = dilation
        self.groups = groups

        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate
        self.use_spectral = use_spectral
        conv=SeparableConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier,
                        strides=self.strides, auto_pad=self.auto_pad, padding_mode=self.padding_mode,
                        activation=None, use_bias=self.use_bias, dilation=self.dilation, groups=self.groups,
                        name=self._name).to(self.device)
        norm = get_normalization(normalization)
        if isinstance(norm, SpectralNorm):
            self.use_spectral = True
            norm = None
            conv = nn.utils.spectral_norm(conv)
        if (hasattr(self, 'sequence_rank') and self.sequence_rank == 'cna') or not hasattr(self, 'sequence_rank'):
            self.add_module('conv', conv)
            self.add_module('norm', norm)
            self.add_module('activation', get_activation(activation, only_layer=True))

        elif self.sequence_rank == 'nac':
            self.add_module('norm', norm)
            self.add_module('activation', get_activation(activation, only_layer=True))
            self.add_module('conv', conv)

        elif self.sequence_rank == 'acn':
            self.add_module('activation', get_activation(activation, only_layer=True))
            self.add_module('conv', conv)
            self.add_module('norm', norm)


        self.depth_multiplier = depth_multiplier
        self.keep_output = keep_output
        self._name = name

    def build(self, input_shape: TensorShape):
        if not self._built:
            self.num_filters = self.input_filters * self.depth_multiplier if self.num_filters is None else self.num_filters

            self.conv.input_shape = input_shape
            if self.use_spectral:
                self.conv = nn.utils.spectral_norm(self.conv)
                if self.norm is SpectralNorm:
                    self.norm = None
            self.to(self.device)
            self._built = True

    def forward(self, x, **kwargs):

        if not hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'cna')
        if self.add_noise == True and self.training == True:
            noise = self.noise_intensity * torch.randn_like(x, dtype=x.dtype)
            x = x + noise
        for child in list(self.children())[:3]:
            if child is not None:
                x = child(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, depth_multiplier={depth_multiplier}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


class TemporalConv1d_Block(Sequential):
    def __init__(self, kernel_size=2, num_filters=None, num_levels=8,strides=1, activation=None,dropout_rate=0.2,
                 use_bias=False, name=None, depth_multiplier=None, keep_output=False,**kwargs):

        super(TemporalConv1d_Block, self).__init__()
        self.kernel_size=kernel_size
        self.num_filters=num_filters
        self.depth_multiplier=depth_multiplier

        layers = []
        self.num_levels=num_levels
        for i in range(num_levels):
            self.add_module('temporal_block{0}'.format(i),self.temporal_block(num_filters= num_filters, kernel_size=2, strides=1, dilation= 2 ** i,
                                     padding=(kernel_size - 1) * (i*2), dropout_rate=dropout_rate))


    def temporal_block(self,kernel_size=2,num_filters=None,depth_multiplier=1,strides=1,dilation=1,padding=0,dropout_rate=0,activation=None):
        return ShortCut(
            Conv1d_Block(kernel_size=1, num_filters=num_filters, depth_multiplier=1, strides=strides, dilation=1, auto_pad=True,activation=None, dropout_rate=0),
            Sequential(
                Conv1d_Block(kernel_size=kernel_size,num_filters=num_filters,depth_multiplier=depth_multiplier,strides=strides,dilation=dilation,auto_pad=False,padding=(0,padding),activation=activation,dropout_rate=dropout_rate),
                Conv1d_Block(kernel_size=kernel_size,num_filters=num_filters,depth_multiplier=depth_multiplier,strides=strides,dilation=dilation,auto_pad=False,padding=(0,padding),activation=activation,dropout_rate=dropout_rate)
            ),mode='add')




def For(what_range, constructor):
    """
    For(what_range, constructor, name='')
    Layer factory function to create a composite through a pattern similar to Python's `for` statement.
    This layer factory loops over the given range and passes each value to the constructor function.
    It is equivalent to
    ``Sequential([constructor(i) for i in what_range])``.
    It is acceptable that ``constructor`` takes no argument.

    Args:
     what_range (range): a Python range to loop over
     constructor (Python function/lambda with 1 or 0 arguments): lambda that constructs a layer
    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the layers as constructed by ``constructor`` one after another.

    Examples:
     >>> # stack of 3 Dense relu layers
     >>> model = For(range(3), lambda: Dense(200, activation=relu))
     >>> # version of the above that has no activation for the last layer
     >>> model = For(range(3), lambda i: Dense(200, name='dense_{0}'.format(i+1)))
     >>> print(model[2].name)
     dense_3
    """
    # Python 2.7 support requires us to use getargspec() instead of inspect
    takes_arg = len(inspect.getfullargspec(constructor).args) > 0

    # For Python 3, check if it is a python function/lambda
    if not callable(constructor):
        raise ValueError("constructor must be a Python function/lambda")

    # helper to call the layer constructor
    def call(i):
        if takes_arg:
            return constructor(i)  # takes an arg: pass it
        else:
            return constructor()  # takes no arg: call without, that's fine too

    layers = [call(i) for i in what_range]
    return Sequential(layers)


class Highway(Layer):
    """Highway module.
    In highway network, two gates are added to the ordinal non-linear
    transformation (:math:`H(x) = activate(W_h x + b_h)`).
    One gate is the transform gate :math:`T(x) = \\sigma(W_t x + b_t)`, and the
    other is the carry gate :math:`C(x)`.
    For simplicity, the author defined :math:`C = 1 - T`.
    Highway module returns :math:`y` defined as
    .. math::
        y = activate(W_h x + b_h) \\odot \\sigma(W_t x + b_t) +
        x \\odot(1 - \\sigma(W_t x + b_t))
    The output array has the same spatial size as the input. In order to
    satisfy this, :math:`W_h` and :math:`W_t` must be square matrices.
    Args:
        in_out_features (int): Dimension of input and output vectors.
        bias (bool): If ``True``, then this function does use the bias.
        activate: Activation function of plain array. :math:`tanh` is also
            available.
    See:
        `Highway Networks <https://arxiv.org/abs/1505.00387>`_.
    """

    def __init__(self, in_out_features, bias=True, activate=F.relu):
        super(Highway, self).__init__()
        self.in_out_features = in_out_features
        self.bias = bias
        self.activate = activate

        self.plain = nn.Linear(self.in_out_features, self.in_out_features, bias=bias)
        self.transform = nn.Linear(self.in_out_features, self.in_out_features, bias=bias)

    def forward(self, x, **kwargs):
        """Computes the output of the Highway module.
        Args:
            x (~torch.Tensor): Input variable.
        Returns:
            Variable: Output variable. Its array has the same spatial size and
            the same minibatch size as the input array.
        """

        out_plain = self.activate(self.plain(x))
        out_transform = torch.sigmoid(self.transform(x))
        x = out_plain * out_transform + x * (1 - out_transform)
        return x


class Classifier1d(Layer):
    def __init__(self, num_classes=10, is_multilable=False, classifier_type='dense', name=None, keep_output=False,
                 **kwargs):
        super(Classifier1d, self).__init__(name=name)
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.is_multilable = is_multilable
        self.dense = None
        self.global_avgpool = None
        self.conv1x1 = None
        self._name = name
        self.keep_output = keep_output

    def build(self, input_shape: TensorShape):
        if self._built == False or self.conv1x1 is None:
            if self.classifier_type == 'global_avgpool':
                if self.input_filters != self.num_classes:
                    if self.conv1x1 is None:
                        self.conv1x1 = Conv2d((1, 1), num_filters=self.num_classes, strides=1, padding=0,
                                              activation=None, use_bias=False).to(self.device)
                        self.conv1x1.input_shape = input_shape
            self._built = True

    def forward(self, x, **kwargs):

        if self.classifier_type == 'dense':
            x = x.view(x.size(0), x.size(1), -1)
            x = torch.mean(x, -1, False)
            if self.dense is None:
                self.dense = nn.Linear(x.size(1), self.num_classes).to(self.device)
            x = self.dense(x)

        elif self.classifier_type == 'global_avgpool':
            if len(self._input_shape) != 4:
                raise ValueError("GlobalAvgPool2d only accept BCHW shape")
            if self.conv1x1 is not None:
                x = self.conv1x1(x)
            if self.global_avgpool is None:
                self.global_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
            x = self.global_avgpool(x)
            x = x.view(x.size(0), x.size(1))
        x = torch.sigmoid(x)
        return torch.softmax(x, dim=1)

    def extra_repr(self):
        s = ('{num_classes}, classifier_type={classifier_type}')
        return s.format(**self.__dict__)


class ShortCut2d(Layer):
    """ShortCut2d Layer """

    def __init__(self, *args, axis=1, branch_from=None, branch_from_uuid=None, activation=None, mode='add',dropout_rate=0, name=None, keep_output=False,
                 **kwargs):
        """

        Args:
            *args ():
            axis ():
            branch_from ():
            activation ():
            mode (str):  'add' 'dot' 'concate'
            name (str):
            keep_output (bool):
            **kwargs ():

        """
        super(ShortCut2d, self).__init__(name=name)
        self.activation = get_activation(activation)
        self.has_identity = False
        self.mode = mode
        self.axis = axis
        self.dropout_rate=dropout_rate
        self.branch_from = branch_from
        self.branch_from_uuid = branch_from_uuid
        if self.branch_from or self.branch_from_uuid:
            for k, v in self.nodes.item_list:
                if self.branch_from is not None and v.name == self.branch_from:
                    v.keep_output = True
                    self.branch_from_uuid = k
                    break

        self.keep_output = keep_output

        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, (Layer, torch.Tensor, list, dict)):
                if isinstance(arg, list):
                    arg = Sequential(*arg)
                elif isinstance(arg, OrderedDict) and len(args) == 1:
                    for k, v in arg.items():
                        if isinstance(v, Identity):
                            self.has_identity = True
                            self.add_module('Identity', v)
                        else:
                            self.add_module(k, v)
                elif isinstance(arg, dict) and len(args) == 1:
                    keys = sorted(list(arg.keys()))
                    for k in keys:
                        v = arg[k]
                        if isinstance(v, Identity):
                            self.has_identity = True
                            self.add_module('Identity', v)
                        else:
                            self.add_module(str(k), v)
                elif isinstance(arg, (dict, OrderedDict)) and len(args) > 1:
                    raise ValueError('more than one dict argument is not support.')

                elif isinstance(arg, Identity):
                    self.has_identity = True
                    self.add_module('Identity', arg)
                elif isinstance(arg, nn.Module):
                    if len(arg.name) > 0 and arg.name != arg.default_name:
                        self.add_module(arg.name, arg)
                    else:
                        self.add_module('branch{0}'.format(i + 1), arg)
                else:
                    raise ValueError('{0} is not support.'.format(arg.__class__.__name))
        if len(self._modules) == 1 and self.has_identity == False and self.branch_from is None and mode != 'concate':
            self.has_identity = True
            self.add_module('Identity', Identity())
        self.to(self.device)

    def build(self, input_shape: TensorShape):
        if self._built == False:
            if self.branch_from is not None:
                for k, v in self.nodes.item_list:
                    if v.name == self.branch_from:
                        v.keep_output = True
                        self.branch_from_uuid = k

                        break
                if self.branch_from_uuid is None:
                    raise ValueError('Cannot find any layer named {0}'.format(self.branch_from))
            self._built = True

    def forward(self, x, **kwargs):
        current = None
        concate_list = []

        for k, v in self._modules.items():
            new_item = v(x)  # if not isinstance(v, Identity) else x
            if current is None:
                current = new_item
                concate_list.append(current)
            else:
                try:
                    if self.mode == 'add':
                        current = current + new_item
                    elif self.mode == 'dot':
                        current = current * new_item
                    elif self.mode == 'concate':
                        concate_list.append(new_item)
                    else:
                        raise ValueError('Not valid shortcut mode')
                except Exception as e:
                    print(e)

        branch1 = None
        if hasattr(self, 'branch_from_uuid') and self.branch_from_uuid is not None and self.branch_from_uuid in self.nodes.key_list:
            tt = self.nodes.get(self.branch_from_uuid)
            branch1 = identity(self.nodes.get(self.branch_from_uuid)._output_tensor)
            if self.mode == 'add':
                current = current + branch1
            elif self.mode == 'dot':
                current = current * branch1
            elif self.mode == 'concate':
                concate_list.append(branch1)
                if len(concate_list) > 1:
                    print('')
        try:
            if self.mode == 'concate':
                x = concate(concate_list, axis=self.axis)
            else:
                x = current
        except Exception as e:
            print('Layer {0} relative name:{1} concate fails. The input shapes:{2} '.format(self.name, self.relative_name, [int_shape(item) for item in concate_list]))
        if self.activation is not None:
            x = self.activation(x)
        if hasattr(self,'dropout_rate') and self.dropout_rate>0:
            x=torch.dropout(x,self.dropout_rate,self.training)
        return x

    def extra_repr(self):
        s = ('mode={mode}, keep_output={keep_output},axis={axis}')
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        if hasattr(self, 'branch_from') and self.branch_from is not None:
            s += (', branch_from={branch_from}, branch_from_uuid={branch_from_uuid}')
        return s.format(**self.__dict__)


class ShortCut(Layer):
    """ShortCut2d Layer """

    def __init__(self, *args, axis=1, branch_from=None, activation=None, mode='add',dropout_rate=0, name=None, keep_output=False,
                 **kwargs):
        """

        Args:
            *args ():
            axis ():
            branch_from ():
            activation ():
            mode (str):  'add' 'dot' 'concate'
            name (str):
            keep_output (bool):
            **kwargs ():

        """
        super(ShortCut, self).__init__(name=name, keep_output=keep_output)
        valid_mode = ['add', 'subtract', 'concate', 'dot', 'maxout']
        if mode in valid_mode:
            self.mode = mode
        else:
            raise ValueError('{0} is not valid mode. please use one of {1}'.format(mode, valid_mode))
        self.activation = get_activation(activation)
        self.has_identity = False

        self.axis = axis
        self.branch_from = branch_from
        self.branch_from_uuid = None
        self.dropout_rate=dropout_rate
        self.keep_output = keep_output

        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, (Layer, torch.Tensor, list, dict)):
                if isinstance(arg, list):
                    arg = Sequential(*arg)
                elif isinstance(arg, OrderedDict) and len(args) == 1:
                    for k, v in arg.items():
                        if isinstance(v, Identity):
                            self.has_identity = True
                            self.add_module('Identity', v)
                        else:
                            self.add_module(k, v)
                elif isinstance(arg, dict) and len(args) == 1:
                    keys = sorted(list(arg.keys()))
                    for k in keys:
                        v = arg[k]
                        if isinstance(v, Identity):
                            self.has_identity = True
                            self.add_module('Identity', v)
                        else:
                            self.add_module(str(k), v)
                elif isinstance(arg, (dict, OrderedDict)) and len(args) > 1:
                    raise ValueError('more than one dict argument is not support.')

                elif isinstance(arg, Identity):
                    self.has_identity = True
                    self.add_module('Identity', arg)
                elif isinstance(arg, nn.Module):
                    if len(arg.name) > 0 and arg.name != arg.default_name:
                        self.add_module(arg.name, arg)
                    else:
                        self.add_module('branch{0}'.format(i + 1), arg)
                else:
                    raise ValueError('{0} is not support.'.format(arg.__class__.__name))
        if len(self._modules) == 1 and self.has_identity == False and self.branch_from is None and mode != 'concate':
            self.has_identity = True
            self.add_module('Identity', Identity())
        self.to(self.device)

    def build(self, input_shape: TensorShape):
        if self._built == False:
            if self.branch_from is not None:
                for k, v in self.nodes.item_list:
                    if v.name == self.branch_from:
                        v.keep_output = True
                        self.branch_from_uuid = k
                        self.register_buffer('branch_from_tensor', v._output_tensor)
                        print('get {0} output info...'.format(self.branch_from))
                        break
                if self.branch_from_uuid is None:
                    raise ValueError('Cannot find any layer named {0}'.format(self.branch_from))
            self._built = True

    def forward(self, x, **kwargs):

        current = None
        concate_list = []

        for k, v in self._modules.items():
            new_item = v(x)  # if not isinstance(v, Identity) else x
            if current is None:
                current = new_item
                concate_list.append(current)
            else:
                if self.mode == 'add':
                    current = current + new_item
                elif self.mode == 'subtract':
                    current = current - new_item
                elif self.mode == 'dot':
                    current = current * new_item
                elif self.mode == 'concate':
                    concate_list.append(new_item)
                else:
                    raise ValueError('Not valid shortcut mode')

        if hasattr(self,
                   'branch_from_uuid') and self.branch_from_uuid is not None and self.branch_from_uuid in self.nodes:
            self.branch_from_tensor = self.nodes.get(self.branch_from_uuid)._output_tensor

            if self.mode == 'add':
                current = current + self.branch_from_tensor
            elif self.mode == 'subtract':
                current = current - self.branch_from_tensor
            elif self.mode == 'dot':
                current = current * self.branch_from_tensor
            elif self.mode == 'concate':
                concate_list.append(self.branch_from_tensor)

        if self.mode == 'concate':
            x = concate(concate_list, axis=self.axis)
        else:
            x = current
        if self.activation is not None:
            x = self.activation(x)
        if hasattr(self, 'dropout_rate') and self.dropout_rate > 0:
            x = torch.dropout(x, self.dropout_rate, self.training)
        return x

    def extra_repr(self):
        s = ('mode={mode}, keep_output={keep_output},axis={axis}')
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        if hasattr(self, 'branch_from') and self.branch_from is not None:
            s += (', branch_from={branch_from}, branch_from_uuid={branch_from_uuid}')
        return s.format(**self.__dict__)


class Hourglass(Layer):
    """HourGlass Block

        Example:
            >>> input = to_tensor(torch.randn(1,128,128,128))
            >>> from trident.models.pytorch_resnet import bottleneck
            >>> block=bottleneck(num_filters=64,strides=1,expansion = 2,conv_shortcut=False)
            >>> hgnet = Hourglass(block,depth=2,blocks_repeat=2)
            >>> out=hgnet(input)
            >>> print(out.shape)
            torch.Size([1, 128, 128, 128])
            >>> print(list(hgnet.named_modules()))


    """

    def __init__(self, block, depth=2, blocks_repeat=2, keep_output=False, name=None, **kwargs):
        super(Hourglass, self).__init__(keep_output=keep_output, name=name)
        self.depth = depth
        self.block = block
        self.pool = MaxPool2d((2, 2), strides=2)
        self.upsample = Upsampling2d(scale_factor=2, mode='bilinear')

        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, blocks_repeat))
            if i == 0:
                res.append(self._make_residual(block, blocks_repeat))
            hg.append(ModuleList(res))
        self.hg = ModuleList(hg)

    def _make_residual(self, block, blocks_repeat):
        return Sequential([block] * blocks_repeat)

    def _hour_glass_forward(self, depth_id, x):
        up1 = self.hg[depth_id][0](x)
        low1 = self.pool(x)
        low1 = self.hg[depth_id][1](low1)
        if depth_id == 0:
            low2 = self.hg[depth_id][3](low1)
        else:
            low2 = self._hour_glass_forward(depth_id + 1, low1)
        low3 = self.hg[depth_id][2](low2)
        up2 = self.upsample(low3)
        return up1 + up2

    def forward(self, x, **kwargs):
        x = self._hour_glass_forward(0, x)
        return x


class ConcateBlock(Layer):
    def __init__(self, *args, axis=1, activation='relu'):
        """

        Parameters
        ----------
        layer_defs : object
        """
        super(ConcateBlock, self).__init__()
        self.activation = get_activation(activation)
        self.axis = axis
        self.has_identity = False
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, (Layer, list, dict)):
                if isinstance(arg, list):
                    arg = Sequential(*arg)
                elif isinstance(arg, dict) and len(args) == 1:
                    for k, v in arg.items():
                        if isinstance(v, Identity):
                            self.has_identity = True
                            self.add_module('Identity', v)
                        else:
                            self.add_module(k, v)
                elif isinstance(arg, dict) and len(args) > 1:
                    raise ValueError('more than one dict argument is not support.')
                elif isinstance(arg, Identity):
                    self.has_identity = True
                    self.add_module('Identity', arg)
                else:
                    self.add_module('branch{0}'.format(i + 1), arg)
        if len(self._modules) == 1 and self.has_identity == False:
            self.add_module('Identity', Identity())
        self.to(self.device)

    def forward(self, x, **kwargs):

        outs = []
        if 'Identity' in self._modules:
            outs.append(x)
        for k, v in self._modules.items():
            if k != 'Identity':
                out = v(x)
                if len(outs) == 0 or out.size()[2:] == outs[0].size()[2:]:
                    outs.append(out)
                else:
                    raise ValueError(
                        'All branches in shortcut should have the same shape {0} {1}'.format(out.size(), x.size()))
        outs = torch.cat(outs, dim=self.axis)
        if self.activation is not None:
            outs = self.activation(outs)
        return outs


class SqueezeExcite(Layer):
    def __init__(self, se_filters, num_filters=None, is_gather_excite=False, activation='relu', use_bias=False, depth_multiplier=None,name=''):
        super(SqueezeExcite, self).__init__(name=name)
        self.se_filters = se_filters
        self.depth_multiplier=depth_multiplier
        self.num_filters = num_filters

        self.use_bias = use_bias
        self.is_gather_excite = is_gather_excite
        self.activation = get_activation(activation)
        self.pool = GlobalAvgPool2d()

    def build(self, input_shape: TensorShape):
        if self._built == False:
            if self.num_filters is None and isinstance(self.depth_multiplier,numbers.Number):
                self.num_filters=int(self.input_filters*self.depth_multiplier)

            se_filters=None
            if isinstance(self.se_filters,numbers.Integral):
                se_filters=self.se_filters
            if inspect.isfunction(self.se_filters):
                se_filters=self.se_filters(self.input_filters)

            self.squeeze = Conv2d((1, 1), se_filters, strides=1, auto_pad=False, activation=None, use_bias=self.use_bias, name=self.name + '_squeeze')
            self.excite = Conv2d((1, 1), self.num_filters, strides=1, auto_pad=False, activation=None, use_bias=self.use_bias, name=self.name + '_excite')

            self.to(self.device)
            self._built = True

    def forward(self, x, **kwargs):
        s = self.pool(x)
        s = s.view(s.size(0), s.size(1), 1, 1)
        s = self.activation(self.squeeze(s))
        s = torch.sigmoid(self.excite(s))

        if self.is_gather_excite:
            s = F.interpolate(s, size=(x.shape[2], x.shape[3]), mode='nearest')

        x = s * x
        return x

class Siamese(Layer):
    def __init__(self, *arg, cardinality=2,similarity_fn=None,norm='l2',name='siamese'):
        super(Siamese, self).__init__(name=name)
        arg=unpack_singleton(arg)
        self.network = arg
        self.cardinality=cardinality
        if similarity_fn is None:
            self.similarity_fn=nn.functional.cosine_similarity
        self.norm=get_normalization(norm)

    def forward(self, *x, **kwargs):
        n_inputs=len(x)
        x=unpack_singleton(x)
        if len(x)==2 and self.cardinality==3:
            x=(x[0],x[0],x[1])
        elif len(x)==1 and self.cardinality>1:
            x = (x[0])*self.cardinality

        if len(x)!=self.cardinality:
            raise  RuntimeError('Numbers of input({0}) are not matching to cardinality({1})'.format(len(x),self.cardinality))
        results=[]
        for k in range(self.cardinality):
            inp=x[k]
            inp=self.network(inp)
            if self.norm is not None:
                inp=self.norm(inp)
            results.append(inp)
        if self.cardinality==2:
            return self.similarity_fn(*results)
        elif self.cardinality==3:
            return self.similarity_fn(results[1],results[0]),self.similarity_fn(results[2],results[0])







