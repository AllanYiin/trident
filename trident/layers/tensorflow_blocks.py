from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import inspect
import itertools
import math
from functools import reduce
from functools import wraps
from itertools import repeat

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import image_ops

from trident.backend.common import *
from trident.layers.tensorflow_activations import get_activation, Identity
from trident.layers.tensorflow_layers import *
from trident.layers.tensorflow_normalizations import get_normalization,SpectralNorm
from trident.layers.tensorflow_pooling import GlobalAvgPool2d
from trident.backend.tensorflow_backend import *
from trident.backend.tensorflow_ops import *
from trident.layers.tensorflow_layers import *

_tf_data_format = 'channels_last'

__all__ = ['FullConnect_Block','Conv1d_Block', 'Conv2d_Block', 'TransConv2d_Block', 'DepthwiseConv2d_Block', 'SeparableConv2d_Block', 'ShortCut2d', 'ConcateBlock', 'SqueezeExcite', 'For']

_session = get_session()


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
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
                 keep_output=False, sequence_rank='fna'):
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
            fc = SpectralNorm(module=fc)
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

    def fuse(self):
        if 'batchnorm' in self.norm.__class__.__name__.lower() and not self.use_spectral:
            shadow_fc = deepcopy(self.fc)
            if self.sequence_rank == 'fna' or self.sequence_rank == 'afn':
                bn_rv = self.norm.running_var.value().copy()
                bn_rm = self.norm.running_mean.value().copy()
                bn_eps = self.norm.eps
                bn_w = self.norm.weight.value().copy() if self.norm.affine else ones_like(bn_rm)
                bn_b = self.norm.bias.value().copy() if self.norm.affine else zeros_like(bn_rm)
                bn_scale = bn_w / tf.sqrt(bn_rv + bn_eps)
                fc_w = self.fc.weight.value().copy()
                fc_b = self.fc.bias.value().copy() if self.fc.use_bias else zeros_like(bn_rm)

                fused_w = expand_dims(fc_w * bn_scale,-1)
                fused_b = (fc_b - bn_rm) * bn_scale + bn_b
                shadow_fc.weight.assign(fused_w)
                shadow_fc.use_bias=True
                shadow_fc.bias = Parameter(vafused_b)

                # test fusion effect
                dummy_input = random_normal([2] + self.input_shape.dims[1:], dtype=self.fc.weight.dtype).cuda()
                result1 = self.forward(dummy_input.copy())
                result2 = shadow_fc.forward(dummy_input.copy())
                if self.activation is not None:
                    result2 = self.activation(result2)
                diff = to_numpy((result1 - result2).abs())
                ctx.print('diff', diff.mean(), diff.max())
                if diff.mean() < 1e-7 and diff.max() < 1e-5:
                    self.fc.weight.assign(fused_w)
                    self.fc.use_bias = True
                    self.fc.bias = Parameter(fused_b)
                    del self._modules['norm']
                    del dummy_input
                    del result1
                    del result2
                    del shadow_fc

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
                      depth_multiplier=self.depth_multiplier)
        self.use_spectral = use_spectral
        # if isinstance(norm, SpectralNorm):
        #     self.use_spectral = True
        #     norm = None
        #     conv= nn.utils.spectral_norm(conv)
        if (hasattr(self, 'sequence_rank') and self.sequence_rank == 'cna') or not hasattr(self, 'sequence_rank'):
            self.conv = conv
            self.norm = norm
            self.activation = get_activation(activation)
        elif self.sequence_rank == 'nac':
            self.norm = norm
            self.activation = get_activation(activation)
            self.conv = conv

    def build(self, input_shape:TensorShape):
        if self._built == False:
            # if self.use_spectral:
            #     self.conv = nn.utils.spectral_norm(self.conv)
            #     if self.norm is SpectralNorm:
            #         self.norm=None
            self._built = True


    def forward(self, x,**kwargs):
        if hasattr(self, 'sequence_rank'):
            setattr(self, 'sequence_rank', 'cna')
        if self.add_noise == True and self.training == True:
            noise = self.noise_intensity * random_normal_like(x, dtype=x.dtype)
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
        if self.dropout_rate > 0 and self.training:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], tf.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


class Conv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None, keep_output=False, sequence_rank='cna', **kwargs):
        super(Conv2d_Block, self).__init__(name=name)
        if sequence_rank in ['cna', 'nac']:
            self.sequence_rank = sequence_rank
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.keep_output = keep_output
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

        self.use_bias = use_bias
        self.dilation = dilation
        self.groups = groups

        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate
        self.depth_multiplier = depth_multiplier
        self.use_spectral = use_spectral
        if not self.use_spectral:
            self.conv = Conv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                               auto_pad=self.auto_pad, activation=None,
                               use_bias=self.use_bias, dilation=self.dilation, groups=self.groups,
                               depth_multiplier=self.depth_multiplier, padding=self.padding, **kwargs)
            self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self.conv.input_shape = input_shape
            # if self.use_spectral:
            #     conv=self._modules['conv']
            #     self._modules['conv']=nn.utils.spectral_norm(conv)
            #     self.norm=None
            self._built = True


    def forward(self, x, **kwargs):
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1,dtype=x.dtype)
            x += noise
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        return x
    
    def fuse(self):

        if 'batchnorm' in self.norm.__class__.__name__.lower() and not self.use_spectral:
            shadow_conv=deepcopy(self.conv)

            if self.sequence_rank == 'cna' or self.sequence_rank == 'acn':
                # sequential
                # y1 = x * w1 + b1  # linear
                # y2 = (y1 - running_mean) / sqrt(running_var + eps) * gamma + beta  # batchnorm

                # # replace y1
                # y2 = (x * w1 + b1 - running_mean) / sqrt(running_var + eps) * gamma + beta
                bn_rv=self.norm.running_var.value().copy()
                bn_rm=self.norm.running_mean.value().copy()
                bn_eps=self.norm.eps
                conv_w=self.conv.weight.value().copy()
                conv_b = self.conv.bias.value().copy() if self.conv.use_bias else zeros_like(bn_rm)
                bn_w=self.norm.weight.value().copy() if self.norm.affine else ones_like(bn_rm)
                bn_b = self.norm.bias.value().copy() if self.norm.affine else zeros_like(bn_rm)
                bn_var_rsqrt = 1/tf.sqrt(bn_rv + bn_eps)

                conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape( [1] * (len(conv_w.shape) - 1)+[-1])
                conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
                shadow_conv.weight.assign(conv_w)
                shadow_conv.use_bias=True
                shadow_conv.bias=Parameter(conv_b)
                #test fusion effect
                dummy_input=random_normal([2]+self.input_shape.dims[1:],dtype=self.conv.weight.dtype).cuda()
                result1=self.forward(dummy_input.copy())
                result2=shadow_conv.forward(dummy_input.copy())
                if self.activation is not None:
                    result2=self.activation(result2)
                diff=to_numpy((result1-result2).abs())
                ctx.print('diff',diff.mean(),diff.max())
                if diff.mean()<1e-6 and diff.max()<1e-5:
                    self.conv.weight.assign(conv_w)
                    self.conv.use_bias =True
                    self.conv.bias = Parameter(conv_b)
                    del self._modules['norm']
                    del dummy_input
                    del result1
                    del result2
                    del shadow_conv

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


class TransConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None, sequence_rank='cna', **kwargs):
        super(TransConv2d_Block, self).__init__(name=name)
        if sequence_rank in ['cna', 'nac']:
            self.sequence_rank = sequence_rank
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.auto_pad = auto_pad

        self.use_bias = use_bias
        self.dilation = dilation
        self.groups = groups

        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate
        self.depth_multiplier = depth_multiplier
        self.use_spectral = use_spectral
        if not self.use_spectral:
            self.conv = TransConv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                                    auto_pad=self.auto_pad, activation=None,
                                    use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name,
                                    depth_multiplier=self.depth_multiplier)
            self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None

    def build(self, input_shape:TensorShape):
        if self._built == False:
            conv = TransConv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                               auto_pad=self.auto_pad, activation=None,
                               use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name,
                               depth_multiplier=self.depth_multiplier)
            # conv.input_shape = input_shape

            if self.use_spectral:
                # self.conv = nn.utils.spectral_norm(conv)
                self.norm = None
            else:
                self.conv = conv
            self._built = True


    def forward(self, x, **kwargs):
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1,dtype=x.dtype)
            x += noise
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


class TransConv3d_Block(Layer):
    def __init__(self, kernel_size=(3, 3, 3), num_filters=32, strides=1, input_shape=None, auto_pad=True,
                 activation='leaky_relu', normalization=None, use_bias=False, dilation=1, groups=1, add_noise=False,
                 noise_intensity=0.001, dropout_rate=0, name=None, sequence_rank='cna', **kwargs):
        super(TransConv3d_Block, self).__init__(name=name)
        if sequence_rank in ['cna', 'nac']:
            self.sequence_rank = sequence_rank
        if add_noise:
            noise = tf.keras.layers.GaussianNoise(noise_intensity)
            self.add(noise)
        self._conv = TransConv3d(kernel_size=kernel_size, num_filters=num_filters, strides=strides,
                                 input_shape=input_shape, auto_pad=auto_pad, activation=None, use_bias=use_bias,
                                 dilation=dilation, groups=groups)
        self.add(self._conv)

        self.norm = get_normalization(normalization)
        if self.norm is not None:
            self.add(self.norm)

        self.activation = get_activation(snake2camel(activation))
        if self.activation is not None:
            self.add(self.activation)
        if dropout_rate > 0:
            self.drop = Dropout(dropout_rate)
            self.add(self.drop)


    def forward(self, x, **kwargs):
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1,dtype=x.dtype)
            x += noise
        x = self._conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, num_filters={num_filters}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)



class DepthwiseConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), depth_multiplier=1, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, keep_output=False, sequence_rank='cna', **kwargs):
        super(DepthwiseConv2d_Block, self).__init__(name=name)
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
        self.conv = None
        self.norm = get_normalization(normalization)
        self.use_spectral = use_spectral
        self.activation = get_activation(activation)
        self.droupout = None
        self.keep_output = keep_output
        self._name = name

    def build(self, input_shape:TensorShape):
        if self._built == False or self.conv is None:
            conv = DepthwiseConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier,
                                   strides=self.strides, auto_pad=self.auto_pad, padding=self.padding, padding_mode=self.padding_mode,
                                   activation=None, use_bias=self.use_bias, dilation=self.dilation, name=self._name)
            conv.input_shape = input_shape
            if self.use_spectral:
                self.conv = spectral_norm(conv)
            else:
                self.conv = conv

            self._built = True


    def forward(self, x, **kwargs):
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1,dtype=x.dtype)
            x += noise
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        return x

    def fuse(self):

        if 'batchnorm' in self.norm.__class__.__name__.lower() and not self.use_spectral:
            shadow_conv=deepcopy(self.conv)

            if self.sequence_rank == 'cna' or self.sequence_rank == 'acn':
                # sequential
                # y1 = x * w1 + b1  # linear
                # y2 = (y1 - running_mean) / sqrt(running_var + eps) * gamma + beta  # batchnorm

                # # replace y1
                # y2 = (x * w1 + b1 - running_mean) / sqrt(running_var + eps) * gamma + beta
                bn_rv=self.norm.running_var.value().copy()
                bn_rm=self.norm.running_mean.value().copy()
                bn_eps=self.norm.eps
                conv_w=self.conv.weight.value().copy()
                conv_b = self.conv.bias.value().copy() if self.conv.use_bias else zeros_like(bn_rm)
                bn_w=self.norm.weight.value().copy() if self.norm.affine else ones_like(bn_rm)
                bn_b = self.norm.bias.value().copy() if self.norm.affine else zeros_like(bn_rm)
                bn_var_rsqrt = 1/tf.sqrt(bn_rv + bn_eps)

                conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 2)+[-1] )
                conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
                shadow_conv.weight.assign(conv_w)
                shadow_conv.use_bias=True
                shadow_conv.bias=Parameter(conv_b)
                #test fusion effect
                dummy_input=random_normal([2]+self.input_shape.dims[1:],dtype=self.conv.weight.dtype).cuda()
                result1=self.forward(dummy_input.copy())
                result2=shadow_conv.forward(dummy_input.copy())
                if self.activation is not None:
                    result2=self.activation(result2)
                diff=to_numpy((result1-result2).abs())
                ctx.print('diff', diff.mean(), diff.max())
                if diff.mean()<1e-6 and diff.max()<1e-5:
                    self.conv.weight.assign(conv_w)
                    self.conv.use_bias =True
                    self.conv.bias = Parameter(conv_b)
                    del self._modules['norm']
                else:
                    ctx.print('diff',diff.mean(),diff.max())
                del dummy_input
                del result1
                del result2
                del shadow_conv


    def extra_repr(self):
        s = 'kernel_size={kernel_size}, depth_multiplier={depth_multiplier}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


class SeparableConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), depth_multiplier=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, sequence_rank='cna', **kwargs):
        super(SeparableConv2d_Block, self).__init__()
        if sequence_rank in ['cna', 'nac']:
            self.sequence_rank = sequence_rank
        self.kernel_size = kernel_size
        self.strides = strides
        self.auto_pad = auto_pad

        self.use_bias = use_bias
        self.dilation = dilation
        self.groups = groups

        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.dropout_rate = dropout_rate
        self.depth_multiplier = depth_multiplier
        self.use_spectral = use_spectral
        if not self.use_spectral:
            self.conv = SeparableConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier, strides=self.strides,
                                        auto_pad=self.auto_pad, activation=None,
                                        use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name)
            self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None

    def build(self, input_shape:TensorShape):
        if self._built == False:
            conv = SeparableConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier, strides=self.strides,
                                   auto_pad=self.auto_pad, activation=None,
                                   use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name
                                   )
            # conv.input_shape = input_shape

            # if self.use_spectral:
            #     self.conv = spectral_norm(self.conv)
            #     if self.norm is SpectralNorm:
            #         self.norm=None
            self._built = True


    def forward(self, x, **kwargs):
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1,dtype=x.dtype)
            x += noise
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        return x
    
    def fuse(self):

        if 'batchnorm' in self.norm.__class__.__name__.lower() and not self.use_spectral:
            shadow_conv=deepcopy(self.conv)
            with torch.no_grad():
                if self.sequence_rank == 'cna' or self.sequence_rank == 'acn':
                    # sequential
                    # y1 = x * w1 + b1  # linear
                    # y2 = (y1 - running_mean) / sqrt(running_var + eps) * gamma + beta  # batchnorm

                    # # replace y1
                    # y2 = (x * w1 + b1 - running_mean) / sqrt(running_var + eps) * gamma + beta
                    bn_rv=self.norm.running_var.value().copy()
                    bn_rm=self.norm.running_mean.value().copy()
                    bn_eps=self.norm.eps
                    conv_w=self.conv.pointwise.value().copy()
                    conv_b = self.conv.bias.value().copy() if self.conv.use_bias else zeros_like(bn_rm)
                    bn_w=self.norm.weight.value().copy() if self.norm.affine else ones_like(bn_rm)
                    bn_b = self.norm.bias.value().copy() if self.norm.affine else zeros_like(bn_rm)
                    bn_var_rsqrt =1/tf.sqrt(bn_rv + bn_eps)

                    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
                    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
                    shadow_conv.pointwise.assign(conv_w)
                    shadow_conv.use_bias=True
                    shadow_conv.bias=Parameter(conv_b)
                    #test fusion effect
                    dummy_input=random_normal([2]+self.input_shape.dims[1:],dtype=self.conv.weight.dtype).cuda()
                    result1=self.forward(dummy_input.copy())
                    result2=shadow_conv.forward(dummy_input.copy())
                    if self.activation is not None:
                        result2=self.activation(result2)
                    diff=to_numpy((result1-result2).abs())
                    ctx.print('diff',diff.mean(),diff.max())
                    if diff.mean()<1e-6 and diff.max()<1e-5:
                        self.conv.weight.assign(conv_w)
                        self.conv.use_bias =True
                        self.conv.bias = Parameter(conv_b)
                        del self._modules['norm']
                        del dummy_input
                        del result1
                        del result2
                        del shadow_conv

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, depth_multiplier={depth_multiplier}, strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        return s.format(**self.__dict__)


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


class Classifer1d(Sequential):
    def __init__(self, num_classes=10, is_multilable=False, classifier_type=ClassfierType.dense, name=None, **kwargs):
        super(Classifer1d, self).__init__(name=name)
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.is_multilable = is_multilable
        if classifier_type == ClassfierType.dense:
            self.add(Flatten)
            self.add(Dense(num_classes, use_bias=False, activation='sigmoid'))
            if not is_multilable:
                self.add(SoftMax)
        elif classifier_type == ClassfierType.global_avgpool:
            self.add(Conv2d((1, 1), num_classes, strides=1, auto_pad=True, activation=None))
            self.add(GlobalAvgPool2d)
            if not is_multilable:
                self.add(SoftMax)


class ShortCut2d(Layer):
    def __init__(self, *args, axis=-1, branch_from=None, activation=None, mode='add', name=None, keep_output=False, **kwargs):
        """
        Args
        layer_defs : object
        """
        super(ShortCut2d, self).__init__(name=name)
        self.activation = get_activation(activation)
        self.has_identity = False
        self.mode = mode if isinstance(mode, str) else mode
        self.axis = axis
        self.branch_from = branch_from
        self.branch_from_uuid = None
        self.keep_output = keep_output

        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, (Layer, tf.Tensor, list, dict)):
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
                elif isinstance(arg, Layer):
                    if len(arg.name) > 0 and arg.name != arg._name:
                        self.add_module(arg.name, arg)
                    else:
                        self.add_module('branch{0}'.format(i + 1), arg)
                else:
                    raise ValueError('{0} is not support.'.format(arg.__class__.__name))
        if len(self._modules) == 1 and self.has_identity == False and self.branch_from is None:
            self.has_identity = True
            self.add_module('Identity', Identity())

    def build(self, input_shape:TensorShape):
        if self._built == False:
            if self.branch_from is not None:
                for k, v in self.nodes.item_list:
                    if v.name == self.branch_from:
                        v.keep_output = True
                        self.branch_from_uuid = k
                        print('get {0} output info...'.format(self.branch_from))
                        break
                if self.branch_from_uuid is None:
                    raise ValueError('Cannot find any layer named {0}'.format(self.branch_from))
            self._built = True

    def forward(self, x, **kwargs):
        current = None
        concate_list = []

        for k, v in self._modules.items():
            if k!='activation':
                new_item = v(x) if not isinstance(v, Identity) else x
                if current is None:
                    current = new_item
                    concate_list.append(current)
                else:
                    if self.mode == 'add':
                        current = current + new_item
                    elif self.mode == 'dot':
                        current = current * new_item
                    elif self.mode == 'concate':
                        concate_list.append(new_item)
                    else:
                        raise ValueError('Not valid shortcut mode')

        if hasattr(self, 'branch_from_uuid') and self.branch_from_uuid is not None and self.branch_from_uuid in self.nodes:
            new_item = self.nodes.get(self.branch_from_uuid)._output_tensor
            if self.mode == 'add':
                current = current + new_item
            elif self.mode == 'dot':
                current = current * new_item
            elif self.mode == 'concate':
                concate_list.append(new_item)

        if self.mode == 'concate':
            x = concate(concate_list, axis=self.axis)
        else:
            x = current
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = ('mode={mode}, keep_output={keep_output},axis={axis}')
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()

        if hasattr(self, 'branch_from') and self.branch_from is not None:
            s += ', branch_from={0}, branch_from_uuid={1}'.format(self.branch_from, self.branch_from_uuid)
        return s.format(**self.__dict__)



class ShortCut(Layer):
    """ShortCut2d Layer """

    def __init__(self, *args, axis=-1, branch_from=None, activation=None, mode='add', name=None, keep_output=False,
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
        valid_mode = ['add', 'subtract', 'concate', 'dot','maxout']
        if mode in valid_mode:
            self.mode = mode
        else:
            raise ValueError('{0} is not valid mode. please use one of {1}'.format(mode, valid_mode))
        self.activation = get_activation(activation)
        self.has_identity = False

        self.axis = axis
        self.branch_from = branch_from
        self.branch_from_uuid = None

        self.keep_output = keep_output

        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, (Layer, Tensor, list, dict)):
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
                elif isinstance(arg, Layer):
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
            if k != 'activation':
                new_item = v(x)  # if not isinstance(v, Identity) else x
                if current is None:
                    current = new_item
                    concate_list.append(current)
                else:
                    if self.mode == 'add':
                        current = current + new_item
                    elif self.mode == 'subtract':
                        current = current -new_item
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
                current = current -  self.branch_from_tensor
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
        return x

    def extra_repr(self):
        s = ('mode={mode}, keep_output={keep_output},axis={axis}')
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        if hasattr(self, 'branch_from') and self.branch_from is not None:
            s += (', branch_from={branch_from}, branch_from_uuid={branch_from_uuid}')
        return s.format(**self.__dict__)


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

    def forward(self, x, **kwargs):
        outs = []
        if 'Identity' in self._modules:
            outs.append(x)
        for k, v in self._modules.items():
            if k != 'Identity':
                out = v(x)
                if len(outs) == 0 or int_shape(out)[1:-1] == int_shape(outs[0])[1:-1]:
                    outs.append(out)
                else:
                    raise ValueError(
                        'All branches in shortcut should have the same shape {0} {1}'.format(int_shape(out), int_shape(x)))
        outs = tf.concat(outs, axis=self.axis)
        if self.activation is not None:
            outs = self.activation(outs)
        return outs


class SqueezeExcite(Layer):
    def __init__(self, se_filters, num_filters, is_gather_excite=False, use_bias=False, name=''):
        super(SqueezeExcite, self).__init__(name=name)

        self.se_filters = se_filters
        self.num_filters = num_filters
        self.squeeze = None
        self.excite = None
        self.is_gather_excite = is_gather_excite
        self.activation = get_activation('swish')
        self.pool = GlobalAvgPool2d(keepdim=True)
        self.use_bias = use_bias

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self.squeeze = Conv2d((1, 1), self.se_filters, strides=1, auto_pad=False, activation=None, use_bias=self.use_bias, name=self.name + '_squeeze')
            self.excite = Conv2d((1, 1), self.num_filters, strides=1, auto_pad=False, activation=None, use_bias=self.use_bias, name=self.name + '_excite')
            self._built = True

    def forward(self, x, **kwargs):
        s = self.pool(x)
        s = self.activation(self.squeeze(s))
        s = tf.sigmoid(self.excite(s))

        if self.is_gather_excite:
            s = image_ops.resize_images_v2(s, x.shape, method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
        x = s * x
        return x


