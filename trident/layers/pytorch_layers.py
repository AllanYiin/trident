from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
from collections import OrderedDict
from functools import partial, wraps, update_wrapper
from itertools import islice
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # import torch functions
import torch.utils.hooks as hooks
import torchvision
from torch._jit_internal import List
from torch._six import container_abcs
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter

from .pytorch_activations import get_activation
from .pytorch_normalizations import *
from ..backend.common import *
from ..backend.pytorch_backend import *
from ..backend.pytorch_ops import meshgrid

__all__ = ['Dense', 'Flatten', 'Concatenate', 'Concate','SoftMax','Add', 'Subtract', 'Dot', 'Conv1d', 'Conv2d', 'Conv3d',
           'TransConv1d', 'TransConv2d', 'TransConv3d', 'SeparableConv1d', 'SeparableConv2d', 'SeparableConv3d',
           'DepthwiseConv1d', 'DepthwiseConv2d', 'DepthwiseConv3d', 'GcdConv2d', 'GcdConv2d_1', 'Lambda', 'Reshape',
           'CoordConv2d', 'Upsampling2d', 'Dropout', 'AlphaDropout', 'SelfAttention','SingleImageLayer']

_session = get_session()

_epsilon = _session.epsilon


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


class Dense(Layer):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, output_shape, use_bias=True, activation=None,keep_output=False, name=None, **kwargs):
        super(Dense, self).__init__()
        self.rank=0
        if isinstance(output_shape, int):
            self.output_shape = _single(output_shape)
        elif isinstance(output_shape, list):
            self.output_shape = tuple(output_shape)
        elif isinstance(output_shape, tuple):
            self.output_shape = output_shape
        else:
            raise ValueError('output_shape should be integer, list of integer or tuple of integer...')
        self.name = name
        self.keep_output=keep_output
        self.weight = None
        self.bias = None
        self.use_bias = use_bias
        self.activation = get_activation(activation)

    def build(self, input_shape):
        if self._built == False:
            self.weight = Parameter(torch.Tensor(*self.output_shape.tolist(), self.input_filters))
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            # self._parameters['weight'] =self.weight
            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.output_shape.tolist()[0]))
                init.zeros_(self.bias)  # self._parameters['bias']=self.bias
            self.to(self.device)
            self._built = True


    def forward(self, *x):
        x = enforce_singleton(x)
        x = F.linear(x, self.weight, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'output_shape={0}'.format(self.output_shape.tolist()) + ',use_bias={use_bias}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()

        return s.format(**self.__dict__)


class Flatten(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Concate(Layer):
    r"""Concate layer to splice  tensors ."""

    def __init__(self, axis=1):
        super(Concate, self).__init__()
        self.axis = axis
    def forward(self, *x) -> torch.Tensor:
        if not isinstance(x, list) or len(x) < 2:
            raise ValueError('A `Concatenate` layer should be called on a list of at least 2 inputs')

        if all([k.size() is None for k in x]):
            return
        reduced_inputs_shapes = [list(k.size()) for k in x]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError(
                'A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got inputs '
                'shapes: %s' % (shape_set))
        x = torch.cat(x, dim=self.axis)
        return x


Concatenate = Concate


class Add(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=1):
        super(Add, self).__init__()

    def build(self, input_shape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, *x) -> torch.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if isinstance(x, tuple):
            x = unpack_singleton(x)
        out = 0
        for item in x:
            out = torch.add(out, item)
        return out


class Subtract(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=1):
        super(Subtract, self).__init__()

    def build(self, input_shape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, *x) -> torch.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if not isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out = torch.sub(out, item)
        return out


class Dot(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=1):
        super(Dot, self).__init__()

    def build(self, input_shape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, *x) -> torch.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if not isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out = torch.dot(out, item)
        return out


class SoftMax(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=1, add_noise=False, noise_intensity=0.005, **kwargs):
        super(SoftMax, self).__init__()
        self.axis = kwargs.get('dim', axis)

    def forward(self, *x) -> torch.Tensor:
        x = enforce_singleton(x)
        if not hasattr(self, 'add_noise'):
            self.add_noise = False
            self.noise_intensity = 0.005
        if self.training:

            if self.add_noise == True:
                noise = self.noise_intensity * torch.randn_like(x, dtype=torch.float32)
                x = x + noise
            x = F.log_softmax(x, dim=1)
        else:
            x = torch.softmax(x, dim=1)
        return x


_gcd = gcd
_get_divisors = get_divisors
_isprime = isprime


def get_static_padding(rank,input_shape,kernal_shape,strides,dilations):
    if isinstance(strides,int):
        strides= _ntuple(rank)(strides)
    if isinstance(dilations,int):
        dilations= _ntuple(rank)(dilations)

    input_shape=to_numpy(input_shape)[-rank:]
    kernal_shape=to_numpy(list(kernal_shape))[-rank:]
    strides = to_numpy(list(strides))[-rank:]
    dilations= to_numpy(list(dilations))[-rank:]
    output_shape=np.ceil(input_shape/strides)

    raw_padding=np.clip((output_shape-1)*strides+(kernal_shape-1)*dilations+1-input_shape,0,None)
    remainder=np.remainder(raw_padding,np.ones_like(raw_padding)*2)

    raw_padding=raw_padding+(remainder*np.greater(strides,1).astype(np.float32))
    lefttop_pad = np.ceil(raw_padding/2.0).astype(np.int32)
    rightbtm_pad=(raw_padding-lefttop_pad).astype(np.int32)
    static_padding = []
    for k in range(rank):
        static_padding.append(lefttop_pad[-1-k])
        static_padding.append(rightbtm_pad[-1-k])
    return static_padding






class _ConvNd(Layer):
    __constants__ = ['kernel_size', 'num_filters', 'strides', 'auto_pad', 'padding_mode', 'use_bias', 'dilation',
                     'groups', 'transposed']

    def __init__(self, kernel_size, num_filters, strides, auto_pad,padding, padding_mode, use_bias, dilation, groups,
                 transposed, name, depth_multiplier, depthwise=False,separable=False,**kwargs):
        super(_ConvNd, self).__init__(name=name)

        self.num_filters = kwargs.get('out_channels', num_filters)
        self.depth_multiplier=depth_multiplier

        self.kernel_size = kernel_size

        self.strides = kwargs['stride'] if 'stride' in kwargs else strides
        self.auto_pad=auto_pad
        self.padding = padding
        self.padding_mode = padding_mode

        self.dilation = dilation
        self.transposed = transposed
        self.groups = groups
        self.depthwise = depthwise
        self.separable = separable
        if self.separable == True:
            self.depthwise = True
        self.register_parameter('weight',None)
        self.register_parameter('bias', None)

        if groups != 1 and self.num_filters % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.transposed = transposed
        self.use_bias = use_bias
        self.to(self.device)



    def build(self, input_shape):
        if self._built == False:
            self.input_filters=input_shape[0].item()
            if self.depthwise or self.separable:
                self.groups=self.input_filters if self.groups==1 else self.groups
                if self.depth_multiplier is None:
                    self.depth_multiplier = 1

            if self.num_filters is None and self.depth_multiplier is not None:
                self.num_filters = int(round(self.input_filters * self.depth_multiplier,0))
            if self.input_filters % self.groups != 0:
                raise ValueError('in_channels must be divisible by groups')

            if self.depthwise and self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            channel_multiplier = int(self.num_filters // self.groups) if self.depth_multiplier is None else self.depth_multiplier# default channel_multiplier

            # if self.auto_pad==True and all([p==0 for p in self.padding]):
            #     static_padding=get_static_padding(self.rank, input_shape, self.kernel_size, self.strides, self.dilation)
            #     self.padding=tuple(static_padding)



            if self.transposed:
                self.weight = Parameter(torch.Tensor(int(self.input_filters), int(self.num_filters // self.groups), *self.kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(int(self.num_filters), int(self.input_filters // self.groups) , *self.kernel_size))  #
                if self.separable:
                    pointwise_kernel_size = (1,) * len(self.kernel_size)
                    self.pointwise = Parameter(torch.Tensor(int(self.input_filters * self.depth_multiplier), int(self.num_filters),1,1))

            init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            if self.use_bias:
                self.bias = Parameter(torch.Tensor(int(self.num_filters)))
                init.zeros_(self.bias)

            self.to(self.device)
            self._built = True

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={auto_pad},use_bias={use_bias} ,dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self._input_shape is not None:
            s += ', input_shape={0}, input_filter={1}'.format(self._input_shape.clone().tolist(), self.input_filters)
        if self.output_shape is not None:
            s += ', output_shape={0}'.format(self.output_shape if isinstance(self.output_shape, (
            list, tuple)) else self.output_shape.clone().tolist())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(
            state)  # if not hasattr(self, 'padding_mode'):  #     self.padding_mode = 'zeros'


class Conv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)
        padding = kwargs.get('padding', None)
        if isinstance(padding,str) and padding.lower() in ['valid','same']:
            if padding.lower()=='valid':
                auto_pad=False
            elif padding.lower()=='same':
                auto_pad = True
        elif isinstance(padding,int):
            padding = _single(padding)
        elif padding is None:
            padding = _single(0)


        super(Conv1d, self).__init__(kernel_size, num_filters, strides, auto_pad,padding,padding_mode, use_bias, dilation,
                                     groups, False, name, depth_multiplier, **kwargs)
        self.rank=1
        self.activation = get_activation(activation)
        self.auto_pad = auto_pad
        self.padding = kwargs.get('padding', None)
        if self.padding is not None:
            self.padding = _single(self.padding)
        else:
            self.padding = _single(0)



    def conv1d_forward(self, x):
        x = F.pad(x, self.padding,mode='constant' if self.padding_mode=='zero' else self.padding_mode)
        return F.conv1d(x, self.weight, self.bias, self.strides, _single(0), self.dilation, self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv2d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)
        static_padding = get_static_padding(2, (3,224,224), kernel_size, strides, dilation)
        padding = kwargs.get('padding', None)
        if isinstance(padding, str) and padding.lower() in ['valid', 'same']:
            if padding.lower() == 'valid':
                auto_pad = False
            elif padding.lower() == 'same':
                auto_pad = True
        elif isinstance(padding, int):
            padding = _pair(padding)
        if padding is None or auto_pad==True:
            padding = static_padding

        super(Conv2d, self).__init__(kernel_size, num_filters, strides, auto_pad, padding,padding_mode, use_bias, dilation,
                                     groups, False, name, depth_multiplier, **kwargs)
        self.rank=2
        self.activation = get_activation(activation)

    def conv2d_forward(self, x):
        self.rank=2
        if len(self.padding)==self.rank:
            self.padding=(self.padding[1],self.padding[1],self.padding[0],self.padding[0])
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[1] // 2, (self.padding[2] + 1) // 2, self.padding[3] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, self.padding,mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv2d(x, self.weight, self.bias, self.strides, _pair(0), self.dilation, self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        self.rank=3
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(kernel_size, num_filters, strides, auto_pad, padding_mode, use_bias, dilation,
                                     groups, False, name, depth_multiplier, **kwargs)
        self.activation = get_activation(activation)
        if self.padding is not None:
            self.padding = _triple(self.padding)
        else:
            self.padding = _triple(0)

    def get_padding(self, input_shape):
        pad_w = 0
        pad_h = 0
        pad_z = 0
        if self.auto_pad == True:
            iz, ih, iw = list(input_shape)[-3:]
            kz, kh, kw = self.kernel_size[-3:]
            sz, sh, sw = self.strides[-3:]
            dz, dh, dw = self.dilation[-3:]
            oz, oh, ow = math.ceil(iz / sz), math.ceil(ih / sh), math.ceil(iw / sw)
            pad_z = max((oz - 1) * sz + (kz - 1) * dz + 1 - iz, 0)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h % 2 == 1 and sh > 1:
                pad_h += 1
            if pad_w % 2 == 1 and sw > 1:
                pad_w += 1
            if pad_z % 2 == 1 and sz > 1:
                pad_z += 1
        elif len(self.padding) == 3:
            pad_z = self.padding[0] * 2
            pad_h = self.padding[1] * 2
            pad_w = self.padding[2] * 2

        if self.padding_mode == 'circular':
            self.padding = (pad_z // 2, pad_z-(pad_z // 2), pad_h // 2, pad_h-(pad_h // 2), pad_w// 2,pad_w- (pad_w // 2))
        else:
            self.padding = (pad_z // 2, pad_h // 2, pad_w // 2)

    def conv3d_forward(self, x):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2, (self.padding[1] + 1) // 2, self.padding[1] // 2,(self.padding[0] + 1) // 2, self.padding[0] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, (self.padding[2], self.padding[2],self.padding[1], self.padding[1], self.padding[0], self.padding[0]),mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv3d(x, self.weight, self.bias, self.strides, _triple(0), self.dilation, self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv3d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):

        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)
        super(TransConv1d, self).__init__(kernel_size, num_filters, strides, auto_pad, padding_mode, use_bias, dilation,
                                          groups, True, name, depth_multiplier, **kwargs)
        self.rank = 1
        self.activation = get_activation(activation)

        if 'padding' in kwargs:
            self.padding = kwargs['padding']
            self.padding = _single(self.padding)
            self.auto_pad = False
        else:
            self.padding = _single(0)
        self.output_padding = _single(0)

    def conv1d_forward(self, x):
        if self.auto_pad == True:
            iw = list(x.size())[-1]
            kw = self.kernel_size[-1]
            sw = self.strides[-1]
            dw = self.dilation[-1]
            ow = math.ceil(iw / sw), math.ceil(iw / sw)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_w > 0:
                self.output_padding = _single(1)
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2], mode=self.padding_mode)
        return F.conv_transpose1d(x, self.weight, self.bias, self.strides, padding=_single(0),
                                  output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv2d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)

        super(TransConv2d, self).__init__(kernel_size, num_filters, strides, auto_pad, padding_mode, use_bias, dilation,
                                          groups, True, name, depth_multiplier, **kwargs)
        self.rank=2
        self.activation = get_activation(activation)

        if self.padding is not None:
            self.padding = _pair(self.padding)
        else:
            self.padding = _pair(0)
        self.output_padding = _pair(0)

    def get_padding(self, input_shape):
        pad_h = 0
        pad_w = 0
        if self.auto_pad == True:
            ih, iw = list(input_shape)[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = self.dilation[-2:]
            oh, ow = (ih - 1) * sh + (kh - 1) * dh + 1, (iw - 1) * sw + (kw - 1) * dw + 1
            pad_h = max(oh - ih * sh, 0)
            pad_w = max(ow - iw * sw, 0)
            self.padding = (pad_h, pad_w)
            if pad_h != 0 or pad_w != 0:
                self.output_padding = (pad_h % 2 if pad_h > 0 else pad_h, pad_w % 2 if pad_w > 0 else pad_w)

                # x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode=self.padding_mode)

    def conv2d_forward(self, x):
        return F.conv_transpose2d(x, self.weight, self.bias, self.strides, padding=self.padding,
                                  output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)
        super(TransConv3d, self).__init__(kernel_size, num_filters, strides, auto_pad, padding_mode, use_bias, dilation,
                                          groups, True, name, depth_multiplier, **kwargs)
        self.rank=3
        if 'padding' in kwargs:
            self.padding = _triple(kwargs.get('padding'))
            self.auto_pad = False
        else:
            self.padding = _triple(0)
        self.output_padding = _triple(0)

    def conv3d_forward(self, x):
        if self.auto_pad == True:
            iz, ih, iw = list(x.size())[-3:]
            kz, kh, kw = self.kernel_size[-3:]
            sz, sh, sw = self.strides[-3:]
            dz, dh, dw = self.dilation[-3:]
            oz, oh, ow = math.ceil(iz / sz), math.ceil(ih / sh), math.ceil(iw / sw)
            pad_z = max((oz - 1) * sz + (kz - 1) * dz + 1 - iz, 0)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)

            if pad_z > 0 or pad_h > 0 or pad_w > 0:
                self.output_padding = _triple(1)
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2,
                              pad_z - pad_z // 2], mode=self.padding_mode)
        return F.conv_transpose3d(x, self.weight, self.bias, self.strides, padding=_triple(0),
                                  output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv3d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableConv1d(_ConvNd):
    def __init__(self, kernel_size,num_filters=None, depth_multiplier=1, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, **kwargs):
        super(SeparableConv1d, self).__init__()
        self.rank=1
        self.kernel_size = _single(kernel_size)
        self.num_filters = kwargs.get('num_filters',num_filters)
        self.depth_multiplier = depth_multiplier
        self.dilation = _single(dilation)
        self.strides = _single(strides)
        self.use_bias = use_bias
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode

        self.activation = get_activation(activation)
        self.conv1 = None
        self.pointwise = None

    def build(self, input_shape):
        if self._built == False or self.conv1 is None:
            if self.num_filters is None:
                self.num_filters = self.input_filters * self.depth_multiplier
            self.conv1 = DepthwiseConv1d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier,
                                         strides=self.strides, auto_pad=self.auto_pad, padding_mode=self.padding_mode,
                                         dilation=self.dilation, groups=self.input_filters, bias=self.use_bias)
            self.pointwise = Conv2d(kernel_size=1, num_filters=self.num_filters, strides=1, use_bias=self.use_bias, dilation=1,
                                    groups=1)
            self.to(self.device)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1(x)
        x = self.pointwise(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableConv2d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, depth_multiplier=1, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, use_bias=False, dilation=1, groups=1, name=None, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.rank=2
        self.kernel_size = _pair(kernel_size)
        self.num_filters = kwargs.get('num_filters',num_filters)
        self.depth_multiplier = depth_multiplier
        self.dilation = _pair(dilation)
        self.strides = _pair(strides)
        self.use_bias = use_bias
        self.padding_mode = padding_mode

        self.activation = get_activation(activation)
        self.padding = kwargs.get('padding', None)
        self.auto_pad = auto_pad
        if self.padding is not None:
            self.padding = _pair(self.padding)
        else:
            self.padding = _pair(0)
        self.conv1 = None
        self.pointwise = None
        self._built = False

    def build(self, input_shape):
        if self._built == False or self.conv1 is None:
            if self.num_filters is None:
                self.num_filters = self.input_filters * self.depth_multiplier if self.depth_multiplier is not None else self.num_filters
            self.conv1 = DepthwiseConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier,
                                         strides=self.strides, auto_pad=self.auto_pad, padding_mode=self.padding_mode,
                                         activation=self.activation, dilation=self.dilation, use_bias=self.use_bias)
            self.pointwise = Conv2d(kernel_size=(1, 1), num_filters=self.num_filters, strides=1, use_bias=self.use_bias,
                                    dilation=1, groups=1)
            self.to(self.device)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1(x)
        x = self.pointwise(x)

        return x


class SeparableConv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None,depth_multiplier=1, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, **kwargs):
        super(SeparableConv3d, self).__init__()
        self.rank=3
        self.kernel_size = _triple(kernel_size)
        self.num_filters = kwargs.get('num_filters',num_filters)

        self.depth_multiplier = 1
        self.dilation = _triple(dilation)
        self.strides = _triple(strides)
        self.use_bias = use_bias
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode

        self.activation = get_activation(activation)
        self.conv1 = None
        self.pointwise = None

    def build(self, input_shape):
        if self._built == False or self.conv1 is None:
            self.num_filters = self.input_filters * self.depth_multiplier if self.depth_multiplier is not None else self.num_filters
            self.conv1 = DepthwiseConv3d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier,
                                         strides=self.strides, auto_pad=self.auto_pad, padding_mode=self.padding_mode,
                                         dilation=self.dilation, groups=self.input_filters, bias=self.use_bias)
            self.pointwise = Conv3d(kernel_size=(1, 1, 1), depth_multiplier=1, strides=1, use_bias=self.use_bias, dilation=1,
                                    groups=1)

            self.to(self.device)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1(x)
        x = self.pointwise(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseConv1d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, **kwargs):

        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)
        padding = kwargs.get('padding', None)
        if isinstance(padding, str) and padding.lower() in ['valid', 'same']:
            if padding.lower() == 'valid':
                auto_pad = False
            elif padding.lower() == 'same':
                auto_pad = True
        elif isinstance(padding, int):
            padding = _single(padding)
        elif padding is None:
            padding = _single(0)
        super(DepthwiseConv1d, self).__init__(kernel_size, None, strides, auto_pad,padding, padding_mode, use_bias, dilation,
                                     groups=1, transposed=False, name=name, depth_multiplier=depth_multiplier,depthwise=True,separable=False ,**kwargs)
        self.rank=1
        self.activation = get_activation(activation)

    def conv1d_forward(self, x):
        x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)
        return F.conv1d(x, self.weight, self.bias, self.strides, _single(0), self.dilation, self.groups)



    def forward(self, *x):
        x = self.conv1d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseConv2d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, name=None, **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)
        static_padding = get_static_padding(2, (3, 224, 224), kernel_size, strides, dilation)
        padding = kwargs.get('padding', None)
        if isinstance(padding, str) and padding.lower() in ['valid', 'same']:
            if padding.lower() == 'valid':
                auto_pad = False
            elif padding.lower() == 'same':
                auto_pad = True
        elif isinstance(padding, int):
            padding = _pair(padding)
        if padding is None or auto_pad == True:
            padding = static_padding

        super(DepthwiseConv2d, self).__init__(kernel_size, None, strides, auto_pad,padding, padding_mode, use_bias, dilation,
                                     groups=1, transposed=False, name=name, depth_multiplier=depth_multiplier,depthwise=True,separable=False ,**kwargs)
        self.rank=2
        self.activation = get_activation(activation)


    def conv2d_forward(self, x):
        self.rank = 2
        if len(self.padding)==self.rank:
            self.padding=(self.padding[1],self.padding[1],self.padding[0],self.padding[0])
        if self.padding_mode == 'circular':
            expanded_padding = ( (self.padding[0] + 1) // 2, self.padding[1] // 2, (self.padding[2] + 1) // 2, self.padding[3] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv2d(x, self.weight, self.bias, self.strides, _pair(0), self.dilation, self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseConv3d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None,  **kwargs):
        super(DepthwiseConv3d, self).__init__()
        self.rank=3
        self.kernel_size = _triple(kernel_size)
        self.num_filters = kwargs.get('num_filters')

        self.depth_multiplier = depth_multiplier
        self.dilation = _triple(dilation)
        self.strides = _triple(strides)
        self.use_bias = use_bias
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode

        self.activation = get_activation(activation)
        self.conv1 = None
        self._built = False

    def build(self, input_shape):
        if self._built == False or self.conv1 is None:
            self.num_filters = self.input_filters * self.depth_multiplier if self.depth_multiplier is not None else self.num_filters
            self.conv1 = torch.nn.Conv3d(self.input_filters, self.num_filters, kernel_size=self.kernel_size,
                                         stride=self.strides, padding=0, dilation=self.dilation,
                                         groups=self.input_filters, bias=self.use_bias)

            self.to(self.device)
            self._built = True

    def get_padding(self, input_shape):
        pad_w = 0
        pad_h = 0
        pad_z = 0
        if self.auto_pad == True:
            iz, ih, iw = list(input_shape)[-3:]
            kz, kh, kw = self.kernel_size[-3:]
            sz, sh, sw = self.strides[-3:]
            dz, dh, dw = self.dilation[-3:]
            oz, oh, ow = math.ceil(iz / sz), math.ceil(ih / sh), math.ceil(iw / sw)
            pad_z = max((oz - 1) * sz + (kz - 1) * dz + 1 - iz, 0)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h % 2 == 1 and sh > 1:
                pad_h += 1
            if pad_w % 2 == 1 and sw > 1:
                pad_w += 1
            if pad_z % 2 == 1 and sz > 1:
                pad_z += 1
        elif len(self.padding) == 3:
            pad_z = self.padding[0] * 2
            pad_h = self.padding[1] * 2
            pad_w = self.padding[2] * 2

        if self.padding_mode == 'circular':
            self.padding = (
            pad_z // 2, pad_z - (pad_z // 2), pad_h // 2, pad_h - (pad_h // 2), pad_w // 2, pad_w - (pad_w // 2))
        else:
            self.padding = (pad_z // 2, pad_h // 2, pad_w // 2)

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2, (self.padding[1] + 1) // 2, self.padding[1] // 2,(self.padding[0] + 1) // 2, self.padding[0] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                      mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        x = self.conv1(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DeformConv2d(Layer):
    def __init__(self, kernel_size, num_filters=None, strides=1, offset_group=2, auto_pad=True, padding_mode='zero',
                 activation=None, use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        super(DeformConv2d, self).__init__()
        self.rank=2
        self.kernel_size = _pair(kernel_size)
        self.num_filters = kwargs.get('num_filters')
        if self.num_filters is None and depth_multiplier is not None:
            self.depth_multiplier = depth_multiplier

        self.dilation = _pair(dilation)
        self.strides = _pair(strides)
        self.use_bias = use_bias
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        self.activation = get_activation(activation)
        self.padding = kwargs.get('padding', None)

        if self.padding is not None and isinstance(self.padding, int):
            if self.padding > 0:
                self.auto_pad = False
            self.padding = _pair(self.padding)
        else:
            self.padding = _pair(0)
        self.groups = groups
        if self.input_filters % self.groups != 0:
            raise ValueError('in_channels must be divisible by groups')

    def build(self, input_shape):
        if self._built == False:
            if self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            self.offset = Parameter(torch.Tensor(self.input_filters, 2 * self.input_filters, 3, 3))
            init.kaiming_uniform_(self.offset, a=math.sqrt(5))
            self.weight = Parameter(
                torch.Tensor(self.num_filters, self.input_filters // self.groups, *self.kernel_size))
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.use_bias:
                self.bias = Parameter(torch.empty(self.num_filters))
                init.zeros_(self.bias)
            self._built = True

    def offetconv2d_forward(self, x):
        if self.auto_pad == True:
            ih, iw = list(x.size())[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = self.dilation[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode=self.padding_mode)
        return F.conv2d(x, self.offset, None, (1, 1), (0, 0), (1, 1), (1, 1))

    def conv2d_forward(self, x):
        if self.auto_pad == True:
            ih, iw = list(x.size())[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = self.dilation[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode='constant' if self.padding_mode=='zero' else self.padding_mode)
        return F.conv2d(x, self.weight, self.bias, self.strides, self.padding, self.dilation, self.groups)

    def forward(self, *x):
        """
        Arguments:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
                out_height, out_width]): offsets to be applied for each position in the
                convolution kernel.
        """
        x = enforce_singleton(x)
        # B 2*input,H,W
        offset = self.offetconv2d_forward(x).round_()
        # 2,H,W-->B,2,H,W
        grid = meshgrid(x.shape[3], x.shape[2]).unsqueeze(0).repeat(x.size(0))

        offset = grid + offset

        deform_x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class GcdConv1d(Layer):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, divisor_rank=0, self_norm=True, is_shuffle=False, name=None,
                 depth_multiplier=None, **kwargs):
        super(GcdConv1d, self).__init__()
        self.rank=1
        self.kernel_size = _single(kernel_size)
        self.num_filters = num_filters
        if self.num_filters is None and depth_multiplier is not None:
            self.depth_multiplier = depth_multiplier
        self.strides = _single(strides)
        self.auto_pad = auto_pad
        self.padding = 0
        self.padding_mode = padding_mode

        self.activation = get_activation(activation)
        self.dilation = _single(dilation)
        self.self_norm = self_norm
        self.norm = None
        self.is_shuffle = is_shuffle
        self.use_bias = use_bias
        self.divisor_rank = divisor_rank
        self.crossgroup_fusion = False
        self.weight = None
        self.bias = None
        self.groups = 1
        self._built = False

    def calculate_gcd(self):
        if self.input_filters is None or not isinstance(self.input_filters, int):
            raise ValueError('in_channels must be integer ')
        gcd_list = gcd(self.input_filters, self.num_filters)
        if len(gcd_list) == 0:
            self.groups = self.input_filters
            self.num_filters_1 = self.input_filters
        else:
            self.gcd = gcd_list[0]
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]

        if self.input_filters == self.num_filters or self.input_filters == self.gcd or self.num_filters == self.gcd:
            self.groups = gcd_list[min(int(self.divisor_rank + 1), len(gcd_list))]

    def build(self, input_shape):
        if self._built == False:
            self.calculate_gcd()
            print('input:{0} -> output:{1}   {2}  {3}  gcd:{4} group:{5}   通道縮放倍數:{5} '.format(self.input_filters,
                                                                                               self.num_filters,
                                                                                               self.input_filters // self.groups,
                                                                                               self.num_filters // self.groups,
                                                                                               self.gcd, self.groups,
                                                                                               self.num_filters / self.num_filters))

            self.channel_kernal = 2 if self.crossgroup_fusion == True and self.groups > 3 else 1
            self.channel_dilation = 1
            if self.crossgroup_fusion == True and self.groups >= 4 :
                self.channel_dilation = 2
            self.kernel_size = (self.channel_kernal,) + _pair(self.kernel_size)
            self.dilation = (self.channel_dilation,) + _pair(self.dilation)
            self.strides = (1,) + _pair(self.strides)
            reshape_input_shape = [-1, self._input_shape[0] // self.groups, self.groups, self._input_shape[1]]

            self.weight = Parameter(torch.Tensor(self.num_filters // self.groups, self._input_shape[0] // self.groups,
                                                 *self.kernel_size))  #
            init.kaiming_uniform_(self.weight, mode='fan_in')
            self._parameters['weight'] = self.weight

            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.num_filters // self.groups))
                init.zeros_(self.bias)
                self._parameters['bias'] = self.bias

            if self.self_norm == True:
                self.norm = get_normalization('batch')
                init.ones_(self.norm.weight)
                init.zeros_(self.norm.bias)

            self.to(self.device)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.auto_pad:
            ih, iw = x.size()[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = _pair(self.dilation)[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode=self.padding_mode)

        x = x.view(x.size(0), x.size(1) // self.groups, self.groups, x.size(2))
        pad_g = max((self.groups - 1) * 1 + (self.channel_kernal - 1) * self.channel_dilation + 1 - self.groups, 0)
        x = F.pad(x, [0, 0, 0, 0, pad_g // 2, pad_g - pad_g // 2], mode='reflect')

        x = F.conv2d(x, self.weight, self.bias, self.strides, self.padding, self.dilation, 1)
        if self.is_shuffle == True:
            x = x.transpose([2, 1])
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
        if self.self_norm == True:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={auto_pad},use_bias={use_bias} ,dilation={dilation}}'
        if self.gcd != 1:
            s += ', gcd={gcd},divisor_rank={divisor_rank},self_norm={self_norm},crossgroup_fusion={' \
                 'crossgroup_fusion},is_shuffle={is_shuffle} '
        if self._input_shape is not None:
            s += ', input_shape={0}, input_filter={1}'.format(self._input_shape.clone().tolist(), self.input_filters)
        if self.output_shape is not None:
            s += ', output_shape={0}'.format(self.output_shape if isinstance(self.output_shape, (
                list, tuple)) else self.output_shape.clone().tolist())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class GcdConv2d(Layer):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, divisor_rank=0, self_norm=True, is_shuffle=False, crossgroup_fusion=True,
                 name=None, depth_multiplier=None, **kwargs):
        super(GcdConv2d, self).__init__()
        self.rank=2
        self.kernel_size = _pair(kernel_size)
        self.num_filters = num_filters
        if self.num_filters is None and depth_multiplier is not None:
            self.depth_multiplier = depth_multiplier
        self.input_filters = None
        self.strides = _pair(strides)
        self.auto_pad = auto_pad
        self.padding = 0
        self.padding_mode = padding_mode
        self.activation = get_activation(activation)
        self.dilation = _pair(dilation)
        self.norm = None
        if self_norm == True:
            self.self_norm = self_norm
            self.norm = get_normalization('instance')

        self.is_shuffle = is_shuffle
        self.use_bias = use_bias
        self.divisor_rank = divisor_rank
        self.name = name

        self.groups = 1

        self.crossgroup_fusion = crossgroup_fusion
        self._built = False

    def calculate_gcd(self):
        if self.input_filters is None or not isinstance(self.input_filters, int):
            raise ValueError('in_channels must be integer ')
        self.register_buffer('gcd', torch.zeros((1)))

        gcd_list = gcd(self.input_filters, self.num_filters)
        if len(gcd_list) == 0:
            self.groups = self.input_filters
            self.num_filters_1 = self.input_filters
        else:
            self.gcd = torch.tensor(gcd_list[0])
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]

        if self.input_filters == self.num_filters or self.input_filters == self.gcd or self.num_filters == self.gcd:
            self.groups = gcd_list[min(int(self.divisor_rank + 1), len(gcd_list))]

    def get_padding(self, input_shape):
        pad_w = 0
        pad_h = 0
        pad_z = 0
        if self.auto_pad == True:
            iz, ih, iw = list(input_shape)[-3:]
            kz, kh, kw = self.actual_kernel_size[-3:]
            sz, sh, sw = self.actual_strides[-3:]
            dz, dh, dw = self.actual_dilation[-3:]
            oz, oh, ow = math.ceil(iz / sz), math.ceil(ih / sh), math.ceil(iw / sw)
            pad_z = max((oz - 1) * sz + (kz - 1) * dz + 1 - iz, 0)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h % 2 == 1 and sh > 1:
                pad_h += 1
            if pad_w % 2 == 1 and sw > 1:
                pad_w += 1

        elif len(self.padding) == 3:
            pad_z = self.padding[0]
            pad_h = self.padding[1] * 2
            pad_w = self.padding[2] * 2

        if self.padding_mode == 'circular':
            self.padding = (pad_z , 0, pad_h // 2, pad_h - (pad_h // 2), pad_w // 2, pad_w - (pad_w // 2))
        else:
            self.padding = (pad_z , pad_h // 2, pad_w // 2)

    def build(self, input_shape):
        if self._built == False:
            self.input_filters = input_shape[0].item()
            self.calculate_gcd()
            if self.num_filters is None and self.depth_multiplier is not None:
                self.num_filters = int(round(self.input_filters * self.depth_multiplier, 0))
            if self.input_filters % self.groups != 0:
                raise ValueError('in_channels must be divisible by groups')
            print('{0} input:{1} -> output:{2}   {3}  {4}  gcd:{5} group:{6}   通道縮放倍數:{7} '.format(self.name,
                                                                                                   self.input_filters,
                                                                                                   self.num_filters,
                                                                                                   self.input_filters // self.groups,
                                                                                                   self.num_filters // self.groups,
                                                                                                   self.gcd,
                                                                                                   self.groups,
                                                                                                   self.num_filters / float(
                                                                                                       self.input_filters)))


            self.channel_kernal = torch.tensor(2 if self.crossgroup_fusion == True and self.groups > 4 else 1)
            self.channel_dilation = torch.tensor(1)
            # if self.crossgroup_fusion == True and self.groups > 6:
            #     self.channel_dilation = torch.tensor(2)

            self.actual_kernel_size = (self.channel_kernal,) + _pair(self.kernel_size)
            self.actual_dilation = (self.channel_dilation,) + _pair(self.dilation)
            self.actual_strides = (1,) + _pair(self.strides)

            self.get_padding([input_shape[0]//self.groups, self.groups, input_shape[1], input_shape[2]])

            self.weight = Parameter(torch.Tensor(self.num_filters // self.groups, self.input_filters // self.groups, *self.actual_kernel_size))  #
            init.kaiming_uniform_(self.weight, mode='fan_in')

            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.num_filters // self.groups))
                init.zeros_(self.bias)
            else:
                self.register_parameter('bias',None)

            self.to(self.device)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x=x.view(x.size(0), x.size(1) // self.groups, self.groups, x.size(2), x.size(3))
        x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1],0,0),mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        if self.channel_kernal.item() == 2:
            x=torch.cat([x,x[:,:,0:1,:,:]],dim=2)


        x = F.conv3d(x, self.weight, self.bias, self.actual_strides, _triple(0), self.actual_dilation, 1)

        if self.is_shuffle == True:
            x = x.transpose_(2, 1)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        if self.self_norm == True:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation'].__repr__())
        s += ',auto_pad={auto_pad},use_bias={use_bias} ,dilation={dilation}'
        if self.gcd != 1:
            s += ', divisor_rank={divisor_rank},self_norm={self_norm},crossgroup_fusion={crossgroup_fusion},is_shuffle={is_shuffle} '
        # if self._input_shape is not None:
        #     s += ', input_shape={0}, input_filter={1}'.format(self._input_shape.clone().tolist(),
        #                                                       self.input_filters)
        # if self.output_shape is not None:
        #     s += ', output_shape={0}'.format(self.output_shape if isinstance(self.output_shape, (
        #     list, tuple)) else self.output_shape.clone().tolist())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class GcdConv2d_1(Layer):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, divisor_rank=0, dilation=1, self_norm=True, name=None, depth_multiplier=None, **kwargs):
        super(GcdConv2d_1, self).__init__(name=name)
        self.rank=2
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        if self.num_filters is None and depth_multiplier is not None:
            self.depth_multiplier = depth_multiplier
        self.input_filters = 3
        self.strides = _pair(strides)
        self.auto_pad = auto_pad
        self.padding = 0
        self.padding_mode = padding_mode

        self.activation = get_activation(activation)
        self.dilation = dilation
        self.self_norm = self_norm
        self.norm = None
        self.use_bias = use_bias

        self.weight = None
        self.bias = None
        self.divisor_rank = divisor_rank
        self.groups = 1
        self._built = False

    def calculate_gcd(self):
        gcd_list = _gcd(self.input_filters, self.num_filters)
        if len(gcd_list) == 0:
            self.groups = self.input_filters
            self.num_filters_1 = self.input_filters
        else:
            self.gcd = gcd_list[0]
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]

            self.num_filters_1 = self.gcd
            self.num_filters_2 = self.num_filters
            factors = _get_divisors(self.num_filters // self.gcd)

        if self.input_filters == self.num_filters or self.input_filters == self.gcd or self.num_filters == self.gcd:
            self.groups = gcd_list[min(int(self.divisor_rank + 1), len(gcd_list))]

    def get_padding(self, input_shape):
        pad_h = 0
        pad_w = 0
        if self.auto_pad == True:
            ih, iw = to_list(input_shape)[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = self.dilation[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max(round((oh - 1) * sh + (kh - 1) * dh + 1 - ih), 0)
            pad_w = max(round((ow - 1) * sw + (kw - 1) * dw + 1 - iw), 0)
            if pad_h % 2 == 1 and sh > 1:
                pad_h += 1
            if pad_w % 2 == 1 and sw > 1:
                pad_w += 1
        elif len(self.padding) == 2:
            pad_h = self.padding[0] * 2
            pad_w = self.padding[1] * 2

        self.padding = (pad_h // 2, pad_w // 2)

    def build(self, input_shape):
        if self._built == False:
            self.calculate_gcd()
            self.get_padding(input_shape)
            print('input:{0} -> output:{1}   {2}  {3}  gcd:{4} group:{5}   通道縮放倍數:{5} '.format(self.input_filters,
                                                                                               self.num_filters,
                                                                                               self.input_filters // self.groups,
                                                                                               self.num_filters // self.groups,
                                                                                               self.gcd, self.groups,
                                                                                               self.num_filters / self.num_filters))

            self.weight = Parameter(
                torch.Tensor(self.input_filters, self.num_filters // self.groups, *self.kernel_size))
            init.kaiming_uniform_(self.weight, mode='fan_in')
            self._parameters['weight'] = self.weight
            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.num_filters // self.groups))
                init.zeros_(self.bias)
                self._parameters['bias'] = self.bias

            if self.self_norm == True:
                self.norm = nn.BatchNorm2d(self.num_filters, momentum=0.1, affine=True, track_running_stats=True).to(
                    self.device)
                init.ones_(self.norm.weight)
                init.zeros_(self.norm.bias)

            self.to(self.device)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.padding_mode == 'circular':
            expanded_padding = (
            (self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2)
            x = F.pad(input, expanded_padding, mode='circular')
        else:
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                      mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        x = F.conv2d(x, self.weight, self.bias, self.strides, self.padding, self.dilation, self.groups)
        if self.self_norm == True:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={auto_pad},use_bias={use_bias} ,dilation={dilation}}'
        if self.gcd != 1:
            s += ', gcd={gcd},divisor_rank={divisor_rank},self_norm={self_norm},crossgroup_fusion={crossgroup_fusion},is_shuffle={is_shuffle} '
        if self._input_shape is not None:
            s += ', input_shape={0}, input_filter={1}'.format(self._input_shape.clone().tolist(), self.input_filters)
        if self.output_shape is not None:
            s += ', output_shape={0}'.format(self.output_shape if isinstance(self.output_shape, (
                list, tuple)) else self.output_shape.clone().tolist())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class Lambda(Layer):
    """
    Applies a lambda function on forward()
    Args:
        lamb (fn): the lambda function
    """

    def __init__(self, function, name=None):
        super(Lambda, self).__init__(name=name)
        self.function = function

    def forward(self, *x):
        return self.function(*x)


class Reshape(Layer):
    """
    Reshape the input volume
    Args:
        *shape (ints): new shape, WITHOUT specifying batch size as first
        dimension, as it will remain unchanged.
    """

    def __init__(self, target_shape, name=None):
        super(Reshape, self).__init__(name=name)
        if isinstance(target_shape, tuple):
            self.target_shape = to_tensor([target_shape[i] for i in range(len(target_shape))])
        elif isinstance(target_shape, list):
            self.target_shape = to_tensor(target_shape)

    def forward(self, *x):
        x = enforce_singleton(x)
        shp = self.target_shape.tolist().copy()
        shp.insert(0, x.size(0))
        shp = tuple(shp)
        return torch.reshape(x, shp)


class SelfAttention(Layer):
    """ Self attention Layer"""

    def __init__(self, reduction_factor=8, name=None):
        super(SelfAttention, self).__init__(name=name)
        self.rank=2
        # self.activation = activation
        self.reduction_factor = reduction_factor
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.attention = None
        self.gamma = nn.Parameter(torch.zeros(1))
        init.zeros_(self.gamma)
        self._parameters['gamma'] = self.gamma

        self.softmax = nn.Softmax(dim=-1)  #

    def build(self, input_shape):
        self.query_conv = nn.Conv2d(in_channels=self.input_filters,
                                    out_channels=self.input_filters // self.reduction_factor, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.input_filters,
                                  out_channels=self.input_filters // self.reduction_factor, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.input_filters, out_channels=self.input_filters, kernel_size=1)
        self.to(self.device)

    def forward(self, *x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = enforce_singleton(x)
        B, C, width, height = x.size()
        proj_query = self.query_conv(x).view(B, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(B, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        self.attention = self.softmax(energy).clone()  # BX (N) X (N)
        proj_value = self.value_conv(x).view(B, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, self.attention.permute(0, 2, 1))
        out = out.view(B, C, width, height)

        out = self.gamma * out.clone() + x
        return out


"""
Implementation of the CoordConv modules from https://arxiv.org/abs/1807.03247
"""


def _append_coords(input_tensor, with_r=False):
    batch_size, _, x_dim, y_dim = input_tensor.size()

    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

    ret = torch.cat([input_tensor, xx_channel.type_as(input_tensor), yy_channel.type_as(input_tensor), ], dim=1, )

    if with_r:
        rr = torch.sqrt(
            torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
        ret = torch.cat([ret, rr], dim=1)

    return ret


"""
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
"""


class CoordConv2d(Layer):
    def __init__(self, kernel_size, num_filters, strides, auto_pad=True, activation=None, use_bias=False, group=1,
                 dilation=1, with_r=False, name=None, **kwargs):
        super().__init__(name=name)
        self.rank=2
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.auto_pad = auto_pad
        self.use_bias = use_bias
        self.group = group
        self.dilation = dilation
        self._conv_settings = kwargs
        self.activation = get_activation(activation)
        self.addcoords = partial(_append_coords, with_r=with_r)
        self.conv = None

    def build(self, input_shape):
        if self._built == False:
            self.conv = Conv2d(self.kernel_size, self.num_filters, self.strides, auto_pad=self.auto_pad,
                               activation=self.activation, use_bias=self.use_bias, group=self.group,
                               dilation=self.dilation, **self._conv_settings)
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class Upsampling2d(Layer):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True, name=None):
        super(Upsampling2d, self).__init__(name=name)
        self.rank=2
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.mode == 'pixel_shuffle':
            return F.pixel_shuffle(x, int(self.scale_factor))
        elif self.mode == 'nearest':
            return F.interpolate(x, self.size, self.scale_factor, self.mode, None)
        else:
            return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class Dropout(Layer):
    def __init__(self, dropout_rate=0, name=None):
        super(Dropout, self).__init__(name=name)
        self.inplace = True
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, ""but got {}".format(dropout_rate))
        self.dropout_rate = dropout_rate

    def forward(self, *x):
        x = enforce_singleton(x)
        return F.dropout(x, self.dropout_rate, self.training, self.inplace)

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.dropout_rate, self.inplace)


class AlphaDropout(Layer):
    '''
     .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    '''

    def __init__(self, dropout_rate=0, name=None):
        super(AlphaDropout, self).__init__(name=name)
        self.inplace = True
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, ""but got {}".format(dropout_rate))
        self.dropout_rate = dropout_rate

    def forward(self, *x):
        x = enforce_singleton(x)
        return F.alpha_dropout(x, self.dropout_rate, self.training, self.inplace)

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.dropout_rate, self.inplace)



class SingleImageLayer(Layer):
    def __init__(self, image,is_recursive=False,name=None):
        super(SingleImageLayer, self).__init__(name=name)
        self.rank=2
        if isinstance(image,(np.ndarray,torch.Tensor)):
            self.origin_image = to_tensor(image).squeeze()
            self.input_shape = image.shape[1:]

    def build(self, input_shape):
        if self._built == False:
            self.weight = Parameter(self.origin_image.clone(), requires_grad=True)
            self.input_filters = input_shape[0]
            self._built = True
    def forward(self,x):
        return self.weight.data.unsqueeze(0)

    def extra_repr(self):
        return 'is_recursive={0}'.format(self.is_recursive)
