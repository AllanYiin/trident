from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import builtins
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

from torch._jit_internal import List
from torch._six import container_abcs
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter

from trident.layers.pytorch_activations import get_activation
from trident.layers.pytorch_normalizations import *
from trident.backend.common import *
from trident.backend.pytorch_backend import *
from trident.backend.pytorch_ops import *

__all__ = ['Dense', 'Flatten', 'Concatenate', 'Concate','SoftMax','Add', 'Subtract', 'Dot', 'Conv1d', 'Conv2d', 'Conv3d',
           'TransConv1d', 'TransConv2d', 'TransConv3d', 'SeparableConv1d', 'SeparableConv2d', 'SeparableConv3d',
           'DepthwiseConv1d', 'DepthwiseConv2d', 'DepthwiseConv3d', 'GcdConv2d', 'Lambda', 'Reshape',
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
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

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

    Examples:

        >>> m = Dense(30)
        >>> input = to_tensor(torch.randn(2, 20))
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([2, 30])
    """

    def __init__(self, num_filters, use_bias=True, activation=None,keep_output=False, name=None, **kwargs):
        super(Dense, self).__init__()
        self.rank=0
        if isinstance(num_filters, int):
            self.num_filters = num_filters
        elif isinstance(num_filters, tuple):
            self.num_filters=unpack_singleton(num_filters)
            #elif isinstance(num_filters, list):
        #    self.num_filters= tuple(num_filters)
        #elif isinstance(num_filters, tuple):
        #    self.num_filters = num_filters
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
            self.weight = Parameter(torch.Tensor(self.num_filters, self.input_filters))
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            # self._parameters['weight'] =self.weight
            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.num_filters))
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
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Concate(Layer):
    """Concate layer to splice  tensors ."""

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
    """Flatten layer to flatten a tensor after convolution."""

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
    """Flatten layer to flatten a tensor after convolution."""

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
    """Flatten layer to flatten a tensor after convolution."""

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
    """SoftMax layer

    SoftMax layer is designed for accelerating  classification model training
    In training stage, it will process the log_softmax transformation (get log-likelihood for a single instance ).
    In testing/ evaluation/ infer stage, it will process the 'so-called' softmax transformation.
    All transformation is processed across 'asix (default=1)'

    And you also can setting add_noise and noise_intensity arugments to imprement output noise.
    output noise can force model make every output probability should large enough or small enough, otherwise it will confused within output noise.
    It;s a regularzation technique for classification model training.

    """
    def __init__(self, axis=1, add_noise=False, noise_intensity=0.005, name=None ,**kwargs):
        """
        Args:
            axis (int,default=1): The axis all the transformation processed across.
            add_noise (bool, default=False): If True, will add (output) noise  in this layer.
            noise_intensity (float, default=0.005): The noise intensity (is propotional to mean of actual output.

        """
        super(SoftMax, self).__init__(name=name)
        self.axis = kwargs.get('dim', axis)
        self.add_noise=add_noise
        self.noise_intensity=noise_intensity

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



def get_static_padding(rank,kernal_shape,strides,dilations,input_shape=None,transpose=False):
    """ Calcualte the actual padding we need in different rank and different convlution settings.

    Args:
        rank (int):
        kernal_shape (tuple of integer):
        strides (tuple of integer):
        dilations (tuple of integer):
        input_shape (None or tuple of integer):
        transpose (bool): whether transposed

    Returns: the padding we need (shape: 2*rank )

    Examples
    >>> get_static_padding(1,(3,),(2,),(2,))
    (2, 2)
    >>> get_static_padding(2,(3,3),(2,2),(1,1),(224,224))
    (1, 1, 1, 1)
    >>> get_static_padding(2,(3,3),(2,2),(1,1),(224,224),True)
    ((1, 1, 1, 1), (1, 1))
    >>> get_static_padding(2,(5,5),(1,1),(2,2))
    (4, 4, 4, 4)
    >>> get_static_padding(2,(5,5),(1,1),(1,1))
    (2, 2, 2, 2)
    >>> get_static_padding(2,(2,2),(1,1),(1,1))
    (1, 0, 1, 0)
    >>> get_static_padding(3,(5,5,5),(1,1,1),(2,2,2))
    (4, 4, 4, 4, 4, 4)
    """
    if input_shape is None:
        input_shape=[224]*rank
    if isinstance(kernal_shape,int):
        kernal_shape= _ntuple(rank)(kernal_shape)
    if isinstance(strides,int):
        strides= _ntuple(rank)(strides)
    if isinstance(dilations,int):
        dilations= _ntuple(rank)(dilations)

    input_shape=to_numpy(input_shape)
    kernal_shape=to_numpy(list(kernal_shape))
    strides = to_numpy(list(strides)).astype(np.float32)
    dilations= to_numpy(list(dilations))
    if transpose==False:
        output_shape=np.ceil(input_shape/strides)
        raw_padding=np.clip((output_shape-1)*strides+(kernal_shape-1)*dilations+1-input_shape,a_min=0,a_max=np.inf)
        remainder=np.remainder(raw_padding,np.ones_like(raw_padding)*2)

        raw_padding=raw_padding+(remainder*np.greater(strides,1).astype(np.float32))
        lefttop_pad = np.ceil(raw_padding/2.0).astype(np.int32)
        rightbtm_pad=(raw_padding-lefttop_pad).astype(np.int32)
        static_padding = []
        for k in range(rank):
            static_padding.append(lefttop_pad[-1-k])
            static_padding.append(rightbtm_pad[-1-k])
        return static_padding
    else:
        output_shape = input_shape * strides
        raw_padding = np.clip( ((input_shape - 1) * strides + (kernal_shape - 1) * dilations + 1 )-input_shape * strides, a_min=0, a_max=np.inf)
        remainder = np.remainder(raw_padding, np.ones_like(raw_padding) * 2)

        raw_padding= raw_padding + (remainder * np.greater(strides, 1).astype(np.float32))
        lefttop_pad = np.ceil(raw_padding / 2.0).astype(np.int32)
        rightbtm_pad = (raw_padding- lefttop_pad).astype(np.int32)

        out_pad=output_shape-((input_shape - 1) * strides + (kernal_shape - 1) * dilations + 1-lefttop_pad-rightbtm_pad)

        static_padding = []

        for k in range(rank):
            static_padding.append(lefttop_pad[-1 - k])
            static_padding.append(rightbtm_pad[-1 - k])
        return tuple(static_padding), tuple(out_pad.astype(np.int32).tolist())


class _ConvNd(Layer):
    __constants__ = ['kernel_size', 'num_filters', 'strides', 'auto_pad', 'padding_mode', 'use_bias', 'dilation',
                     'groups', 'transposed']

    def __init__(self, rank,kernel_size, num_filters, strides, auto_pad,padding, padding_mode, use_bias, dilation, groups,
                 transposed=False, name=None, depth_multiplier=1, depthwise=False,separable=False,**kwargs):
        super(_ConvNd, self).__init__(name=name)
        self.rank=rank
        self.num_filters = num_filters
        self.depth_multiplier=depth_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.transposed = transposed
        if self.transposed:
            self.output_padding=_ntuple(rank)(0)
        self.groups = groups
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        if padding is not None:
            self.padding = normalize_padding(padding,rank)
        else:
            self.padding=None

        self.depthwise = depthwise
        self.separable = separable
        if self.separable == True:
            self.depthwise = True

        self.register_parameter('weight',None)
        self.register_parameter('bias', None)

        self.transposed = transposed
        self.use_bias = use_bias
        self.to(self.device)

    def build(self, input_shape):
        if self._built == False:
            self.input_filters=input_shape[0].item()
            if self.auto_pad:
                if self.transposed==False:
                    padding = get_static_padding(self.rank, self.kernel_size, self.strides, self.dilation,input_shape.tolist()[1:])
                    self.padding=tuple(padding)
                else:
                    self.padding ,self.output_padding= get_static_padding(self.rank, self.kernel_size, self.strides, self.dilation, input_shape.tolist()[1:], self.transposed)
            else:
                if self.padding is None:
                    self.padding = [0] * (2 * self.rank)
                elif isinstance(self.padding, int):
                    self.self.padding = [self.padding] * (2 * self.rank)
                elif len(self.padding) == self.rank:
                    self.padding = list(self.padding)*2
                elif len(self.padding) == 2 * self.rank:
                    pass


            if self.depthwise or self.separable:
                if self.depth_multiplier is None:
                    self.depth_multiplier = 1
                if self.groups>1:
                    pass
                elif self.depth_multiplier<1:
                    self.groups=int(builtins.round(self.input_filters*self.depth_multiplier,0))
                else:
                    self.groups=self.input_filters if self.groups==1 else self.groups


            if self.num_filters is None and self.depth_multiplier is not None:
                self.num_filters = int(builtins.round(self.input_filters * self.depth_multiplier,0))

            if self.groups != 1 and self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            if self.depthwise and self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            channel_multiplier = int(self.num_filters // self.groups) if self.depth_multiplier is None else self.depth_multiplier# default channel_multiplier

            if self.transposed:
                self.weight = Parameter(torch.Tensor(int(self.input_filters), int(self.num_filters // self.groups), *self.kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(int(self.num_filters), int(self.input_filters // self.groups) , *self.kernel_size))  #

                if self.separable:
                    self.pointwise = Parameter(torch.Tensor(int(self.input_filters * self.depth_multiplier), int(self.num_filters),1,1))

            init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            if self.use_bias:
                self.bias = Parameter(torch.Tensor(int(self.num_filters)))
                init.zeros_(self.bias)

            self.to(get_device())
            self._built = True

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, num_filters={num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={auto_pad}'
        if hasattr(self,'padding') and self.padding is not None:
            s += ', padding={0}, padding_mode={1}'.format(self.padding, self.padding_mode)
        s += ',use_bias={use_bias} ,dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if hasattr(self,'_input_shape') and self._input_shape is not None:
            s += ', input_shape={0}, input_filter={1}'.format(to_numpy(self._input_shape).tolist(), self.input_filters)
        if hasattr(self,'_output_shape') and self._output_shape is not None:
            s += ', output_shape={0}'.format(self._output_shape if isinstance(self._output_shape, (
            list, tuple)) else self._output_shape.clone().tolist())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(
            state)  # if not hasattr(self, 'padding_mode'):  #     self.padding_mode = 'zeros'


class Conv1d(_ConvNd):
    """Applies to create a 1D convolution layer

        Args:
            kernel_size :(int or tupleof ints)
                shape (spatial extent) of the receptive field

            num_filters :(int  or None, default to None)
                number of output channel (filters)`, sometimes in backbond design output channel is propotional to input channel.
                But in trident all layer is shape  delay inferred

            strides:(int or tupleof ints ,default to 1)
                 stride of the convolution (increment when sliding the filter over the input)

            auto_pad:bool
                if `False`, then the filter will be shifted over the "valid" area of input, that is, no value outside the area is used. If ``pad=True`` means 'same

            *padding (optional)
                auto_pad can help you calculate the pad you need.
                if you have special need , you still can use the paddding
                implicit paddings on both sides of the input. Can be a single number or a double tuple (padH, padW)
                or quadruple(pad_left, pad_right, pad_top, pad_btm )

            padding_mode:string (default is 'zero', available option are 'reflect', 'replicate','constant','circular')

            activation: (None, string, function or Layer)
                activation function after the convolution operation for apply non-linearity.

            use_bias:bool
                the layer will have no bias if `False` is passed here

            dilation:(int or tupleof ints)
                the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1

            groups
                split input into groups, \text{in\_channels}in_channels should be divisible by the number of groups. Default: 1
            depth_multiplier: (int of decimal)


            name
                name of the layer

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

        Examples:
            >>> input = to_tensor(torch.randn(1,64,32))
            >>> conv1= Conv1d(3,64,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
            >>> output = conv1(input)
            >>> print(output.size())
            torch.Size([1, 64, 16])
            >>> print(conv1.weight.size())
            torch.Size([64, 64, 3])
            >>> print(conv1.padding)
            (1, 1)
            >>> conv2= Conv1d(3, 256, strides=2, auto_pad=False, padding=1)
            >>> output = conv2(input)
            >>> print(output.size())
            torch.Size([1, 256, 16])
            >>> print(conv2.weight.size())
            torch.Size([256, 64, 3])
            >>> print(conv2.padding)
            (1, 1)
            >>> conv3= Conv1d(5,64,strides=1,activation=mish, auto_pad=True,use_bias=False,dilation=4,groups=16)
            >>> output = conv3(input)
            >>> print(output.size())
            torch.Size([1, 64, 32])
            >>> print(conv3.weight.size())
            torch.Size([64, 4, 5])
            >>> print(conv3.padding)
            (8, 8)
            >>> input = to_tensor(torch.randn(1,32,37))
            >>> conv4= Conv1d(3,64,strides=2,activation=mish, auto_pad=True,use_bias=False)
            >>> output = conv4(input)
            >>> print(output.size())
            torch.Size([1, 64, 19])

        """
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None,padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride',strides))
        dilation = _single(kwargs.get('dilation_rate',dilation))
        num_filters=kwargs.get('filters',kwargs.get('out_channels',num_filters))
        use_bias=kwargs.get('bias',use_bias)
        padding_mode = padding_mode.lower().replace('zeros','zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad==False:
            #avoid someone use the definition as keras padding
            auto_pad = (padding.lower()  == 'same')
            auto_pad = False
        elif isinstance(padding, int):
            padding = _single(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            pass
        super(Conv1d, self).__init__(rank,kernel_size, num_filters, strides, auto_pad,padding,padding_mode, use_bias, dilation,
                                     groups, transposed=False, name=name, depth_multiplier=depth_multiplier,depthwise=False,separable=False, **kwargs)

        self.activation = get_activation(activation)


    def conv1d_forward(self, x):
        x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)
        return F.conv1d(x, self.weight, self.bias, self.strides, _single(0), self.dilation, self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv2d(_ConvNd):
    """Applies to create a 2D convolution layer

        Args:
            kernel_size :(int or tupleof ints)
                shape (spatial extent) of the receptive field

            num_filters :(int  or None, default to None)
                number of output channel (filters)`, sometimes in backbond design output channel is propotional to input channel.
                But in trident all layer is shape  delay inferred

            strides:(int or tupleof ints ,default to 1)
                 stride of the convolution (increment when sliding the filter over the input)

            auto_pad:bool
                if `False`, then the filter will be shifted over the "valid" area of input, that is, no value outside the area is used. If ``pad=True`` means 'same

            *padding (optional)
                auto_pad can help you calculate the pad you need.
                if you have special need , you still can use the paddding
                implicit paddings on both sides of the input. Can be a single number or a double tuple (padH, padW)
                or quadruple(pad_left, pad_right, pad_top, pad_btm )

            padding_mode:string (default is 'zero', available option are 'reflect', 'replicate','constant','circular')

            activation: (None, string, function or Layer)
                activation function after the convolution operation for apply non-linearity.

            use_bias:bool
                the layer will have no bias if `False` is passed here

            dilation:(int or tupleof ints)
                the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1

            groups
                split input into groups, \text{in\_channels}in_channels should be divisible by the number of groups. Default: 1
            depth_multiplier: (int of decimal)

            name
                name of the layer

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

        Examples:
            >>> input = to_tensor(torch.randn(1,32,32,32))
            >>> conv1= Conv2d((3,3),64,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
            >>> output = conv1(input)
            >>> print(output.size())
            torch.Size([1, 64, 16, 16])
            >>> print(conv1.weight.size())
            torch.Size([64, 32, 3, 3])
            >>> print(conv1.padding)
            (1, 1, 1, 1)
            >>> conv2= Conv2d((3, 3), 256, strides=(2, 2), auto_pad=False, padding=((1, 0), (1, 0)))
            >>> output = conv2(input)
            >>> print(output.size())
            torch.Size([1, 256, 16, 16])
            >>> print(conv2.weight.size())
            torch.Size([256, 32, 3, 3])
            >>> print(conv2.padding)
            (1, 0, 1, 0)
            >>> conv3= Conv2d((3,5),64,strides=(1,2),activation=mish, auto_pad=True,use_bias=False,dilation=4,groups=16)
            >>> output = conv3(input)
            >>> print(output.size())
            torch.Size([1, 64, 32, 16])
            >>> print(conv3.weight.size())
            torch.Size([64, 2, 3, 5])
            >>> print(conv3.padding)
            (8, 8, 4, 4)
            >>> input = to_tensor(torch.randn(1,32,608,608))
            >>> conv4= Conv2d((3,3),64,strides=2,activation=mish, auto_pad=True,use_bias=False)
            >>> output = conv4(input)
            >>> print(output.size())
            torch.Size([1, 64, 304, 304])
            >>> print(conv4.padding)
            (1, 1, 1, 1)

        """

    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None,padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str):
            if padding.lower() == 'same':
                auto_pad =True
                padding = None
            elif padding.lower() == 'valid':
                auto_pad = False
                padding =_ntuple(self.rank)(0)
        elif isinstance(padding, int) and padding>0:
            padding = _pair(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            auto_pad = False
            pass
        super(Conv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, **kwargs)

        self.activation = get_activation(activation)
        self.rank=2

    def conv2d_forward(self, x):
        #for backward compatibility
        if len(self.padding)!=len(self.kernel_size)+2:
            self.padding=normalize_padding(self.padding,len(self.kernel_size))
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
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None,padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        rank = 3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str)and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _triple(padding)
        elif isinstance(padding, tuple):
            pass
        super(Conv3d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, **kwargs)

        self.activation = get_activation(activation)

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
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None,padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride', strides))
        dilation = _single(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str)and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _single(padding)
        elif isinstance(padding, tuple):
            pass
        super(TransConv1d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=True, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, **kwargs)

        self.activation = get_activation(activation)
        self.output_padding = _single(0)

    def conv1d_forward(self, x):
        x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)
        if self.padding>0:
            self.output_padding = _single(1)
        return F.conv_transpose1d(x, self.weight, self.bias, self.strides, padding=_single(0),output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv2d(_ConvNd):
    """
    Examples:
        >>> input = to_tensor(torch.randn(1,32,128,128))
        >>> conv1= TransConv2d((3,3),64,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
        >>> output = conv1(input)
        >>> conv1.padding
        (1, 1, 1, 1)
        >>> conv1.output_padding
        (1, 1)
        >>> print(output.size())
        torch.Size([1, 64, 256, 256])

    """
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None,padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str)and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _pair(padding)
        elif isinstance(padding, tuple):
            pass
        super(TransConv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=True, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, **kwargs)

        self.activation = get_activation(activation)
        self.output_padding = _pair(0)

    # def get_padding(self, input_shape):
    #     pad_h = 0
    #     pad_w = 0
    #     if self.auto_pad == True:
    #         ih, iw = list(input_shape)[-2:]
    #         kh, kw = self.kernel_size[-2:]
    #         sh, sw = self.strides[-2:]
    #         dh, dw = self.dilation[-2:]
    #         oh, ow = (ih - 1) * sh + (kh - 1) * dh + 1, (iw - 1) * sw + (kw - 1) * dw + 1
    #         pad_h = max(oh - ih * sh, 0)
    #         pad_w = max(ow - iw * sw, 0)
    #         self.padding = (pad_h, pad_w)
    #         if pad_h != 0 or pad_w != 0:
    #             self.output_padding = (pad_h % 2 if pad_h > 0 else pad_h, pad_w % 2 if pad_w > 0 else pad_w)

    def conv2d_forward(self, x):
        # if len(self.padding) == self.rank:
        #     self.padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        # if self.padding_mode == 'circular':
        #     expanded_padding = (
        #     (self.padding[0] + 1) // 2, self.padding[1] // 2, (self.padding[2] + 1) // 2, self.padding[3] // 2)
        #     x = F.pad(x, expanded_padding, mode='circular')
        # else:
        #     x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv_transpose2d(x, self.weight, self.bias, self.strides, padding=(self.padding[0],self.padding[2]),
                                  output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None,padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        rank = 3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str)and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _triple(padding)
        elif isinstance(padding, tuple):
            pass
        super(TransConv3d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=True, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, **kwargs)

        self.activation = get_activation(activation)
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
    def __init__(self, kernel_size,num_filters=None, depth_multiplier=1, strides=1, auto_pad=True,padding=None, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, **kwargs):
        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride', strides))
        dilation = _single(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str)and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _single(padding)
        elif isinstance(padding, tuple):
            pass
        super(SeparableConv1d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=True, separable=True, **kwargs)

        self.activation = get_activation(activation)
        self.conv1 = None
        self.pointwise = None

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1(x)
        x = self.pointwise(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableConv2d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, depth_multiplier=1, strides=1, auto_pad=True, padding=None,padding_mode='zero',
                 activation=None, use_bias=False, dilation=1, groups=1, name=None, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str)and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _pair(padding)
        elif isinstance(padding, tuple):
            pass
        super(SeparableConv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=True, separable=True, **kwargs)

        self.activation = get_activation(activation)
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
    def __init__(self, kernel_size, num_filters=None,depth_multiplier=1, strides=1, auto_pad=True,padding=None, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, **kwargs):
        rank = 3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str)and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _triple(padding)
        elif isinstance(padding, tuple):
            pass
        super(SeparableConv3d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=True, separable=True, **kwargs)

        self.activation = get_activation(activation)
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
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True,padding=None, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1,  name=None, **kwargs):

        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride', strides))
        dilation = _single(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', kwargs.get('num_filters', None)))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _single(padding)
        elif isinstance(padding, tuple):
            pass
        groups=1
        super(DepthwiseConv1d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias, dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier, depthwise=True, separable=False,
                                              **kwargs)

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
    """
    Applies to create a 2D  Depthwise convolution layer
    Depthwise convolution performs just the first step of a depthwise spatial convolution (which acts on each input channel separately).

     Args:
         kernel_size :(int or tupleof ints)
             shape (spatial extent) of the receptive field

         depth_multiplier:(int , decimal or None, default to None)
             The number of depthwise convolution output filters for each input filters.
             The total number of depthwise convolution output filters will be equal to input_filters * depth_multiplier

         strides:(int or tupleof ints ,default to 1)
              stride of the convolution (increment when sliding the filter over the input)

         auto_pad:bool
             if `False`, then the filter will be shifted over the "valid" area of input, that is,
             no value outside the area is used. If ``pad=True`` means 'same

         *padding (optional)
             auto_pad can help you calculate the pad you need.
             if you have special need , you still can use the paddding
             implicit paddings on both sides of the input. Can be a single number or a double tuple (padH, padW)
             or quadruple(pad_left, pad_right, pad_top, pad_btm )

         padding_mode:string (default is 'zero', available option are 'reflect', 'replicate','constant','circular')

         activation: (None, string, function or Layer)
             activation function after the convolution operation for apply non-linearity.

         use_bias:bool
             the layer will have no bias if `False` is passed here

         dilation:(int or tupleof ints)
             the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1

         groups
             split input into groups, \text{in\_channels}in_channels should be divisible by the number of groups.
             Default: 1


         name
             name of the layer

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

     Examples:
         >>> input = to_tensor(torch.randn(1,32,32,32))
         >>> conv1= DepthwiseConv2d((3,3),depth_multiplier=2,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
         >>> output = conv1(input)
         >>> print(output.size())
         torch.Size([1, 64, 16, 16])
         >>> print(conv1.weight.size())
         torch.Size([64, 1, 3, 3])
         >>> print(conv1.padding)
         (1, 1, 1, 1)
         >>> print(conv1.num_filters)
         64

     """

    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding=None,padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, name=None, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters =kwargs.get('filters', kwargs.get('out_channels', kwargs.get('num_filters', None)))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _pair(padding)
        elif isinstance(padding, tuple):
            pass
        groups=1
        super(DepthwiseConv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias, dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier, depthwise=True, separable=False,
                                              **kwargs)

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
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding=None,padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None,  **kwargs):
        rank = 3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters =kwargs.get('filters', kwargs.get('out_channels', kwargs.get('num_filters', None)))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _triple(padding)
        elif isinstance(padding, tuple):
            pass
        super(DepthwiseConv3d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, **kwargs)

        self.activation = get_activation(activation)
        self._built = False



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


# class MixConv2d(Layer):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
#     def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
#         super(MixConv2d, self).__init__()
#
#         groups = len(k)
#         if method == 'equal_ch':  # equal channels per group
#             i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
#             ch = [(i == g).sum() for g in range(groups)]
#         else:  # 'equal_params': equal parameter count per group
#             b = [out_ch] + [0] * groups
#             a = np.eye(groups + 1, groups, k=-1)
#             a -= np.roll(a, 1, axis=1)
#             a *= np.array(k) ** 2
#             a[0] = 1
#             ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b
#
#         self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch,
#                                           out_channels=ch[g],
#                                           kernel_size=k[g],
#                                           stride=stride,
#                                           padding=k[g] // 2,  # 'same' pad
#                                           dilation=dilation,
#                                           bias=bias) for g in range(groups)])
#
#     def forward(self, x):
#         return torch.cat([m(x) for m in self.m], 1)

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
        Args:
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


class GcdConv2d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, divisor_rank=0, self_norm=True, is_shuffle=False, crossgroup_fusion=True,
                 name=None, depth_multiplier=None, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', kwargs.get('num_filters', None)))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode
        padding = kwargs.get('padding', None)
        if isinstance(padding, str) and auto_pad==False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _pair(padding)
        elif isinstance(padding, tuple):
            pass
        groups = 1
        super(GcdConv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias, dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier, depthwise=True, separable=False,
                                              **kwargs)

        self.activation = get_activation(activation)

        self.norm = None
        if self_norm == True:
            self.self_norm = self_norm
            self.norm = get_normalization('instance')

        self.is_shuffle = is_shuffle
        self.divisor_rank = divisor_rank
        self.crossgroup_fusion = crossgroup_fusion


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




class Lambda(Layer):
    def __init__(self, function, name=None):
        """
        Applies a lambda function on forward()
        Args:
            function (fn): the lambda function
        """
        super(Lambda, self).__init__(name=name)
        self.function = function

    def forward(self, *x):
        return self.function(*x)


class Reshape(Layer):
    def __init__(self, target_shape, name=None):
        """
        Reshape the input volume
        Args:
            *shape (ints): new shape, WITHOUT specifying batch size as first
            dimension, as it will remain unchanged.
        """
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
    """ Self attention Laye"""

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





def _append_coords(input_tensor, with_r=False):
    """
    An alternative implementation for PyTorch with auto-infering the x-y dimensions.
    https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
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





class CoordConv2d(Layer):
    """
    Implementation of the CoordConv modules from https://arxiv.org/abs/1807.03247
    """
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
    """
     .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

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
    

class DropBlock2d(Layer):
    r"""Randomly zeroes spatial blocks of the input tensor.


    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, dropout_rate=0.1, block_size=7):
        super(DropBlock2d, self).__init__()
        self.dropout_rate = dropout_rate
        self.block_size = block_size

    def forward(self, *x):
        x = enforce_singleton(x)
        if not self.training or self.dropout_rate == 0:
            return x
        gamma = self.dropout_rate / self.block_size ** 2
        for sh in x.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(x) * gamma)
        Msum = F.conv2d(M,
                        torch.ones((x.shape[1], 1, self.block_size, self.block_size)).to(device=x.device,dtype=x.dtype),
                        padding=self.block_size // 2,
                        groups=x.shape[1])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=x.device, dtype=x.dtype)
        return x * mask * mask.numel() /mask.sum() #TODO x * mask * self.keep_prob ?



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
