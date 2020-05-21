from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import builtins
import collections
import inspect
import itertools
import math
from itertools import repeat
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import gen_math_ops, image_ops, math_ops
from tensorflow.python.ops import nn, nn_ops, array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops

from trident.backend.common import *
from trident.backend.load_backend import *
from trident.backend.tensorflow_backend import Layer, Sequential,normalize_padding
from trident.backend.tensorflow_ops import *
from trident.layers.tensorflow_activations import get_activation

_tf_data_format = 'channels_last'


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

__all__ = ['Dense', 'Flatten', 'Concatenate', 'Concate', 'Add', 'Subtract', 'Conv1d', 'Conv2d', 'Conv3d', 'TransConv1d',
           'TransConv2d', 'TransConv3d', 'DepthwiseConv1d', 'DepthwiseConv2d', 'DepthwiseConv3d', 'SeparableConv2d',
           'Upsampling2d', 'Reshape', 'Dropout', 'Lambda', 'SoftMax', 'Noise']

_session = get_session()

_device = 'CPU'
for device in device_lib.list_local_devices():
    if tf.DeviceSpec.from_string(device.name).device_type == 'GPU':
        _device = 'GPU'
        break

_epsilon = _session.epsilon


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


def get_layer_repr(layer):
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    if hasattr(layer, 'extra_repr') and callable(layer.extra_repr):
        extra_repr = layer.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
    child_lines = []
    if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)) and layer.layers is not None:
        for module in layer.layers:
            mod_str = repr(module)
            mod_str = addindent(mod_str, 2)
            child_lines.append('(' + module.name + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = layer.__class__.__name__ + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str


class Dense(Layer):
    def __init__(self, num_filters, use_bias=True, activation=None, keep_output=False, name=None, **kwargs):
        super(Dense, self).__init__()
        if isinstance(num_filters, int):
            self.num_filters = _single(num_filters)
        elif isinstance(num_filters, list):
            self.num_filters = tuple(num_filters)
        elif isinstance(num_filters, tuple):
            self.num_filters = num_filters
        else:
            raise ValueError('output_shape should be integer, list of integer or tuple of integer...')
        self._name = name
        self.weight = None
        self.bias = None
        self.use_bias = use_bias
        self.activation = get_activation(activation)
        self.keep_output = keep_output

    def build(self, input_shape):
        if self._built == False:
            shape = tf.TensorShape([s.value for s in input_shape.dims] + list(self.num_filters))
            self.weight = tf.Variable(tf.random.normal(shape=shape, mean=0, stddev=1) * 0.02, name='weight')
            if self.use_bias:
                self.bias = tf.Variable(to_tensor(np.zeros((self.num_filters))), name='bias')

            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)

        x = dot(x, self.weight)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x


class Flatten(Layer):
    def __init__(self, keep_output=False, name=None, **kwargs):
        super(Flatten, self).__init__()
        self._name = name
        self.keep_output = keep_output
        self.keep_output = keep_output

    def build(self, input_shape):
        if self._built == False:
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x = tf.reshape(x, [x.get_shape().as_list()[0], -1])
        return x


class Concate(Layer):
    """Concate layer to splice  tensors ."""

    def __init__(self, axis=-1):
        super(Concate, self).__init__()
        self.axis = axis

    def forward(self, *x) -> tf.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError('A `Concatenate` layer should be called on a list of at least 2 tensor  inputs')

        if all([k.size() is None for k in x]):
            return

        reduced_inputs_shapes = [k.get_shape().as_list() for k in x]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError(
                'A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got inputs '
                'shapes: %s' % shape_set)
        x = tf.concat(x, axis=self.axis)
        return x


Concatenate = Concate


class Add(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self):
        super(Add, self).__init__()

    def build(self, input_shape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, *x) -> tf.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out += item
        return out


class Subtract(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self):
        super(Subtract, self).__init__()

    def build(self, input_shape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, *x) -> tf.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out -= item
        return out


class Dot(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=1):
        super(Dot, self).__init__()

    def build(self, input_shape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, *x) -> tf.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out *= item
        return out


class SoftMax(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=-1, add_noise=False, noise_intensity=0.005, name=None, **kwargs):
        super(SoftMax, self).__init__(name=name)
        self.axis = axis
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity

    def forward(self, *x):
        x = enforce_singleton(x)
        if not hasattr(self, 'add_noise'):
            self.add_noise = False
            self.noise_intensity = 0.005
        if self.training:
            if self.add_noise == True:
                noise = self.noise_intensity * tf.random.normal(shape=x.get_shape(), mean=1, stddev=1)
                x = x + noise
            x = tf.math.log(tf.math.softmax(x, self.axis))
        else:
            x = tf.math.softmax(x, self.axis)
        return x


# def get_static_padding(rank,input_shape,kernal_shape,strides,dilations):
#     if isinstance(strides,int):
#         strides= _ntuple(rank)(strides)
#     if isinstance(dilations,int):
#         dilations= _ntuple(rank)(dilations)
#
#     input_shape=to_numpy(input_shape)[-rank:]
#     kernal_shape=to_numpy(list(kernal_shape))[-rank:]
#     strides = to_numpy(list(strides))[-rank:]
#     dilations= to_numpy(list(dilations))[-rank:]
#     output_shape=np.ceil(input_shape/strides)
#
#     raw_padding=np.clip((output_shape-1)*strides+(kernal_shape-1)*dilations+1-input_shape,0,None)
#     remainder=np.remainder(raw_padding,np.ones_like(raw_padding)*2)
#
#     raw_padding=raw_padding+(remainder*np.greater(strides,1).astype(np.float32))
#     lefttop_pad = np.ceil(raw_padding/2.0).astype(np.int32)
#     rightbtm_pad=(raw_padding-lefttop_pad).astype(np.int32)
#     static_padding = []
#     for k in range(rank):
#         static_padding.append(lefttop_pad[-1-k])
#         static_padding.append(rightbtm_pad[-1-k])
#     return static_padding

def get_static_padding(rank, kernal_shape, strides, dilations, input_shape=None):
    """ Calcualte the actual padding we need in different rank and different convlution settings.

    Args:
        rank (int):
        kernal_shape (tuple of integer):
        strides (tuple of integer):
        dilations (tuple of integer):
        input_shape (None or tuple of integer):

    Returns: the padding we need (shape: (rank,2) )

    Examples
    >>> get_static_padding(1,(3,),(2,),(2,))
    ((2, 2),)
    >>> get_static_padding(2,(3,3),(2,2),(1,1),(224,224))
    ((1, 1), (1, 1))
    >>> get_static_padding(2,(5,5),(1,1),(2,2))
    ((4, 4), (4, 4))
    >>> get_static_padding(4,(1,5,5,1),(1,1,1,1),(1,1,1,1))
    ((0, 0), (2, 2), (2, 2), (0, 0))
    >>> get_static_padding(2,(2,2),(1,1),(1,1))
    ((1, 0), (1, 0))
    >>> get_static_padding(3,(5,5,5),(1,1,1),(2,2,2))
    ((4, 4), (4, 4), (4, 4))
    """
    if input_shape is None:
        input_shape = [224] * rank
    if isinstance(kernal_shape, int):
        kernal_shape = _ntuple(rank)(kernal_shape)
    if isinstance(strides, int):
        strides = _ntuple(rank)(strides)
    if isinstance(dilations, int):
        dilations = _ntuple(rank)(dilations)

    input_shape = to_numpy(input_shape)
    kernal_shape = to_numpy(list(kernal_shape))
    strides = to_numpy(list(strides)).astype(np.float32)
    dilations = to_numpy(list(dilations))

    output_shape = np.ceil(input_shape / strides)
    raw_padding = np.clip((output_shape - 1) * strides + (kernal_shape - 1) * dilations + 1 - input_shape, a_min=0,
                          a_max=np.inf)
    remainder = np.remainder(raw_padding, np.ones_like(raw_padding) * 2)

    raw_padding = raw_padding + (remainder * np.greater(strides, 1).astype(np.float32))
    lefttop_pad = np.ceil(raw_padding / 2.0).astype(np.int32)
    rightbtm_pad = (raw_padding - lefttop_pad).astype(np.int32)
    static_padding = np.concatenate([np.expand_dims(lefttop_pad, -1), np.expand_dims(rightbtm_pad, -1)], -1)

    return tuple([(static_padding[i, 0], static_padding[i, 1]) for i in range(rank)])


class _ConvNd(Layer):
    __constants__ = ['kernel_size', 'num_filters', 'strides', 'auto_pad', 'padding_mode', 'use_bias', 'dilation',
                     'groups', 'transposed']

    def __init__(self, rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias, dilation,
                 groups=1, transposed=False, name=None, depth_multiplier=1, depthwise=False, separable=False, **kwargs):
        super(_ConvNd, self).__init__(name=name)
        self.rank = rank
        self.num_filters = num_filters
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.transposed = transposed
        self.groups = groups
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        if padding is not None:
            self.padding = normalize_padding(padding, rank)
        else:
            self.padding = None

        self.depthwise = depthwise
        self.separable = separable
        if self.separable == True:
            self.depthwise = True

        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

        self.transposed = transposed
        self.use_bias = use_bias

    def build(self, input_shape):
        if self._built == False:
            self.input_filters = input_shape.as_list()[-1]
            if self.depthwise or self.separable:
                if self.depth_multiplier is None:
                    self.depth_multiplier = 1
                    # ex. self.depth_multiplier=0.5  input 128==>output 64  groups=64
                    # ex. self.depth_multiplier=2  input 64==>output 128  groups=64
                self.groups = int(builtins.round(self.input_filters * builtins.min(builtins.max(self.depth_multiplier, 0), 1), 0))


            if self.num_filters is None and self.depth_multiplier is not None:
                self.num_filters=int(builtins.round(self.input_filters* self.depth_multiplier))

            if self.groups != 1 and self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            if self.depthwise and self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            if self.auto_pad:
                self.padding = get_static_padding(self.rank, self.kernel_size ,
                                                   self.strides ,  self.dilation ,
                                                   tuple(input_shape.as_list()[:-1]))
            else:
                self.padding = normalize_padding(self.padding, self.rank)



                # elif self.depth_multiplier < 1:
                #
                #     self.groups = int(builtins.round(self.input_filters * self.depth_multiplier, 0))
                #     self.num_filters=int(self.groups)
                # else:




            channel_multiplier = int(self.num_filters // self.groups) if self.groups>1 else int(self.num_filters)#if self.depth_multiplier is None else self.depth_multiplier  # default channel_multiplier

            if self.transposed:
                # filter_height, filter_width,  out_channels in_channels,
                self.weight = tf.Variable(
                    tf.random.normal(shape=[*self.kernel_size, int(channel_multiplier), int(self.input_filters)], mean=0,  stddev=1) * 0.02, name='weight')
            else:

                # [filter_height, filter_width, in_channels, out_channels]`
                self.weight = tf.Variable(  tf.random.normal(shape=[*self.kernel_size, int(self.input_filters), int(channel_multiplier)], mean=0,    stddev=1) * 0.02, name='weight')

                if self.separable:
                    pointwise_kernel_size = (1,) * len(self.kernel_size)
                    self.pointwise = tf.Variable(tf.random.normal(
                        shape=[*pointwise_kernel_size, int(self.input_filters * channel_multiplier),int(self.num_filters)], mean=0, stddev=1) * 0.02, name='weight')

            if self.use_bias:
                self.bias = tf.Variable(tf.zeros([int(self.num_filters)]), name='bias')
            else:
                self.register_parameter('bias', None)
            self._built = True

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={auto_pad},use_bias={use_bias} ,dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self._input_shape is not None:
            s += ', input_shape={0}, input_filters={1}'.format(self._input_shape.as_list(), self.input_filters)
        if self.output_shape is not None:
            s += ', output_shape={0}'.format(
                self.output_shape if isinstance(self.output_shape, (list, tuple)) else self.output_shape.as_list())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(
            state)  # if not hasattr(self, 'padding_mode'):  #     self.padding_mode = 'zeros'


class Conv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None, use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride', strides))
        dilation = _single(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
            auto_pad = False
        elif isinstance(padding, int):
            padding = _single(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            pass
        super(Conv1d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, **kwargs)

        self.activation = get_activation(activation)
        self.rank=1

    def conv1d_forward(self, x):
        x = tf.pad(x, [[0], [self.padding[0]], [self.padding[1]], [0]])
        return tf.nn.conv1d(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding='VALID',
                            data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, num_filters={num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], tf.keras.layers.Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={0}'.format(self.padding == 'same') + ',use_bias={use_bias} ,dilation={dilation_rate}'
        if self.groups != 1:
            s += ', groups={groups}'

        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class Conv2d(_ConvNd):
    """
    Applies to create a 2D convolution layer

    Args:
        kernel_size :(int or tupleof ints)
            shape (spatial extent) of the receptive field

        num_filters :(int  or None, default to None)
            number of output channel (filters), sometimes in backbond design output channel is propotional
            to input channel, so setting depth_multiplier instead of num_filters (num_filters=depth_multiplier*input_filters).

        strides:(int or tupleof ints ,default to 1)
             strides of the convolution (increment when sliding the filter over the input)

        auto_pad:bool
            if `False`, then the filter will be shifted over the "valid" area of input, that is,
            no value outside the area is used. If ``pad=True`` means 'same

        *padding (optional)
            auto_pad can help you calculate the pad you need.
            if you have special need , you still can use the paddding
            implicit paddings on both sides of the input. Can be a single number or a double tuple (padH, padW)
            or quadruple(pad_left, pad_right, pad_top, pad_btm )

        padding_mode:string (default is 'zero', available option are 'reflect', 'replicate','constant',
        'circular')
        mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)

        activation: (None, string, function or Layer)
            activation function after the convolution operation for apply non-linearity.

        use_bias:bool
            the layer will have no bias if `False` is passed here

        dilation:(int or tupleof ints)
            the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1

        groups
            split input into groups, \text{in\_channels}in_channels should be divisible by the number of
            groups. Default: 1
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

    Examples::
    >>> input =to_tensor(np.random.random((1,128,128,32)))
    >>> conv1= Conv2d((3,3),64,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
    >>> output = conv1(input)
    >>> print(output.shape)
    (1, 64, 64, 64)
    >>> print(conv1.weight.shape)
    (3, 3, 32, 64)
    >>> print(conv1.padding)
    [1, 1, 1, 1]
    >>> conv2= Conv2d((3, 3), 256, strides=(2, 2), auto_pad=False, padding=((1, 0), (1, 0)))
    >>> output = conv2(input)
    >>> print(output.shape)
    (1, 64, 64, 256)
    >>> print(conv2.weight.shape)
    (3, 3, 32, 256)
    >>> print(conv2.padding)
    (1, 0, 1, 0)
    >>> conv3= Conv2d((3,5),64,strides=(1,2),activation=mish, auto_pad=True,use_bias=False,dilation=4,groups=16)
    >>> output = conv3(input)
    >>> print(output.shape)
    (1, 136, 60, 4)
    >>> print(conv3.weight.shape)
    (3, 5, 32, 4)
    >>> print(conv3.padding)
    [8, 8, 4, 4]
    >>> input = to_tensor(np.random.random((1,37,37,32)))
    >>> conv4= Conv2d((3,3),64,strides=2,activation=mish, auto_pad=True,use_bias=False)
    >>> output = conv4(input)
    >>> print(output.shape)
    (1, 19, 19, 64)

    """

    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None, use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):

        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode
        if isinstance(padding, str):
            if padding.lower() == 'same':
                auto_pad = True
                padding = None
            elif padding.lower() == 'valid':
                auto_pad = False
                padding = _ntuple(self.rank)(0)
        elif isinstance(padding, int) and padding > 0:
            padding = _pair(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            auto_pad = False
            pass

        super(Conv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, **kwargs)

        self.activation = get_activation(activation)
        self.rank = 2


    def conv2d_forward(self, x):
        if self.auto_pad == True and len(self.padding) == self.rank + 2:
            x = tf.pad(x, self.padding, mode='CONSTANT')
        else:
            padlist = list(self.padding)
            padlist.insert(0, (0, 0))
            padlist.append((0, 0))

            x = tf.pad(x, tuple(padlist), mode='CONSTANT')

        return tf.nn.conv2d(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding='VALID',
                            data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False, dilation=1,
                 groups=1, depth_multiplier=None, name=None, **kwargs):
        rank=3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode
        if isinstance(padding, str):
            if padding.lower() == 'same':
                auto_pad = True
                padding = None
            elif padding.lower() == 'valid':
                auto_pad = False
                padding = _ntuple(self.rank)(0)
        elif isinstance(padding, int) and padding > 0:
            padding = _triple(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            auto_pad = False
            pass

        super(Conv3d, self).__init__(rank,num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad, padding=padding, padding_mode='zero', dilation=dilation,
                                     groups=groups, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                     transposed=False, name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank = 3

    def conv3d_forward(self, x):
        return tf.nn.conv3d(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding='VALID',
                            data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)


class TransConv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False, dilation=1,
                 groups=1, depth_multiplier=None, name=None, **kwargs):
        rank=1
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)

        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(TransConv1d, self).__init__(rank,num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                          auto_pad=auto_pad, padding=padding, padding_mode='zero', dilation=dilation,
                                          groups=groups, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                          transposed=True, name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank = 1

    def conv1d_forward(self, x):
        in_shape = x.get_shape().as_list()
        in_shape[1] *= self.strides[0]
        in_shape[-1] = self.num_filters
        return tf.nn.conv1d_transpose(x, filters=self.weight, output_shape=in_shape, strides=(1,) + self.strides + (1,),
                                      padding=self.padding, data_format="NHWC", dilations=(1,) + self.dilation + (1,),
                                      name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], tf.keras.layers.Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={0}'.format(self.padding == 'same') + ',use_bias={use_bias} ,dilation={dilation_rate}'
        if self.groups != 1:
            s += ', groups={groups}'

        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class TransConv2d(_ConvNd):
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False, dilation=1,
                 groups=1, depth_multiplier=None, name=None, **kwargs):
        rank=2
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)

        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(TransConv2d, self).__init__(rank,num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                          auto_pad=auto_pad, padding=padding, padding_mode='zero', dilation=dilation,
                                          groups=groups, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                          transposed=True, name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank=2

    def conv2d_forward(self, x):
        in_shape = x.get_shape().as_list()
        in_shape[1] *= self.strides[0]
        in_shape[2] *= self.strides[1]
        in_shape[3] = self.num_filters
        return tf.nn.conv2d_transpose(x, filters=self.weight, output_shape=in_shape, strides=(1,) + self.strides + (1,),
                                      padding=self.padding, data_format="NHWC", dilations=(1,) + self.dilation + (1,),
                                      name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], tf.keras.layers.Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={0}'.format(self.padding == 'same') + ',use_bias={use_bias} ,dilation={dilation_rate}'
        if self.groups != 1:
            s += ', groups={groups}'

        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class TransConv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False, dilation=1,
                 groups=1, depth_multiplier=None, name=None, **kwargs):
        rank=3
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)

        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(TransConv3d, self).__init__(rank,num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                          auto_pad=auto_pad, padding=padding, padding_mode='zero', dilation=dilation,
                                          groups=groups, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                          transposed=True, name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank=3

    def conv3d_forward(self, x):
        in_shape = x.get_shape().as_list()
        in_shape[1] *= self.strides[0]
        in_shape[2] *= self.strides[1]
        in_shape[3] *= self.strides[2]
        in_shape[-1] = self.num_filters
        return tf.nn.conv3d_transpose(x, filters=self.weight, output_shape=in_shape, strides=(1,) + self.strides + (1,),
                                      padding=self.padding, data_format="NHWC", dilations=(1,) + self.dilation + (1,),
                                      name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv3d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], tf.keras.layers.Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={0}'.format(self.padding == 'same') + ',use_bias={use_bias} ,dilation={dilation_rate}'
        if self.groups != 1:
            s += ', groups={groups}'

        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class DepthwiseConv1d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False,
                 dilation=1, groups=1, name=None, **kwargs):
        rank=1
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)

        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(DepthwiseConv1d, self).__init__(rank,num_filters=None, kernel_size=kernel_size, strides=strides,
                                              auto_pad=auto_pad, padding=padding, padding_mode='zero',
                                              dilation=dilation, groups=groups, use_bias=use_bias,
                                              depth_multiplier=depth_multiplier, transposed=False, depthwise=True,
                                              name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank=1

    def conv1d_forward(self, x):
        return tf.nn.convolution(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding=self.padding,
                                 data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.use_bias:
            x += self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseConv2d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1,auto_pad=True, padding=None, padding_mode='zero',activation=None, use_bias=False,
                 dilation=1, name=None, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode
        if isinstance(padding, str):
            if padding.lower() == 'same':
                auto_pad = True
                padding = None
            elif padding.lower() == 'valid':
                auto_pad = False
                padding = _ntuple(self.rank)(0)
        elif isinstance(padding, int) and padding > 0:
            padding = _pair(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            auto_pad = False
            pass
        super(DepthwiseConv2d, self).__init__(rank,num_filters=None, kernel_size=kernel_size, strides=strides,
                                              auto_pad=auto_pad, padding=padding, padding_mode='zero',
                                              dilation=dilation, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                              groups=None,transposed=False, depthwise=True, name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank=2

    def conv2d_forward(self, x):
        if self.auto_pad == True and len(self.padding) == self.rank + 2:
            x = tf.pad(x, self.padding, mode='CONSTANT')
        else:
            padlist = list(self.padding)
            padlist.insert(0, (0, 0))
            padlist.append((0, 0))

            x = tf.pad(x, tuple(padlist), mode='CONSTANT')

        return tf.nn.depthwise_conv2d(x,filter=self.weight, strides=(1,) + self.strides + (1,), padding='VALID', data_format="NHWC", dilations= self.dilation, name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseConv3d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False,
                 dilation=1, groups=1, name=None, **kwargs):
        rank=3
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)

        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(DepthwiseConv3d, self).__init__(rank,num_filters=None, kernel_size=kernel_size, strides=strides,
                                              auto_pad=auto_pad, padding=padding, padding_mode='zero',
                                              dilation=dilation, groups=groups, use_bias=use_bias,
                                              depth_multiplier=depth_multiplier, transposed=False, depthwise=True,
                                              name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank=3

    def conv3d_forward(self, x):
        return tf.nn.convolution(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding='VALID',
                                 data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv3d_forward(x)
        if self.use_bias:
            x += self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableConv2d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1,auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False,
                 dilation=1, groups=1, name=None, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode
        if isinstance(padding, str):
            if padding.lower() == 'same':
                auto_pad = True
                padding = None
            elif padding.lower() == 'valid':
                auto_pad = False
                padding = _ntuple(self.rank)(0)
        elif isinstance(padding, int) and padding > 0:
            padding = _pair(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            auto_pad = False
            pass
        super(SeparableConv2d, self).__init__(rank,num_filters=None, kernel_size=kernel_size, strides=strides,
                                              auto_pad=auto_pad, padding=padding, padding_mode='zero',
                                              dilation=dilation, groups=groups, use_bias=use_bias,
                                              depth_multiplier=depth_multiplier, transposed=False, separable=True,
                                              name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank=2

    def conv2d_forward(self, x):
        if self.auto_pad == True and len(self.padding) == self.rank + 2:
            x = tf.pad(x, self.padding, mode='CONSTANT')
        else:
            padlist = list(self.padding)
            padlist.insert(0, (0, 0))
            padlist.append((0, 0))

            x = tf.pad(x, tuple(padlist), mode='CONSTANT')

        return tf.nn.separable_conv2d(x, depthwise_filter=self.weight, pointwise_filter=self.pointwise,
                                      strides=(1,) + self.strides + (1,), padding='VALID', data_format="NHWC",
                                      dilations=(1,) + self.dilation + (1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x


class Upsampling2d(Layer):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True, keep_output=False, name=None):
        super(Upsampling2d, self).__init__(name=name)
        if mode not in {'nearest', 'bilinear', 'area', 'pixel_shuffle'}:
            raise ValueError('`mode` argument should be one of `"nearest"` '
                             'or `"bilinear"`.')
        self.rank = 2
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = scale_factor
        else:
            self.scale_factor = (float(scale_factor), float(scale_factor))
        self.mode = mode
        self.align_corners = align_corners
        self.keep_output = keep_output

    def forward(self, *x):
        x = enforce_singleton(x)
        new_shape = x.shape.as_list()[1:-1]

        if self.scale_factor is not None and isinstance(self.scale_factor, tuple):
            new_shape[0] = int(new_shape[0] * self.scale_factor[0])
            new_shape[1] = int(new_shape[1] * self.scale_factor[1])
        new_shape = to_tensor(new_shape, dtype=tf.int32)
        if self.mode == 'pixel_shuffle':
            return tf.nn.depth_to_space(x, int(self.scale_factor[0]))
        elif self.mode == 'nearest':
            return image_ops.resize_images_v2(x, new_shape, method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
        elif self.mode == 'bilinear':
            return image_ops.resize_images_v2(x, new_shape, method=image_ops.ResizeMethod.BILINEAR)
        elif self.mode == 'area':
            return image_ops.resize_images_v2(x, new_shape, method=image_ops.ResizeMethod.AREA)
        else:
            return image_ops.resize_images_v2(x, new_shape, method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class Lambda(Layer):
    """
    Applies a lambda function on forward()
    Args:
        lamb (fn): the lambda function
    """

    def __init__(self, function, name=''):
        super(Lambda, self).__init__(name=name)
        self.function = function

    def build(self, input_shape):
        if self._built == False:
            self._built = True

    def forward(self, *x):
        return self.function(*x)

    def extra_repr(self):
        s = 'function={0}'.format("".join(inspect.getsourcelines(self.function)[0]))


class Reshape(Layer):
    def __init__(self, target_shape, name=None, **kwargs):
        super(Reshape, self).__init__(name=name)
        self.target_shape = target_shape

    def build(self, input_shape):
        if self._built == False:
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x = tf.reshape(x, tf.constant((x.get_shape()[0], *self.target_shape), dtype=tf.int32))
        return x

    def extra_repr(self):
        s = 'target_shape={0}'.format(self.target_shape)


class Dropout(Layer):
    def __init__(self, dropout_rate=0, keep_output=False, name=None, **kwargs):
        super(Dropout, self).__init__(name=name)
        self._name = name
        self.dropout_rate = dropout_rate
        self.keep_output = keep_output

    def build(self, input_shape):
        if self._built == False:
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.training:
            x = tf.nn.dropout(x, self.dropout_rate)
        return x

    def extra_repr(self):
        s = 'dropout_rate={0}'.format(self.dropout_rate)


class Noise(tf.keras.layers.GaussianNoise):
    def __init__(self, stddev=0.1, name=None):
        super(Noise, self).__init__(stddev=stddev, name=name)

    def __repr__(self):
        return get_layer_repr(self)

    def extra_repr(self):
        s = 'stddev={0}'.format(self.stddev)


class SingleImageLayer(Layer):
    def __init__(self, image, is_recursive=False, name=''):
        super(SingleImageLayer, self).__init__(name=name)
        if isinstance(image, (np.ndarray, tf.Tensor)):
            self.origin_image = to_tensor(image)
            self.input_shape = image.shape[1:]

    def build(self, input_shape):
        if self._built == False:
            self.weight = tf.Variable(to_tensor(self.origin_image.clone()), name='weight')
            self.input_filters = input_shape.get_shape().as_list()[-1]
            self._built = True

    def forward(self, x):
        return expand_dims(self.weight, 0)

    def extra_repr(self):
        return 'is_recursive={0}'.format(self.is_recursive)
