from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn, nn_ops, array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops

from .tensorflow_activations import get_activation
from ..backend.common import *
from ..backend.load_backend import *
from ..backend.tensorflow_backend import Layer, to_numpy, to_tensor, is_tensor, Sequential

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


__all__ = [ 'Dense', 'Flatten', 'Concatenate', 'Concate', 'Add', 'Subtract', 'Conv1d', 'Conv2d', 'Conv3d','TransConv1d',
           'TransConv2d', 'TransConv3d','DepthwiseConv1d','DepthwiseConv2d','DepthwiseConv3d','SeparableConv2d', 'Reshape', 'Dropout', 'Lambda', 'SoftMax', 'Noise']

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
            shape =tf.TensorShape([s.value for s in input_shape.dims] + list(self.num_filters))
            self.weight = tf.Variable(tf.random.normal(shape=shape, mean=0, stddev=1) * 0.02, name='weight')
            if self.use_bias:

                self.bias = tf.Variable(to_tensor(np.zeros((self.num_filters))), name='bias')

            self._built = True


    def forward(self, *x):
        x = enforce_singleton(x)

        x =dot(x, self.weight)
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
        self.keep_output=keep_output
    def build(self, input_shape):
        if self._built == False:
            self._built = True
    def forward(self, *x):
        x = enforce_singleton(x)
        x = tf.reshape(x, [x.get_shape().as_list()[0], -1], dtype=tf.int32)
        return x


class Concate(Layer):
    r"""Concate layer to splice  tensors ."""

    def __init__(self, axis=-1):
        super(Concate, self).__init__()
        self.axis = axis
    def forward(self, *x) ->tf.Tensor:
        if not isinstance(x, (list,tuple)) or len(x) < 2:
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
        x =tf.concat(x, axis=self.axis)
        return x


Concatenate = Concate



class Add(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

    def __init__(self):
        super(Add, self).__init__()

    def build(self, input_shape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, *x) ->tf.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out += item
        return out


class Subtract(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

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
            out -=item
        return out


class Dot(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

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
            out*=item
        return out


class SoftMax(Layer):
    r"""Flatten layer to flatten a tensor after convolution."""

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
            x = tf.math.log(tf.math.softmax(x, -1))
        else:
            x = tf.math.softmax(x, -1)
        return x


class _ConvNd(Layer):
    __constants__ = ['kernel_size', 'num_filters', 'strides', 'auto_pad', 'padding_mode', 'use_bias', 'dilation',
                     'groups', 'transposed']

    def __init__(self, kernel_size, num_filters, strides, auto_pad, padding,padding_mode, use_bias, dilation, groups,
                 transposed, name, depth_multiplier,depthwise=False,separable=False, **kwargs):
        super(_ConvNd, self).__init__(name=name)

        self.num_filters = None
        if num_filters is None :
            self.depth_multiplier = depth_multiplier if depth_multiplier is not None else 1
        else:
            self.num_filters =num_filters
        self.kernel_size = kernel_size
        self.padding = padding# padding if padding is not None else 0in_channel
        self.strides =  strides
        if self.padding is not None:
            self.auto_pad = None
        else:
            self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        self.static_padding = None
        self.dilation = dilation
        self.transposed = transposed
        self.depthwise=depthwise
        self.separable =separable
        if self.separable==True:
            self.depthwise=True
        self.groups = groups



        self.transposed = transposed

        self.use_bias = use_bias

        # self.input_filters = kwargs.get('in_channels', None)
        #         # if self.input_filters is not None:
        #         #     self.build_once(self.input_filters)
        #
        # if self.input_filters is not None and self.input_filters % groups != 0:
        #     raise ValueError('in_channels must be divisible by groups')
        # if self.num_filters % groups != 0:
        #     raise ValueError('out_channels must be divisible by groups')



    def get_padding(self, input_shape):
        pass

    def build(self, input_shape):
        if self._built == False:
            self.input_filters =input_shape.as_list()[-1]

            if (self.depthwise or self.separable) and self.depth_multiplier is None:
                self.depth_multiplier = 1
            if self.num_filters is None and self.depth_multiplier is not None:
                self.num_filters = int(round(self.input_filters * self.depth_multiplier,0))
            if self.input_filters % self.groups != 0:
                raise ValueError('in_channels must be divisible by groups')
            if self.depthwise and self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')
            channel_multiplier=int(self.num_filters// self.groups)   #default channel_multiplier

            if self.depthwise and self.depth_multiplier%1==0:
                self.num_filters= int(round(self.input_filters * self.depth_multiplier,0))
                self.groups=self.input_filters

            self.get_padding(input_shape)
            if self.transposed:
                self.weight = tf.Variable(tf.random.normal(shape=[*self.kernel_size,channel_multiplier ,int(self.input_filters) ], mean=0, stddev=1) * 0.02, name='weight')

            #[filter_height, filter_width, in_channels, out_channels]`
            else:
                self.weight = tf.Variable(tf.random.normal(shape=[*self.kernel_size, int(self.input_filters),channel_multiplier], mean=0, stddev=1) * 0.02, name='weight')

                if self.separable:
                    pointwise_kernel_size=(1,)*len(self.kernel_size)
                    self.pointwise = tf.Variable(tf.random.normal(shape=[*pointwise_kernel_size, int(self.input_filters*channel_multiplier), int(self.num_filters) ], mean=0, stddev=1) * 0.02, name='weight')

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
            s += ', output_shape={0}'.format(self.output_shape if isinstance(self.output_shape, (
            list, tuple)) else self.output_shape.as_list())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(
            state)  # if not hasattr(self, 'padding_mode'):  #     self.padding_mode = 'zeros'


class Conv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters, strides=1,  auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1, depth_multiplier=None, name=None, **kwargs):
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(Conv1d, self).__init__(self,num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero',dilation=dilation,groups=groups, use_bias=use_bias,depth_multiplier=depth_multiplier,transposed=False,
                                      name=name, **kwargs)

    def conv1d_forward(self, x):
        return tf.nn.conv1d(x, filters=self.weight, strides=(1,)+self.strides+(1,), padding=self.padding, data_format="NHWC", dilations=(1,)+self.dilation+(1,), name=self._name)

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
    def __init__(self, kernel_size, num_filters, strides=1,  auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1, depth_multiplier=None, name=None, **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)

        self.activation = get_activation(activation)
        auto_pad = auto_pad
        groups = groups
        super(Conv2d, self).__init__(num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=None, padding_mode='zero',dilation=dilation,groups=groups, use_bias=use_bias,depth_multiplier=depth_multiplier,transposed=False,
                                      name=name, **kwargs)

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

        self.padding= (pad_h // 2,pad_w // 2)

    def conv2d_forward(self, x):
        x=tf.pad(x,  [[0, 0], [self.padding[0],self.padding[0]], [self.padding[1],self.padding[1]], [0, 0]])
        return tf.nn.conv2d(x, filters=self.weight, strides=(1,)+self.strides+(1,),padding='VALID', data_format="NHWC",dilations=(1,)+self.dilation+(1,), name=self._name)


    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, activation=None, use_bias=False, dilation=1,
                 groups=1, depth_multiplier=None, name=None, **kwargs):
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(Conv3d, self).__init__(num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad, padding=padding, padding_mode='zero', dilation=dilation,
                                     groups=groups, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                     transposed=False, name=name, **kwargs)


    def conv3d_forward(self, x):
        return tf.nn.conv3d(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding=self.padding,data_format="NHWC", dilations=(1,)+self.dilation+(1,), name=self._name)

class TransConv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1, depth_multiplier=None, name=None, **kwargs):
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(TransConv1d, self).__init__(num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero', dilation=dilation, groups=groups,
                                      use_bias=use_bias, depth_multiplier=depth_multiplier,
                                     transposed=True, name=name, **kwargs)


    def conv1d_forward(self, x):
        in_shape=x.get_shape().as_list()
        in_shape[1]*=self.strides[0]
        in_shape[-1]=self.num_filters
        return tf.nn.conv1d_transpose(x, filters=self.weight, output_shape=in_shape,strides=(1,)+self.strides+(1,), padding=self.padding, data_format="NHWC",   dilations=(1,)+self.dilation+(1,), name=self._name)


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
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1, depth_multiplier=None, name=None, **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(TransConv2d, self).__init__(num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero', dilation=dilation, groups=groups,
                                      use_bias=use_bias, depth_multiplier=depth_multiplier,
                                     transposed=True, name=name, **kwargs)


    def conv2d_forward(self, x):
        in_shape=x.get_shape().as_list()
        in_shape[1]*=self.strides[0]
        in_shape[2] *= self.strides[1]
        in_shape[3]=self.num_filters
        return tf.nn.conv2d_transpose(x, filters=self.weight, output_shape=in_shape,strides=(1,)+self.strides+(1,), padding=self.padding, data_format="NHWC",
                            dilations=(1,)+self.dilation+(1,), name=self._name)


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
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1, depth_multiplier=None, name=None, **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(TransConv3d, self).__init__(num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero', dilation=dilation, groups=groups,
                                      use_bias=use_bias, depth_multiplier=depth_multiplier,
                                     transposed=True, name=name, **kwargs)
    def conv3d_forward(self, x):
        in_shape=x.get_shape().as_list()
        in_shape[1]*=self.strides[0]
        in_shape[2] *= self.strides[1]
        in_shape[3] *= self.strides[2]
        in_shape[-1]=self.num_filters
        return tf.nn.conv3d_transpose(x, filters=self.weight, output_shape=in_shape,strides=(1,)+self.strides+(1,), padding=self.padding, data_format="NHWC",
                            dilations=(1,)+self.dilation+(1,), name=self._name)


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
    def __init__(self, kernel_size, depth_multiplier=1, strides=1,  auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1,  name=None, **kwargs):
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(DepthwiseConv1d, self).__init__(num_filters=None, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero',dilation=dilation,groups=groups, use_bias=use_bias,depth_multiplier=depth_multiplier,transposed=False,depthwise=True,
                                      name=name, **kwargs)

    def conv1d_forward(self, x):
        return tf.nn.convolution(x, filters=self.weight, strides=(1,)+self.strides+(1,),padding=self.padding, data_format="NHWC",dilations=(1,)+self.dilation+(1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.use_bias:
            x+=self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x

class DepthwiseConv2d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1,  auto_pad=True, activation=None,
                 use_bias=False, dilation=1, name=None, **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'
        super(DepthwiseConv2d, self).__init__(num_filters=None, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero',dilation=dilation, use_bias=use_bias,depth_multiplier=depth_multiplier,transposed=False,depthwise=True,
                                      name=name, **kwargs)

    def conv2d_forward(self, x):
        return tf.nn.depthwise_conv2d(x, filters=self.weight, strides=(1,)+self.strides+(1,),padding=self.padding, data_format="NHWC",dilations=(1,)+self.dilation+(1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x

class DepthwiseConv3d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1,  auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1,  name=None, **kwargs):
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(DepthwiseConv3d, self).__init__(num_filters=None, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero',dilation=dilation,groups=groups, use_bias=use_bias,depth_multiplier=depth_multiplier,transposed=False,depthwise=True,
                                      name=name, **kwargs)

    def conv3d_forward(self, x):
        return tf.nn.convolution(x, filters=self.weight, strides=(1,)+self.strides+(1,),padding=self.padding, data_format="NHWC",dilations=(1,)+self.dilation+(1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv3d_forward(x)
        if self.use_bias:
            x+=self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableConv2d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1,  auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1,  name=None, **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)
        self.activation = get_activation(activation)
        auto_pad = auto_pad
        padding = 'VALID'
        if auto_pad == True:
            padding = 'SAME'

        groups = groups
        super(SeparableConv2d, self).__init__(num_filters=None, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero',dilation=dilation,groups=groups, use_bias=use_bias,depth_multiplier=depth_multiplier,transposed=False,separable=True,
                                      name=name, **kwargs)

    def conv2d_forward(self, x):
        return tf.nn.separable_conv2d(x, depthwise_filter=self.weight,pointwise_filter=self.pointwise, strides=(1,)+self.strides+(1,),padding=self.padding, data_format="NHWC",dilations=(1,)+self.dilation+(1,), name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x





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
        self.target_shape=target_shape

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
    def __init__(self, image,is_recursive=False,name=''):
        super(SingleImageLayer, self).__init__(name=name)
        if isinstance(image,(np.ndarray,tf.Tensor)):
            self.origin_image = to_tensor(image)
            self.input_shape = image.shape[1:]

    def build(self, input_shape):
        if self._built == False:
            self.weight =tf.Variable(to_tensor(self.origin_image.clone()), name='weight')
            self.input_filters = input_shape.get_shape().as_list()[-1]
            self._built = True
    def forward(self,x):
        return expand_dims(self.weight,0)

    def extra_repr(self):
        return 'is_recursive={0}'.format(self.is_recursive)
