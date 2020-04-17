from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import itertools
import math
from itertools import repeat

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


__all__ = ['InputLayer', 'Dense', 'Flatten', 'Concatenate', 'Concate', 'Add', 'Subtract', 'Conv1d', 'Conv2d', 'Conv3d',
           'TransConv2d', 'TransConv3d', 'Reshape', 'Dropout', 'Lambda', 'SoftMax', 'Noise']

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


class InputLayer(tf.keras.layers.InputLayer):
    def __init__(self, input_shape: (list, tuple, int) = None, batch_size=None, name=None, **kwargs):
        if isinstance(input_shape, int):
            input_shape = (input_shape),
        elif isinstance(input_shape, list):
            input_shape = tuple(input_shape)
        super(InputLayer, self).__init__(input_shape=input_shape, batch_size=batch_size, name=name, **kwargs)

    def __repr__(self):
        return get_layer_repr(self)

    def extra_repr(self):
        s = 'input_shape={input_shape},batch_size= {batch_size},name={name}'
        return s.format(**self.__dict__)


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
            shape = [s.value for s in input_shape.dims] + list(self.num_filters)
            self.weight = tf.Variable(tf.random.normal(shape=shape, mean=0, stddev=1) * 0.02, name='weight')
            if self.use_bias:
                self.bias = tf.Variable(tf.zeros([self.num_filters]), name='bias')

            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x = tf.matmul(x, self.weight)
        if self.use_bias:
            x += self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class Flatten(Layer):
    def __init__(self, keep_output=False, name=None, **kwargs):
        super(Flatten, self).__init__()
        self._name = name
        self.keep_output = keep_output

    def build(self, input_shape):
        if self._built == False:
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        x = tf.reshape(x, tf.constant((x.get_shape()[0], -1), dtype=tf.int32))
        return x


class Concate(tf.keras.layers.Concatenate):
    def __init__(self, axis=-1, name=None):
        super(Concate, self).__init__(axis=axis, name=name)

    def __repr__(self):
        return get_layer_repr(self)

    def extra_repr(self):
        return ''


Concatenate = Concate


class Add(tf.keras.layers.Add):
    def __init__(self, name=None):
        super(Add, self).__init__(name=name)

    def __repr__(self):
        return get_layer_repr(self)

    def extra_repr(self):
        return ''


class Subtract(tf.keras.layers.Subtract):
    def __init__(self, name=None):
        super(Subtract, self).__init__(name=name)

    def __repr__(self):
        return get_layer_repr(self)

    def extra_repr(self):
        return ''


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
                 transposed, name, depth_multiplier, **kwargs):
        super(_ConvNd, self).__init__(name=name)

        self.num_filters = None
        if num_filters is None and depth_multiplier is not None:
            self.depth_multiplier = depth_multiplier
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
        self.groups = groups

        if groups != 1 and self.num_filters % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.transposed = transposed

        self.weight = None
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
            if self.num_filters is None and self.depth_multiplier is not None:
                self.num_filters = int(round(self.input_filters * self.depth_multiplier,0))
            if self.input_filters % self.groups != 0:
                raise ValueError('in_channels must be divisible by groups')



            if self.transposed:
                self.weight = tf.Variable(tf.random.normal(shape=[*self.kernel_size,int(self.num_filters) ,int(self.input_filters)// self.groups ], mean=0, stddev=1) * 0.02, name='weight')

            #[filter_height, filter_width, in_channels, out_channels]`
            else:
                self.weight = tf.Variable(tf.random.normal(shape=[*self.kernel_size, int(self.input_filters) // self.groups,int(self.num_filters)], mean=0, stddev=1) * 0.02, name='weight')

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
            s += ', input_shape={0}, input_filters={1}'.format(self._input_shape.clone().tolist(), self.input_filters)
        if self.output_shape is not None:
            s += ', output_shape={0}'.format(self.output_shape if isinstance(self.output_shape, (
            list, tuple)) else self.output_shape.clone().tolist())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(
            state)  # if not hasattr(self, 'padding_mode'):  #     self.padding_mode = 'zeros'


class Conv1d(tf.keras.layers.Conv1D):
    def __init__(self, kernel_size, num_filters, strides, input_shape=None, auto_pad=True, padding_mode='replicate',
                 activation=None, use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, **kwargs):
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)
        self.activation = get_activation(activation)
        if num_filters is None and depth_multiplier is not None:
            num_filters = depth_multiplier
        super(Conv1d, self).__init__(self,kernel_size, num_filters, strides, input_shape=None, auto_pad=True,
                 use_bias=False, dilation=1, groups=1, depth_multiplier=None, name=None, **kwargs,
                                     **kwargs)

    def conv1d_forward(self, x):
        return tf.nn.conv1d(x, filters=self.weight, strides=self.strides, padding=self.padding, data_format="NHWC",
                            dilations=self.dilation, name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv1d_forward(x)
        if self.use_bias:
            x += self.bias
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
    def __init__(self, kernel_size, num_filters, strides,  auto_pad=True, activation=None,
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
        super(Conv2d, self).__init__(num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     auto_pad=auto_pad,padding=padding, padding_mode='zero',dilation=dilation,groups=groups, use_bias=use_bias,depth_multiplier=depth_multiplier,transposed=False,
                                      name=name, **kwargs)

    def conv2d_forward(self, x):

        return tf.nn.conv2d(x, filters=self.weight, strides=(1,)+self.strides+(1,),padding=self.padding, data_format="NHWC",dilations=self.dilation, name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x+=self.bias
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3d(tf.keras.layers.Conv3D):
    def __init__(self, kernel_size, num_filters, strides,auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, filter_rate=None, **kwargs):
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)
        activation = get_activation(activation)
        super(Conv3d, self).__init__(filters=num_filters, kernel_size=kernel_size, strides=strides,
                                     padding='same' if auto_pad else 'valid', dilation_rate=dilation,
                                     activation=activation, use_bias=use_bias, data_format='channels_last',
                                     kernel_initializer=tf.keras.initializers.he_normal(), name=name, **kwargs)
        self.groups = groups

        inp_shape = kwargs.get('input_shape')
        if inp_shape is not None:
            self.input_spec = inp_shape

    @property
    def num_filters(self):
        return super().filters

    @num_filters.setter
    def num_filters(self, value):
        self.filters = _triple(value)

    @property
    def auto_pad(self):
        return super().padding == 'same'

    @auto_pad.setter
    def auto_pad(self, value):
        self.padding = 'same' if value else 'valid'

    @property
    def dilation(self):
        return super().dilation_rate

    @dilation.setter
    def dilation(self, value):
        self.dilation_rate = _triple(value)

    def __repr__(self):
        return get_layer_repr(self)

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
    def __init__(self, kernel_size, num_filters, strides, auto_pad=True, activation=None,
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
                            dilations=self.dilation, name=self._name)

    def forward(self, *x):
        x = enforce_singleton(x)
        x = self.conv2d_forward(x)
        if self.use_bias:
            x += self.bias
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


class TransConv3d(tf.keras.layers.Conv3DTranspose):
    def __init__(self, kernel_size, num_filters, strides, input_shape=None, auto_pad=True, activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, filter_rate=None, **kwargs):
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)
        activation = get_activation(activation)
        super(TransConv3d, self).__init__(filters=num_filters, kernel_size=kernel_size, strides=strides,
                                          input_shape=input_shape, padding='same' if auto_pad else 'valid',
                                          dilation_rate=dilation, activation=activation, use_bias=use_bias,
                                          data_format='channels_last', name=name,
                                          kernel_initializer=tf.keras.initializers.he_normal(), **kwargs)
        self.groups = groups

        @property
        def num_filters(self):
            return super().filters

        @num_filters.setter
        def num_filters(self, value):
            self.filters = _triple(value)

        @property
        def auto_pad(self):
            return super().padding == 'same'

        @auto_pad.setter
        def auto_pad(self, value):
            self.padding = 'same' if value else 'valid'

        @property
        def dilation(self):
            return super().dilation_rate

        @dilation.setter
        def dilation(self, value):
            self.dilation_rate = _triple(value)

    def __repr__(self):
        return get_layer_repr(self)

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
