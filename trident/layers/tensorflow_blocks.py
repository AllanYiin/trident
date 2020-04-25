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
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import image_ops
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import layer_utils as trackable_layer_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import keras_export

from .tensorflow_activations import get_activation, Identity
from .tensorflow_layers import *
from .tensorflow_normalizations import get_normalization
from .tensorflow_pooling import get_pooling, GlobalAvgPool2d
from ..backend.common import *
from ..backend.tensorflow_backend import *
from ..layers.tensorflow_layers import *

_tf_data_format = 'channels_last'

__all__ = ['Conv2d_Block', 'TransConv2d_Block','DepthwiseConv2d_Block','SeparableConv2d_Block', 'ShortCut2d','SqueezeExcite','For']

_session = get_session()


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

#
# class Conv1d_Block(tf.keras.Sequential):
#     def __init__(self, kernel_size=(3), num_filters=32, strides=1, input_shape=None, auto_pad=True,
#                  activation='leaky_relu', normalization=None, use_bias=False, dilation=1, groups=1, add_noise=False,
#                  noise_intensity=0.001, dropout_rate=0, name=None, **kwargs):
#         super(Conv1d_Block, self).__init__(name=name)
#         if add_noise:
#             noise = tf.keras.layers.GaussianNoise(noise_intensity)
#             self.add(noise)
#         self._conv = Conv1d(kernel_size=kernel_size, num_filters=num_filters, strides=strides, input_shape=input_shape,
#                             auto_pad=auto_pad, activation=None, use_bias=use_bias, dilation=dilation, groups=groups)
#         self.add(self._conv)
#
#         self.norm = get_normalization(normalization)
#         if self.norm is not None:
#             self.add(self.norm)
#
#         self.activation = get_activation(snake2camel(activation))
#         if self.activation is not None:
#             self.add(self.activation)
#         if dropout_rate > 0:
#             self.drop = Dropout(dropout_rate)
#             self.add(self.drop)
#


class Conv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None, **kwargs):
        super(Conv2d_Block, self).__init__()
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
            self.conv= Conv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                              auto_pad=self.auto_pad, activation=None,
                              use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name,
                              depth_multiplier=self.depth_multiplier)
            self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None



    def build(self, input_shape):
        if self._built == False:
            conv = Conv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                          auto_pad=self.auto_pad, activation=None,
                          use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name,
                          depth_multiplier=self.depth_multiplier)
            #conv.input_shape = input_shape

            if self.use_spectral:
                #self.conv = nn.utils.spectral_norm(conv)
                self.norm=None
            else:
                self.conv = conv
            self._built=True

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1)
            x +=noise
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0 :
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


#
# class Conv3d_Block(tf.keras.Sequential):
#     def __init__(self, kernel_size=(3, 3, 3), num_filters=32, strides=1, input_shape=None, auto_pad=True,
#                  activation='leaky_relu', normalization=None, use_bias=False, dilation=1, groups=1, add_noise=False,
#                  noise_intensity=0.001, dropout_rate=0, name=None, **kwargs):
#         super(Conv3d_Block, self).__init__(name=name)
#         if add_noise:
#             noise = tf.keras.layers.GaussianNoise(noise_intensity)
#             self.add(noise)
#         self._conv = Conv3d(kernel_size=kernel_size, num_filters=num_filters, strides=strides, input_shape=input_shape,
#                             auto_pad=auto_pad, activation=None, use_bias=use_bias, dilation=dilation, groups=groups)
#         self.add(self._conv)
#
#         self.norm = get_normalization(normalization)
#         if self.norm is not None:
#             self.add(self.norm)
#
#         self.activation = get_activation(snake2camel(activation))
#         if self.activation is not None:
#             self.add(self.activation)
#         if dropout_rate > 0:
#             self.drop = Dropout(dropout_rate)
#             self.add(self.drop)
#
#     @property
#     def conv(self):
#         return self._conv
#
#     @conv.setter
#     def conv(self, value):
#         self._conv = value


#
# class TransConv1d_Block(Sequential):
#     def __init__(self, kernel_size=(3), num_filters=32, strides=1, auto_pad=True,activation='leaky_relu',
#     normalization=None,  use_bias=False,dilation=1, groups=1,add_noise=False,noise_intensity=0.001,dropout_rate=0,
#     **kwargs ):
#         super(TransConv1d_Block, self).__init__()
#         if add_noise:
#             noise = tf.keras.layers.GaussianNoise(noise_intensity)
#             self.add(noise)
#         self._conv = TransConv1d(kernel_size=kernel_size, num_filters=num_filters, strides=strides, auto_pad=auto_pad,
#                       activation=None, use_bias=use_bias, dilation=dilation, groups=groups)
#         self.add(self._conv)
#
#         self.norm = get_normalization(normalization)
#         if self.norm is not None:
#             self.add(self.norm)
#
#         self.activation = get_activation(activation)
#         if self.activation is not None:
#             self.add(activation)
#         if dropout_rate > 0:
#             self.drop = Dropout(dropout_rate)
#             self.add(self.drop)
#     @property
#     def conv(self):
#         return self._conv
#     @conv.setter
#     def conv(self,value):
#         self._conv=value
#
#
#     def __repr__(self):
#         return get_layer_repr(self)



class TransConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None, **kwargs):
        super(TransConv2d_Block, self).__init__()
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
            self.conv= TransConv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                              auto_pad=self.auto_pad, activation=None,
                              use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name,
                              depth_multiplier=self.depth_multiplier)
            self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None



    def build(self, input_shape):
        if self._built == False:
            conv = TransConv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                          auto_pad=self.auto_pad, activation=None,
                          use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name,
                          depth_multiplier=self.depth_multiplier)
            #conv.input_shape = input_shape

            if self.use_spectral:
                #self.conv = nn.utils.spectral_norm(conv)
                self.norm=None
            else:
                self.conv = conv
            self._built=True

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1)
            x +=noise
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0 :
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


class TransConv3d_Block(tf.keras.Sequential):
    def __init__(self, kernel_size=(3, 3, 3), num_filters=32, strides=1, input_shape=None, auto_pad=True,
                 activation='leaky_relu', normalization=None, use_bias=False, dilation=1, groups=1, add_noise=False,
                 noise_intensity=0.001, dropout_rate=0, name=None, **kwargs):
        super(TransConv3d_Block, self).__init__(name=name)
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

    @property
    def conv(self):
        return self._conv

    @conv.setter
    def conv(self, value):
        self._conv = value


class DepthwiseConv2d_Block(Layer):
    def __init__(self, kernel_size=(3,3),depth_multiplier=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None,  **kwargs):
        super(DepthwiseConv2d_Block, self).__init__()
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
            self.conv= DepthwiseConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier, strides=self.strides,
                              auto_pad=self.auto_pad, activation=None,
                              use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name)
            self.norm = get_normalization(normalization)
            self.conv=None
        self.activation = get_activation(activation)
        self.droupout = None



    def build(self, input_shape):
        if self._built == False:
            conv = DepthwiseConv2d(kernel_size=self.kernel_size,depth_multiplier=self.depth_multiplier, strides=self.strides,
                          auto_pad=self.auto_pad, activation=None,
                          use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name)
            #conv.input_shape = input_shape

            if self.use_spectral:
                #self.conv = nn.utils.spectral_norm(conv)
                self.norm=None
            else:
                self.conv = conv
            self._built=True

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1)
            x +=noise
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0 :
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

class SeparableConv2d_Block(Layer):
    def __init__(self, kernel_size=(3,3),depth_multiplier=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None,  **kwargs):
        super(SeparableConv2d_Block, self).__init__()
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
            self.conv= SeparableConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier, strides=self.strides,
                              auto_pad=self.auto_pad, activation=None,
                              use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name)
            self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None



    def build(self, input_shape):
        if self._built == False:
            conv = SeparableConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier, strides=self.strides,
                          auto_pad=self.auto_pad, activation=None,
                          use_bias=self.use_bias, dilation=self.dilation, groups=self.groups, name=self._name
                          )
            #conv.input_shape = input_shape

            if self.use_spectral:
                #self.conv = nn.utils.spectral_norm(conv)
                self.norm=None
            else:
                self.conv = conv
            self._built=True

    def forward(self, *x):
        x = enforce_singleton(x)
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1)
            x +=noise
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.training and self.dropout_rate > 0 :
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



def For(what_range, constructor):
    '''
    For(what_range, constructor, name='')
    Layer factory function to create a composite through a pattern similar to Python's `for` statement.
    This layer factory loops over the given range and passes each value to the constructor function.
    It is equivalent to
    ``Sequential([constructor(i) for i in what_range])``.
    It is acceptable that ``constructor`` takes no argument.
    Example:
     >>> from cntk.layers import *
     >>> from cntk.ops import relu
     >>> # stack of 3 Dense relu layers
     >>> model = For(range(3), lambda: Dense(2000, activation=relu))
     >>> # version of the above that has no activation for the last layer
     >>> model = For(range(3), lambda i: Dense(2000, activation=relu if i < 2 else identity))
     >>> # complex example that uses For() inside Sequential()
     >>> with default_options(activation=relu, pad=True):  # default activation is relu
     ...     model = Sequential([
     ...          For(range(2), lambda : [
     ...              Convolution2D((3,3), 64),
     ...              Convolution2D((3,3), 64),
     ...              MaxPooling((3,3), strides=2)
     ...          ]),
     ...          Label('ndfeat'),              # name this specific value
     ...          For(range(2), lambda i: [     # this passes a nested list to Sequential
     ...              Dense([256,128][i]),      # layer index i used to index into an array of parameters
     ...              Dropout(0.5)
     ...          ]),
     ...          Label('hidden'),
     ...          Dense(10, activation=None)    # activation parameter overrides default (which was set to relu)
     ...      ])
     >>> model.update_signature((3,32,32))      # RGB, 32 x 32 pixels
     >>> model.ndfeat.shape                     # shape at top of convo/pooling pyramid
         (64, 8, 8)
     >>> model.hidden.shape                     # shape before classifier
         (128,)
    Args:
     what_range (range): a Python range to loop over
     constructor (Python function/lambda with 1 or 0 arguments): lambda that constructs a layer
    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the layers as constructed by ``constructor`` one after another.
    '''
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
            return constructor()   # takes no arg: call without, that's fine too

    layers = [call(i) for i in what_range]
    sequential = Sequential(layers)
    return sequential



class Classifer1d(tf.keras.Sequential):
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

    def __repr__(self):
        return get_layer_repr(self)


#
# class ShortCut2d(Layer):
#     def __init__(self, *args, activation='relu', name="ShortCut2d", **kwargs):
#         """
#
#         Parameters
#         ----------
#         layer_defs : object
#         """
#         super(ShortCut2d, self).__init__(name=name, **kwargs)
#         self.activation = get_activation(activation)
#         self.has_identity = False
#         self.add_layer=Add()
#         for i in range(len(args)):
#             arg = args[i]
#             if isinstance(arg, (tf.keras.layers.Layer, list, dict)):
#                 if isinstance(arg, list):
#                     arg = Sequential(*arg)
#                     self.add(arg)
#                 elif isinstance(arg, dict) and len(args) == 1:
#                     for k, v in arg.items():
#                         if v is Identity:
#                             self.has_identity = True
#                         self.add(v)
#                 elif isinstance(arg, dict) and len(args) > 1:
#                     raise ValueError('more than one dict argument is not support.')
#                 elif arg is  Identity:
#                     self.has_identity = True
#                     self.add(arg)
#                 else:
#                     # arg.name='branch{0}'.format(i + 1)
#                     self.add(arg)
#         if len(self.layers) == 1 and self.has_identity == False:
#             self.add(Identity(name='Identity'))
#
#         # Add to the model any layers passed to the constructor.
#
#
#
#     @property
#     def layers(self):
#         return self._layers
#
#     def add(self, layer):
#         self._layers.append(layer)
#
#
#     def compute_output_shape(self, input_shape):
#         shape = input_shape
#         shape = self.layers[0].compute_output_shape(shape)
#         return shape
#
#     def call(self, inputs, training=None, mask=None):
#         x = enforce_singleton(inputs)
#         result=[]
#         if 'Identity' in self._layers:
#             result.append(x)
#         for layer in self._layers:
#             if layer is not Identity:
#                 out = layer(x)
#                 result.append(out)
#         result=self.add_layer(result)
#         if self.activation is not None:
#             result = self.activation(result)
#         return result
#
#     def __repr__(self):
#         return get_layer_repr(self)



class ShortCut2d(Layer):
    def __init__(self, *args, output_idx=None,activation=None, mode='add', name='shortcut', **kwargs):
        """

        Parameters
        ----------
        layer_defs : object
        """
        super(ShortCut2d, self).__init__(name=name)
        self.activation = get_activation(activation)
        self.has_identity = False
        self.mode = mode if isinstance(mode, str) else mode
        self.output_idx=output_idx
        self.skip_tensor=None
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, (Layer, list, dict)):
                if isinstance(arg, list):
                    arg = Sequential(*arg)
                elif isinstance(arg, (dict,OrderedDict)) and len(args) == 1:
                    for k, v in arg.items():
                        if isinstance(v, Identity):
                            self.has_identity = True
                            self.add_module('Identity', v)
                        else:
                            self.add_module(k, v)
                elif isinstance(arg,  (dict,OrderedDict)) and len(args) > 1:
                    raise ValueError('more than one dict argument is not support.')
                elif is_tensor(arg):
                    raise ValueError('only layer can be branch of shortcut, not tensor...')
                elif isinstance(arg, Identity):
                    self.has_identity = True
                    self.add_module('Identity', arg)
                elif isinstance(arg, Layer):
                    if len(arg.name)>0 :
                        self.add_module(arg.name, arg)
                    else:
                        self.add(arg)
                else:
                    raise ValueError('{0} is not support.'.format(arg.__class__.__name))
        if len(self._modules) == 1 and self.has_identity == False and self.output_idx is None:
            self.has_identity = True
            self.add_module('Identity', Identity())



    def forward(self, *x):
        x = enforce_singleton(x)

        current=None
        if self.has_identity == True:
            current=x
        for k, v in self._modules.items():
            if not isinstance(v, Identity):
                if current is None:
                    current=v(x)
                else:
                    if not hasattr(self, 'mode') or self.mode == 'add':
                        current=current+v(x)
                    elif self.mode == 'dot':
                        current =current* v(x)
                    elif self.mode == 'concate':
                        current=tf.concat([current,v(x)], axis=1)
                    else:
                        raise ValueError('Not valid shortcut mode')
        x=current
        if hasattr(self,'skip_tensor') and self.skip_tensor is not None:
            if not hasattr(self, 'mode') or self.mode == 'add':
                x = x+self.skip_tensor
            elif self.mode == 'dot':
                x =x* self.skip_tensor
            elif self.mode == 'concate':
                x =tf.concat([x, self.skip_tensor], axis=-1)
            else:
                raise ValueError('Not valid shortcut mode')
        if self.activation is not None:
            x = self.activation(x)
        return x




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

    def build(self, input_shape):
        if self._built == False :
            self.squeeze = Conv2d((1, 1), self.se_filters, strides=1, auto_pad=False, activation=None,use_bias=self.use_bias, name=self.name + '_squeeze')
            self.excite = Conv2d((1, 1), self.num_filters, strides=1, auto_pad=False, activation=None, use_bias=self.use_bias, name=self.name + '_excite')
            self.to(self.device)
            self._built = True

    def forward(self, x):
        s = self.pool(x)
        s = self.activation(self.squeeze(s))
        s = tf.sigmoid(self.excite(s))

        if self.is_gather_excite:
            s=image_ops.resize_images_v2(s, x.shape, method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
        x = s * x
        return x


