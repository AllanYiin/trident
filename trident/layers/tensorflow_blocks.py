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

__all__ = ['Conv2d_Block', 'TransConv2d_Block', 'ShortCut2d']

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
        self.conv = None
        self.use_spectral = use_spectral
        self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None
        self.depth_multiplier = depth_multiplier


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

        super(TransConv2d_Block, self).__init__(name=name)
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
        self.conv = None
        self.use_spectral = use_spectral
        self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None
        self.depth_multiplier = depth_multiplier

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


class ShortCut2d(training.Model):
    @trackable.no_automatic_dependency_tracking
    def __init__(self, *layers, activation=None, mode='add', name=None, **kwargs):
        super(ShortCut2d, self).__init__(name=name)
        self._build_input_shape = None
        self._layer_call_argspecs = {}
        has_identity = False
        if len(layers) > 1:
            for layer in layers:
                if 'identity' in str(layer.__class__.__name__).lower():
                    has_identity = True
                if isinstance(layer, tf.keras.Sequential):
                    self.add(layer)
                elif isinstance(layer, (tuple, list)):
                    self.add(Sequential(list(layer)))
                else:
                    self.add(layer)
        elif len(layers) == 1 and isinstance(layers[0], tf.keras.layers.Layer):
            self.add(layers[0])
        elif len(layers) == 1 and isinstance(layers[0], list):
            for layer in layers[0]:
                self.add(layer)
        elif len(layers) == 1 and isinstance(layers[0], OrderedDict):
            for k, v in layers[0].items():
                v.__name__ = k
                self.add(v)

        self.rank = 2
        self.activation = get_activation(activation)
        self.merge = None
        self.mode=mode
        if not hasattr(self, 'mode') or mode == 'add':
            self.merge = tf.keras.layers.Add()
        elif mode == 'dot':
            self.merge = tf.keras.layers.Dot(axes=-1)
        elif mode == 'concate':
            self.merge = tf.keras.layers.Concatenate(axis=-1)

    @trackable.no_automatic_dependency_tracking
    def add(self, layer):
        """Adds a layer instance on top of the layer stack.

        Arguments:
            layer: layer instance.

        Raises:
            TypeError: If `layer` is not a layer instance.
            ValueError: In case the `layer` argument does not
                know its input shape.
            ValueError: In case the `layer` argument has
                multiple output tensors, or is already connected
                somewhere else (forbidden in `Sequential` models).
        """
        # If we are passed a Keras tensor created by keras.Input(), we can extract
        # the input layer from its keras history and use that without any loss of
        # generality.
        if hasattr(layer, '_keras_history'):
            origin_layer = layer._keras_history[0]
            if isinstance(origin_layer, input_layer.InputLayer):
                layer = origin_layer

        if not isinstance(layer, base_layer.Layer):
            raise TypeError('The added layer must be '
                            'an instance of class Layer. '
                            'Found: ' + str(layer))

        tf_utils.assert_no_legacy_layers([layer])

        # This allows the added layer to broadcast mutations to the current
        # layer, which is necessary to ensure cache correctness.
        layer._attribute_sentinel.add_parent(self._attribute_sentinel)

        self.built = False
        self._layers.append(layer)
        if self._layers:
            self._track_layers(self._layers)

        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
        # Different Model types add to `._layers` in different ways, so for safety
        # we do a cache invalidation to make sure the changes are reflected.
        self._attribute_sentinel.invalidate_all()

    @property
    def layers(self):
        layers = super(ShortCut2d, self).layers
        if layers and isinstance(layers[0], input_layer.InputLayer):
            return layers[1:]
        return layers[:]

    @property
    @trackable_layer_utils.cache_recursive_attribute('dynamic')
    def dynamic(self):
        return any(layer.dynamic for layer in self.layers)


    def build(self, input_shape):
        if self._is_graph_network:
            self._init_graph_network(self.inputs, self.outputs, name=self.name)
        else:
            if input_shape is None:
                raise ValueError('You must provide an `input_shape` argument.')
            input_shape = tuple(input_shape)
            input_filters = int(input_shape[-1])
            self._build_input_shape = input_shape
            super(ShortCut2d, self).build(input_shape)
        self.built = True

    def call(self, inputs, training=tf.keras.backend.learning_phase(),**kwargs):
        if self._is_graph_network:
            if not self.built:
                self._init_graph_network(self.inputs, self.outputs, name=self.name)
            return super(ShortCut2d, self).call(inputs, )

        outputs = []  # handle the corner case where self.layers is empty
        for layer in self.layers:
            # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
            # are the outputs of `layer` applied to `inputs`. At the end of each
            # iteration `inputs` is set to `outputs` to prepare for the next layer.
            kwargs = {}
            argspec = self._layer_call_argspecs[layer].args
            if 'training' in argspec:
                kwargs['training'] = training
            if 'identity' in str(layer.__class__.__name__).lower():
                outputs.append(inputs)
            else:
                outputs.append(layer(inputs, **kwargs))
        if self.merge is not None and len(outputs)>=2:
            outputs = self.merge(outputs)
        else:
            raise ValueError('Not valid shortcut mode')

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(4)

        output_shapes = []
        for layer in self.layers:
            shape = layer.compute_output_shape(input_shape)
            output_shapes.append(shape)
        if self.mode == 'add' or self.mode == 'dot':
            output_shapes = list(set(output_shapes))
            if len(output_shapes) == 1:
                return output_shapes[0]

        elif self.mode == 'concate':
            output_shape = list(output_shapes[0])
            for shape in output_shapes[1:]:
                if output_shape[3] is None or shape[3] is None:
                    output_shape[3] = None
                    break
                output_shape[3] += shape[3]
            return tuple(output_shape)

    @property
    def input_spec(self):
        if self.layers and hasattr(self.layers[0], 'input_spec'):
            return self.layers[0].input_spec
        return None

    @property
    def _trackable_saved_model_saver(self):
        return model_serialization.SequentialSavedModelSaver(self)

    def get_config(self):
        layer_configs = []
        for layer in self.layers:
            layer_configs.append(generic_utils.serialize_keras_object(layer))
        # When constructed using an `InputLayer` the first non-input layer may not
        # have the shape information to reconstruct `Sequential` as a graph network.
        if (self._is_graph_network and layer_configs and 'batch_input_shape' not in layer_configs[0][
            'config'] and isinstance(self._layers[0], input_layer.InputLayer)):
            batch_input_shape = self._layers[0]._batch_input_shape
            layer_configs[0]['config']['batch_input_shape'] = batch_input_shape

        config = {'name': self.name,
                  'layers': copy.deepcopy(layer_configs),
                  'activation': generic_utils.serialize_keras_object(self.activation),
                  'mode': self.mode,
                  'merge': generic_utils.serialize_keras_object(self.merge), }
        if self._build_input_shape:
            config['build_input_shape'] = self._build_input_shape
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        activation = None
        merge = None
        if 'name' in config:
            name = config['name']
            build_input_shape = config.get('build_input_shape')
            layer_configs = config['layers']
            activation = layer_module.deserialize( config['activation'])
            merge =  layer_module.deserialize(config['merge'])
        else:
            name = None
            build_input_shape = None
            layer_configs = config
        model = cls(name=name)
        model.activation = activation
        model.merge = merge
        for layer_config in layer_configs:
            layer = layer_module.deserialize(layer_config, custom_objects=custom_objects)
            model.add(layer)
        if not model.inputs and build_input_shape:
            model.build(build_input_shape)
        return model
