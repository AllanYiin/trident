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
from trident.layers.tensorflow_normalizations import get_normalization
from trident.layers.tensorflow_pooling import  GlobalAvgPool2d
from trident.backend.tensorflow_backend import *
from trident.backend.tensorflow_ops import *
from trident.layers.tensorflow_layers import *

_tf_data_format = 'channels_last'

__all__ = ['Conv2d_Block', 'TransConv2d_Block','DepthwiseConv2d_Block','SeparableConv2d_Block', 'ShortCut2d','SqueezeExcite','For']

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




class Conv2d_Block(Layer):
    def __init__(self,kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None, keep_output=False, sequence_rank='cna', **kwargs):
        super(Conv2d_Block, self).__init__(name=name)
        if sequence_rank in ['cna','nac']:
            self.sequence_rank=sequence_rank
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
            self.conv= Conv2d(kernel_size=self.kernel_size, num_filters=self.num_filters, strides=self.strides,
                              auto_pad=self.auto_pad, activation=None,
                              use_bias=self.use_bias, dilation=self.dilation, groups=self.groups,
                              depth_multiplier=self.depth_multiplier,padding=self.padding, **kwargs)
            self.norm = get_normalization(normalization)
        self.activation = get_activation(activation)
        self.droupout = None



    def build(self, input_shape):
        if self._built == False:
            self.conv.input_shape = input_shape
            if self.use_spectral:
                conv=self._modules['conv']
                self._modules['conv']=nn.utils.spectral_norm(conv)
                self.norm=None
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





class TransConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, depth_multiplier=None,sequence_rank='cna', **kwargs):
        super(TransConv2d_Block, self).__init__(name=name)
        if sequence_rank in ['cna','nac']:
            self.sequence_rank=sequence_rank
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


class TransConv3d_Block(Layer):
    def __init__(self, kernel_size=(3, 3, 3), num_filters=32, strides=1, input_shape=None, auto_pad=True,
                 activation='leaky_relu', normalization=None, use_bias=False, dilation=1, groups=1, add_noise=False,
                 noise_intensity=0.001, dropout_rate=0, name=None,sequence_rank='cna', **kwargs):
        super(TransConv3d_Block, self).__init__(name=name)
        if sequence_rank in ['cna','nac']:
            self.sequence_rank=sequence_rank
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


class DepthwiseConv2d_Block1(Layer):
    def __init__(self,kernel_size=(3,3),depth_multiplier=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None,  sequence_rank='cna', **kwargs):
        super(DepthwiseConv2d_Block, self).__init__()
        if sequence_rank in ['cna','nac']:
            self.sequence_rank=sequence_rank
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

class DepthwiseConv2d_Block(Layer):
    def __init__(self, kernel_size=(3, 3), depth_multiplier=1, strides=1, auto_pad=True, padding=None,padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, keep_output=False, sequence_rank='cna',**kwargs):
        super(DepthwiseConv2d_Block, self).__init__(name=name)
        if sequence_rank in ['cna','nac']:
            self.sequence_rank=sequence_rank
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
    def build(self, input_shape):
        if self._built == False or self.conv is None:
            conv = DepthwiseConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier,
                                   strides=self.strides, auto_pad=self.auto_pad, padding=self.padding,padding_mode=self.padding_mode,
                                   activation=None, use_bias=self.use_bias, dilation=self.dilation, name=self._name)
            conv.input_shape = input_shape
            if self.use_spectral:
                self.conv = spectral_norm(conv)
            else:
                self.conv = conv

            self._built = True



    def forward(self, *x):
        x = enforce_singleton(x)
        if self.training and self.add_noise == True:
            noise = self.noise_intensity * tf.random.normal(shape=x.shape, mean=0, stddev=1)
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


class SeparableConv2d_Block(Layer):
    def __init__(self, kernel_size=(3,3),depth_multiplier=None, strides=1, auto_pad=True, padding_mode='zero',
                 activation=None, normalization=None, use_spectral=False, use_bias=False, dilation=1, groups=1,
                 add_noise=False, noise_intensity=0.005, dropout_rate=0, name=None, sequence_rank='cna', **kwargs):
        super(SeparableConv2d_Block, self).__init__()
        if sequence_rank in ['cna','nac']:
            self.sequence_rank=sequence_rank
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
                self.conv = spectral_norm(self.conv)
                if self.norm is SpectralNorm:
                    self.norm=None
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
            return constructor()   # takes no arg: call without, that's fine too

    layers = [call(i) for i in what_range]
    return Sequential(layers)



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



class ShortCut2d(Layer):
    def __init__(self, *args, axis=-1,branch_from=None,activation=None, mode='add', name=None, keep_output=False,**kwargs):
        """
        Args
        layer_defs : object
        """
        super(ShortCut2d, self).__init__(name=name)
        self.activation = get_activation(activation)
        self.has_identity = False
        self.mode = mode if isinstance(mode, str) else mode
        self.axis=axis
        self.branch_from=branch_from
        self.branch_from_uuid=None
        self.keep_output=keep_output

        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, (Layer,tf.Tensor, list, dict)):
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
                elif isinstance(arg,  (dict,OrderedDict)) and len(args) > 1:
                    raise ValueError('more than one dict argument is not support.')
                elif isinstance(arg, Identity):
                    self.has_identity = True
                    self.add_module('Identity', arg)
                elif isinstance(arg, Layer):
                    if len(arg.name)>0 and arg.name!=arg._name:
                        self.add_module(arg.name, arg)
                    else:
                        self.add_module('branch{0}'.format(i + 1), arg)
                else:
                    raise ValueError('{0} is not support.'.format(arg.__class__.__name))
        if len(self._modules) == 1 and self.has_identity == False and self.branch_from is None:
            self.has_identity = True
            self.add_module('Identity', Identity())

    def build(self, input_shape):
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
    def forward(self, *x):
        x = enforce_singleton(x)
        current = None
        concate_list = []

        for k, v in self._modules.items():
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
        s = ('mode={mode}, keep_output={keep_output}, activation={activation.__name__},axis={axis}')
        if hasattr(self, 'branch_from') and self.branch_from is not None:
            s += ', branch_from={0}, branch_from_uuid={1}'.format(self.branch_from, self.branch_from_uuid)
        return s.format(**self.__dict__)











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
            self._built = True

    def forward(self, x):
        s = self.pool(x)
        s = self.activation(self.squeeze(s))
        s = tf.sigmoid(self.excite(s))

        if self.is_gather_excite:
            s=image_ops.resize_images_v2(s, x.shape, method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
        x = s * x
        return x


