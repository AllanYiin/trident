from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import builtins
from typing import Optional,List,Tuple
import collections
import inspect
import itertools
import math
import numbers
from itertools import repeat
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.ops import embedding_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

from tensorflow.python.ops import gen_math_ops, image_ops, math_ops
from tensorflow.python.ops import nn, nn_ops, array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops

from trident.backend.common import *
from trident.backend.load_backend import *
from trident.backend.tensorflow_backend import Layer, Sequential, Parameter,normalize_padding, get_device
from trident.backend.tensorflow_ops import *
from trident.layers.tensorflow_activations import get_activation
from trident.layers.tensorflow_initializers import *
from trident.backend import dtype

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

__all__ = ['Dense','Embedding' ,'Flatten', 'Concatenate', 'Concate', 'Add', 'Subtract','Scale', 'Aggregation','Conv1d', 'Conv2d', 'Conv3d', 'TransConv1d',
           'TransConv2d', 'TransConv3d', 'DepthwiseConv1d', 'DepthwiseConv2d', 'DepthwiseConv3d', 'SeparableConv2d',
           'GatedConv2d','Upsampling2d', 'Reshape', 'Dropout', 'Lambda', 'SoftMax', 'Noise','Permute']

_session = get_session()

_device =get_device()
_epsilon = _session.epsilon




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
    def __init__(self, num_filters, use_bias=True, activation=None,kernel_regularizer=None, keep_output=False, name=None, **kwargs):
        super(Dense, self).__init__(name=name,keep_output=keep_output)
        self.rank = 0
        if isinstance(num_filters, int):
            self.num_filters = num_filters
        elif isinstance(num_filters, tuple):
            self.num_filters = unpack_singleton(num_filters)
        else:
            raise ValueError('output_shape should be integer, list of integer or tuple of integer...')
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

        self.use_bias = use_bias
        if kernel_regularizer == 'l2':
            self.kernel_regularizer = l2_normalize
        else:
            self.kernel_regularizer = None

        self.activation = get_activation(activation)


    def build(self, input_shape:TensorShape):
        if not self._built:
            with tf.device(get_device()):
                with self.name_scope:
                    if len(input_shape.dims) == 1:
                        self.input_filters = input_shape.dims[0]
                    else:
                        self.input_filters = input_shape[self.filter_index]
                    self.weight =Parameter(data=random_normal(shape=(self.input_filters,self.num_filters), mean=0., std=0.2) , name='weight')
                    kaiming_uniform(self.weight, a=math.sqrt(5))
                    if self.use_bias:
                        self.bias =Parameter(data=random_normal(shape=(self.num_filters), mean=0., std=0.002) , name='bias')
                    else:
                        self.bias=None

                self._built = True


    def forward(self, x, **kwargs) :

        if hasattr(self, 'kernel_regularizer') and self.kernel_regularizer is not None:
            x = tf.matmul(x, self.kernel_regularizer(self.weight))
        else:
            x = tf.matmul(x, self.weight)
        if self.use_bias:
            x=x+ self.bias

        if self.activation is not None:
            x = self.activation(x)
        return x

class Embedding(Layer):
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, padding_idx: Optional[int] = 0,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[tf.Tensor] = None, filter_index=-1, keep_output: bool = False, name: Optional[str] = None, add_noise=False,
                 noise_intensity=0.005) -> None:
        super(Embedding, self).__init__(keep_output=keep_output, name=name)
        self.filter_index = filter_index
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity

        self.register_parameter('weight', None)
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is not None and int_shape(_weight)[-1] == embedding_dim and len( int_shape(_weight))==2:
            with tf.device(get_device()):
                with self.name_scope:
                    self.weight =Parameter(tf.identity(_weight),trainable=True, name='weight')
            self.num_embeddings = int_shape(self.weight)[0]
            self._built = True
        elif _weight is not None :
            raise  ValueError('Shape[-1] of weight does not match embedding_dim')
        elif _weight is None and self.num_embeddings is not None:
            with tf.device(get_device()):
                with self.name_scope:
                    self.weight =Parameter(tf.identity(tf.random.normal(shape=(self.num_embeddings, self.embedding_dim), mean=0, stddev=1) * 0.02),trainable=True, name='weight')
            self._built = True
        if self._built:
           # self.to(self.device)
            #init.normal_(self.weight)
            if self.padding_idx is not None:
                padding_tensor=np.ones((self.num_embeddings,1))
                padding_tensor[self.padding_idx]=0
                self.weight.assign(self.weight.value()*to_tensor(padding_tensor))
        self.sparse = sparse


    def build(self, input_shape:TensorShape):
        if not self._built:
            with tf.device(get_device()):
                with self.name_scope:
                    if len(input_shape.dims) == 1:
                        self.input_filters = input_shape.dims[0]
                    else:
                        self.input_filters = input_shape[self.filter_index]
                    self.weight =Parameter(tf.random.normal(shape=(self.num_embeddings, self.embedding_dim), mean=0, stddev=1) * 0.02, name='weight')
                    if self.use_bias:
                        self.bias =Parameter(to_tensor(np.zeros((self.num_filters))), name='bias')

                    self._built = True


    def forward(self, x, **kwargs) :
        dtype = x.dtype
        if self.padding_idx is not None:
            padding_tensor=np.ones((self.num_embeddings,1))
            padding_tensor[self.padding_idx]=0
            self.weight.assign(self.weight.value()*to_tensor(padding_tensor))

        if dtype != tf.int32 :
            x = math_ops.cast(x,tf.int32)
        if isinstance(self.weight, sharded_variable.ShardedVariable):
            x = embedding_ops.embedding_lookup_v2(self.weight.variables,x,max_norm=self.max_norm,name=self.name)
        else:
            x = embedding_ops.embedding_lookup_v2(self.weight, x,max_norm=self.max_norm,name=self.name)

        if self.add_noise == True and self.training == True:
            _mean=x.mean()
            _std=x.std()
            if is_abnormal_number(_mean):
                _mean=0
            if is_abnormal_number(_std) or _std is None or _std<0.02:
                _std=0.02
            noise = self.noise_intensity * random_normal_like(x, mean=_mean, std=_std, dtype=x.dtype).detach().to(x.device)
            x = x + noise
        return x



class Flatten(Layer):
    def __init__(self, keep_output=False, name=None, **kwargs):
        super(Flatten, self).__init__()
        self._name = name
        self.keep_output = keep_output


    def build(self, input_shape:TensorShape):
        if self._built == False:

            self._built = True


    def forward(self, x, **kwargs) :

        x = tf.reshape(x, [x.get_shape().as_list()[0], -1])
        return x


class Concate(Layer):
    """Concate layer to splice  tensors ."""

    def __init__(self, axis=-1):
        super(Concate, self).__init__()
        self.axis = axis


    def forward(self, x, **kwargs) -> tf.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError('A `Concatenate` layer should be called on a list of at least 2 tensor  inputs')

        if all([int_shape(k) is None for k in x]):
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

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True


    def forward(self, x, **kwargs) -> tf.Tensor:
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

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True


    def forward(self, x, **kwargs) -> tf.Tensor:
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

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True


    def forward(self, x, **kwargs) -> tf.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out *= item
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
    def __init__(self, axis=-1, add_noise=False, noise_intensity=0.005, name=None,keep_output=False, **kwargs):
        """
         Args:
             axis (int,default=-1): The axis all the transformation processed across.
             add_noise (bool, default=False): If True, will add (output) noise  in this layer.
             noise_intensity (float, default=0.005): The noise intensity (is propotional to mean of actual output.

         """
        super(SoftMax, self).__init__(name=name,keep_output=keep_output)
        self.axis = axis
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity


    def forward(self, x, **kwargs) :

        if not hasattr(self, 'add_noise'):
            self.add_noise = False
            self.noise_intensity = 0.005
        if self.training:
            if self.add_noise:
                _mean=x.mean()
                _std=x.var().sqrt()
                if is_nan(_mean):
                    _mean=0.0
                if _std is None or _std < 0.02:
                    _std = 0.02
                noise = self.noise_intensity * random_normal_like(x,mean=_mean, std=_std,dtype=x.dtype).to(x.device).detach()
                x = x + noise
            x = tf.math.log_softmax(x, self.axis)
        else:
            x = tf.math.softmax(x, self.axis)
        return x



class Scale(Layer):
    """The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.
         Examples:
                >>> x = to_tensor(ones((2,2,2,4)))
                >>> layer1=Scale(scale=2,shift=0.5,power=1,mode='uniform')
                >>> output1 = layer1(x)
                >>> tf.reduce_all(output1==(x*2+0.5)**1)
                <tf.Tensor: shape=(), dtype=bool, numpy=True>
                >>> x = to_tensor(ones((2,2,2,4)))
                >>> layer2=Scale(scale=to_tensor([1,2,3,4]),shift=0.5,power=1.2,mode='channel')
                >>> output2 = layer2(x)
                >>> tf.reduce_all(output2==pow((x*(to_tensor([1,2,3,4]).reshape((1,1,1,4)))+0.5),1.2))
                <tf.Tensor: shape=(), dtype=bool, numpy=True>
    """
    def __init__(self, scale:(float,Tensor)=1.0, shift:(float,Tensor)=0.0, power:(float,Tensor)=1.0,mode='uniform',keep_output: bool=False,name:Optional[str] = None):
        super(Scale, self).__init__(keep_output=keep_output,name=name)
        self._scale=scale
        self._shift=shift
        self._power=power

        if mode == 'uniform' and (numel(self._scale)!=1 or numel(self._shift)!=1or numel(self._power)!=1):
            raise ValueError('Scale/ Shift/ Power should float, 0d Tensor or One element Tensor whem mode=uniform')
        if mode in [ 'uniform', 'channel', 'elementwise']:
            self.mode = mode
        else :
            raise ValueError('Only [uniform,channel,elementwise] is valid value for mode ')

    def build(self, input_shape:TensorShape):
        def remove_from(name:str,*dicts):
            for d in dicts:
                if name in d:
                    del d[name]
        if self._built == False:
            if self.mode == 'uniform':
                self.weight_scale = Parameter(ones((1)).to(self.get_root().device) * self._scale, trainable=True)
                self.weight_shift = Parameter(ones((1)).to(self.get_root().device) * self._shift, trainable=True)
                self.weight_power = Parameter(ones((1)).to(self.get_root().device) * self._power, trainable=True)
            elif self.mode == 'constant':
                self.weight_scale = ones((1)).to(self.get_root().device) * self._scale
                self.weight_shift = ones((1)).to(self.get_root().device)  * self._shift
                self.weight_power = ones((1)).to(self.get_root().device) *  self._power
            elif self.mode == 'channel':
                new_shape=[1,]*(input_shape.rank)
                new_shape[self.filter_index]=self.input_filters
                new_shape=tuple(new_shape[1:])
                self.weight_scale = Parameter(ones(new_shape).to(self.get_root().device)*self._scale,trainable=True)
                self.weight_shift = Parameter(ones(new_shape).to(self.get_root().device)*self._shift,trainable=True)
                self.weight_power = Parameter(ones(new_shape).to(self.get_root().device)*self._power,trainable=True)
            elif self.mode == 'elementwise':
                new_shape = input_shape.dims[1:]
                self.weight_scale = Parameter(ones(new_shape).to(self.get_root().device) * self._scale, trainable=True)
                self.weight_shift = Parameter(ones(new_shape).to(self.get_root().device) * self._shift, trainable=True)
                self.weight_power = Parameter(ones(new_shape).to(self.get_root().device) * self._power, trainable=True)

            self._built = True


    def forward(self, x, **kwargs) -> tf.Tensor:
        x = x * self.scale + self.shift
        x=sign(x)* pow(abs(x), self.weight_power)
        return x


class Aggregation(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self, mode='mean', axis=-1, keepdims=True, keep_output: bool = False, name: Optional[str] = None):
        super(Aggregation, self).__init__(name=name, keep_output=keep_output)
        valid_mode = ['mean', 'sum', 'max', 'min', 'first', 'last']
        if mode in valid_mode:
            self.mode = mode
        else:
            raise ValueError('{0} is not valid mode. please use one of {1}'.format(mode, valid_mode))
        self.axis = unpack_singleton(axis)
        if mode in ['first', 'last'] and isinstance(self.axis, (list, tuple)):
            raise ValueError('{0} only can reduction along one axis.'.format(mode))

        self.keepdims = keepdims

    def build(self, input_shape: TensorShape):
        if self._built == False:
            dims = input_shape.dims
            if self.keepdims == True:
                dims[self.axis] = 1
            else:
                dims.pop(self.axis)
            self.output_shape = TensorShape(dims)
            self._built = True

    def forward(self, x, **kwargs) -> Tensor:
        if self.mode == 'mean':
            x = tf.math.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)
        elif self.mode == 'sum':
            x =  tf.math.reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        elif self.mode == 'max':
            x =  tf.math.reduce_max(x, axis=self.axis, keepdims=self.keepdims)
        elif self.mode == 'min':
            x =  tf.math.reduce_min(x, axis=self.axis, keepdims=self.keepdims)
        elif self.mode == 'first':
            begin=[0]*len(int_shape(x))

            size=list(int_shape(x))
            size[self.axis]=1
            x=tf.slice(x,begin=begin,size=size)

            if self.keepdims:
                x = expand_dims(x, axis=self.axis)
        elif self.mode == 'last':
            begin = [0] * len(int_shape(x))
            begin[self.axis] = -1
            size = list(int_shape(x))
            size[self.axis] = 1
            x = tf.slice(x, begin=begin, size=size)
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

    def __init__(self, rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias=False, dilation=1,
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





        self.transposed = transposed
        self.use_bias = use_bias
        self.register_parameter('weight', None)
        if self.use_bias:
            self.register_parameter('bias', None)

    def build(self, input_shape:TensorShape):
        if self._built == False:
            with tf.device(get_device()):
                with self.name_scope:
                    self.input_filters =input_shape[self.filter_index]

                    if self.auto_pad:
                        if self.transposed == False:
                            padding = get_static_padding(self.rank, self.kernel_size, self.strides, self.dilation, input_shape[1:-1])
                            self.padding = tuple(padding)
                        else:
                            self.padding= get_static_padding(self.rank, self.kernel_size, self.strides, self.dilation, input_shape[1:-1])
                    else:
                        if self.padding is None:
                            self.padding = [0] * (2 * self.rank)
                        elif isinstance(self.padding, int):
                            self.self.padding = [self.padding] * (2 * self.rank)
                        elif len(self.padding) == self.rank:
                            self.padding = list(self.padding) * 2
                        elif len(self.padding) == 2 * self.rank:
                            pass

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

                    # if self.auto_pad:
                    #     self.padding = get_static_padding(self.rank, self.kernel_size ,
                    #                                        self.strides ,  self.dilation ,
                    #                                        tuple(to_list(input_shape)[:-1]))
                    # else:
                    #     self.padding = normalize_padding(self.padding, self.rank)



                        # elif self.depth_multiplier < 1:
                        #
                        #     self.groups = int(builtins.round(self.input_filters * self.depth_multiplier, 0))
                        #     self.num_filters=int(self.groups)
                        # else:




                    channel_multiplier = int(self.num_filters // self.groups) if self.groups>1 else int(self.num_filters)#if self.depth_multiplier is None else self.depth_multiplier  # default channel_multiplier

                    if self.transposed:
                        # filter_height, filter_width,  out_channels in_channels,
                        self.weight =Parameter(tf.random.normal(shape=[*self.kernel_size,  self.num_filters // self.groups, self.input_filters], mean=0,  stddev=1) * 0.02, trainable=True, name='weight')

                    else:

                        # [filter_height, filter_width, in_channels, out_channels]`
                        self.weight =Parameter( data=tf.random.normal(shape=[*self.kernel_size, self.input_filters,  self.num_filters // self.groups], mean=0,    stddev=1) * 0.02, trainable=True, name='weight')

                        if self.separable:
                            pointwise_kernel_size = (1,) * len(self.kernel_size)
                            self.pointwise =Parameter(data=tf.random.normal( shape=[*pointwise_kernel_size, self.input_filters * channel_multiplier,self.num_filters], mean=0, stddev=1) * 0.02, trainable=True, name='weight')

                    if self.use_bias:
                        self.bias =Parameter(data=tf.random.normal([int(self.num_filters)]), name='bias')


                self._built = True

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], Layer):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={auto_pad}'
        if hasattr(self,'padding') and self.padding is not None:
            s += ', padding={0}, padding_mode={1}'.format(self.padding, self.padding_mode)
        s += ',use_bias={use_bias} ,dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if hasattr(self,'_input_shape') and self._input_shape is not None:
            s += ', input_shape={0}, input_filters={1}'.format(to_list(self._input_shape), self.input_filters)
        if hasattr(self,'_output_shape') and self._output_shape is not None:
            s += ', output_shape={0}'.format(
                self._output_shape if isinstance(self._output_shape, (list, tuple)) else to_list(self._output_shape))
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


    def conv1d_forward(self, x, **kwargs):
        return tf.nn.conv1d(x, filters=self.weight, stride=(1,) + self.strides + (1,), padding='SAME' if self.auto_pad else 'VALID',
                            data_format="NWC", dilations=(1,) + self.dilation + (1,), name=self._name)


    def forward(self, x, **kwargs) :

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
        s += ',auto_pad={self.auto_pad}' + ',use_bias={use_bias} ,dilation={dilation_rate}'
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

    Examples:
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


    def conv2d_forward(self, x, **kwargs):
        if self.auto_pad == True and len(self.padding) == self.rank+2 :
            x = tf.pad(x, self.padding, mode='CONSTANT')
        else:
            if len(self.padding)==4:
                self.padding=((self.padding[0],self.padding[1]),(self.padding[2],self.padding[3]))
            padlist = list(self.padding)
            padlist.insert(0, (0, 0))
            padlist.append((0, 0))

            x = tf.pad(x, tuple(padlist), mode='CONSTANT')

        return tf.nn.conv2d(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding='VALID',
                            data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)


    def forward(self, x, **kwargs) :

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


    def conv3d_forward(self, x, **kwargs):
        return tf.nn.conv3d(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding='VALID',
                            data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)


    def forward(self, x, **kwargs) :

        x = self.conv3d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False, dilation=1,
                 groups=1, depth_multiplier=None, name=None, **kwargs):
        rank=1
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)

        auto_pad = auto_pad
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

        groups = groups
        super(TransConv1d, self).__init__(rank,num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                          auto_pad=auto_pad, padding=padding, padding_mode='zero', dilation=dilation,
                                          groups=groups, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                          transposed=True, name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank = 1


    def conv1d_forward(self, x, **kwargs):
        in_shape = x.get_shape().as_list()
        in_shape[1] *= self.strides[0]
        in_shape[-1] = self.num_filters

        return tf.nn.conv1d_transpose(x, filters=self.weight, output_shape=in_shape, strides=(1,) + self.strides + (1,),
                                      padding='SAME' if self.auto_pad else 'VALID', data_format="NHWC", dilations=(1,) + self.dilation + (1,),
                                      name=self._name)


    def forward(self, x, **kwargs) :

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
        s += ',auto_pad={self.auto_pad}' + ',use_bias={use_bias} ,dilation={dilation_rate}'
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



        groups = groups
        super(TransConv2d, self).__init__(rank,num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                          auto_pad=auto_pad, padding=padding, padding_mode='zero', dilation=dilation,
                                          groups=groups, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                          transposed=True, name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank=2


    def conv2d_forward(self, x, **kwargs):
        in_shape = to_list(int_shape(x))
        in_shape[1] *= self.strides[0]
        in_shape[2] *= self.strides[1]
        in_shape[3] = self.num_filters
        return tf.nn.conv2d_transpose(x, filters=self.weight, output_shape=in_shape, strides=(1,) + self.strides + (1,),
                                      padding='SAME' if self.auto_pad else 'VALID', data_format="NHWC", dilations=(1,) + self.dilation + (1,),
                                      name=self._name)


    def forward(self, x, **kwargs) :

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
        s += ',auto_pad={self.auto_pad}' + ',use_bias={use_bias} ,dilation={dilation_rate}'
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

        groups = groups
        super(TransConv3d, self).__init__(rank,num_filters=num_filters, kernel_size=kernel_size, strides=strides,
                                          auto_pad=auto_pad, padding=padding, padding_mode='zero', dilation=dilation,
                                          groups=groups, use_bias=use_bias, depth_multiplier=depth_multiplier,
                                          transposed=True, name=name, **kwargs)
        self.activation = get_activation(activation)
        self.rank=3


    def conv3d_forward(self, x, **kwargs):
        in_shape = x.get_shape().as_list()
        in_shape[1] *= self.strides[0]
        in_shape[2] *= self.strides[1]
        in_shape[3] *= self.strides[2]
        in_shape[-1] = self.num_filters
        return tf.nn.conv3d_transpose(x, filters=self.weight, output_shape=in_shape, strides=(1,) + self.strides + (1,),
                                      padding='SAME' if self.auto_pad else 'VALID', data_format="NHWC", dilations=(1,) + self.dilation + (1,),
                                      name=self._name)


    def forward(self, x, **kwargs) :

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
        s += ',auto_pad={self.auto_pad}' + ',use_bias={use_bias} ,dilation={dilation_rate}'
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


    def conv1d_forward(self, x, **kwargs):
        return tf.nn.convolution(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding=self.padding,
                                 data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)


    def forward(self, x, **kwargs) :

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


    def conv2d_forward(self, x, **kwargs):
        if self.auto_pad == True and len(self.padding) == self.rank + 2:
            x = tf.pad(x, self.padding, mode='CONSTANT')
        else:
            padlist = list(self.padding)
            padlist.insert(0, (0, 0))
            padlist.append((0, 0))

            x = tf.pad(x, tuple(padlist), mode='CONSTANT')

        return tf.nn.depthwise_conv2d(x,filter=self.weight, strides=(1,) + self.strides + (1,), padding='VALID', data_format="NHWC", dilations= self.dilation, name=self._name)


    def forward(self, x, **kwargs) :

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


    def conv3d_forward(self, x, **kwargs):
        return tf.nn.convolution(x, filters=self.weight, strides=(1,) + self.strides + (1,), padding='VALID',
                                 data_format="NHWC", dilations=(1,) + self.dilation + (1,), name=self._name)


    def forward(self, x, **kwargs) :

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


    def conv2d_forward(self, x, **kwargs):
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


    def forward(self, x, **kwargs) :

        x = self.conv2d_forward(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format='NHWC')
        if self.activation is not None:
            x = self.activation(x)
        return x



class GatedConv2d(Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support zero-size tensor and more features.
    """

    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, depth_multiplier=None, norm=l2_normalize, name=None,**kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        super(GatedConv2d, self).__init__(kernel_size=kernel_size, num_filters=num_filters, strides=strides, auto_pad=auto_pad, padding=padding, padding_mode=padding_mode, activation=activation,
                 use_bias=use_bias, dilation=dilation, groups=groups, name=name, depth_multiplier=depth_multiplier,
                                              **kwargs)

        self.norm = norm


    def forward(self, x, **kwargs):
        x = super().forward(x)
        if self.strides==(2,2):
            x = x[:, 1::2, 1::2,:]

        x, g =split(x,axis=-1,num_splits=2)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x * sigmoid(g)
        return x


class Upsampling2d(Layer):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True, keep_output=False, name=None):
        super(Upsampling2d, self).__init__(name=name)
        if mode not in {'nearest', 'bilinear', 'area', 'pixel_shuffle'}:
            raise ValueError('`mode` argument should be one of `"nearest"` '
                             'or `"bilinear"`.')
        self.rank = 2
        self.size = size
        if scale_factor is not None:
            if isinstance(scale_factor, tuple):
                self.scale_factor = scale_factor
            else:
                self.scale_factor = (scale_factor, scale_factor)
        else:
            self.scale_factor=1
        self.mode = mode
        self.align_corners = align_corners
        self.keep_output = keep_output


    def forward(self, x, **kwargs) :

        new_shape =list(int_shape(x)[1:])

        if self.scale_factor is not None and isinstance(self.scale_factor, tuple):
            new_shape[0] = int(new_shape[0] * self.scale_factor[0])
            new_shape[1] = int(new_shape[1] * self.scale_factor[1])

        if self.mode == 'pixel_shuffle':
            return tf.nn.depth_to_space(x, int(self.scale_factor[0]))
        elif self.mode == 'nearest':
            return image_ops.resize_images_v2(x, [new_shape[1],new_shape[0] ], method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
        elif self.mode == 'bilinear':
            return image_ops.resize_images_v2(x, [new_shape[1],new_shape[0] ], method=image_ops.ResizeMethod.BILINEAR)
        elif self.mode == 'area':
            return image_ops.resize_images_v2(x, [new_shape[1],new_shape[0] ], method=image_ops.ResizeMethod.AREA)
        else:
            return image_ops.resize_images_v2(x, [new_shape[1],new_shape[0] ], method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)

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

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self._built = True


    def forward(self, x, **kwargs) :
        return self.function(*x)

    def extra_repr(self):
        s = 'function={0}'.format("".join(inspect.getsourcelines(self.function)[0]))


class Reshape(Layer):
    def __init__(self, target_shape, name=None, **kwargs):
        super(Reshape, self).__init__(name=name)
        if isinstance(target_shape, numbers.Integral):
            self.target_shape = (target_shape,)
        elif isinstance(target_shape, list):
            self.target_shape = tuple(target_shape)
        else:
            self.target_shape =target_shape

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self._built = True


    def forward(self, x, **kwargs) :
        shp = self.target_shape
        if -1 in shp:
            return tf.reshape(x, tf.constant((int_shape(x)[0],) + tuple(shp), dtype=tf.int32))
        else:
            return  tf.reshape(x, tf.constant((-1,) + tuple(shp), dtype=tf.int32))


    def extra_repr(self):
        s = 'target_shape={0}'.format(self.target_shape)

class Permute(Layer):
    """Permute Layer

    """

    def __init__(self, *args, name=None):
        """
        Permute the input tensor
        Args:
            *shape (ints): new shape, WITHOUT specifying batch size as first
            dimension, as it will remain unchanged.
        """
        super(Permute, self).__init__(name=name)
        self.pattern = args

    def forward(self, x, **kwargs):
        return permute(x, self.pattern)



class Dropout(Layer):
    def __init__(self, dropout_rate=0, keep_output=False, name=None, **kwargs):
        super(Dropout, self).__init__(name=name)
        self._name = name
        self.dropout_rate = dropout_rate
        self.keep_output = keep_output

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self._built = True


    def forward(self, x, **kwargs) :

        if self.training:
            x = tf.nn.dropout(x, self.dropout_rate)
        return x

    def extra_repr(self):
        s = 'dropout_rate={0}'.format(self.dropout_rate)


class Noise(Layer):
    def __init__(self, stddev=0.1, name=None):
        super(Noise, self).__init__(stddev=stddev, name=name)
        if stddev is None or stddev < 0.02:
            stddev = 0.02
        self.stddev=stddev
    def build(self, input_shape:TensorShape):
        if self._built == False:
            self._built = True


    def forward(self, x, **kwargs) :

        if self.training:
            noise = random_normal_like(x,mean=mean(x),std=self.stddev)
            x=x+noise
        return x

    def extra_repr(self):
        s = 'stddev={0}'.format(self.stddev)


class SingleImageLayer(Layer):
    def __init__(self, image, is_recursive=False, name=''):
        super(SingleImageLayer, self).__init__(name=name)
        if isinstance(image, (np.ndarray, tf.Tensor)):
            self.origin_image = to_tensor(image)
            self.input_shape = image.shape[1:]

    def build(self, input_shape:TensorShape):
        if self._built == False:
            with tf.device(get_device()):
                self.weight =Parameter(to_tensor(self.origin_image.clone()), name='weight')
                self.input_filters =input_shape[self.filter_index]
                self._built = True


    def forward(self, x, **kwargs):
        return expand_dims(self.weight, 0)

    def extra_repr(self):
        return 'is_recursive={0}'.format(self.is_recursive)
