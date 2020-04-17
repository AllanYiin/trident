from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from itertools import repeat

import cntk as C
import numpy as np
from cntk.internal import _as_tuple
from cntk.layers.blocks import _inject_name, _INFERRED
from cntk.ops import *
from cntk.ops.functions import Function, BlockFunction

from .cntk_activations import get_activation
# from backend.common import  epsilon
from .cntk_layers import *
from .cntk_normalizations import get_normalization
from ..backend.cntk_backend import Sequential
from ..backend.common import get_session, gcd, get_divisors, isprime, next_prime, prev_prime, nearest_prime

__all__ = ['Conv2d_Block','TransConv2d_Block']

_session = get_session()
_device =  "cuda" if 'GPU' in str(C.all_devices()[0]) else "cpu"
_epsilon=_session.epsilon

def _ntuple(n):
    def parse(x):
        if hasattr(x, "__iter__") and not isinstance(x,str):
            return tuple([i for i in x])
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def conv2d_block(x,kernel_size=(3,3), num_filters=32, strides=1, auto_pad=True,activation='relu6',
                 normalization='instance',  use_bias=False,dilation=1, groups=1,add_noise=False,noise_intensity=0.005,dropout_rate=0,name=''):
    normalization_fn = get_normalization(normalization)
    activation_fn = get_activation(activation)
    kernel_size= _pair(kernel_size)
    strides= _pair(strides)
    dilation = _pair(dilation)


    conv = Conv2d(kernel_size=kernel_size, num_filters=num_filters, activation=None, strides=strides, auto_pad=auto_pad,use_bias=use_bias,dilation=dilation, input_shape=x.shape,name=name+'_conv')

    if add_noise == True :
        noise = noise_intensity *C.random.uniform_like(x)
        x = x + noise
    x = conv(x)
    if normalization_fn != None:
        x = normalization_fn(x)
    if activation_fn != None:
        x = activation_fn(x)
    if dropout_rate > 0:
        x = C.dropout(x,dropout_rate=dropout_rate)
    return x



def Conv2d_Block(kernel_size=(3, 3), num_filters=32, strides=1, auto_pad=True, activation='relu6',
                 normalization='instance', use_bias=False, dilation=1, groups=1, add_noise=False, noise_intensity=0.005,
                 dropout_rate=0):
    def internal_op(x):
        return conv2d_block(x,kernel_size=kernel_size, num_filters=num_filters, strides=strides, auto_pad=auto_pad, activation=activation,
                            normalization=normalization, use_bias=use_bias, dilation=dilation, groups=groups, add_noise=add_noise,
                            noise_intensity=noise_intensity, dropout_rate=dropout_rate)

    return internal_op


def trans_conv2d_block(x,kernel_size=(3,3), num_filters=32, strides=1, auto_pad=True,activation='relu6',
                 normalization='instance',  use_bias=False,dilation=1, groups=1,add_noise=False,noise_intensity=0.005,dropout_rate=0, name=''):
    normalization_fn = get_normalization(normalization)
    activation_fn = get_activation(activation)
    kernel_size= _pair(kernel_size)
    strides= _pair(strides)
    dilation = _pair(dilation)


    conv = TransConv2d(kernel_size=kernel_size, num_filters=num_filters, activation=None, strides=strides, auto_pad=auto_pad,use_bias=use_bias,dilation=dilation, input_shape=x.shape,name=name+'_conv')

    if add_noise == True :
        noise = noise_intensity *C.random.uniform_like(x, dtype=np.float32)
        x = x + noise
    x = conv(x)
    if normalization_fn != None:
        x = normalization_fn(x)
    if activation_fn != None:
        x = activation_fn(x)
    if dropout_rate > 0:
        x = C.dropout(x,dropout_rate=dropout_rate)
    return x



def TransConv2d_Block(kernel_size=(3, 3), num_filters=32, strides=1, auto_pad=True, activation='relu6',
                 normalization='batch', use_bias=False, dilation=1, groups=1, add_noise=False, noise_intensity=0.005,
                 dropout_rate=0):
    def internal_op(x):
        return trans_conv2d_block(x, kernel_size=kernel_size, num_filters=num_filters, strides=strides, auto_pad=auto_pad, activation=activation,
                            normalization=normalization, use_bias=use_bias, dilation=dilation, groups=groups, add_noise=add_noise,
                            noise_intensity=noise_intensity, dropout_rate=dropout_rate)

    return internal_op









#
#
# def gcdconv2d_block(x,kernel_size, num_filters, strides=(1, 1), init=C.he_normal(0.02),activation='leaky_relu6', normalization=None,padding='same', use_bias=False, dilation=1, divisor_rank=0,dropout_rate=0.0, add_noise=False,self_norm=True,name=''):
#     normalization_fn = get_normalization(normalization)
#     activation_fn = get_activation(activation)
#
#     x =gcd_conv2d(x,kernel_size=kernel_size, num_filters=num_filters, activation=None, strides=strides, init=init, padding=padding, use_bias=use_bias,dilation=dilation,divisor_rank=divisor_rank, self_norm=self_norm,name=name+'_conv')
#     if normalization is not None:
#         x = normalization_fn()(x)
#     if activation is not None:
#         x = activation_fn()(x)
#     if add_noise:
#         x=x + C.random.normal_like(x,mean=0.,stddev= 0.005)
#     if dropout_rate > 0:
#         x = C.layers.Dropout(dropout_rate)(x)
#     return  x
#
#
# def GcdConv2d_Block(kernel_size=(3,3),
#                  num_filters=32,
#                  strides=(1, 1),
#                  auto_pad=True,
#                  activation='leaky_relu6',
#                  normalization=None,
#                  init=C.he_normal(0.02),
#                  divisor_rank=0,
#                  use_bias=False, init_bias=0, dilation=1, add_noise=False,
#                  dropout_rate=0,
#                  self_norm=True,
#                  weights_contraint=None):
#     # normalization_fn = get_normalization(normalization)
#     # activation_fn = get_activation(activation)
#
#     def internal_op(x):
#         padding = 'same'
#         if not auto_pad:
#             padding='valid'
#         activation_fn=None
#         if activation=='leaky_relu6':
#             activation_fn=Leaky_Relu6
#         return gcdconv2d_block(x,kernel_size=kernel_size, num_filters=num_filters, strides=strides, init=init, activation=activation_fn,
#                  normalization=normalization, padding=padding, use_bias=use_bias, dilation=dilation,divisor_rank=divisor_rank, dropout_rate=dropout_rate,add_noise=False,self_norm=self_norm)
#     return internal_op
#
#
#
# def gcdconv2d_block_1(x,kernel_size, num_filters, strides=(1, 1), init=C.he_normal(0.02),activation='leaky_relu6', normalization=None,padding='same', use_bias=False, dilation=1, divisor_rank=0,dropout_rate=0.0, add_noise=False,self_norm=True,name=''):
#     normalization_fn = get_normalization(normalization)
#     activation_fn = get_activation(activation)
#
#     x =gcd_conv2d_1(x,kernel_size=kernel_size, num_filters=num_filters, activation=None, strides=strides, init=init, padding=padding, use_bias=use_bias,dilation=dilation,divisor_rank=divisor_rank, self_norm=self_norm,name=name+'_conv')
#     if normalization is not None:
#         x = normalization_fn()(x)
#     if activation is not None:
#         x = activation_fn()(x)
#     if add_noise:
#         x=x + C.random.normal_like(x,mean=0.,stddev= 0.005)
#     if dropout_rate > 0:
#         x = C.layers.Dropout(dropout_rate)(x)
#     return  x
#
#
# def GcdConv2d_Block_1(kernel_size=(3,3),
#                  num_filters=32,
#                  strides=(1, 1),
#                  auto_pad=True,
#                  activation='leaky_relu6',
#                  normalization=None,
#                  init=C.he_normal(0.02),
#                  divisor_rank=0,
#                  use_bias=False, init_bias=0, dilation=1, add_noise=False,
#                  dropout_rate=0,
#                  self_norm=True,
#                  weights_contraint=None):
#     # normalization_fn = get_normalization(normalization)
#     # activation_fn = get_activation(activation)
#
#     def internal_op(x):
#         padding = 'same'
#         if not auto_pad:
#             padding='valid'
#         activation_fn=None
#         if activation=='leaky_relu6':
#             activation_fn=Leaky_Relu6
#         return gcdconv2d_block_1(x,kernel_size=kernel_size, num_filters=num_filters, strides=strides, init=init, activation=activation_fn,
#                  normalization=normalization, padding=padding, use_bias=use_bias, dilation=dilation,divisor_rank=divisor_rank, dropout_rate=dropout_rate,add_noise=False,self_norm=self_norm)
#     return internal_op
#
#
#
#
# def sepatable_conv2d_block(x,
#                            kernel_size,  depth_multiplier=1, strides=(1, 1), init=C.xavier(0.5),activation='leaky_relu6', normalization=None,
#                            padding='same', use_bias=False, dilation=1,dropout_rate=0.0,
#                            add_noise=False,
#                            name=''):
#     normalization_fn = get_normalization(normalization)
#     activation_fn = get_activation(activation)
#     input_num_filters = x.shape[0]
#     x =sepatable_conv2d(x,kernel_size=kernel_size,depth_multiplier=depth_multiplier,  activation=None, strides=strides, init=init, padding=padding, use_bias=use_bias,dilation=dilation,input_num_filters=input_num_filters, name=name+'_conv')
#     if normalization is not None:
#         x = normalization_fn()(x)
#     if activation is not None:
#         x = activation_fn()(x)
#     if add_noise:
#         x=x + C.random.normal_like(x,mean=0.,stddev= 0.005)
#     if dropout_rate > 0:
#         x = C.layers.Dropout(dropout_rate)(x)
#     return  x
#
#
# def SepatableConv2d_Block(kernel_size=(3,3),
#                  depth_multiplier=1,
#                  strides=(1, 1),
#                  auto_pad=True,
#                  activation='leaky_relu6',
#                  normalization='instance',
#                  init=C.he_normal(0.02),
#                  use_bias=False, init_bias=0, dilation=1, add_noise=False, dropout_rate=0,
#                  weights_contraint=None):
#     def internal_op(x):
#
#         padding='valid'
#         if auto_pad:
#             padding='same'
#         activation_fn=None
#         if activation=='leaky_relu6':
#             activation_fn=Leaky_Relu6
#         return sepatable_conv2d_block(x,kernel_size=kernel_size,depth_multiplier=depth_multiplier, strides=strides, init=init, activation=activation_fn,
#                  normalization=normalization, padding=padding, use_bias=use_bias, dilation=dilation, dropout_rate=dropout_rate,add_noise=False)
#     return internal_op
#
#
#
# def shortcut(x,layers):
#     input_shape=x.shape
#     branches = collections.OrderedDict()
#     from functools import reduce
#     if isinstance(layers,dict):
#         idx=0
#         for k,v in layers.items():
#             if hasattr(v,'__iter__'):
#                 out = x
#                 for f in v:
#                     if callable(f):
#                         out = f(out)
#                 branches[idx]=out
#                 idx+=1
#         branches_agg=None
#         for k,v in branches.items():
#             if branches_agg is None:
#                 branches_agg=v
#             else:
#                 branches_agg=C.plus(branches_agg,v)
#         if x.shape==branches_agg.shape:
#             branches_agg=C.plus(x,branches_agg)
#         return branches_agg
#     else:
#         raise  ValueError('')
#
#
#
# def ShortCut(layers):
#     def internal_op(x):
#        return shortcut(x,layers)
#     return internal_op
#
#
# def classifier(x,num_classes, is_multiselect, classifier_type):
#
#     if classifier_type == 'dense':
#         x =C.reduce_mean(x,[1,2],keepdims=False)
#         x=C.layers.Dense(num_classes,activation=sigmoid,init_bias=C.he_normal(),bias=False)(x)
#         return x
#     elif classifier_type == 'global_avgpool':
#         if len(x.shape) != 3:
#             raise ValueError("GlobalAvgPooling only accept CHW shape")
#         if x.shape[0]!=num_classes:
#             x=conv2d(x,(1,1), num_filters=num_classes, strides=1, padding='valid', activation=None, use_bias=False)
#         x=C.reduce_mean(x,[1,2],keepdims=False)
#         x=C.exp(x)/reduce_sum((C.exp(x)))
#         return x
#     elif classifier_type == 'gcd_conv':
#         x=C.reduce_mean(x,[1,2],keepdims=True)
#         x=gcd_conv2d(x,(1,1), num_filters=num_classes, strides=1, padding=0, activation=Sigmoid, use_bias=False)
#         x=C.softmax(squeeze(x))
#         return x
#     else:
#         raise ValueError("only dense and global_avgpool are valid classifier_type")
#
#
#
#
#
# def Classifier( num_classes=10, is_multiselect=default_override_or(False), classifier_type=default_override_or('dense')):
#     is_multiselect = C.get_default_override(Classifier, is_multiselect=is_multiselect)
#     classifier_type = C.get_default_override(Classifier, classifier_type=classifier_type)
#     def internal_op(x):
#         return classifier(x,num_classes=num_classes, is_multiselect=is_multiselect, classifier_type=classifier_type)
#     return internal_op
#
#
#
#
#


#
# def resnet_bottleneck(input, out_num_filters, inter_out_num_filters):
#     c1 = conv_bn_relu(input, (1, 1), inter_out_num_filters)
#     c2 = conv_bn_relu(c1, (3, 3), inter_out_num_filters)
#     c3 = conv_bn(c2, (1, 1), out_num_filters, bn_init_scale=0)
#     p = c3 + input
#     return C.relu(p)
#
#
# def resnet_bottleneck_inc(input, out_num_filters, inter_out_num_filters, stride1x1, stride3x3):
#     c1 = conv_bn_relu(input, (1, 1), inter_out_num_filters, strides=stride1x1)
#     c2 = conv_bn_relu(c1, (3, 3), inter_out_num_filters, strides=stride3x3)
#     c3 = conv_bn(c2, (1, 1), out_num_filters, bn_init_scale=0)
#     stride = np.multiply(stride1x1, stride3x3)
#     s = conv_bn(input, (1, 1), out_num_filters, strides=stride) # Shortcut
#     p = c3 + s
#     return C.relu(p)