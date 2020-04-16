from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
import cntk as C
import six
from cntk.layers.blocks import _inject_name

from backend.common import epsilon



def _moments(x, axes=None,  keep_dims=True):
    _axes = list(tuple(axes))

    norm_mean = C.reduce_mean(x,axis=_axes)
    norm_variance=C.reduce_mean(C.square(x-C.stop_gradient(norm_mean)),axis=_axes)
    return norm_mean, norm_variance

@C.typemap
def BatchNormalization(
        axis=0,
        epsilon=1e-5,
        name=''):
    return C.layers.BatchNormalization(map_rank=1,use_cntk_engine=False,disable_regularization=False,epsilon=epsilon, name=name)

def LayerNormalization(
        axis=0,
        center=True,
        scale=True,
        epsilon=1e-5,
        name=''):
    return C.layers.LayerNormalization(initial_scale=1, initial_bias=0, epsilon=epsilon, name=name)

@C.typemap
def InstanceNormalization(
        axis=0,
        center=False,
        scale=False,
        epsilon=1e-5,
        name=''):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
       Normalize the activations of the previous layer at each step,
       i.e. applies a transformation that maintains the mean activation
       close to 0 and the activation standard deviation close to 1.
      """
    
    def instance_normalization(x):
        reduction_axes = list(range(0, len(x.shape)))
        if (axis is not None):
            del reduction_axes[axis]
        mean = C.reduce_mean(x, axis=reduction_axes, keepdims=True)
        stddev =C.reduce_mean(C.square(x-C.stop_gradient(mean)),axis=reduction_axes, keepdims=True)
        normed = (x - mean) / (stddev+epsilon)
        if scale:
            gamma =C.parameter(mean.shape,init=1)
            normed = normed * gamma
        if center:
            beta = C.parameter(mean.shape, init=0)
            normed=normed+beta
        return _inject_name(normed,name=name)
    return instance_normalization


InstanceNorm2d=InstanceNormalization

@C.typemap
def GroupNormalization(
                 groups=32,
                 axis=0,
                 epsilon=1e-5,
                 center=False,
                 scale=False,
                 name=''):
    def group_normalization(x):
        reduction_axes = list(range(0, len(x.shape)))
        if (axis is not None):
            del reduction_axes[axis]
        if len(x.shape) != 3 and len(x.shape) != 1:
            raise ValueError('Inputs should have rank ' + str(4) + " or " + str(2) + '; Received input shape:',str(x.shape))
        if len(x.shape) == 3:
            c, h, w = x.shape
            if c < groups:
                raise ValueError('Input channels should be larger than group size' + '; Received input channels: ' + str(c) + '; Group size: ' + str(groups))
            x = C.reshape(x, (groups, c // groups, h, w))
            group_mean,group_variance= _moments(x, axes=reduction_axes,keep_dims=True)
            group_std = C.sqrt(group_variance )+ epsilon
            normed = (x - group_mean) /group_std
            normed = C.reshape(normed, (c, h, w))

            if scale:
                gamma = C.parameter(normed.shape, init=1)
                normed = normed * gamma
            if center:
                beta = C.parameter(normed.shape, init=0)
                normed = normed + beta
            return  _inject_name(normed,name=name)
        elif len(x.shape) == 1:
            c= x.shape
            x = C.reshape(x, (groups, c // groups))
            group_mean, group_variance= _moments(x,keep_dims=True)
            group_std = C.sqrt(group_variance) + epsilon
            normed = (x - group_mean) / group_std
            normed = C.reshape(normed, c)
            if scale:
                gamma = C.parameter(group_mean.shape, init=1)
                normed = normed * gamma
            if center:
                beta = C.parameter(group_mean.shape, init=0)
                normed = normed + beta
            return  _inject_name(normed,name=name)
    return group_normalization




def L2Normalization(axis=None,epsilon=epsilon, name=''):
    def apply_x(x):
        x=x / C.sqrt(C.reduce_sum(C.square(x),axis=axis) + epsilon)
        return _inject_name(x,name=name)
    return apply_x

def get_normalization(identifier):
    if identifier is None:
        return None
    if isinstance(identifier,str):
        if identifier.lower().strip() in ['instance','in','i']:
            return InstanceNormalization
        elif  identifier.lower().strip() in ['batch','b']:
            return InstanceNormalization
        elif  identifier.lower().strip() in ['group','g']:
            return GroupNormalization
    else:
        raise ValueError('Not valid normalization functions name : ' + str(identifier))


