from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cntk as C
import six
from cntk.layers.blocks import _inject_name

from trident.backend.common import epsilon


def _moments(x, axes=None,  keep_dims=True):
    _axes = list(tuple(axes))

    norm_mean = C.reduce_mean(x,axis=_axes)
    norm_variance=C.reduce_mean(C.square(x-C.stop_gradient(norm_mean)),axis=_axes)
    return norm_mean, norm_variance

# def moments2(x, axes=None, shift=None, keep_dims=False):
#     _axes = tuple(axes)
#     if shift is None:
#         shift = x
#         # Compute true mean while keeping the dims for proper broadcasting.
#         for axis in _axes:
#             shift = C.reduce_mean(shift, axis=axis)
#     shift = C.stop_gradient(shift)
#     shifted_mean = C.minus(x, shift)
#     for axis in _axes:
#         shifted_mean = C.reduce_mean(shifted_mean, axis=axis)
#     variance_mean = C.square(C.minus(x, shift))
#     for axis in _axes:
#         variance_mean = C.reduce_mean(variance_mean, axis=axis)
#     variance = C.minus(variance_mean, C.square(shifted_mean))
#     mean = C.plus(shifted_mean, shift)
#     if not keep_dims:
#         mean = C.squeeze(mean, _axes)
#         variance = C.squeeze(variance, _axes)
#     return mean, variance

def BatchNormalization(
        axis=0,
        center=True,
        scale=True,
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


def InstanceNormalization(
        axis=0,
        center=True,
        scale=True,
        epsilon=1e-5,
        name=''):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
       Normalize the activations of the previous layer at each step,
       i.e. applies a transformation that maintains the mean activation
       close to 0 and the activation standard deviation close to 1.
      """
    def apply_x(x):
        reduction_axes = list(range(0, len(x.shape)))
        if (axis is not None):
            del reduction_axes[axis]

        mean = C.reduce_mean(x, axis=reduction_axes, keepdims=True)
        stddev =C.reduce_mean(C.square(x-C.stop_gradient(mean)),axis=reduction_axes, keepdims=True)
        normed = (x - mean) / (stddev+epsilon)

        gamma =C.parameter(normed.shape,init=1)
        beta =C.parameter(normed.shape,init=0)
        if scale:
            normed=normed*gamma
        if center:
            normed=normed+beta
        return _inject_name(normed,name=name)
    return apply_x


def GroupNormalization(
                 groups=32,
                 axis=0,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 name=''):
    def apply_x(x):
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
            gamma = C.parameter(normed.shape, init=1)
            beta = C.parameter(normed.shape, init=0)
            if scale:
                normed = normed * gamma
            if center:
                normed = normed + beta
            return  _inject_name(normed,name=name)
        elif len(x.shape) == 1:
            c= x.shape
            x = C.reshape(x, (groups, c // groups))
            group_mean, group_variance= _moments(x,keep_dims=True)
            group_std = C.sqrt(group_variance) + epsilon
            normed = (x - group_mean) / group_std
            normed = C.reshape(normed, c)
            gamma = C.parameter(normed.shape, init=1)
            beta = C.parameter(normed.shape, init=0)
            if scale:
                normed = normed * gamma
            if center:
                normed = normed + beta
            return  _inject_name(normed,name=name)
    return apply_x




def L2Normalization(axis=None,epsilon=epsilon, name=''):
    def apply_x(x):
        x=x / C.sqrt(C.reduce_sum(C.square(x),axis=axis) + epsilon)
        return _inject_name(x,name=name)
    return apply_x

def get_normalization(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        fn=BatchNormalization
        mod=fn.__module__
        obj_dict = fn.__globals__
        for k, v in obj_dict.items():
            if (k == identifier or k.replace('Normalization','').lower()==identifier.lower() or (k!='L2Normalization' and str(k[0]).lower()==str(identifier[0]).lower() and len(identifier)<=2)) and mod=='trident.backend.cntk_normalizations':
                fn = v
                return fn
        raise ValueError('Not valid normalization functions name : ' + str(identifier))


