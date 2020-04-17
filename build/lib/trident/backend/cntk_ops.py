from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cntk as C
import numpy as np

from .common import epsilon


def cumsum(x, axis: int = -1):
    """ Calculates the cumulative sum across a static axis
    Arguments:
        x: input tensor
        axis (int): static axis of tensor to cumsum over
    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    d = x.shape[axis]
    u = C.constant(np.triu(np.ones((d, d))).astype(x.dtype))
    if axis != -1:
        x = C.swapaxes(x, -1, axis)
    z = C.times(x, u)
    if axis != -1:
        z = C.swapaxes(z, -1, axis)
    return z

def l2_normalize(x,axis=0,keepdims=False):
    x = x / C.sqrt(C.reduce_sum(C.square(x), axis=axis,keepdims=keepdims) + epsilon)
    return x

def element_cosine_distance(a, b,axis=0,keepdims=False):
    """ Calculates the pairewise cumulative sum across a static axis
    Arguments:
        x: input tensor
        axis (int): static axis of tensor to cumsum over
    Returns:
        :class:`~cntk.ops.functions.Function`
    """
    if a.shape[0]!=b.shape[0]:
        raise ValueError('Input tensures should have the same length at axis 0!')
    normalize_a = l2_normalize(a, axis=axis,keepdims=keepdims)
    normalize_b = l2_normalize(b, axis=axis,keepdims=keepdims)
    distance = 1 - tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance



def gram(x):
    features = C.minus(C.reshape(x,(x.shape[0],-1)), C.reduce_mean(x))
    return C.times_transpose(features, features)