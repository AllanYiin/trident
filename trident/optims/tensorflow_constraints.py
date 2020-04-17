import functools

import numpy as np
import tensorflow as tf
from  tensorflow.keras import backend as K

from ..backend.common import get_session

__all__ = ['max_norm', 'non_neg_norm', 'unit_norm', 'min_max_norm', 'maxnorm', 'nonnegnorm', 'unitnorm', 'minmaxnorm', 'get_constraint']

_session=get_session()
_epsilon=_session.epsilon


def  max_norm(model,max_value=3, axis=0):
    ws = model.weights()
    for i in range(len(ws)):
        w = ws[i]
        norms = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(w), axis=axis, keepdims=True))
        desired = tf.math.clip(norms, 0, max_value)
        w=w * (desired / (K.epsilon() + norms))

def  non_neg_norm(model):
    ws = model.get_weights()
    for i in range(len(ws)):
        w = ws[i]
        w=w * K.cast(K.greater_equal(w, 0.), K.floatx())

def  unit_norm(model,axis):
    ws = model.get_weights()
    for i in range(len(ws)):
        w = ws[i]
        w=w / (K.epsilon() + K.sqrt(tf.math.reduce_sum(tf.math.square(w),axis=axis.axis,keepdims=True)))

def  min_max_norm(model,min_value=0.0, max_value=1.0, rate=3.0, axis=0):
        ws = model.get_weights()
        for i in range(len(ws)):
            w=ws[i]
            norms = K.sqrt(tf.math.reduce_sum(tf.math.square(w), axis=axis, keepdims=True))
            desired = (rate * K.clip(norms, min_value, max_value) + (1 - rate) * norms)
            w= w * (desired / (K.epsilon() + norms))







# Legacy aliases.

maxnorm = max_norm
nonnegnorm = non_neg_norm
unitnorm = unit_norm
minmaxnorm=min_max_norm
default_constrains=functools.partial(min_max_norm,functools.partial)

def get_constraint(constraint):
    if constraint in ['maxnorm','max_norm']:
        return max_norm
    elif constraint in ['non_neg','nonneg']:
        return non_neg_norm
    elif constraint in ['unit_norm','unitnorm']:
        return unit_norm
    elif constraint in ['min_max_norm', 'minmaxnorm']:
        return min_max_norm
    else:
        return None

