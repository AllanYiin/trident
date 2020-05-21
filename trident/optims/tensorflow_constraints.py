from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
from trident.backend.common import get_session,epsilon
from trident.backend.tensorflow_ops import *

__all__ = ['max_norm', 'non_neg_norm', 'unit_norm', 'min_max_norm', 'maxnorm', 'nonnegnorm', 'unitnorm', 'minmaxnorm', 'get_constraint']

_session=get_session()
_epsilon=_session.epsilon


def  max_norm(model,max_value=3, axis=0):
    """
    MaxNorm weight constraint.
    Constrains the weights incident to each hidden unit to have a norm less than or equal to a desired value.
    Args:
        model : the model contains  weights need to setting the constraints.
        max_value (float):the maximum norm value for the incoming weights
        axis (int):axis along which to calculate weight norms.

    """
    ws = model.weights()
    for i in range(len(ws)):
        w = ws[i]
        norms = sqrt(reduce_sum(square(w), axis=axis, keepdims=True))
        desired = clip(norms, 0, max_value)
        w=w * (desired / (epsilon() + norms))

def  non_neg_norm(model):
    """
    Constrains the weights to be non-negative.
    Args:
        model : the model contains  weights need to setting the constraints.

    """
    ws = model.get_weights()
    for i in range(len(ws)):
        w = ws[i]
        w=w * tf.cast(greater_equal(w, 0.), tf.float32)

def  unit_norm(model,axis=0):
    """
    Constrains the weights incident to each hidden unit to have unit norm.
    Args:
        axis (int):axis along which to calculate weight norms.
        model : the model contains  weights need to setting the constraints.

    """
    ws = model.get_weights()
    for i in range(len(ws)):
        w = ws[i]
        w=w / (epsilon() +sqrt(reduce_sum(square(w),axis=axis,keepdims=True)))

def  min_max_norm(model,min_value=0.0, max_value=1.0, rate=3.0, axis=0):
    """
    MinMaxNorm weight constraint.
    Constrains the weights incident to each hidden unit to have the norm between a lower bound and an upper bound.

    Args:
        model : the model contains  weights need to setting the constraints.
        min_value (float):the minimum norm for the incoming weights.
        max_value ()float:the maximum norm for the incoming weights.
        rate (float):rate for enforcing the constraint: weights will be rescaled to yield (1 - rate) * norm + rate * norm.clip(min_value, max_value). Effectively, this means that rate=1.0 stands for strict enforcement of the constraint, while rate<1.0 means that weights will be rescaled at each step to slowly move towards a value inside the desired interval.
        axis (int): axis along which to calculate weight norms
    """


    ws = model.get_weights()
    for i in range(len(ws)):
        w=ws[i]
        norms = sqrt(reduce_sum(square(w), axis=axis, keepdims=True))
        desired = (rate * clip(norms, min_value, max_value) + (1 - rate) * norms)
        w= w * (desired / (epsilon() + norms))




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

