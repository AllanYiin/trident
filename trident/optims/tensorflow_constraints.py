from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
from trident.backend.tensorflow_backend import Parameter, Layer

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

    def apply_constraint(t: Tensor):
        w_data = None
        if isinstance(t, tf.Variable):
            w_data = t.value().detach()
        else:
            w_data = t.copy().detach()
        norms = sqrt(reduce_sum(square(w_data), axis=axis, keepdims=True))
        desired = clip(norms, 0, max_value)
        param_applied = w_data * (desired / (epsilon() + norms))
        param_applied = param_applied.detach()
        return param_applied

    if is_tensor(model):
        model = apply_constraint(model)
    elif isinstance(model, Layer):
        for name, param in model.named_parameters():
            if 'bias' not in name and param is not None and param.trainable == True:
                param.assign(apply_constraint(param))


def  non_neg_norm(model):
    """
    Constrains the weights to be non-negative.
    Args:
        model : the model contains  weights need to setting the constraints.

    """


    def apply_constraint(t: Tensor):
        w_data = None
        if isinstance(t, tf.Variable):
            w_data = t.value().detach()
        else:
            w_data = t.copy().detach()
        param_applied =w_data * tf.cast(greater_equal(param, 0.), tf.float32)
        param_applied = param_applied.detach()
        return param_applied

    if is_tensor(model):
        model = apply_constraint(model)
    elif isinstance(model, Layer):
        for name, param in model.named_parameters():
            if 'bias' not in name and param is not None and param.trainable == True:
                param.assign(apply_constraint(param))


def  unit_norm(model,axis=0):
    """
    Constrains the weights incident to each hidden unit to have unit norm.
    Args:
        axis (int):axis along which to calculate weight norms.
        model : the model contains  weights need to setting the constraints.

    """
    def apply_constraint(t: Tensor):
        w_data = None
        if isinstance(t, tf.Variable):
            w_data = t.value().detach()
        else:
            w_data = t.copy().detach()
        param_applied = w_data/ (epsilon() +sqrt(reduce_sum(square(w_data),axis=axis,keepdims=True)))
        param_applied = param_applied.detach()
        return param_applied

    if is_tensor(model):
        model = apply_constraint(model)
    elif isinstance(model, Layer):
        for name, param in model.named_parameters():
            if 'bias' not in name and param is not None and param.trainable == True:
                param.assign(apply_constraint(param))


def  min_max_norm(model,min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    """
    MinMaxNorm weight constraint.
    Constrains the weights incident to each hidden unit to have the norm between a lower bound and an upper bound.

    Args:
        model : the model contains  weights need to setting the constraints.
        min_value (float):the minimum norm for the incoming weights.
        max_value ()float:the maximum norm for the incoming weights.
        rate (float):rate for enforcing the constraint: weights will be rescaled to yield (1 - rate) * norm + rate * norm.clip(min_value, max_value). Effectively, this means that rate=1.0 stands for strict enforcement of the constraint, while rate<1.0 means that weights will be rescaled at each step to slowly move towards a value inside the desired interval.
        axis (int): axis along which to calculate weight norms

    Examples:
        >>> t=random_normal((2,64,64,32),mean=5,std=10)
        >>> print(t)
        >>> t1=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)(t)
        >>> print(t1)
        >>> t2=min_max_norm(t,min_value=0.0, max_value=1.0, rate=1.0, axis=0)
        >>> print(t2)
        >>> np.testing.assert_almost_equal(to_numpy(t1),to_numpy(t2),decimal=6,verbose=True)



    """
    def apply_constraint(t:Tensor):
        w_data=None
        if isinstance(t, tf.Variable):
            w_data = t.value().detach()
        else:
            w_data=t.copy().detach()
        norms = sqrt(reduce_sum(square(w_data), axis=axis, keepdims=True))
        desired = (rate * clip(norms, min_value, max_value) + (1 - rate) * norms)
        param_applied = w_data * (desired / (epsilon() + norms))
        return param_applied

    if is_tensor(model):
        model=apply_constraint(model)
    elif isinstance(model,Layer):
        for name, param in model.named_parameters():
            if 'bias' not in name and param is not None  and param.trainable==True:
                param.assign(apply_constraint(param))


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

