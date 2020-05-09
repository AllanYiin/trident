from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trident.backend.common import get_session
from trident.backend.pytorch_ops import *

__all__ = ['max_norm', 'non_neg_norm', 'unit_norm', 'min_max_norm', 'maxnorm', 'nonnegnorm', 'unitnorm', 'minmaxnorm', 'get_constraint']

_session=get_session()
_epsilon=_session.epsilon

def max_norm(model, max_value=3,axis=0):
    '''
    MaxNorm weight constraint.
    Constrains the weights incident to each hidden unit to have a norm less than or equal to a desired value.
    Args:
        model : the model contains  weights need to setting the constraints.
        max_value (float):the maximum norm value for the incoming weights
        axis (int):axis along which to calculate weight norms.

    '''
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=axis, keepdim=True)
            desired = torch.clamp(norm, 0, max_value)
            param = param * (desired / (_epsilon + norm))

def non_neg_norm(model):
    '''
    Constrains the weights to be non-negative.
    Args:
        model : the model contains  weights need to setting the constraints.

    '''
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param = torch.clamp(param, 0, torch.inf)

def unit_norm(model,axis=0):
    '''
    Constrains the weights incident to each hidden unit to have unit norm.
    Args:
        axis (int):axis along which to calculate weight norms.
        model : the model contains  weights need to setting the constraints.

    '''
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=axis, keepdim=True)
            param = param/ (_epsilon + norm)

def min_max_norm(model,min_value=0, max_value=1, rate=3.0, axis=0):
    '''
    MinMaxNorm weight constraint.
    Constrains the weights incident to each hidden unit to have the norm between a lower bound and an upper bound.

    Args:
        model : the model contains  weights need to setting the constraints.
        min_value (float):the minimum norm for the incoming weights.
        max_value ()float:the maximum norm for the incoming weights.
        rate (float):rate for enforcing the constraint: weights will be rescaled to yield (1 - rate) * norm + rate * norm.clip(min_value, max_value). Effectively, this means that rate=1.0 stands for strict enforcement of the constraint, while rate<1.0 means that weights will be rescaled at each step to slowly move towards a value inside the desired interval.
        axis (int): axis along which to calculate weight norms


    '''
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=axis, keepdim=True)
            desired = rate *torch.clamp(norm, min_value, max_value)+ (1 - rate) * norm
            param = param * (desired / (_epsilon + norm))



maxnorm = max_norm
nonnegnorm = non_neg_norm
unitnorm = unit_norm
minmaxnorm=min_max_norm


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