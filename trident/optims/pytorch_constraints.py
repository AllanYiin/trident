from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..backend.common import  get_session
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['max_norm', 'non_neg_norm', 'unit_norm', 'min_max_norm', 'maxnorm', 'nonnegnorm', 'unitnorm', 'minmaxnorm', 'get_constraints']


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_session=get_session()
_epsilon=_session.epsilon

def max_norm(model, max_value=3):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_value)
            param = param * (desired / (_epsilon + norm))

def non_neg_norm(model):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param = torch.clamp(param, 0, torch.inf)

def unit_norm(model):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            param = param/ (_epsilon + norm)

def min_max_norm(model,min_value=0.0, max_value=1.0, rate=3.0, axis=0):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=axis, keepdim=True)
            desired = rate *torch.clamp(norm, min_value, max_value)+ (1 - rate) * norm
            param = param * (desired / (_epsilon + norm))



maxnorm = max_norm
nonnegnorm = non_neg_norm
unitnorm = unit_norm
minmaxnorm=min_max_norm


def get_constraints(constraint):
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