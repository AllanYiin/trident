from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from ..backend.common import  get_session
import numpy as np
import cntk as C

__all__ = ['max_norm','non_neg','unit_norm','min_max_norm','maxnorm','nonneg','unitnorm','minmaxnorm','get_constraints']

_session=get_session()
_epsilon=_session.epsilon


def  max_norm(model,max_value=3, axis=0):
    for p in model.para:
        norms = C.sqrt(C.reduce_sum(C.square(p.value), axis, keepdims=True))
        desired = C.clip(norms, 0 , max_value)
        C.assign(p,p * (desired / (_epsilon + norms)))


def  non_neg(model):
    for p in model.para:
        C.assign(p,p * C.greater_equal(p, 0.))

def  unit_norm(model,axis):
    for p in model.para:
        C.assign(p, p / (_epsilon+ C.sqrt(C.sum(C.square(p.value), axis,keepdims=True))))

def  min_max_norm(model,min_value=0.0, max_value=1.0, rate=3.0, axis=0):
    for p in model.para:
        norms = C.sqrt(C.reduce_sum(C.square(p.value), axis, keepdims=True))
        desired = (rate * C.clip(norms, min_value, max_value) + (1 - rate) * norms)
        C.assign(p,p * (desired / (_epsilon + norms)))



# Legacy aliases.

maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm
minmaxnorm=min_max_norm
default_constrains=functools.partial(min_max_norm,functools.partial)

def get_constraints(constraint):
    if constraint in ['maxnorm','max_norm']:
        return max_norm
    elif constraint in ['non_neg','nonneg']:
        return non_neg
    elif constraint in ['unit_norm','unitnorm']:
        return unit_norm
    elif constraint in ['min_max_norm', 'minmaxnorm']:
        return min_max_norm
    else:
        return None
