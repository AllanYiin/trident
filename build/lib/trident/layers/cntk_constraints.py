from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from backend.common import  epsilon,floatx
import numpy as np
import cntk as C

def  default_constrains(w):
    w=C.clip(w,-1e5,1e5)
    return w

def  max_norm(w,max_value=2, axis=0):
    w=default_constrains(w)
    norms = C.sqrt(C.reduce_sum(C.square(w), axis, keepdims=True))
    desired = C.clip(norms, epsilon() , max_value)
    return w * (desired / (epsilon() + norms))

def  non_neg(w:C.Parameter):
    w = default_constrains(w)
    return w * C.greater_equal(w, 0.)

def  unit_norm(w:C.Parameter,axis):
    w = default_constrains(w)
    return w / (epsilon() + C.sqrt(C.sum(C.square(w), axis,keepdims=True)))

def  min_max_norm(w:C.Parameter,min_value=0.0, max_value=1.0, rate=3.0, axis=0):
    w = default_constrains(w)
    norms = C.sqrt(C.reduce_sum(C.square(w), axis, keepdims=True))
    desired = (rate * C.clip(norms, min_value, max_value) + (1 - rate) * norms)
    return w * (desired / (epsilon() + norms))



# Legacy aliases.

maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm
minmaxnorm=min_max_norm


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
