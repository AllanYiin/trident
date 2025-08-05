from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import inspect
import os
import sys
import builtins
import random
import shutil
import string
import sys
import time
import uuid
import json
from typing import Callable, Any

import numpy as np
from trident.context import split_path, make_dir_if_need, sanitize_path

from trident.backend.common import to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict,  if_none
from trident.backend.tensorspec import *
_session = get_session()
_backend = _session.backend
if _backend == 'pytorch':
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *



class Metric(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name=kwargs.get('name')
        self.format=kwargs.get('format')
        self.aggregate = kwargs.get('aggregate')


    def _forward_unimplemented(self, output: Any, target: Any, **kwargs) -> None:
        raise NotImplementedError
    calculate_metric=_forward_unimplemented



def as_metric(format=None,aggregate='mean',name=None):
    def _f(fun):
        m=Metric
        m.calculate_metric = fun
        m.__name__ = m.__qualname__ = if_none(name,fun.__name__),
        m.format=format
        m.aggregate=aggregate
        m.__doc__ = m.__doc__
        m._signature = inspect.Signature(fun)
    return _f



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count