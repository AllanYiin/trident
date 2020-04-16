from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import collections
import itertools
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn_ops
from tensorflow.python.client import device_lib
from .tensorflow_activations import get_activation
from itertools import repeat
import inspect

from collections import OrderedDict
from ..backend.common import get_session, gcd, get_divisors, isprime, next_prime, prev_prime, nearest_prime,unpack_singleton,enforce_singleton

_tf_data_format= 'channels_last'

__all__ = ['Dense', 'Flatten', 'Concatenate','Concate','Add','Subtract', 'Conv1d', 'Conv2d', 'Conv3d',  'TransConv2d', 'TransConv3d','Reshape','Dropout','Lambda']


_session = get_session()

_device='CPU'
for device in device_lib.list_local_devices():
      if tf.DeviceSpec.from_string(device.name).device_type == 'GPU':
          _device='GPU'
          break

_epsilon = _session.epsilon


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)



class MaxPool1D(tf.keras.layers.MaxPool1D):
    def __init__(self,kernel_size, stride=None, auto_pad=True, **kwargs ):

        super(MaxPool1D, self).__init__( kernel_size,stride,
                 padding='valid', data_format='channels_last', **kwargs)

    @property
    def kernel_size(self):
        return super().pool_size

    @kernel_size.setter
    def kernel_size(self,value):
        self.pool_size=_single(value)