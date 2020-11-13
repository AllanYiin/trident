from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
from trident.backend.common import get_session,epsilon
from trident.backend.tensorflow_ops import *

__all__ = []

_session=get_session()
_epsilon=_session.epsilon