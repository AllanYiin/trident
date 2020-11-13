from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import math
import os
import random
import re
import builtins
import time
from itertools import repeat
from functools import partial
import inspect
import cv2
import numpy as np
import six
from trident.backend.common import *

__all__ = ['text_backend_adaption','reverse_text_backend_adaption']
_session = get_session()
_backend = _session.backend

def text_backend_adaption(text):
    if  _session.backend == 'tensorflow':
        if text.ndim==2: #gray-scale image
            text=np.expand_dims(text,-1).astype(np.float32)
        elif text.ndim in (3,4):
            text=text.astype(np.float32)
    else:
        if text.ndim==2: #gray-scale image
            text=np.expand_dims(text,0).astype(np.float32)
        elif text.ndim==3:
            text = np.transpose(text, [2, 0, 1]).astype(np.float32)
        elif text.ndim==4:
            text = np.transpose(text, [0, 3, 1, 2]).astype(np.float32)
    return text


def reverse_text_backend_adaption(text):
    if _session.backend in ['pytorch', 'cntk'] and text.ndim == 3 and text.shape[0] in [3, 4]:
        text = np.transpose(text, [1, 2, 0]).astype(np.float32)
    elif _session.backend in ['pytorch', 'cntk'] and text.ndim == 4 and text.shape[1] in [3, 4]:
        text = np.transpose(text, [0, 2, 3, 1]).astype(np.float32)
    return text