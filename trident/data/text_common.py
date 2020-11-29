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


def text_backend_adaption(text):
    if  get_backend() == 'tensorflow':
        if text.dtype==np.int64 and text.ndim ==1:
            pass
        elif text.ndim ==2:
            text=text.astype(np.float32)
    else:
        if text.dtype == np.int64:
            pass
        elif text.ndim ==2:
            text=text.astype(np.float32)
    return text


def reverse_text_backend_adaption(text):
    # if get_backend() == 'tensorflow':
    #     if text.dtype == np.int64 and text.ndim == 1:
    #         pass
    #     elif text.ndim == 2:
    #         text =argmax(text,-1)
    # else:
    #     if text.dtype == np.int64:
    #         pass
    #     elif text.ndim == 2:
    #         text = argmax(text, -1)
    return text