from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import math
import os
import random
import re
import builtins
import string
import time
from itertools import repeat
from functools import partial
import inspect
import cv2
import numpy as np
import six
from trident.backend.common import *

__all__ = ['text_backend_adaption','reverse_text_backend_adaption']


if get_backend()== 'pytorch':
    from trident.backend.pytorch_backend import to_numpy, to_tensor, ObjectType
    from trident.backend.pytorch_ops import int_shape
    from trident.layers.pytorch_layers import Embedding
    import torch
elif get_backend()== 'tensorflow':
    from trident.backend.tensorflow_backend import to_numpy, to_tensor,ObjectType
    from trident.backend.tensorflow_ops import int_shape



def chinese_full2half():
    """Convert all fullwidth Chinese characters to halfwidth .

    Returns:

    """
    def string_op(input_str:str):
        rstring = ""
        for uchar in input_str:
            u_code = ord(uchar)
            if u_code == 0x3000 or u_code == 12288 or uchar == string.whitespace:
                u_code = 32
            elif 65281 <= u_code <= 65374:
                u_code -= 65248
            rstring += chr(u_code)
        return rstring
    return string_op

def chinese_half2full():
    """Convert all halfwidth Chinese characters to fullwidth .

    Returns:

    """
    def string_op(input_str:str):
        rstring = ""
        for uchar in input_str:
            u_code = ord(uchar)
            if u_code == 32:
                u_code = 12288
            elif 33 <= u_code <= 126:
                u_code += 65248
            rstring += chr(u_code)
        return rstring
    return string_op

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



# def  char2embedding(embedding:Embedding):
#     def img_op(sentence:str,**kwargs):
#         sentence = reverse_image_backend_adaption(image)
#         norm_mean = mean
#         norm_std = std
#         if isinstance(norm_mean, tuple):
#             norm_mean = list(norm_mean)
#
#         if isinstance(norm_std, tuple):
#             norm_std = list(norm_std)
#
#         if isinstance(norm_mean, (float, int)) and isinstance(norm_std, (float, int)) and image.ndim == 3:
#             return image * float(norm_std) + float(norm_mean)
#         elif isinstance(norm_mean, list) and isinstance(norm_std, list) and len(norm_mean) == 1 and len(norm_std) == 1:
#             return image * float(norm_std[0]) + float(norm_mean[0])
#         elif isinstance(norm_mean, list) and isinstance(norm_std, list) and len(norm_mean) == 3 and len(norm_std) == 3:
#             norm_mean = np.reshape(np.array(norm_mean), (1, 1, 3))
#             norm_std = np.reshape(np.array(norm_std), (1, 1, 3))
#             return image * norm_std + norm_mean
#         return image
#
#     return img_op
