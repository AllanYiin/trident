from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import re

import numpy as np
import six
import numbers
from trident.backend.common import *
from trident.backend.tensorspec import *
from trident.backend.load_backend import *

__all__ = ['label_backend_adaptive','get_onehot','check_is_onehot']



if get_backend()== 'pytorch':
    from trident.backend.pytorch_backend import to_numpy, to_tensor
    from trident.backend.pytorch_ops import int_shape
    import torch
elif get_backend()== 'tensorflow':
    from trident.backend.tensorflow_backend import to_numpy, to_tensor
    from trident.backend.tensorflow_ops import int_shape
from trident.backend.tensorspec import *

def get_onehot(idx,len):
    if idx>=len:
        raise ValueError('')
    arr=np.zeros(len,dtype=np.float32)
    arr[idx]=1
    return arr


def check_is_onehot(label):
    label1=label.copy()
    label1[label1 > 0] = 1
    mean_lable1=label1.mean()
    if mean_lable1 < 2 * 1 / float(label.shape[-1]):
        return True
    else:
        return False

def label_backend_adaptive(label,label_mapping=None,object_type=None,**kwargs):
    if get_backend()== 'pytorch':
        if isinstance(label,np.ndarray):
            # binary mask
            if object_type == ObjectType.binary_mask:
                if label.ndim==2 :
                    label[label > 0] = 1
                    return label.astype(np.int64)
                elif label.ndim==3 and label.shape[-1] in [1,2]:
                    if label.shape[-1] ==2:
                        label=label[:,:,1]
                    elif label.shape[-1] ==1:
                        label = label[:, :,0]
                    label[label > 0] = 1
                    return label.astype(np.int64)
            elif object_type == ObjectType.label_mask:
                if label.ndim==2 :
                    return label.astype(np.int64)
                if label.ndim == 3 and label.shape[-1] >2:
                    if check_is_onehot(label):
                        label=np.argmax(label,-1).astype(np.int64)
                        return label
            label = label.astype(np.int64)
        elif isinstance(label, int):
            return label
        return label
    elif get_backend()== 'tensorflow':
        if isinstance(label, numbers.Integral):
            if isinstance(label_mapping, dict) and len(label_mapping) > 0:
                label_mapping = list(label_mapping.values())[0]
                label = get_onehot(label, len(label_mapping))
                return label
            elif label_mapping is None:
                return label
    return label


