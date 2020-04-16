import os
import sys
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import numpy as np
from .common import get_session,addindent,get_time_prefix,get_class

__all__ = ['argmax','expand_dims','meshgrid',]


def argmax(t:torch.Tensor,axis=1):
    _, idx = t.max(axis)
    return idx

def expand_dims(t:torch.Tensor,axis=0):
    return t.unsqueeze(axis)


def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)