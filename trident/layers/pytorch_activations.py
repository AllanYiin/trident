#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
from .load_backend import *
import torch
import torch.nn as nn
import torch.nn.functional as F # import torch functions


class identity(nn.Module):
    def __init__(self):
        super(identity,self).__init__()
        self.name = self.__class__.__name__

    def forward(self, x):
        return x


class sigmoid(nn.Module):
    def __init__(self):
        super(sigmoid, self).__init__()
        self.name = self.__class__.__name__
    def forward(self, x):
        return F.sigmoid(x)

class tanh(nn.Module):
    def __init__(self):
        super(tanh, self).__init__()
        self.name = self.__class__.__name__
    def forward(self, x):
        return F.tanh(x)


class relu(nn.Module):
    def __init__(self):
        super(relu, self).__init__()
        self.name = self.__class__.__name__
    def forward(self, x,upper_limit=None,):
        if upper_limit <= 0:
            raise ValueError('Upper limit should greater than 0!')
        if upper_limit is not None:
            return torch.clamp(F.relu(x), min=0, max=upper_limit)
        return F.relu(x)



