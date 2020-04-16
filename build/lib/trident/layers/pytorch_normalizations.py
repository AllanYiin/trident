from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backend.common import epsilon,get_function
import numpy as np

__all__ = ['InstanceNorm2d','BatchNorm2d','GroupNorm2d','LayerNorm2d','get_normalization']

class InstanceNorm2d(nn.Module):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True):
        """
        http://pytorch.org/docs/stable/nn.html#batchnorm1d

        Args:
            dim: 1d, 2d, or 3d BatchNorm
         eps: nn.BatchNorm parameter
            momentum: nn.BatchNorm parameter
            affine: nn.BatchNorm parameter
            track_running_stats: nn.BatchNorm parameter
        """
        super().__init__()
        self.norm_kwargs = dict(eps=eps,
            momentum=momentum,
            affine=affine)
        self._NormClass = nn.InstanceNorm2d
        self._normalizer = None

    def forward(self, x):
        if self._normalizer is None:
            channel =x.size(1)
            self._normalizer = self._NormClass(channel, **self.norm_kwargs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self._normalizer(x)

class BatchNorm2d(nn.Module):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        """
        http://pytorch.org/docs/stable/nn.html#batchnorm1d

        Args:
            dim: 1d, 2d, or 3d BatchNorm
         eps: nn.BatchNorm parameter
            momentum: nn.BatchNorm parameter
            affine: nn.BatchNorm parameter
            track_running_stats: nn.BatchNorm parameter
        """
        super().__init__()
        self.norm_kwargs = dict(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self._NormClass = nn.BatchNorm2d
        self._normalizer = None

    def forward(self, x):
        if self._normalizer is None:
            channel =x.size(1)
            self._normalizer = self._NormClass(channel, **self.norm_kwargs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self._normalizer(x)

class GroupNorm2d(nn.Module):
    def __init__(self, num_groups, eps=1e-5, affine=True):
        super().__init__()
        self.norm_kwargs = dict(eps=eps, affine=affine)
        self.num_groups = num_groups
        self._normalizer = None

    def forward(self, x):
        if self._normalizer is None:
            channel =x.size(1)
            assert channel % self.num_groups == 0, 'number of groups {} must divide number of channels {}'.format( self.num_groups, channel)
            self._normalizer = nn.GroupNorm(self.num_groups, channel, **self.norm_kwargs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self._normalizer(x)


class LayerNorm2d(nn.Module):
    def __init__(self, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        """
        http://pytorch.org/docs/stable/nn.html#batchnorm1d

        Args:
            dim: 1d, 2d, or 3d BatchNorm
         eps: nn.BatchNorm parameter
            momentum: nn.BatchNorm parameter
            affine: nn.BatchNorm parameter
            track_running_stats: nn.BatchNorm parameter
        """
        super().__init__()
        self.norm_kwargs = dict(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self._NormClass = _LayerNorm
        self._normalizer = None

    def forward(self, x):
        if self._normalizer is None:
            channel =x.size(1)
            self._normalizer = self._NormClass(channel, **self.norm_kwargs).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self._normalizer(x)

class _LayerNorm(nn.Module):
    """
    Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf).
    """
    def __init__(self, last_dim_size, eps=1e-6):
        """
        :param last_dim_size: Size of last dimension.
        :param eps: Small number for numerical stability (avoid division by zero).
        """
        super(_LayerNorm, self).__init__()
        self._a_2 = nn.Parameter(torch.ones(last_dim_size))
        self._b_2 = nn.Parameter(torch.zeros(last_dim_size))
        self._eps = eps
    def forward(self, x):
        """
        :param x: Tensor to be layer normalized.
        :return: Layer normalized Tensor.
        """
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = x.std(dim=-1, keepdim=True).detach()
        return self._a_2 * (x - mean) / (std + self._eps) + self._b_2

def get_normalization(fn_name):
    if fn_name is None:
        return None
    if isinstance(fn_name, str):
        if fn_name.lower().strip() in ['instance','in','i']:
            return InstanceNorm2d
        elif  fn_name.lower().strip() in ['batch','b']:
            return BatchNorm2d
        elif  fn_name.lower().strip() in ['group','g']:
            return GroupNorm2d
    fn_modules = ['torch', 'torch.nn','torch.nn.functional', 'trident.layers.pytorch_normalizations']
    normalization_fn_ = get_function(fn_name, fn_modules)
    normalization_fn = normalization_fn_
    return normalization_fn
