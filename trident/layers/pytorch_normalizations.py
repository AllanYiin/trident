from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from trident.backend.common import epsilon, get_function, get_session, enforce_singleton,get_class
from trident.backend.pytorch_backend import Layer,get_device
from trident.backend.pytorch_ops import *

__all__ = ['InstanceNorm','BatchNorm','BatchNorm2d','BatchNorm3d','GroupNorm','GroupNorm2d','GroupNorm3d','LayerNorm2d','SpectralNorm','EvoNormB0','EvoNormS0','get_normalization']
_session = get_session()
_epsilon=_session.epsilon

def instance_std(x, eps=1e-5):
    rank=len(x.shape)-2
    new_shape=range(len(x.shape))
    _, var = moments(x, axis=new_shape[2:], keepdims=True)
    return sqrt(var + eps)

def group_std(x, groups, eps = 1e-5):
    rank = len(x.shape) - 2
    spaceshape=x.shape[2:]
    N = x.size(0)
    C = x.size(1)
    x1 = x.reshape(N,groups,-1)
    var = (x1.var(dim=-1, keepdim = True)+eps).reshape(N,groups,-1)
    return (x1 / var.sqrt()).reshape((N,C,)+spaceshape)


class BatchNorm(Layer):
    def __init__(self,  momentum=0.1, affine=True, track_running_stats=True, eps=1e-5, **kwargs):
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

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine :
            init.ones_(self.weight)
            init.zeros_(self.bias)
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)


    def build(self, input_shape):
        if self._built == False:
            self.input_filters= int(input_shape[0])
            if self.affine:
                self.weight = Parameter(torch.Tensor(self.input_filters))
                self.bias = Parameter(torch.Tensor(self.input_filters))
                init.ones_(self.weight)
                init.zeros_(self.bias)
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

            if self.track_running_stats:
                self.register_buffer('running_mean', torch.zeros(self.input_filters))
                self.register_buffer('running_var', torch.ones(self.input_filters))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            else:
                self.register_parameter('running_mean', None)
                self.register_parameter('running_var', None)
                self.register_parameter('num_batches_tracked', None)
            self.reset_parameters()
            self.to(get_device())
            self._built = True
    def forward(self, *x):
        x = enforce_singleton(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)
    def extra_repr(self):
        return '{input_filters}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
BatchNorm1d=BatchNorm
BatchNorm2d=BatchNorm
BatchNorm3d=BatchNorm



class GroupNorm(Layer):
    def __init__(self, num_groups,affine=True, eps=1e-5, **kwargs):
        super().__init__()
        self.affine=affine
        self.num_groups = num_groups
        self.num_filters = None
        self.eps = eps
        self.affine = affine


    def build(self, input_shape):
        if self._built == False :
            assert  self.input_filters % self.num_groups == 0, 'number of groups {} must divide number of channels {}'.format(self.num_groups,  self.input_filters)
            if self.affine:
                self.weight = Parameter(torch.Tensor(self.input_filters))
                self.bias = Parameter(torch.Tensor(self.input_filters))
                init.ones_(self.weight)
                init.zeros_(self.bias)
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

                self._built = True
    def forward(self, *x):
        x = enforce_singleton(x)
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
GroupNorm2d=GroupNorm
GroupNorm3d=GroupNorm


class InstanceNorm(Layer):
    def __init__(self,momentum=0.1, affine=True, track_running_stats=True, eps=1e-5, **kwargs):
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
        self.eps = _epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine :
            init.ones_(self.weight)
            init.zeros_(self.bias)
    def build(self, input_shape):
        if self._built == False:
            if self.affine:
                self.weight = Parameter(torch.Tensor(self.input_filters))
                self.bias = Parameter(torch.Tensor(self.input_filters))
                init.ones_(self.weight)
                init.zeros_(self.bias)
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

            if self.track_running_stats:
                self.register_buffer('running_mean', torch.zeros(self.input_filters))
                self.register_buffer('running_var', torch.ones(self.input_filters))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            else:
                self.register_parameter('running_mean', None)
                self.register_parameter('running_var', None)
                self.register_parameter('num_batches_tracked', None)
            self.reset_running_stats()
            self.to(get_device())
            self._built = True
    def forward(self, *x):
        x = enforce_singleton(x)
        return F.instance_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

InstanceNorm2d=InstanceNorm
InstanceNorm3d=InstanceNorm


class LayerNorm2d(Layer):
    """
    Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf).

    http://pytorch.org/docs/stable/nn.html#batchnorm1d

    Args:

        eps: nn.BatchNorm parameter
        momentum: nn.BatchNorm parameter
        affine: nn.BatchNorm parameter
        axis: 1d, 2d, or 3d BatchNorm
    """
    def __init__(self, momentum=0.1, affine=True, axis=-1,  eps=_epsilon):
        super().__init__()
        self.momentum=momentum
        self.affine=affine
        self.eps=eps
        self.axis=axis
    def build(self, input_shape):
        if self._built == False :
            if self.affine:
                self.weight = Parameter(torch.Tensor(self.input_filters))
                self.bias = Parameter(torch.Tensor(self.input_filters))
                init.ones_(self.weight)
                init.zeros_(self.bias)
            self._built=True
    def forward(self, *x):
        x = enforce_singleton(x)
        mean = x.mean(dim=self.axis, keepdim=True).detach()
        std = x.std(dim=self.axis, keepdim=True).detach()
        return self.weight * (x - mean) / (std + self._eps) +self.bias








class SpectralNorm(Layer):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations


    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2_normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2_normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2_normalize(u.data)
        v.data = l2_normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)
    def build(self, input_shape):
        if self._built == False:
            if not self._made_params():
                self._make_params()
            self._built = True
    def forward(self, *x):
        x=enforce_singleton(x)
        self._update_u_v()
        return self.module(x)


class EvoNormB0(Layer):
    def __init__(self,rank=2,nonlinear=True,momentum=0.9,eps = 1e-5):
        super(EvoNormB0, self).__init__()
        self.rank=rank
        self.nonlinear = nonlinear
        self.momentum = momentum
        self.eps = eps

    def build(self, input_shape):
        if self._built == False :
            newshape=np.ones(self.rank+2)
            newshape[1]=self.input_filters
            newshape=tuple(newshape.astype(np.int32).tolist())
            self.weight = Parameter(ones(newshape))
            self.bias = Parameter(zeros(newshape))
            if self.nonlinear:
                self.v = Parameter(ones(newshape))
            self.register_buffer('running_var', ones(newshape))
            self._built=True

    def forward(self, x):
        if self.training:
            permute_pattern=np.arange(0,self.rank+2)
            permute_pattern[0]=1
            permute_pattern[1]=0
            x1 = x.permute(tuple(permute_pattern))
            _, x_var = moments(x1, [1, 2, 3], keepdims=True)
            x_var=x_var.permute(tuple(permute_pattern))
            self.running_var=self.momentum * self.running_var + (1 - self.momentum) * x_var
        else:
            x_var=self.running_var
        if self.nonlinear:
            den =maximum((x_var+self.eps).sqrt(), self.v * x + instance_std(x))
            return (x* self.weight )/ den + self.bias
        else:
            return (x * self.weight) + self.bias


class EvoNormS0(Layer):
    def __init__(self,rank=2,groups=8,nonlinear=True):
        super(EvoNormS0, self).__init__()
        self.nonlinear = nonlinear
        self.groups = groups

    def build(self, input_shape):
        if self._built == False :
            self.input_filters=input_shape[0].item
            self.weight = Parameter(ones((1, self.input_filters)+(1,)*self.rank))
            self.bias = Parameter(zeros((1, self.input_filters)+(1,)*self.rank))
            if self.nonlinear:
                self.v = Parameter(ones((1, self.input_filters)+(1,)*self.rank))
            self.register_buffer('running_var', ones((1, self.input_filters)+(1,)*self.rank))
            self._built=True

    def forward(self, x):
        if self.nonlinear:
            num = torch.sigmoid(self.v * x)
            std = group_std(x,self.groups)
            return num * std * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta


def get_normalization(fn_name):
    if fn_name is None:
        return None
    elif isinstance(fn_name,Layer) and 'Norm' in fn_name.__class__.__name__:
        return fn_name
    elif inspect.isclass(fn_name):
        return fn_name
    elif isinstance(fn_name, str):
        if fn_name.lower().strip() in ['instance_norm','instance','in','i']:
            return InstanceNorm()
        elif  fn_name.lower().strip() in ['batch_norm','batch','bn','b']:
            return BatchNorm()
        elif  fn_name.lower().strip() in ['group_norm','group','gn','g']:
            return GroupNorm(num_groups=16)
        elif fn_name.lower().strip() in ['evo_normb0', 'evo-b0', 'evob0']:
            return EvoNormB0()
        elif fn_name.lower().strip() in ['evo_norms0', 'evo-s0', 'evos0']:
            return EvoNormS0()
        elif fn_name.lower().strip() in ['spectral_norm','spectral','spec','sp' ,'s']:
            return SpectralNorm

    elif inspect.isclass(fn_name):
        return fn_name
    fn_modules = ['trident.layers.pytorch_normalizations']
    normalization_fn_ = get_class(fn_name, fn_modules)
    normalization_fn = normalization_fn_
    return normalization_fn
