from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numbers
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

__all__ = ['InstanceNorm','InstanceNorm2d','InstanceNorm3d','BatchNorm','BatchNorm2d','BatchNorm3d','GroupNorm','GroupNorm2d','GroupNorm3d','LayerNorm','LayerNorm2d','LayerNorm3d','L2Norm','PixelNorm','SpectralNorm','EvoNormB0','EvoNormS0','get_normalization']
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
    """Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs

    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    References:
    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167

    """
    _version = 2
    def __init__(self,  momentum=0.1, affine=True, track_running_stats=True, eps=1e-5,in_sequence=False,name=None, **kwargs):
        """
        Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

        Examples:
            >>> bn=BatchNorm2d(affine=False)
            >>> input = torch.randn(2, 64, 128, 128)
            >>> print(int_shape(bn(input)))
            (2, 64, 128, 128)

        """

        super().__init__(in_sequence=in_sequence,name=name)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats


    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()


    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

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

    def build(self, input_shape):
        if self._built == False:
            if self.affine:
                self.weight = Parameter(ones(self.input_filters))
                self.bias = Parameter(zeros(self.input_filters))

            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

            if self.track_running_stats:
                self.register_buffer('running_mean', zeros(self.input_filters))
                self.register_buffer('running_var', ones(self.input_filters))
                self.register_buffer('num_batches_tracked',to_tensor(0, dtype=torch.long))
            else:
                self.register_buffer('running_mean', None)
                self.register_buffer('running_var', None)
                self.register_buffer('num_batches_tracked', None)

            self.reset_parameters()
            self.to(get_device())
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x=x.permute(0, 2, 1)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
                Mini-batch stats are used in training mode, and in eval mode when buffers are None.
                """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
                passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
                used for normalization (i.e. in eval mode when buffers are not None).
                """
        x= F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

        if hasattr(self,'in_sequence') and self.in_sequence:
            x=x.permute(0, 2, 1)
        return x

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
        self.eval()
BatchNorm1d=BatchNorm
BatchNorm2d=BatchNorm
BatchNorm3d=BatchNorm



class GroupNorm(Layer):
    """Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    References:
    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494

    """

    def __init__(self, num_groups=16,affine=True, eps=1e-5,in_sequence=False,name=None, **kwargs):
        """
        Args:
            num_groups (int): number of groups to separate the channels into
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            affine: a boolean value that when set to ``True``, this module
                has learnable per-channel affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.

        Examples:
            >>> gn=GroupNorm(affine=False)
            >>> input = torch.randn(2, 64, 128, 128)
            >>> print(int_shape(gn(input)))
            (2, 64, 128, 128)

        """
        super().__init__(in_sequence=in_sequence,name=name)
        self.affine=affine
        self.num_groups = num_groups
        self.eps = eps

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
        if hasattr(self,'in_sequence') and self.in_sequence:
            x = x.permute(0, 2, 1)
        x= F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x=x.permute(0, 2, 1)
        return x

GroupNorm2d=GroupNorm
GroupNorm3d=GroupNorm


class InstanceNorm(Layer):
    """Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Instance Normalization: The Missing Ingredient for Fast Stylization`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm2d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm2d` is applied
        on each channel of channeled data like RGB images, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm2d` usually don't apply affine
        transform.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    References:
    .. _`Instance Normalization: The Missing Ingredient for Fast Stylization`:
        https://arxiv.org/abs/1607.08022
    """

    def __init__(self,momentum=0.1, affine=True, track_running_stats=True, eps=1e-5,in_sequence=False,axis=1,name=None, **kwargs):
        """
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, H, W)`
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            momentum: the value used for the running_mean and running_var computation. Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters, initialized the same way as done for batch normalization.
                Default: ``False``.
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``False``

        Examples:
            >>> innorm=InstanceNorm(affine=False)
            >>> input = torch.randn(2, 64, 128, 128)
            >>> print(int_shape(innorm(input)))
            (2, 64, 128, 128)

        """
        super().__init__(in_sequence=in_sequence,name=name)
        self.eps = _epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.axis=axis

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
        if hasattr(self,'in_sequence') and self.in_sequence:
            x = x.permute(0, 2, 1)
        x= F.instance_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

        if hasattr(self,'in_sequence') and self.in_sequence:
            x = x.permute(0, 2, 1)
        return x

InstanceNorm2d=InstanceNorm
InstanceNorm3d=InstanceNorm


class LayerNorm(Layer):
    """Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)



    References:
    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450

    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,in_sequence=False,name=None, **kwargs):
        """
    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Examples:

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

        """
        super().__init__(in_sequence=in_sequence,name=name)
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine



    def build(self, input_shape):
        if self._built == False :
            self.register_parameter('weight',Parameter(ones((self.normalized_shape))))
            self.register_parameter('bias', Parameter(zeros((self.normalized_shape))))
            self._built=True
    def forward(self, *x):
        x = enforce_singleton(x)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x = x.permute(0, 2, 1)
        x= F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x=x.permute(0, 2, 1)
        return x
        # mean = x.mean(dim=self.axis, keepdim=True).detach()
        # std = x.std(dim=self.axis, keepdim=True).detach()
        # return self.weight * (x - mean) / (std + self._eps) +self.bias

LayerNorm2d=LayerNorm
LayerNorm3d=LayerNorm


class L2Norm(Layer):
    def __init__(self,in_sequence=False, axis=1,name=None, **kwargs):
        super().__init__(in_sequence=in_sequence,name=name)
        self.eps=epsilon()
        self.axis=axis

    def build(self, input_shape):
        if self._built == False :
            self._built = True
    def forward(self, *x):
        x = enforce_singleton(x)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x = x.permute(0, 2, 1)
        x= l2_normalize(x,axis=self.axis,keepdims=True)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x = x.permute(0, 2, 1)
        return x



class PixelNorm(Layer):
    def __init__(self,in_sequence=False,name=None, **kwargs):
        super(PixelNorm, self).__init__(in_sequence=in_sequence,name=name)

    def forward(self, x):
        x = enforce_singleton(x)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x = x.permute(0, 2, 1)
        x= x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x=x.permute(0, 2, 1)
        return x


class SpectralNorm(Layer):
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Examples::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    def __init__(self, module, name='weight', power_iterations=1,in_sequence=False,**kwargs):
        super(SpectralNorm, self).__init__(in_sequence=in_sequence,name=name)
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
        x = enforce_singleton(x)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x = x.permute(0, 2, 1)
        self._update_u_v()
        x= self.module(x)
        if hasattr(self,'in_sequence') and self.in_sequence:
            x=x.permute(0, 2, 1)
        return x


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
