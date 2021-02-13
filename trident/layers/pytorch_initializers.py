from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from functools import partial

import torch.nn as nn
from torch.nn import init



__all__ = ['uniform','normal','ones','zeros','kaiming_uniform', 'kaiming_normal','xavier_uniform','xavier_normal','trunc_normal']

from trident.backend.common import get_function, camel2snake


def uniform(tensor, a=0., b=1.):
    # type: (Tensor, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    """

    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad == True and 'bias' not in name:
                init.uniform_(weight, a=a,b=b)
    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad:
            init.uniform_(tensor, a=a,b=b)


def normal(tensor, mean=0., std=1.):
    # type: (Tensor, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad==True and 'bias' not in name:
                init.normal(weight,mean=mean,std=std)
    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad:
            init.normal(tensor,mean=mean,std=std)




def zeros(tensor):
    # type: (Tensor) -> Tensor
    r"""Fills the input Tensor with the scalar value `0`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """

    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad:
                init.zeros_(weight)
    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad:
            init.zeros_(tensor)


def ones(tensor):
    # type: (Tensor) -> Tensor
    r"""Fills the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad==True and 'bias' not in name:
                init.ones_(weight)
    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad:
            init.ones_(tensor)


def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad==True and 'bias' not in name and weight.dim()>=2:
                kaiming_uniform(weight, a, mode, nonlinearity)
    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad and tensor.dim()>=2:
            init.kaiming_uniform_(tensor, a, mode, nonlinearity)
        elif tensor.requires_grad and tensor.dim()<2:
            init.kaiming_uniform_(tensor.unsqueeze_(0).unsqueeze_(0), a, mode, nonlinearity)
            tensor.squeeze_(0).squeeze_(0)


def kaiming_normal(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad==True and 'bias' not in name and weight.dim()>=2:
                kaiming_normal(weight, a, mode, nonlinearity)
    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad and tensor.dim()>=2:
            init.kaiming_normal_(tensor, a, mode, nonlinearity)
        elif tensor.requires_grad and tensor.dim()<2:
            init.kaiming_normal_(tensor.unsqueeze_(0).unsqueeze_(0), a, mode, nonlinearity)
            tensor.squeeze_(0).squeeze_(0)


def xavier_uniform(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """

    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad == True and 'bias' not in name and weight.dim()>=2:
                init.xavier_uniform(weight,gain=gain)
    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad and tensor.dim()>=2:
            init.xavier_uniform(tensor, gain=gain)


def xavier_normal(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad==True and 'bias' not in name and weight.dim()>=2:
                init.xavier_normal_(weight,gain=gain)
    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad and tensor.dim()>=2:
            init.xavier_normal_(tensor, gain=gain)

def trunc_normal(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    if isinstance(tensor,nn.Module):
        for name,weight in tensor.named_parameters():
            if weight.requires_grad==True and 'bias' not in name:
                init.trunc_normal_(weight,mean=0., std=1., a=-2., b=2)

    elif isinstance(tensor, nn.Parameter):
        if tensor.requires_grad:
            init.trunc_normal_(tensor,mean=0., std=1., a=-2., b=2)



def get_initializer(initializer,**kwargs):
    if isinstance(initializer,str):
        initializer_fn = get_function(camel2snake(initializer), ['trident.backend.pytorch_initializers'])
        initializer_fn=partial(initializer_fn,**kwargs) if len(kwargs)>0 else initializer_fn
        return initializer_fn
    elif inspect.isfunction(initializer) and getattr(initializer, '__module__', None) =='trident.backend.pytorch_initializers':
        initializer = partial(initializer, **kwargs) if len(kwargs) > 0 else initializer
        return initializer
