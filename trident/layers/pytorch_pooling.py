from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
from collections import OrderedDict
from functools import partial, wraps, update_wrapper
from itertools import islice
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # import torch functions
import torch.utils.hooks as hooks
from torch._jit_internal import List
from torch._six import container_abcs
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter

from trident.backend.common import *
from trident.backend.pytorch_backend import  Layer, Sequential

__all__ = ['MaxPool2d', 'MaxPool1d', 'MaxPool3d', 'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d', 'AvgPool1d', 'AvgPool2d',
           'AvgPool3d', 'GlobalAvgPool2d', 'AdaptiveAvgPool2d']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon = _session.epsilon


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class _PoolNd(Layer):
    __constants__ = ['kernel_size', 'strides', 'auto_pad', 'padding', 'dilation', 'ceil_mode']

    def __init__(self, kernel_size, strides=None, auto_pad=True, padding_mode='replicate', dilation=1, name='',
                 **kwargs):
        super(_PoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides or kernel_size
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        self.padding = 0
        self.dilation = dilation
        self.return_indices = kwargs.get('return_indices', False)
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.count_include_pad = kwargs.get('count_include_pad', False)
        self.name = name

    def build(self, input_shape):
        if self._built == False:
            self.get_padding(input_shape)
            self._built = True

    def extra_repr(self):
        return 'kernel_size={kernel_size}, strides={strides}, padding={padding}' \
               ', dilation={dilation}'.format(**self.__dict__)


class MaxPool1d(_PoolNd):
    r"""Applies a 1D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, strides \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        kernel_size: the size of the window to take a max over
        stride: the strides of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the strides of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Examples::

        >>> # pool of size=3, strides=2
        >>> m = nn.MaxPool1d(3, strides=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, padding_mode='replicate', name='', **kwargs):
        super(MaxPool1d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _single(kernel_size)
        self.strides = _single(strides if strides is not None else kernel_size)
        self.padding = _single(self.padding)
        self.padding_mode = padding_mode
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.return_indices = kwargs.get('return_indices', False)

    def forward(self, *x):
        x = enforce_singleton(x)
        return F.max_pool1d(x, self.kernel_size, self.strides, self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)


class MaxPool2d(_PoolNd):
    r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the strides of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the strides of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, strides=2
        >>> m = nn.MaxPool2d(3, strides=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), strides=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, padding_mode='zero', name='', **kwargs):
        super(MaxPool2d, self).__init__(kernel_size, strides, auto_pad, padding_mode, 1, name, **kwargs)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides if strides is not None else kernel_size)
        self.padding = _pair(0)
        if padding_mode == 'zero':
            self.padding_mode = 'constant'
        else:
            self.padding_mode = padding_mode
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.return_indices = kwargs.get('return_indices', False)

    def get_padding(self, input_shape):
        pad_h = 0
        pad_w = 0
        if self.auto_pad == True:
            ih, iw = input_shape[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]

            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) + 1 - iw, 0)
            if pad_h % 2 == 1 and sh > 1:
                pad_h += 1
            if pad_w % 2 == 1 and sw > 1:
                pad_w += 1

        elif len(self.padding) == 2:
            pad_h = self.padding[0] * 2
            pad_w = self.padding[1] * 2

        self.padding = (int(pad_h / 2), int(pad_w / 2))

    def forward(self, *x):
        x = enforce_singleton(x)
        # self.get_padding(x.size()[1:])
        # if self.padding[0] > 0 or self.padding[1] > 0:
        #     x = F.pad(x, (self.padding[1] , self.padding[1], self.padding[0] , self.padding[0] ), mode='constant'
        #     if self.padding_mode == 'zero' else self.padding_mode)
        return F.max_pool2d(x, self.kernel_size, self.strides, self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)


class MaxPool3d(_PoolNd):
    r"""Applies a 3D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, 
            kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w 
                                                             + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the strides of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        dilation: a parameter that controls the strides of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, strides=2
        >>> m = nn.MaxPool3d(3, strides=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), strides=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50,44, 31)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """  # noqa: E501

    def __init__(self, kernel_size, strides=None, auto_pad=True, name='', **kwargs):
        super(MaxPool3d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _triple(kernel_size)
        self.strides = _triple(strides if strides is not None else kernel_size)
        self.padding = _triple(self.padding)
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.return_indices = kwargs.get('return_indices', False)

    def forward(self, input):
        return F.max_pool3d(input, self.kernel_size, self.strides, self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)


class MaxUnpool1d(_PoolNd):
    r"""Computes a partial inverse of :class:`MaxPool1d`.

    :class:`MaxPool1d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool1d` takes in as input the output of :class:`MaxPool1d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: :class:`MaxPool1d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        strides (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool1d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in})`
        - Output: :math:`(N, C, H_{out})`, where

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{kernel\_size}[0]

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> pool = nn.MaxPool1d(2, strides=2, return_indices=True)
        >>> unpool = nn.MaxUnpool1d(2, strides=2)
        >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])

        >>> # Example showcasing the use of output_size
        >>> input = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8, 9]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices, output_size=input.size())
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.,  0.]]])

        >>> unpool(output, indices)
        tensor([[[ 0.,  2.,  0.,  4.,  0.,  6.,  0., 8.]]])
    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, name='', **kwargs):
        super(MaxUnpool1d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _single(kernel_size)
        self.strides = _single(strides if strides is not None else kernel_size)
        self.padding = _single(self.padding)

    def forward(self, x, indices, output_size=None):
        return F.max_unpool1d(x, indices, self.kernel_size, self.strides, self.padding, output_size)


class MaxUnpool2d(_PoolNd):
    r"""Computes a partial inverse of :class:`MaxPool2d`.

    :class:`MaxPool2d` is not fully invertible, since the non-maximal values are lost.

    :class:`MaxUnpool2d` takes in as input the output of :class:`MaxPool2d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: :class:`MaxPool2d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs and Example below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        strides (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool2d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
            H_{out} = (H_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
            W_{out} = (W_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> pool = nn.MaxPool2d(2, strides=2, return_indices=True)
        >>> unpool = nn.MaxUnpool2d(2, strides=2)
        >>> input = torch.tensor([[[[ 1.,  2,  3,  4],
                                    [ 5,  6,  7,  8],
                                    [ 9, 10, 11, 12],
                                    [13, 14, 15, 16]]]])
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        tensor([[[[  0.,   0.,   0.,   0.],
                  [  0.,   6.,   0.,   8.],
                  [  0.,   0.,   0.,   0.],
                  [  0.,  14.,   0.,  16.]]]])

        >>> # specify a different output size than input size
        >>> unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
        tensor([[[[  0.,   0.,   0.,   0.,   0.],
                  [  6.,   0.,   8.,   0.,   0.],
                  [  0.,   0.,   0.,  14.,   0.],
                  [ 16.,   0.,   0.,   0.,   0.],
                  [  0.,   0.,   0.,   0.,   0.]]]])
    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, name='', **kwargs):
        super(MaxUnpool2d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides or kernel_size)
        self.auto_pad = auto_pad
        self.padding = _pair(0)

    def forward(self, x, indices, output_size=None):
        return F.max_unpool2d(x, indices, self.kernel_size, self.strides,
                              (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), output_size)


class MaxUnpool3d(_PoolNd):
    r"""Computes a partial inverse of :class:`MaxPool3d`.

    :class:`MaxPool3d` is not fully invertible, since the non-maximal values are lost.
    :class:`MaxUnpool3d` takes in as input the output of :class:`MaxPool3d`
    including the indices of the maximal values and computes a partial inverse
    in which all non-maximal values are set to zero.

    .. note:: :class:`MaxPool3d` can map several input sizes to the same output
              sizes. Hence, the inversion process can get ambiguous.
              To accommodate this, you can provide the needed output size
              as an additional argument :attr:`output_size` in the forward call.
              See the Inputs section below.

    Args:
        kernel_size (int or tuple): Size of the max pooling window.
        strides (int or tuple): Stride of the max pooling window.
            It is set to :attr:`kernel_size` by default.
        padding (int or tuple): Padding that was added to the input

    Inputs:
        - `input`: the input Tensor to invert
        - `indices`: the indices given out by :class:`~torch.nn.MaxPool3d`
        - `output_size` (optional): the targeted output size

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = (D_{in} - 1) \times \text{stride[0]} - 2 \times \text{padding[0]} + \text{kernel\_size[0]}

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride[1]} - 2 \times \text{padding[1]} + \text{kernel\_size[1]}

          .. math::
              W_{out} = (W_{in} - 1) \times \text{stride[2]} - 2 \times \text{padding[2]} + \text{kernel\_size[2]}

          or as given by :attr:`output_size` in the call operator

    Example::

        >>> # pool of square window of size=3, strides=2
        >>> pool = nn.MaxPool3d(3, strides=2, return_indices=True)
        >>> unpool = nn.MaxUnpool3d(3, strides=2)
        >>> output, indices = pool(torch.randn(20, 16, 51, 33, 15))
        >>> unpooled_output = unpool(output, indices)
        >>> unpooled_output.size()
        torch.Size([20, 16, 51, 33, 15])
    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, name='', **kwargs):
        super(MaxUnpool3d, self).__init__(self, kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _triple(kernel_size)
        self.strides = _triple(strides if strides is not None else kernel_size)
        self.padding = _triple(self.padding)

    def forward(self, x, indices, output_size=None):
        return F.max_unpool3d(x, indices, self.kernel_size, self.strides, self.padding, output_size)


class AvgPool1d(_PoolNd):
    r"""Applies a 1D average pooling over an input signal composed of several
    input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,
    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`
    can be precisely described as:

    .. math::

        \text{out}(N_i, C_j, l) = \frac{1}{k} \sum_{m=0}^{k-1}
                               \text{input}(N_i, C_j, \text{stride} \times l + m)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be
    an ``int`` or a one-element tuple.

    Args:
        kernel_size: the size of the window
        stride: the strides of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} +
              2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor

    Examples::

        >>> # pool with window of size=3, strides=2
        >>> m = nn.AvgPool1d(3, strides=2)
        >>> m(torch.tensor([[[1.,2,3,4,5,6,7]]]))
        tensor([[[ 2.,  4.,  6.]]])
    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, name='', **kwargs):
        super(AvgPool1d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _single(kernel_size)
        self.strides = _single(strides if strides is not None else kernel_size)
        self.padding = _single(self.padding)
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.count_include_pad = kwargs.get('count_include_pad', False)

    def forward(self, input):
        return F.avg_pool1d(input, self.kernel_size, self.strides, self.padding, self.ceil_mode, self.count_include_pad)


class AvgPool2d(_PoolNd):
    r"""Applies a 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the strides of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise attr:`kernel_size` will be used

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, strides=2
        >>> m = nn.AvgPool2d(3, strides=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), strides=(2, 1))
        >>> input = torch.randn(20, 16, 50, 32)
        >>> output = m(input)
    """
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']

    def __init__(self, kernel_size, strides=None, auto_pad=True,count_include_pad=True, divisor_override=None, name='', **kwargs):
        super(AvgPool2d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides if strides is not None else kernel_size)
        self.auto_pad == auto_pad

        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.count_include_pad = kwargs.get('count_include_pad', True)
        self.divisor_override = kwargs.get('divisor_override', None)

    def get_padding(self, input_shape):
        pad_h = 0
        pad_w = 0
        if self.auto_pad == True:
            ih, iw = input_shape[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]

            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) + 1 - iw, 0)
            if pad_h % 2 == 1 and sh > 1:
                pad_h += 1
            if pad_w % 2 == 1 and sw > 1:
                pad_w += 1

        elif len(self.padding) == 2:
            pad_h = self.padding[0] * 2
            pad_w = self.padding[1] * 2

        self.padding = (int(pad_h / 2), int(pad_w / 2))

    def forward(self, *x):
        x = enforce_singleton(x)

        return F.avg_pool2d(x, self.kernel_size, self.strides, self.padding, self.ceil_mode, self.count_include_pad,
                            self.divisor_override)


class AvgPool3d(_PoolNd):
    r"""Applies a 3D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1} \\
                                              & \frac{\text{input}(N_i, C_j, \text{stride}[0] \times d + k,
                                                      \text{stride}[1] \times h + m, \text{stride}[2] \times w + n)}
                                                     {kD \times kH \times kW}
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides
    for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the strides of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad: when True, will include the zero-padding in the averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise attr:`kernel_size` will be used

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] -
                    \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -
                    \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -
                    \text{kernel\_size}[2]}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, strides=2
        >>> m = nn.AvgPool3d(3, strides=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool3d((3, 2, 2), strides=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50,44, 31)
        >>> output = m(input)
    """
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']

    def __init__(self, kernel_size, strides=None, auto_pad=True, name='', **kwargs):
        super(AvgPool3d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _triple(kernel_size)
        self.strides = _triple(strides if strides is not None else kernel_size)
        self.padding = _triple(self.padding)
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.count_include_pad = kwargs.get('count_include_pad', False)
        self.divisor_override = kwargs.get('divisor_override', None)

    def forward(self, input):
        return F.avg_pool3d(input, self.kernel_size, self.strides, self.padding, self.ceil_mode, self.count_include_pad,
                            self.divisor_override)

    def __setstate__(self, d):
        super(AvgPool3d, self).__setstate__(d)
        self.__dict__.setdefault('padding', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)


class GlobalAvgPool2d(Layer):
    def __init__(self, keepdim=False, name='avg_pool'):
        super(GlobalAvgPool2d, self).__init__()
        self.keepdim = keepdim
        self.name = name

    def build(self, input_shape):
        if self._built == False:
            if self.keepdim == True:
                output_shape = input_shape.clone()
                output_shape[1] = 1
                output_shape[2] = 1
                self.output_shape = output_shape
            else:
                self.output_shape = input_shape[0]
            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        N,C,H,W=x.size()
        x = x.view(N, C, -1).mean(dim=-1, keepdim=self.keepdim)
        return x


class AdaptiveAvgPool2d(Layer):
    def __init__(self, output_size, name='adaptive_avg_pool'):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = _pair(output_size)
        self.name = name

    def forward(self, *x):
        x = enforce_singleton(x)
        return F.adaptive_avg_pool2d(x, self.output_size)

#
# class GlobalAvgPool1d(Layer):
#     def __init__(self):
#         super(GlobalAvgPool1d, self).__init__()
#
#     def build(self, input_shape):
#         pass
#
#     def forward(self, *x):
#         x = enforce_singleton(x)
#         assert len(x.size()) == 3, x.size()
#         B, C, L = x.size()
#         return F.avg_pool1d(x, L)
#
#
# class GlobalAvgPool2d(Layer):
#     def __init__(self):
#         super(GlobalAvgPool2d, self).__init__()
#
#     def build(self, input_shape):
#         pass
#
#     def forward(self, *x):
#         x = enforce_singleton(x)
#         assert len(x.size()) == 4, x.size()
#         B, C, W, H = x.size()
#         return F.avg_pool2d(x, (W, H))
