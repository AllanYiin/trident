from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F  # import torch functions
from torch._six import container_abcs
from torch._jit_internal import List
from itertools import repeat

from ..backend.common import get_session
from .pytorch_activations import get_activation
from .pytorch_normalizations import get_normalization
import numpy as np
__all__ = ['Flatten','Conv1d','Conv2d','Conv3d','SeparableConv2d','GcdConv2d','GcdConv2d_1','Lambda','Reshape','CoordConv2d']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class Flatten(nn.Module):
    r"""Flatten layer to flatten a tensor after convolution."""

    def forward(self,  # type: ignore
                x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size()[0], -1)


def _gcd(x, y):
    gcds = []
    gcd = 1
    if x % y == 0:
        gcds.append(int(y))
    for k in range(int(y // 2), 0, -1):
        if x % k == 0 and y % k == 0:
            gcd = k
            gcds.append(int(k))
    return gcds


def _get_divisors(n):
    return [d for d in range(2, n // 2) if n % d == 0]


def _isprime(n):
    divisors = [d for d in range(2, int(math.sqrt(n))) if n % d == 0]
    return all(n % od != 0 for od in divisors if od != n)


class _ConvNd(nn.Module):
    __constants__ = ['strides', 'padding', 'dilation', 'groups', 'use_bias']

    def __init__(self, kernel_size, num_filters, strides, auto_pad, init, use_bias, init_bias, dilation,
                 groups, weights_contraint, padding_mode, transposed, **kwargs):
        super(_ConvNd, self).__init__()

        self.input_filters = kwargs[
            'in_channels'] if 'in_channels' in kwargs else None  # if  self.in_channelsi is not None else input_filters
        self.num_filters = kwargs[
            'out_channels'] if 'out_channels' in kwargs else num_filters  # self.in_channels i#f self.in_channelsi is not None else num_filters

        if self.input_filters is not None and self.input_filters % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if self.num_filters % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        #
        # self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = 0  # padding if padding is not None else 0in_channel
        self.strides = kwargs['stride'] if 'stride' in kwargs else strides
        self.auto_pad = auto_pad
        self.dilation = dilation
        self.transposed = transposed
        self.groups = groups
        self.init =init
        self.init_bias = init_bias
        self.divisor_rank = 0
        self.transposed = transposed
        self.padding_mode = padding_mode
        self._is_built = False
        self.weight = None
        self.use_bias = use_bias
        self.to(_device)

    def reset_parameters(self):
        if self.init is not None:
            self.init(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, mode='fan_in')
        if self.use_bias==True and self.bias is not None and self.init_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            self.init_bias(self.bias, -bound, bound)

    def build_once(self, input_shape):
        if self._is_built == False:
            self.input_shape = input_shape
            self.input_filters = input_shape[1]
            if self.transposed:
                self.weight = Parameter(torch.Tensor(self.input_filters, self.num_filters // self.groups, *self.kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(self.num_filters, self.input_filters // self.groups, *self.kernel_size))  #
            self.reset_parameters()
            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.num_filters))
            else:
                self.register_parameter('bias', None)


            self.to(_device)
            self._is_built = True

    def extra_repr(self):
        s = (
            'kernel_size={kernel_size}, {num_filters}, strides={strides}, activation={activation}, auto_pad={auto_pad} , dilation={dilation}')
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', use_bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(
            state)  # if not hasattr(self, 'padding_mode'):  #     self.padding_mode = 'zeros'


class Conv1d(_ConvNd):
    r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters,
          of size
          :math:`\left\lfloor\frac{C_\text{out}}{C_\text{in}}\right\rfloor`

    .. note::

        Depending of the size of your kernel, several (of the last)
        columns of the input might be lost, because it is a valid
        `cross-correlation`_, and not a full `cross-correlation`_.
        It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(C_\text{in}=C_{in}, C_\text{out}=C_{in} \times K, ..., \text{groups}=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size). The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{C_\text{in} * \text{kernel\_size}}`

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, num_filters, strides, auto_pad, activation, init, use_bias, init_bias, dilation=1,
                 groups=1, weights_contraint=None, padding_mode='zero', **kwargs):
        kernel_size = _single(kernel_size)
        strides = _single(strides)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(kernel_size, num_filters, strides, auto_pad, init, use_bias, init_bias,
                                     dilation, groups, weights_contraint, padding_mode, False, **kwargs)

    def conv1d_forward(self, x):
        self.input_filters = x.size(1)
        if self.auto_pad == True:
            iw = x.size()[-1]
            kw = self.weight.size()[-1]
            sw = self.strides[-1]
            dw = self.dilation[-1]
            ow = math.ceil(iw / sw), math.ceil(iw / sw)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2], mode='replicate')
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x):
        self.build_once(x.shape)
        result = self.conv1d_forward(x)
        if torch.isnan(self.weight).any() or torch.isnan(result).any():
            print(self.__module__ + '  nan detected!!')
        return result


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        input_filters (int): Number of channels in the input image
        num_filters (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        strides (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        use_bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        use_bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, num_filters, strides=1, auto_pad=True, activation=None, init=None, use_bias=False,
                 init_bias=0, dilation=1, groups=1, weights_contraint=None, padding_mode='zero', **kwargs):
        kernel_size = _pair(kernel_size)
        strides = _pair(strides)
        dilation = _pair(dilation)



        super(Conv2d, self).__init__(kernel_size, num_filters, strides, auto_pad, init, use_bias, init_bias,
                                     dilation, groups, weights_contraint, padding_mode, False, **kwargs)
        self.activation=get_activation(activation)
        if 'in_channels' in kwargs:
            self.input_filters = kwargs['in_channels']
            self.build_once(self.input_filters)
        if 'out_channels' in kwargs:
            self.num_filters = kwargs['out_channels']
        if 'padding' in kwargs:
            self.padding = kwargs['padding']
            self.padding = _pair(self.padding)
            self.auto_pad = False
        else:
            self.padding = _pair(0)

    def conv2d_forward(self, x):
        self.input_filters = x.size(1)
        if self.auto_pad == True:
            ih, iw = x.size()[-2:]
            kh, kw = self.weight.size()[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = self.dilation[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode='replicate')
        return F.conv2d(x, self.weight, self.bias, self.strides, self.padding, self.dilation, self.groups)

    def forward(self, x):
        self.build_once(x.shape)
        x = self.conv2d_forward(x)
        if self.activation is not None:
            x=self.activation(x)
        if torch.isnan(self.weight).any() or torch.isnan(x).any():
            print(self.__module__ + '  nan detected!!')
        return x


class Conv3d(_ConvNd):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)

    where :math:`\star` is the valid 3D `cross-correlation`_ operator

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`,
         a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
         :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, num_filters, strides, auto_pad, activation, init, use_bias, init_bias, dilation=1,
                 groups=1, weights_contraint=None, padding_mode='zero', **kwargs):
        kernel_size = _triple(kernel_size)
        strides = _triple(strides)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(kernel_size, num_filters, strides, auto_pad, init, use_bias, init_bias,
                                     dilation, groups, weights_contraint, padding_mode, False, **kwargs)
        if 'in_channels' in kwargs:
            self.input_filters = kwargs['in_channels']
            self.build_once(self.input_filters)
        if 'out_channels' in kwargs:
            self.num_filters = kwargs['out_channels']
        if 'padding' in kwargs:
            self.padding = kwargs['padding']
            self.padding = _triple(self.padding)
            self.auto_pad = False
        else:
            self.padding = _triple(0)

    def conv3d_forward(self, x):
        self.input_filters = x.size(1)
        if self.auto_pad == True:
            ih, iw = x.size()[-2:]
            kh, kw = self.weight.size()[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = self.dilation[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode='replicate')
        return F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x):
        self.build_once(x.shape)
        result = self.conv3d_forward(x)
        if torch.isnan(self.weight).any() or torch.isnan(result).any():
            print(self.__module__ + '  nan detected!!')
        return result


class SeparableConv2d(nn.Module):
    def __init__(self, kernel_size=(3, 3), num_filters=None, strides=1, auto_pad=True, padding=None, activation=None,
                 init=None, use_bias=False, init_bias=0, dilation=1, groups=1, weights_contraint=None,
                 padding_mode='zeros', transposed=False, ):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.num_filters = num_filters
        self.dilation = _pair(dilation)
        self.strides = _pair(strides)
        self.use_bias = use_bias
        self.conv1 = None
        self.pointwise = None

    def forward(self, x):
        if self.conv1 is None:
            self.conv1 = nn.Conv2d(x.size(1), self.num_filters, kernel_size=self.kernel_size, stride=self.strides,
                                   padding=0, dilation=self.dilation, groups=x.size(1), bias=self.use_bias)
            self.pointwise = nn.Conv2d(x.size(1), self.num_filters, 1, 1, 0, 1, 1, bias=self.use_bias)
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class GcdConv2d(nn.Module):
    def __init__(self, kernel_size, num_filters, strides, auto_pad=True, activation=None, init=None, use_bias=False,
                 init_bias=0, divisor_rank=0, dilation=1, self_norm=True, is_shuffle=False, weights_contraint=None,
                 padding_mode='zeros', **kwargs):
        super(GcdConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.input_filters = 3
        self.strides = _pair(strides)
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        self.activation = get_activation(activation)
        self.dilation = _pair(dilation)
        self.self_norm = self_norm
        self.is_shuffle = is_shuffle
        self.init = init
        self.use_bias = use_bias
        self.init_bias = init_bias
        self.group_conv = None
        self.group_conv2 = None
        self.divisor_rank = divisor_rank

        self.groups = 1
        self.weights_contraint = weights_contraint

        self.pointwise = None
        self.input_shape = None
        self.is_shape_inferred = False

    def calculate_gcd(self):
        gcd_list = _gcd(self.input_filters, self.num_filters)
        if len(gcd_list) == 0:
            self.groups = self.input_filters
            self.num_filters_1 = self.input_filters
        else:
            self.gcd = gcd_list[0]
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]

            self.num_filters_1 = self.gcd
            self.num_filters_2 = self.num_filters
            factors = _get_divisors(self.num_filters // self.gcd)

            if self.input_filters == self.gcd:
                self.groups = self.input_filters
                self.num_filters_1 = self.input_filters
            elif self.num_filters == self.gcd:
                self.groups = self.gcd
                self.num_filters_1 = self.num_filters
            elif len(factors) == 0:
                self.groups = self.gcd
                self.num_filters_1 = self.gcd
            else:
                self.num_filters_1 = self.gcd * factors[-1]

    def forward(self, x):
        input_shape = x.size()
        if self.is_shape_inferred == True and self.input_shape[1] != input_shape[1]:
            raise ValueError(
                'You have do dynamic shape inferred once. Current shape {0} channel is not the same with the shape {1} channel'.format(
                    input_shape, self.input_shape))
        elif self.is_shape_inferred == False:
            self.input_filters = x.size(1)
            self.input_shape = input_shape
            self.calculate_gcd()
            print('input:{0}   output:{1}->{2}  gcd:{3} group:{4}   放大因子:{5} '.format(self.input_filters,
                                                                                      self.num_filters_1,
                                                                                      self.num_filters_2, self.gcd,
                                                                                      self.groups,
                                                                                      self.num_filters_1 // self.groups))

            self.group_conv = Conv2d(kernel_size=self.kernel_size, input_filters=self.input_filters,
                                     num_filters=self.num_filters_1, in_channels=self.input_filters,
                                     out_channels=self.num_filters_1, strides=self.strides, padding=0,
                                     auto_pad=self.auto_pad, dilation=self.dilation, groups=self.groups,
                                     use_bias=self.use_bias, transposed=False).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.add_module('gcd_conv1', self.group_conv)
            torch.nn.init.xavier_normal_(self.group_conv.weight, gain=0.01)
            self.group_conv2 = Conv2d(kernel_size=self.kernel_size, input_filters=self.num_filters_1,
                                      num_filters=self.num_filters_2, in_channels=self.num_filters_1,
                                      out_channels=self.num_filters_2, strides=1, padding=0, auto_pad=self.auto_pad,
                                      dilation=self.dilation, groups=self.groups, use_bias=self.use_bias,
                                      transposed=False).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.add_module('gcd_conv2', self.group_conv2)
            torch.nn.init.xavier_normal_(self.group_conv2.weight, gain=0.01)

            self.pointwise_conv = Conv2d(kernel_size=(1, 1), input_filters=self.num_filters_2,
                                         num_filters=self.num_filters_2, in_channels=self.num_filters_2,
                                         out_channels=self.num_filters_2, strides=(1, 1), padding=0, dilation=1,
                                         groups=1, use_bias=self.use_bias, transposed=False).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.add_module('pointwise_conv', self.pointwise_conv)
            torch.nn.init.xavier_normal_(self.pointwise_conv.weight, gain=0.01)

            self.is_shape_inferred = True

        # if self.auto_pad:
        #     #test_shape=self.group_conv.forward(x).size()
        #     ih, iw = x.size()[-2:]
        #     #oh,ow=self.group_conv.forward(x).size()[-2:]
        #     kh, kw = self.group_conv.weight.size()[-2:]
        #     sh, sw = self.strides
        #     oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        #     pad_h = max((oh - 1) * self.strides[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        #     pad_w = max((ow - 1) * self.strides[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        #     if pad_h > 0 or pad_w > 0:
        #         x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        x = self.group_conv(x)
        x = self.group_conv2(x)
        N, G, C, H, W = x.size()
        if self.self_norm:
            x = x.view(x.size(0), x.size(1), -1)
            mean = x.mean(dim=2, keepdim=True)
            mean.requires_grade = False
            std = x.std(dim=2, keepdim=True)
            x = (x - mean) / (std + 1e-8)
        x = x.view(N, C * G, H, W)
        # x = self.pointwise_conv(x)
        return x


class GcdConv2d_1(nn.Module):
    def __init__(self, kernel_size, num_filters, strides, auto_pad=True, activation=None, init=None, use_bias=False,
                 init_bias=0, divisor_rank=0, dilation=1, self_norm=True, is_shuffle=False, weights_contraint=None,
                 padding_mode='zeros', **kwargs):
        super(GcdConv2d_1, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.input_filters = 3
        self.strides = _pair(strides)
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        self.activation = get_activation(activation)
        self.dilation = dilation
        self.self_norm = self_norm
        self.is_shuffle = is_shuffle
        self.init = init
        self.use_bias = use_bias
        self.init_bias = init_bias
        self.gcd_conv3d = None

        self.divisor_rank = divisor_rank

        self.groups = 1
        self.weights_contraint = weights_contraint

        self.pointwise = None
        self.input_shape = None
        self.is_shape_inferred = False

    def calculate_gcd(self):
        gcd_list = _gcd(self.input_filters, self.num_filters)
        if len(gcd_list) == 0:
            self.groups = self.input_filters
            self.num_filters_1 = self.input_filters
        else:
            self.gcd = gcd_list[0]
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]

            self.num_filters_1 = self.gcd
            self.num_filters_2 = self.num_filters
            factors = _get_divisors(self.num_filters // self.gcd)

            if self.input_filters == self.num_filters or self.input_filters == self.gcd or self.num_filters == self.gcd:
                self.groups = gcd_list[min(int(self.divisor_rank + 1), len(gcd_list))]

    def forward(self, x):
        input_shape = x.size()
        if self.is_shape_inferred == True and self.input_shape[1] != input_shape[1]:
            raise ValueError(
                'You have do dynamic shape inferred once. Current shape {0} channel is not the same with the shape {1} channel'.format(
                    input_shape, self.input_shape))
        elif self.is_shape_inferred == False:
            self.input_filters = x.size(1)
            self.input_shape = input_shape
            self.calculate_gcd()
            print('input:{0}   output:{1}->{2}  gcd:{3} group:{4}   放大因子:{5} '.format(self.input_filters,
                                                                                      self.num_filters_1,
                                                                                      self.num_filters_2, self.gcd,
                                                                                      self.groups,
                                                                                      self.num_filters_1 // self.groups))

            # if _pair(self.kernel_size)==(1,1):
            #     self.gcd_conv3d = nn.Conv3d(self.input_filters // self.groups, self.num_filters // self.groups,
            #                                 (1,) + _pair(self.kernel_size), (1,) + _pair(self.strides), padding=0,
            #                                 dilation=(2, 1, 1), groups=1, bias=False).to(
            #         torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            #
            #     # self.gcd_conv3d = nn.Conv2d(self.input_filters , self.num_filters,
            #     #                              _pair(self.kernel_size), _pair(self.strides), padding=0,
            #     #                             dilation=( 1, 1), groups=1, bias=False).to(
            #     #     torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # else:
            self.gcd_conv3d = nn.Conv3d(self.input_filters // self.groups, self.num_filters // self.groups,
                                        (2,) + _pair(self.kernel_size), (1,) + _pair(self.strides), padding=(1, 0, 0),
                                        dilation=(2,) + _pair(self.dilation), groups=1, bias=False).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.add_module('gcd_conv3d', self.gcd_conv3d)
            torch.nn.init.xavier_normal_(self.gcd_conv3d.weight, gain=0.01)
            if self.self_norm == True:
                self.norm = nn.BatchNorm2d(self.num_filters, _session.epsilon, momentum=0.1, affine=True,
                                           track_running_stats=True).to(_device)

                torch.nn.init.ones_(self.norm.weight)
                torch.nn.init.zeros_(self.norm.bias)

            #
            # self.pointwise_conv = nn.Conv2d(self.num_filters, self.num_filters,1,1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # self.add_module('pointwise_conv', self.pointwise_conv)
            # torch.nn.init.xavier_normal_(self.pointwise_conv.weight,gain=0.01)

            self.is_shape_inferred = True

        if self.auto_pad:
            # test_shape=self.group_conv.forward(x).size()
            ih, iw = x.size()[-2:]
            # oh,ow=self.group_conv.forward(x).size()[-2:]
            kh, kw = self.gcd_conv3d.weight.size()[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = _pair(self.dilation)[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode='replicate')

        x = x.view(x.size(0), x.size(1) // self.groups, self.groups, x.size(2), x.size(3))
        if torch.isnan(x).any():
            print(self._get_name() + '  nan detected!!')

        if torch.isnan(self.gcd_conv3d.weight).any():
            result = x.cpu().detach().numpy()
            p = self.gcd_conv3d.weight.data.cpu().detach().numpy()
            print('x   mean: {0} max:{1} min:n {2}'.format(result.mean(), result.max(), result.min()))
            print('parameters mean: {0} max:{1} min:n {2}'.format(p.mean(), p.max(), p.min()))
            item = torch.isnan(self.gcd_conv3d.weight).float()
            data = self.gcd_conv3d.weight.data
            data[item == 1] = 1e-8
            self.gcd_conv3d.weight.data = data
            print(self._get_name() + '  nan fix!!')
        x = self.gcd_conv3d(x)
        if torch.isnan(self.gcd_conv3d.weight).any():
            print(self._get_name() + '  nan detected!!')
        x = torch.transpose(x, 1, 2).contiguous()  # N, G,C, H, W

        # x= torch.transpose(x, 1, 2).contiguous()

        # if self.self_norm:
        #     reshape_x=x.view(x.size(0),x.size(1),-1) #N, G,C*H,W
        #     mean =reshape_x.mean(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1).detach() #N, G,1,1,1
        #     std = reshape_x.std(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1).detach()#N, G,1,1,1
        #     x=(x - mean) / (std +_session.epsilon)  #N, G,C, H, W

        if self.is_shuffle == False:
            x = torch.transpose(x, 1, 2).contiguous()  # N, C,G, H, W

        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        if self.self_norm == True:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)
        return x


def gcdconv2d(x, kernel_size=(1, 1), num_filters=None, strides=1, padding=0, activation=None, init=None, use_bias=False,
              init_bias=0, divisor_rank=0, dilation=1, weights_contraint=None):
    conv = GcdConv2d(kernel_size=kernel_size, num_filters=num_filters, strides=strides, padding=padding,
                     activation=activation, init=init, use_bias=False, init_bias=0, divisor_rank=divisor_rank,
                     dilation=dilation, weights_contraint=None, padding_mode='zeros', transposed=False)
    return conv(x)


class Lambda(nn.Module):
    """
    Applies a lambda function on forward()
    Args:
        lamb (fn): the lambda function
    """

    def __init__(self, lam):
        super(Lambda, self).__init__()
        self.lam = lam

    def forward(self, x):
        return self.lam(x)


class Reshape(nn.Module):
    """
    Reshape the input volume
    Args:
        *shape (ints): new shape, WITHOUT specifying batch size as first
        dimension, as it will remain unchanged.
    """

    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)



"""
Implementation of the CoordConv modules from https://arxiv.org/abs/1807.03247
"""
def _append_coords(input_tensor, with_r=False):
    batch_size, _, x_dim, y_dim = input_tensor.size()

    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

    ret = torch.cat(
        [
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor),
        ],
        dim=1,
    )

    if with_r:
        rr = torch.sqrt(
            torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
            + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
        )
        ret = torch.cat([ret, rr], dim=1)

    return ret


"""
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
"""



class CoordConv2d(nn.Module):
    def __init__(self,kernel_size, num_filters, strides, auto_pad=True, activation=None, init=None, use_bias=False,
                 init_bias=0, group=1, dilation=1,  weights_contraint=None,
                 padding_mode='zeros', with_r=False, **kwargs):
        super().__init__()
        self.addcoords = partial(_append_coords,with_r=with_r)
        self.conv = Conv2d(kernel_size, num_filters, strides, auto_pad=True, activation=None, init=None, use_bias=False,
                 init_bias=0, group=1,dilation=1,  weights_contraint=None,
                 padding_mode='zeros',  **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret