"""Tensorflow pooling layers definition in trident"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from itertools import repeat

import tensorflow as tf
from tensorflow.python.client import device_lib

from trident.backend.common import *
from trident.backend.tensorflow_backend import *
from trident.backend.tensorflow_ops import *

_tf_data_format = 'channels_last'

__all__ = ['MaxPool2d', 'MaxPool1d', 'MaxPool3d', 'MaxUnpool2d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
           'GlobalAvgPool1d','GlobalAvgPool2d']

_session = get_session()

_device =get_device()

_epsilon = _session.epsilon


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class _PoolNd(Layer):
    __constants__ = ['kernel_size', 'strides', 'auto_pad', 'padding', 'dilation']

    def __init__(self, kernel_size, strides=None, auto_pad=True, padding_mode='zero', dilation=1, name=None, **kwargs):
        super(_PoolNd, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.strides = strides or kernel_size
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        self.padding = 0
        self.dilation = dilation
        self.return_indices = kwargs.get('return_indices', False)
        # self.ceil_mode = kwargs.get('ceil_mode', False)
        # self.count_include_pad = kwargs.get('count_include_pad', False)


    def extra_repr(self):
        return 'kernel_size={kernel_size}, strides={strides}, padding={padding}' \
               ', dilation={dilation}'.format(**self.__dict__)


class MaxPool1d(_PoolNd):
    """Applies a 1D max pooling over an input signal composed of several inputplanes.

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

    Examples:

        >>> # pool of size=3, strides=2
        >>> m = MaxPool1d(3, strides=2)
        >>> input = randn(20, 16, 50)
        >>> output = m(input)

    References
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


    def forward(self, x) :

        return tf.nn.max_pool1d(x, self.kernel_size, self.strides, 'SAME' if self.auto_pad else 'VALID',
                                data_format="NWC", name=None)


class MaxPool2d(_PoolNd):
    """Applies a 2D max pooling over an input signal composed of several input
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
        return_indices: if ``True``, will return the max indices along with the outputs. Useful for
        :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              'H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}\times (\text{
              kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor'

          .. math::
              'W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}\times (\text{
              kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor'

    Examples:
        >>> # pool of square window of size=3, strides=2
        >>> m = MaxPool2d(3, strides=2)
        >>> # pool of non-square window
        >>> m = MaxPool2d((3, 2), strides=(2, 1))
        >>> input = tf.random.normal((20, 16, 50, 32))
        >>> output = m(input)

    References:
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


    def forward(self, x) :

        return tf.nn.max_pool2d(x, self.kernel_size, self.strides, 'SAME' if self.auto_pad else 'VALID',
                                data_format="NHWC", name=None)


class MaxPool3d(_PoolNd):
    """Applies a 3D max pooling over an input signal composed of several input
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
        return_indices: if ``True``, will return the max indices along with the outputs.Useful for
        :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              'D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times(\text{
              kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor'

          .. math::
              'H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times (\text{
              kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor'

          .. math::
              'W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times (\text{
              kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor'

    Examples:
        >>> # pool of square window of size=3, strides=2
        >>> m = MaxPool3d(3, strides=2)
        >>> # pool of non-square window
        >>> m = MaxPool3d((3, 2, 2), strides=(2, 1, 2))
        >>> input = torch.randn(20, 16, 50,44, 31)
        >>> output = m(input)

    References:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, name='', **kwargs):
        super(MaxPool3d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _triple(kernel_size)
        self.strides = _triple(strides if strides is not None else kernel_size)
        self.padding = _triple(self.padding)
        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.return_indices = kwargs.get('return_indices', False)


    def forward(self, input):
        return tf.nn.max_pool3d(input, self.kernel_size, self.strides, 'SAME' if self.auto_pad else 'VALID',
                                data_format="NDHWC", name=None)


class MaxUnpool2d(_PoolNd):
    """Computes a partial inverse of :class:`MaxPool2d`.

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

    Examples:

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
        return tf.nn.max_pool_with_argmax(x, self.kernel_size, self.strides, 'SAME' if self.auto_pad else 'VALID',
                                          data_format="NHWC", output_dtype=tf.int64, include_batch_in_index=False,
                                          name=None)


class AvgPool1d(_PoolNd):
    """Applies a 1D average pooling over an input signal composed of several
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

    Examples:

        >>> # pool with window of size=3, strides=2
        >>> m = AvgPool1d(3, strides=2)
        >>> m(to_tensor([[[1.,2,3,4,5,6,7]]]))
        tensor([[[ 2.,  4.,  6.]]])
    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, name='', **kwargs):
        super(AvgPool1d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _single(kernel_size)
        self.auto_pad = auto_pad
        self.strides = _single(strides if strides is not None else kernel_size)
        self.padding = _single(
            self.padding)  # self.ceil_mode = kwargs.get('ceil_mode', False)  # self.count_include_pad = kwargs.get(
        # 'count_include_pad', False)


    def forward(self, input):
        return tf.nn.avg_pool1d(input, self.kernel_size, self.strides, 'SAME' if self.auto_pad else 'VALID',
                                data_format="NWC", name=None)


class AvgPool2d(_PoolNd):
    """Applies a 2D average pooling over an input signal composed of several input
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

    Examples:

        >>> # pool of square window of size=3, strides=2
        >>> m1 = AvgPool2d(3, strides=2)
        >>> # pool of non-square window
        >>> m2 = AvgPool2d((3, 2), strides=(2, 1))
        >>> input = tf.random.normal((1,128,128,32))
        >>> output = m2(input)
        >>> print(int_shape(output))
        [1, 64, 128, 32]

    """

    def __init__(self, kernel_size, strides=None, auto_pad=True, count_include_pad=True, divisor_override=None, name='',
                 **kwargs):
        super(AvgPool2d, self).__init__(kernel_size, strides, auto_pad, 1, name, **kwargs)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides if strides is not None else kernel_size)
        self.auto_pad = auto_pad

        self.ceil_mode = kwargs.get('ceil_mode', False)
        self.count_include_pad = kwargs.get('count_include_pad', True)
        self.divisor_override = kwargs.get('divisor_override', None)


    def forward(self, x) :


        return tf.nn.avg_pool2d(x, self.kernel_size, self.strides, 'SAME' if self.auto_pad else 'VALID',
                                data_format="NHWC", name=None)


class AvgPool3d(_PoolNd):
    """Applies a 3D average pooling over an input signal composed of several input planes.

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
            'D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] -\text{kernel\_size}[0]}{\text{stride}[
            0]} + 1\right\rfloor'

        .. math::
            'H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] -\text{kernel\_size}[1]}{\text{stride}[
            1]} + 1\right\rfloor'

        .. math::
            'W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] -\text{kernel\_size}[2]}{\text{stride}[
            2]} + 1\right\rfloor'

    Examples:
        >>> # pool of square window of size=3, strides=2
        >>> m = AvgPool3d(3, strides=2)
        >>> # pool of non-square window
        >>> m = AvgPool3d((3, 2, 2), strides=(2, 1, 2))
        >>> input = tf.random.normal((1,128,128,128,32))
        >>> output = m(input)
        >>> print(int_shape(output))
        [1, 64, 128, 64, 32]

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
        return tf.nn.avg_pool3d(input, self.kernel_size, self.strides, 'SAME' if self.auto_pad else 'VALID',
                                data_format="NDHWC", name=None)

    def __setstate__(self, d):
        super(AvgPool3d, self).__setstate__(d)
        self.__dict__.setdefault('padding', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)


class GlobalAvgPool1d(Layer):
    def __init__(self, keepdims=False, name='global_avg_pool',**kwargs):
        super(GlobalAvgPool1d, self).__init__(name=name)
        self.keepdims = kwargs.get('keepdim',keepdims)

    def build(self, input_shape:TensorShape):
        if self._built == False:
            if self.keepdims == True:
                output_shape = input_shape.dims
                output_shape[1] = 1
                self.output_shape =TensorShape(output_shape)
            else:
                self.output_shape =TensorShape([input_shape[0],input_shape[-1]])
            self._built = True

    def forward(self, x) :
        x = tf.reduce_mean(x, axis=1, keepdims=self.keepdims)
        return x

class GlobalAvgPool2d(Layer):
    def __init__(self, keepdims=False, name='global_avg_pool',**kwargs):
        super(GlobalAvgPool2d, self).__init__(name=name)
        self.keepdims = kwargs.get('keepdim',keepdims)

    def build(self, input_shape:TensorShape):
        if self._built == False:
            if self.keepdims == True:
                output_shape = input_shape.dims
                output_shape[1] = 1
                output_shape[2] = 1
                self.output_shape = TensorShape(output_shape)
            else:
                self.output_shape =TensorShape([input_shape[0],input_shape[-1]])
            self._built = True

    def forward(self, x) :

        x = tf.reduce_mean(x, [1, 2], keepdims=self.keepdims)
        return x
