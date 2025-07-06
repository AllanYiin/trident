from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
import inspect
import math
import numbers
from abc import ABCMeta
from itertools import repeat
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # import torch functions
from torch import Tensor
from collections import abc
from torch.nn import init
from trident.backend import dtype
from trident.backend.common import *
from trident.backend.common import TensorShape
from trident.backend.pytorch_backend import *
from trident.backend.pytorch_ops import *
from trident.layers.pytorch_activations import get_activation
from trident.layers.pytorch_initializers import *
from trident.layers.pytorch_normalizations import get_normalization

__all__ = ['Dense', 'Embedding', 'Flatten', 'Concatenate', 'Concate', 'SoftMax', 'Add', 'Subtract', 'Dot', 'Scale',
           'Conv1d', 'Conv2d', 'Conv3d',
           'TransConv1d', 'TransConv2d', 'TransConv3d', 'SeparableConv1d', 'SeparableConv2d', 'SeparableConv3d',
           'DepthwiseConv1d', 'DepthwiseConv2d', 'DepthwiseConv3d', 'GatedConv2d', 'GcdConv2d', 'Lambda', 'Reshape','Squeeze',
           'Permute',
           'CoordConv2d', 'Upsampling1d', 'Upsampling2d', 'Upsampling3d', 'Dropout', 'AlphaDropout', 'SelfAttention',
           'SingleImageLayer', 'Aggregation']

_session = get_session()

_epsilon = _session.epsilon


def _ntuple(n):
    def parse(x):
        if isinstance(x, abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Dense(Layer):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples:

        >>> m = Dense(30)
        >>> input = to_tensor(torch.randn(2, 20))
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([2, 30])

        >>> m = Dense(128)
        >>> input = to_tensor(torch.randn(1, 32,64))
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([1, 32,128])
    """

    def __init__(self, num_filters=None, use_bias=True, activation=None, weights_norm=None, keep_output=False,
                 name=None, filter_index=-1, depth_multiplier=1, **kwargs):
        super(Dense, self).__init__(name=name, keep_output=keep_output)
        self.rank = 0
        self.filter_index = -1
        self.depth_multiplier = depth_multiplier
        if isinstance(num_filters, int) or num_filters is None:
            self.num_filters = num_filters
        elif isinstance(num_filters, tuple):
            self.num_filters = unpack_singleton(num_filters)
        else:
            raise ValueError('output_shape should be integer, list of integer or tuple of integer...')
        self.weight = None
        self.bias = None
        self.use_bias = use_bias
        if weights_norm == 'l2':
            self.weights_norm = l2_normalize
        else:
            self.weights_norm = None
        self.activation = get_activation(activation)
        self.decision_boundary_optimal = False

    def build(self, input_shape: TensorShape):
        if not self._built:
            if len(input_shape.dims) == 1:
                self.input_filters = input_shape.dims[0]
            else:
                self.input_filters = input_shape[self.filter_index]
            if self.num_filters is None:
                self.num_filters = int(self.input_filters * self.depth_multiplier)
            self.weight = Parameter(torch.Tensor(self.num_filters, self.input_filters))
            kaiming_uniform(self.weight, a=math.sqrt(5))
            # self._parameters['weight'] =self.weight
            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.num_filters))
                init.zeros_(self.bias)  # self._parameters['bias']=self.bias

            self.to(get_device())
            self._built = True

    def forward(self, x, **kwargs):
        # x=enforce_singleton(x)
        if hasattr(self, 'weights_norm') and self.weights_norm is not None:
            x = F.linear(x, self.weights_norm(self.weight), self.bias)
        else:
            x = F.linear(x, self.weight, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self):
        s = 'output_shape={0}'.format(self.output_shape.tolist()) + ',use_bias={use_bias}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()

        return s.format(**self.__dict__)


class Embedding(Layer):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.
    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`
    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`
    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)
    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.
    Examples::
        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],
                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])
        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0,2,0,5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])
    """
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    embedding_dim: int
    padding_idx: int
    max_norm: float
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool
    keep_output: bool
    name: str

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, padding_idx: Optional[int] = 0,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, keep_output: bool = False,
                 name: Optional[str] = None, add_noise=False,
                 noise_intensity=0.005) -> None:
        super(Embedding, self).__init__(keep_output=keep_output, filter_index=1, name=name)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity

        self.register_parameter('weight', None)

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is not None and int_shape(_weight)[-1] == embedding_dim and len(int_shape(_weight)) == 2:
            self.weight = Parameter(_weight.to(self.get_root().device))
            self.weight.requires_grad = False
            self.num_embeddings = int_shape(self.weight)[0]
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)
            self._built = True
        elif _weight is not None:
            raise ValueError('Shape[-1] of weight does not match embedding_dim')
        elif _weight is None and self.num_embeddings is not None and self.embedding_dim is not None:
            self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim).to(self.get_root().device))
            init.normal_(self.weight)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)
            self._built = True

        self.sparse = sparse

    def build(self, input_shape: TensorShape):
        if self._built == False and self.sparse == False:
            raise ValueError('Only sparse embedding support shape inferred, please setting num_embeddings manually. ')
        elif not self._built:
            if len(input_shape.dims) == 1:
                self.input_filters = input_shape.dims[0]
            else:
                self.input_filters = input_shape[self.filter_index]
            if self.weight is None:
                self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
                init.normal_(self.weight)
                if self.padding_idx is not None:
                    with torch.no_grad():
                        self.weight[self.padding_idx].fill_(0)
            self.to(self.get_root().device)
            self._built = True

    def forward(self, x: torch.Tensor) -> Tensor:
        if self.sparse and x.dtype != dtype.long and int_shape(x)[-1] == self.num_embeddings:
            x = argmax(x, -1)
        elif not self.sparse and x.dtype != dtype.long and int_shape(x)[-1] != self.num_embeddings:
            x = x.long()
        if self.weight.device != x.device:
            self.weight.to(x.device)
        embed = F.embedding(x, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq,
                            self.sparse)
        if self.add_noise == True and self.training == True:
            std = embed.std()
            if std is None or std < 0.02:
                std = 0.02
            noise = self.noise_intensity * random_normal_like(embed, mean=embed.mean(), std=std,
                                                              dtype=embed.dtype).detach().to(embed.device)
            embed = embed + noise
        return embed

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.
        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): See module initialization documentation.
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (boolean, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.
        Examples::
            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding


class Flatten(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self, keep_output: bool = False, name: Optional[str] = None):
        super(Flatten, self).__init__(name=name, keep_output=keep_output)
        super().__setattr__('_built', True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Concate(Layer):
    """Concate layer to splice  tensors ."""

    def __init__(self, axis=1, keep_output: bool = False, name: Optional[str] = None):
        super(Concate, self).__init__(name=name, keep_output=keep_output)
        self.axis = axis
        super().__setattr__('_built', True)

    def forward(self, x, **kwargs) -> torch.Tensor:
        if not isinstance(x, list) or len(x) < 2:
            raise ValueError('A `Concatenate` layer should be called on a list of at least 2 inputs')

        if all([k.size() is None for k in x]):
            return
        reduced_inputs_shapes = [list(k.size()) for k in x]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError(
                'A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got inputs '
                'shapes: %s' % (shape_set))
        x = torch.cat(x, dim=self.axis)
        return x


Concatenate = Concate


class Add(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=1, keep_output: bool = False, name: Optional[str] = None):
        super(Add, self).__init__(name=name, keep_output=keep_output)

    def build(self, input_shape: TensorShape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, x, **kwargs) -> torch.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if isinstance(x, tuple):
            x = unpack_singleton(x)
        out = 0
        for item in x:
            out = torch.add(out, item)
        return out


class Subtract(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=1, keep_output: bool = False, name: Optional[str] = None):
        super(Subtract, self).__init__(name=name, keep_output=keep_output)

    def build(self, input_shape: TensorShape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, x, **kwargs) -> torch.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if not isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out = torch.sub(out, item)
        return out


class Dot(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self, axis=1, keep_output: bool = False, name: Optional[str] = None):
        super(Dot, self).__init__(name=name, keep_output=keep_output)

    def build(self, input_shape: TensorShape):
        if self._built == False:
            self.output_shape = input_shape
            self._built = True

    def forward(self, x, **kwargs) -> torch.Tensor:
        if not isinstance(x, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if not isinstance(x, tuple):
            x = list(x)
        out = 0
        for item in x:
            out = torch.dot(out, item)
        return out


class SoftMax(Layer):
    """SoftMax layer

    SoftMax layer is designed for accelerating  classification model training
    In training stage, it will process the log_softmax transformation (get log-likelihood for a single instance ).
    In testing/ evaluation/ infer stage, it will process the 'so-called' softmax transformation.
    All transformation is processed across 'asix (default=1)'

    And you also can setting add_noise and noise_intensity arugments to imprement output noise.
    output noise can force model make every output probability should large enough or small enough, otherwise it will confused within output noise.
    It;s a regularzation technique for classification model training.

    """

    def __init__(self, axis=1, temperature=1,add_noise=False, noise_intensity=0.005, keep_output: bool = False,
                 name: Optional[str] = None, **kwargs):
        """
        Args:
            axis (int,default=1): The axis all the transformation processed across.
            add_noise (bool, default=False): If True, will add (output) noise  in this layer.
            noise_intensity (float, default=0.005): The noise intensity (is propotional to mean of actual output.

        """
        super(SoftMax, self).__init__(name=name, keep_output=keep_output)
        self.axis = kwargs.get('dim', axis)
        self.temperature=builtins.min(builtins.max(temperature,epsilon()),100)
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity

    def forward(self, x, **kwargs) -> torch.Tensor:
        if not hasattr(self, 'add_noise'):
            self.add_noise = False
            self.noise_intensity = 0.005
        if self.training:
            if self.add_noise:
                _mean = x.mean()
                _std = x.var().sqrt()
                if is_nan(_mean):
                    _mean = 0.0
                if is_nan(_std) or _std < 0.02:
                    _std = 0.02

                noise = self.noise_intensity * random_normal_like(x, mean=_mean, std=_std, dtype=x.dtype).to(x.device).detach()
                x = x + noise
            return F.log_softmax(x/self.temperature, dim=self.axis)
        else:
            return torch.softmax(x, dim=self.axis)


class Scale(Layer):
    """The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

          output=(input∗scale+shift)power

         Examples:
                >>> x = to_tensor(ones((2,4,2,2)))
                >>> layer1=Scale(scale=2,shift=0.5,power=1,mode='uniform')
                >>> output1 = layer1(x)
                >>> (output1==(x*2+0.5)**1).all().to('cpu')
                tensor(True)
                >>> layer2=Scale(scale=to_tensor([1,2,3,4]),shift=0.5,power=1.2,mode='channel')
                >>> output2 = layer2(to_tensor(ones((2,4,2,2))))
                >>> (output2.to('cpu')==pow((x*(to_tensor([1,2,3,4]).reshape((1,4,1,1)))+0.5),1.2).to('cpu')).all()
                tensor(True)
    """

    def __init__(self, scale: (float, Tensor) = 1.0, shift: (float, Tensor) = 0.0, power: (float, Tensor) = 1.0,
                 mode='uniform', keep_output: bool = False,
                 name: Optional[str] = None):
        super(Scale, self).__init__(keep_output=keep_output, name=name)
        self._scale = scale
        self._shift = shift
        self._power = power

        if mode in ['uniform', 'constant', 'channel', 'elementwise']:
            self.mode = mode
        else:
            raise ValueError('Only [uniform,channel,elementwise] is valid value for mode ')

    def build(self, input_shape: TensorShape):
        def remove_from(name: str, *dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if not self._built:

            self.input_filters = input_shape[self.filter_index]
            if self.mode == 'uniform':
                self.register_parameter('weight_scale', Parameter(ones(1, dtype=dtype.float32)))
                self.register_parameter('weight_shift', Parameter(zeros(1, dtype=dtype.float32)))
                self.register_parameter('weight_power', Parameter(ones(1, dtype=dtype.float32)))

            elif self.mode == 'constant':
                self.register_buffer('weight_scale', ones(1, dtype=dtype.float32))
                self.register_buffer('weight_shift', zeros(1, dtype=dtype.float32))
                self.register_buffer('weight_power', ones(1, dtype=dtype.float32))
            elif self.mode == 'channel':

                self.register_parameter('weight_scale', Parameter(ones(self.input_filters, dtype=dtype.float32)))
                self.register_parameter('weight_shift', Parameter(zeros(self.input_filters, dtype=dtype.float32)))
                self.register_parameter('weight_power', Parameter(ones(self.input_filters, dtype=dtype.float32)))
            elif self.mode == 'elementwise':
                new_shape = input_shape.dims[1:]
                self.register_parameter('weight_scale', Parameter(ones(new_shape, dtype=dtype.float32)))
                self.register_parameter('weight_shift', Parameter(zeros(new_shape, dtype=dtype.float32)))
                self.register_parameter('weight_power', Parameter(ones(new_shape, dtype=dtype.float32)))
        self.weight_scale.data.fill_(1.00)
        self.weight_shift.data.fill_(0.00)
        self.weight_power.data.fill_(1.00)
        self._built = True

    def forward(self, x, **kwargs) -> torch.Tensor:
        # scale,shift,power=split(self.weight,3,axis=0)

        x = x.multiply(self.weight_scale)
        x = x.add(self.weight_shift)
        x = torch.sign(x) * torch.pow(torch.abs(x), self.weight_power)
        return x


class Aggregation(Layer):
    """Flatten layer to flatten a tensor after convolution."""

    def __init__(self, mode='mean', axis=1, keepdims=True, keep_output: bool = False, name: Optional[str] = None):
        super(Aggregation, self).__init__(name=name, keep_output=keep_output)
        valid_mode = ['mean', 'sum', 'max', 'min', 'first', 'last']
        if mode in valid_mode:
            self.mode = mode
        else:
            raise ValueError('{0} is not valid mode. please use one of {1}'.format(mode, valid_mode))
        self.axis = unpack_singleton(axis)
        if mode in ['first', 'last'] and isinstance(self.axis, (list, tuple)):
            raise ValueError('{0} only can reduction along one axis.'.format(mode))

        self.keepdims = keepdims

    def build(self, input_shape: TensorShape):
        if not self._built:
            dims = input_shape.dims
            if self.keepdims:
                if isinstance(self.axis, (list, tuple)):
                    for a in self.axis:
                        dims[a] = 1
                elif isinstance(self.axis, numbers.Integral):
                    dims[self.axis] = 1
            else:
                if isinstance(self.axis, (list, tuple)):
                    for a in self.axis:
                        dims.pop(self.axis)
                elif isinstance(self.axis, numbers.Integral):
                    dims.pop(self.axis)
            self.output_shape = TensorShape(dims)
            self._built = True

    def forward(self, x, **kwargs) -> torch.Tensor:
        if self.mode == 'mean':
            x = reduce_mean(x, axis=self.axis, keepdims=self.keepdims)
        elif self.mode == 'sum':
            x = reduce_sum(x, axis=self.axis, keepdims=self.keepdims)
        elif self.mode == 'max':
            x = reduce_max(x, axis=self.axis, keepdims=self.keepdims)
        elif self.mode == 'min':
            x = reduce_min(x, axis=self.axis, keepdims=self.keepdims)
        elif self.mode == 'first':
            x = x.select(dim=self.axis, index=0)
            if self.keepdims:
                x = expand_dims(x, axis=self.axis)
        elif self.mode == 'last':
            x = x.select(dim=self.axis, index=-1)
            if self.keepdims:
                x = expand_dims(x, axis=self.axis)
        return x


_gcd = gcd
_get_divisors = get_divisors
_isprime = isprime


def get_static_padding(rank, kernal_shape, strides, dilations, input_shape=None, transpose=False):
    """ Calcualte the actual padding we need in different rank and different convlution settings.

    Args:
        rank (int):
        kernal_shape (tuple of integer):
        strides (tuple of integer):
        dilations (tuple of integer):
        input_shape (None or tuple of integer):
        transpose (bool): whether transposed

    Returns: the padding we need (shape: 2*rank )

    Examples
    >>> get_static_padding(1,(3,),(2,),(2,))
    (2, 2)
    >>> get_static_padding(2,(3,3),(2,2),(1,1),(224,224))
    (1, 1, 1, 1)
    >>> get_static_padding(2,(3,3),(2,2),(1,1),(224,224),True)
    ((1, 1, 1, 1), (1, 1))
    >>> get_static_padding(2,(5,5),(1,1),(2,2))
    (4, 4, 4, 4)
    >>> get_static_padding(2,(5,5),(1,1),(1,1))
    (2, 2, 2, 2)
    >>> get_static_padding(2,(2,2),(1,1),(1,1))
    (1, 0, 1, 0)
    >>> get_static_padding(3,(5,5,5),(1,1,1),(2,2,2))
    (4, 4, 4, 4, 4, 4)
    """
    if input_shape is None:
        input_shape = [224] * rank

    if isinstance(kernal_shape, int):
        kernal_shape = _ntuple(rank)(kernal_shape)
    if isinstance(strides, int):
        strides = _ntuple(rank)(strides)
    if isinstance(dilations, int):
        dilations = _ntuple(rank)(dilations)

    input_shape = to_numpy(input_shape)
    kernal_shape = to_numpy(list(kernal_shape))
    strides = to_numpy(list(strides)).astype(np.float32)
    dilations = to_numpy(list(dilations))
    if not transpose:
        output_shape = np.ceil(input_shape / strides)
        raw_padding = np.clip((output_shape - 1) * strides + (kernal_shape - 1) * dilations + 1 - input_shape, a_min=0,
                              a_max=np.inf)
        remainder = np.remainder(raw_padding, np.ones_like(raw_padding) * 2)

        raw_padding = raw_padding + (remainder * np.greater(strides, 1).astype(np.float32))
        lefttop_pad = np.ceil(raw_padding / 2.0).astype(np.int32)
        rightbtm_pad = (raw_padding - lefttop_pad).astype(np.int32)
        static_padding = []
        for k in range(rank):
            static_padding.append(lefttop_pad[-1 - k])
            static_padding.append(rightbtm_pad[-1 - k])
        return static_padding
    else:
        output_shape = input_shape * strides
        raw_padding = np.clip(
            ((input_shape - 1) * strides + (kernal_shape - 1) * dilations + 1) - input_shape * strides, a_min=0,
            a_max=np.inf)
        remainder = np.remainder(raw_padding, np.ones_like(raw_padding) * 2)

        raw_padding = raw_padding + (remainder * np.greater(strides, 1).astype(np.float32))
        lefttop_pad = np.ceil(raw_padding / 2.0).astype(np.int32)
        rightbtm_pad = (raw_padding - lefttop_pad).astype(np.int32)

        out_pad = output_shape - (
                    (input_shape - 1) * strides + (kernal_shape - 1) * dilations + 1 - lefttop_pad - rightbtm_pad)

        static_padding = []

        for k in range(rank):
            static_padding.append(int(lefttop_pad[-1 - k]))
            static_padding.append(int(rightbtm_pad[-1 - k]))
        return tuple(static_padding), tuple([int(item) for item in out_pad.astype(np.int32)])


class _ConvNd(Layer):
    __constants__ = ['kernel_size', 'num_filters', 'strides', 'auto_pad', 'padding_mode', 'use_bias', 'dilation',
                     'groups', 'transposed']

    def __init__(self, rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias, dilation,
                 groups,
                 transposed=False, name=None, depth_multiplier=1, depthwise=False, separable=False, keep_output=False,
                 **kwargs):
        super(_ConvNd, self).__init__(keep_output=keep_output, name=name)
        self.rank = rank
        self.num_filters = num_filters
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation = dilation
        self.transposed = transposed
        if self.transposed:
            self.output_padding = _ntuple(rank)(0)
        self.groups = groups
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        if padding is not None:
            self.padding = normalize_padding(padding, rank)
        else:
            self.padding = None

        self.depthwise = depthwise
        self.separable = separable
        if self.separable:
            self.depthwise = True

        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

        self.transposed = transposed
        self.use_bias = use_bias
        #self.to(self.device)

    def build(self, input_shape: TensorShape):
        if not self._built and input_shape is not None:
            self.input_filters = int(input_shape[self.filter_index])
            self._input_shape = input_shape if isinstance(input_shape, TensorShape) else TensorShape(input_shape)
            if self.auto_pad:
                if not self.transposed:
                    self.padding = get_static_padding(self.rank, self.kernel_size, self.strides, self.dilation,
                                                      input_shape.dims[2:])
                else:
                    self.padding, self.output_padding = get_static_padding(self.rank, self.kernel_size, self.strides,
                                                                           self.dilation, input_shape.dims[2:],
                                                                           self.transposed)
            else:
                if self.padding is None:
                    self.padding = [0] * (2 * self.rank)
                elif isinstance(self.padding, int):
                    self.self.padding = [self.padding] * (2 * self.rank)
                elif len(self.padding) == self.rank:
                    self.padding = list(self.padding) * 2
                elif len(self.padding) == 2 * self.rank:
                    pass

            if self.depthwise or self.separable:
                if self.depth_multiplier is None:
                    self.depth_multiplier = 1
                if self.groups > 1:
                    pass
                elif self.depth_multiplier < 1:
                    self.groups = int(builtins.round(self.input_filters * self.depth_multiplier, 0))
                else:
                    self.groups = self.input_filters if self.groups == 1 else self.groups

            if self.num_filters is None and self.depth_multiplier is not None:
                self.num_filters = int(builtins.round(self.input_filters * self.depth_multiplier, 0))

            if self.groups != 1 and self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            if self.depthwise and self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            # channel_multiplier = int(self.num_filters // self.groups) if self.depth_multiplier is None else self.depth_multiplier  # default channel_multiplier

            if self.transposed:
                self.weight = Parameter(
                    torch.Tensor(int(self.input_filters), self.num_filters // self.groups, *self.kernel_size).to(self.get_root().device))
            else:
                self.weight = Parameter(
                    torch.Tensor(int(self.num_filters), self.input_filters // self.groups, *self.kernel_size).to(self.device))  #

                if self.separable:
                    self.pointwise = Parameter(
                        torch.Tensor(int(self.input_filters * self.depth_multiplier), int(self.num_filters), 1, 1).to(self.get_root().device))

            xavier_uniform(self.weight, gain=1)

            if self.use_bias:
                self.bias = Parameter(torch.Tensor(int(self.num_filters)).to(self.get_root().device))
                init.zeros_(self.bias)



            self._built = True

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, num_filters={num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={auto_pad}'
        if hasattr(self, 'padding') and self.padding is not None:
            s += ', padding={0}, padding_mode={1}'.format(self.padding, self.padding_mode)
        s += ',use_bias={use_bias} ,dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if hasattr(self, '_input_shape') and self._input_shape is not None:
            s += ', input_shape={0}, input_filter={1}'.format(to_numpy(self._input_shape).tolist(), self.input_filters)
        if hasattr(self, '_output_shape') and self._output_shape is not None:
            s += ', output_shape={0}'.format(
                self._output_shape._dims if isinstance(self._output_shape, TensorShape) else self._output_shape)
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(
            state)  # if not hasattr(self, 'padding_mode'):  #     self.padding_mode = 'zeros'


class Conv1d(_ConvNd, metaclass=ABCMeta):
    """Applies to create a 1D convolution layer

        Args:
            kernel_size :(int or tupleof ints)
                shape (spatial extent) of the receptive field

            num_filters :(int  or None, default to None)
                number of output channel (filters)`, sometimes in backbond design output channel is propotional to input channel.
                But in trident all layer is shape  delay inferred

            strides:(int or tupleof ints ,default to 1)
                 stride of the convolution (increment when sliding the filter over the input)

            auto_pad:bool
                if `False`, then the filter will be shifted over the "valid" area of input, that is, no value outside the area is used. If ``pad=True`` means 'same

            *padding (optional)
                auto_pad can help you calculate the pad you need.
                if you have special need , you still can use the paddding
                implicit paddings on both sides of the input. Can be a single number or a double tuple (padH, padW)
                or quadruple(pad_left, pad_right, pad_top, pad_btm )

            padding_mode:string (default is 'zero', available option are 'reflect', 'replicate','constant','circular')

            activation: (None, string, function or Layer)
                activation function after the convolution operation for apply non-linearity.

            use_bias:bool
                the layer will have no bias if `False` is passed here

            dilation:(int or tupleof ints)
                the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1

            groups
                split input into groups, \text{in\_channels}in_channels should be divisible by the number of groups. Default: 1
            depth_multiplier: (int of decimal)


            name
                name of the layer

        Shape:
            - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
              additional dimensions and :math:`H_{in} = \text{in\_features}`
            - Output: :math:`(N, *, H_{out})` where all but the last dimension
              are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

        Attributes:
            weight: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in\_features}}`
            bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                    If :attr:`bias` is ``True``, the values are initialized from
                    :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                    :math:`k = \frac{1}{\text{in\_features}}`

        Examples:
            >>> input = to_tensor(torch.randn(1,64,32))
            >>> conv1= Conv1d(3,64,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
            >>> output = conv1(input)
            >>> print(output.size())
            torch.Size([1, 64, 16])
            >>> print(conv1.weight.size())
            torch.Size([64, 64, 3])
            >>> print(conv1.padding)
            (1, 1)
            >>> conv2= Conv1d(3, 256, strides=2, auto_pad=False, padding=1)
            >>> output = conv2(input)
            >>> print(output.size())
            torch.Size([1, 256, 16])
            >>> print(conv2.weight.size())
            torch.Size([256, 64, 3])
            >>> print(conv2.padding)
            (1, 1)
            >>> conv3= Conv1d(5,64,strides=1,activation=mish, auto_pad=True,use_bias=False,dilation=4,groups=16)
            >>> output = conv3(input)
            >>> print(output.size())
            torch.Size([1, 64, 32])
            >>> print(conv3.weight.size())
            torch.Size([64, 4, 5])
            >>> print(conv3.padding)
            (8, 8)
            >>> input = to_tensor(torch.randn(1,32,37))
            >>> conv4= Conv1d(3,64,strides=2,activation=mish, auto_pad=True,use_bias=False)
            >>> output = conv4(input)
            >>> print(output.size())
            torch.Size([1, 64, 19])

        """

    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, keep_output=False, **kwargs):
        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride', strides))
        dilation = _single(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            # avoid someone use the definition as keras padding
            auto_pad = (padding.lower() == 'same')
            auto_pad = False
        elif isinstance(padding, int):
            padding = _single(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            pass
        super(Conv1d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation,
                                     groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)

    def conv1d_forward(self, x, **kwargs):
        x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)
        return F.conv1d(x, self.weight, self.bias, self.strides, _single(0), self.dilation, self.groups)

    def forward(self, x, **kwargs):

        x = self.conv1d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv2d(_ConvNd):
    """Applies to create a 2D convolution layer

        Args:
            kernel_size :(int or tupleof ints)
                shape (spatial extent) of the receptive field

            num_filters :(int  or None, default to None)
                number of output channel (filters)`, sometimes in backbond design output channel is propotional to input channel.
                But in trident all layer is shape  delay inferred

            strides:(int or tupleof ints ,default to 1)
                 stride of the convolution (increment when sliding the filter over the input)

            auto_pad:bool
                if `False`, then the filter will be shifted over the "valid" area of input, that is, no value outside the area is used. If ``pad=True`` means 'same

            *padding (optional)
                auto_pad can help you calculate the pad you need.
                if you have special need , you still can use the paddding
                implicit paddings on both sides of the input. Can be a single number or a double tuple (padH, padW)
                or quadruple(pad_left, pad_right, pad_top, pad_btm )

            padding_mode:string (default is 'zero', available option are 'reflect', 'replicate','constant','circular')

            activation: (None, string, function or Layer)
                activation function after the convolution operation for apply non-linearity.

            use_bias:bool
                the layer will have no bias if `False` is passed here

            dilation:(int or tupleof ints)
                the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1

            groups
                split input into groups, \text{in\_channels}in_channels should be divisible by the number of groups. Default: 1
            depth_multiplier: (int of decimal)

            name
                name of the layer

        Shape:
            - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
              additional dimensions and :math:`H_{in} = \text{in\_features}`
            - Output: :math:`(N, *, H_{out})` where all but the last dimension
              are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

        Attributes:
            weight: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in\_features}}`
            bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                    If :attr:`bias` is ``True``, the values are initialized from
                    :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                    :math:`k = \frac{1}{\text{in\_features}}`

        Examples:
            >>> input = to_tensor(torch.randn(1,32,32,32))
            >>> conv1= Conv2d((3,3),64,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
            >>> output = conv1(input)
            >>> print(output.size())
            torch.Size([1, 64, 16, 16])
            >>> print(conv1.weight.size())
            torch.Size([64, 32, 3, 3])
            >>> print(conv1.padding)
            (1, 1, 1, 1)
            >>> conv2= Conv2d((3, 3), 256, strides=(2, 2), auto_pad=False, padding=((1, 0), (1, 0)))
            >>> output = conv2(input)
            >>> print(output.size())
            torch.Size([1, 256, 16, 16])
            >>> print(conv2.weight.size())
            torch.Size([256, 32, 3, 3])
            >>> print(conv2.padding)
            (1, 0, 1, 0)
            >>> conv3= Conv2d((3,5),64,strides=(1,2),activation=mish, auto_pad=True,use_bias=False,dilation=4,groups=16)
            >>> output = conv3(input)
            >>> print(output.size())
            torch.Size([1, 64, 32, 16])
            >>> print(conv3.weight.size())
            torch.Size([64, 2, 3, 5])
            >>> print(conv3.padding)
            (8, 8, 4, 4)
            >>> input = to_tensor(torch.randn(1,32,608,608))
            >>> conv4= Conv2d((3,3),64,strides=2,activation=mish, auto_pad=True,use_bias=False)
            >>> output = conv4(input)
            >>> print(output.size())
            torch.Size([1, 64, 304, 304])
            >>> print(conv4.padding)
            (1, 1, 1, 1)

        """

    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, keep_output=False, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str):
            if padding.lower() == 'same':
                auto_pad = True
                padding = None
            elif padding.lower() == 'valid':
                auto_pad = False
                padding = _ntuple(self.rank)(0)
        elif isinstance(padding, int) and padding > 0:
            padding = _pair(padding)
            auto_pad = False
        elif isinstance(padding, tuple):
            auto_pad = False
            pass
        super(Conv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)
        # self.rank = 2

    def conv2d_forward(self, x, **kwargs):
        # for backward compatibility

        if self.padding is None or (len(self.padding) != len(self.kernel_size) + 2):
            self.padding = normalize_padding(self.padding, len(self.kernel_size))

        if all([v == 0 for v in self.padding]):
            pass
        elif self.padding_mode == 'zero':
            x = F.pad(x, self.padding, mode='constant', value=0)

        elif self.padding_mode == 'circular':
            expanded_padding = (
            (self.padding[0] + 1) // 2, self.padding[1] // 2, (self.padding[2] + 1) // 2, self.padding[3] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, self.padding, mode=self.padding_mode)

        return F.conv2d(x, self.weight, self.bias, self.strides, _pair(0), self.dilation, self.groups)

    def forward(self, x, **kwargs):

        x = self.conv2d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# class SequenceConv2d(Conv2d):
#     def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero', activation=None, use_bias=False, dilation=1, groups=1,
#     name=None,
#                  depth_multiplier=None, keep_output=False, **kwargs):
#         super().__init__(kernel_size, num_filters, strides, auto_pad, padding, padding_mode, activation, use_bias, dilation, groups, name, depth_multiplier, keep_output,
#         **kwargs)
#         self.
#     def forward(self, x, **kwargs):
#
#         x = self.conv2d_forward(x)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x

class Conv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, keep_output=False, **kwargs):
        rank = 3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _triple(padding)
        elif isinstance(padding, tuple):
            pass
        super(Conv3d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode, use_bias,
                                     dilation, groups, transposed=False, name=name, depth_multiplier=depth_multiplier,
                                     depthwise=False, separable=False, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)

    def conv3d_forward(self, x, **kwargs):
        if self.padding_mode == 'circular':
            expanded_padding = (
                (self.padding[2] + 1) // 2, self.padding[2] // 2, (self.padding[1] + 1) // 2, self.padding[1] // 2,
                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, (
            self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                      mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv3d(x, self.weight, self.bias, self.strides, _triple(0), self.dilation, self.groups)

    def forward(self, x, **kwargs):

        x = self.conv3d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, keep_output=False, **kwargs):
        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride', strides))
        dilation = _single(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _single(padding)
        elif isinstance(padding, tuple):
            pass
        super(TransConv1d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                          use_bias,
                                          dilation, groups, transposed=True, name=name,
                                          depth_multiplier=depth_multiplier,
                                          depthwise=False, separable=False, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)
        self.output_padding = _single(0)

    def conv1d_forward(self, x, **kwargs):
        x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)
        if self.padding > 0:
            self.output_padding = _single(1)
        return F.conv_transpose1d(x, self.weight, self.bias, self.strides, padding=_single(0),
                                  output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)

    def forward(self, x, **kwargs):

        x = self.conv1d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv2d(_ConvNd):
    """
    Examples:
        >>> input = to_tensor(torch.randn(1,32,128,128))
        >>> conv1= TransConv2d((3,3),64,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
        >>> output = conv1(input)
        >>> conv1.padding
        (1, 1, 1, 1)
        >>> conv1.output_padding
        (1, 1)
        >>> print(output.size())
        torch.Size([1, 64, 256, 256])

    """

    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, keep_output=False, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _pair(padding)
        elif isinstance(padding, tuple):
            pass
        super(TransConv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                          use_bias,
                                          dilation, groups, transposed=True, name=name,
                                          depth_multiplier=depth_multiplier,
                                          depthwise=False, separable=False, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)
        self.output_padding = _pair(0)

    # def get_padding(self, input_shape):
    #     pad_h = 0
    #     pad_w = 0
    #     if self.auto_pad == True:
    #         ih, iw = list(input_shape)[-2:]
    #         kh, kw = self.kernel_size[-2:]
    #         sh, sw = self.strides[-2:]
    #         dh, dw = self.dilation[-2:]
    #         oh, ow = (ih - 1) * sh + (kh - 1) * dh + 1, (iw - 1) * sw + (kw - 1) * dw + 1
    #         pad_h = max(oh - ih * sh, 0)
    #         pad_w = max(ow - iw * sw, 0)
    #         self.padding = (pad_h, pad_w)
    #         if pad_h != 0 or pad_w != 0:
    #             self.output_padding = (pad_h % 2 if pad_h > 0 else pad_h, pad_w % 2 if pad_w > 0 else pad_w)

    def conv2d_forward(self, x, **kwargs):
        # if len(self.padding) == self.rank:
        #     self.padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        # if self.padding_mode == 'circular':
        #     expanded_padding = (
        #     (self.padding[0] + 1) // 2, self.padding[1] // 2, (self.padding[2] + 1) // 2, self.padding[3] // 2)
        #     x = F.pad(x, expanded_padding, mode='circular')
        # else:
        #     x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv_transpose2d(x, self.weight, self.bias, self.strides, padding=(self.padding[0], self.padding[2]),
                                  output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)

    def forward(self, x, **kwargs):

        x = self.conv2d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TransConv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None, keep_output=False, **kwargs):
        rank = 3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _triple(padding)
        elif isinstance(padding, tuple):
            pass
        super(TransConv3d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                          use_bias,
                                          dilation, groups, transposed=True, name=name,
                                          depth_multiplier=depth_multiplier,
                                          depthwise=False, separable=False, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)
        self.output_padding = _triple(0)

    def conv3d_forward(self, x, **kwargs):
        if self.auto_pad == True:
            iz, ih, iw = list(x.size())[-3:]
            kz, kh, kw = self.kernel_size[-3:]
            sz, sh, sw = self.strides[-3:]
            dz, dh, dw = self.dilation[-3:]
            oz, oh, ow = math.ceil(iz / sz), math.ceil(ih / sh), math.ceil(iw / sw)
            pad_z = max((oz - 1) * sz + (kz - 1) * dz + 1 - iz, 0)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)

            if pad_z > 0 or pad_h > 0 or pad_w > 0:
                self.output_padding = _triple(1)
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2,
                              pad_z - pad_z // 2], mode=self.padding_mode)
        return F.conv_transpose3d(x, self.weight, self.bias, self.strides, padding=_triple(0),
                                  output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)

    def forward(self, x, **kwargs):

        x = self.conv3d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableConv1d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, depth_multiplier=1, strides=1, auto_pad=True, padding=None,
                 padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, keep_output=False, **kwargs):
        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride', strides))
        dilation = _single(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _single(padding)
        elif isinstance(padding, tuple):
            pass
        super(SeparableConv1d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias,
                                              dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier,
                                              depthwise=True, separable=True, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)


    def conv1d_forward(self, x, **kwargs):
        x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)
        return F.conv1d(x, self.weight, self.bias, self.strides, _single(0), self.dilation, self.groups)

    def forward(self, x, **kwargs):
        x = self.conv1d_forward(x)
        x = F.conv1d(x, self.pointwise, self.bias, 1, _pair(0), 1, 1)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SeparableConv2d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, depth_multiplier=1, strides=1, auto_pad=True, padding=None,
                 padding_mode='zero',
                 activation=None, use_bias=False, dilation=1, groups=1, name=None, keep_output=False, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _pair(padding)
        elif isinstance(padding, tuple):
            pass
        super(SeparableConv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias,
                                              dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier,
                                              depthwise=True, separable=True, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)
        #self.pointwise =Conv2d(kernel_size=(1, 1), num_filters=self.num_filters, strides=1, use_bias=self.use_bias, dilation=1, groups=1)
        self._built = False

    # def build(self, input_shape: TensorShape):
    #     if self._built == False or self.conv1 is None:
    #         if self.num_filters is None:
    #             self.num_filters = self.input_filters * self.depth_multiplier if self.depth_multiplier is not None else self.num_filters
    #         self.conv1 = DepthwiseConv2d(kernel_size=self.kernel_size, depth_multiplier=self.depth_multiplier,
    #                                      strides=self.strides, auto_pad=self.auto_pad, padding_mode=self.padding_mode,
    #                                      activation=self.activation, dilation=self.dilation, use_bias=self.use_bias)
    #         self.pointwise = Conv2d(kernel_size=(1, 1), num_filters=self.num_filters, strides=1, use_bias=self.use_bias,
    #                                 dilation=1, groups=1)
    #         self.to(self.device)
    #         self._built = True

    def conv2d_forward(self, x, **kwargs):
        self.rank = 2
        if len(self.padding) == self.rank:
            self.padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        if self.padding_mode == 'circular':
            expanded_padding = (
            (self.padding[0] + 1) // 2, self.padding[1] // 2, (self.padding[2] + 1) // 2, self.padding[3] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv2d(x, self.weight, self.bias, self.strides, _pair(0), self.dilation, self.groups)
    def forward(self, x, **kwargs):

        x = self.conv2d_forward(x)
        x = F.conv2d(x, self.pointwise, self.bias, 1, _pair(0), 1, 1)

        return x


class SeparableConv3d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, depth_multiplier=1, strides=1, auto_pad=True, padding=None,
                 padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, keep_output=False, **kwargs):
        rank = 3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', num_filters))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _triple(padding)
        elif isinstance(padding, tuple):
            pass
        super(SeparableConv3d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias,
                                              dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier,
                                              depthwise=True, separable=True, keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)



    def conv3d_forward(self, x, **kwargs):
        if self.padding_mode == 'circular':
            expanded_padding = (
                (self.padding[2] + 1) // 2, self.padding[2] // 2, (self.padding[1] + 1) // 2, self.padding[1] // 2,
                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, (
            self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                      mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv3d(x, self.weight, self.bias, self.strides, _triple(0), self.dilation, self.groups)

    def forward(self, x, **kwargs):

        x = self.conv3d_forward(x)
        x = F.conv3d(x, self.pointwise, self.bias, 1, _triple(0), 1, 1)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseConv1d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, name=None, keep_output=False, **kwargs):

        rank = 1
        kernel_size = _single(kernel_size)
        strides = _single(kwargs.get('stride', strides))
        dilation = _single(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', kwargs.get('num_filters', None)))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _single(padding)
        elif isinstance(padding, tuple):
            pass
        groups = 1
        super(DepthwiseConv1d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias, dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier, depthwise=True, separable=False,
                                              keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)

    def conv1d_forward(self, x, **kwargs):
        x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)
        return F.conv1d(x, self.weight, self.bias, self.strides, _single(0), self.dilation, self.groups)

    def forward(self, x, **kwargs):

        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseConv2d(_ConvNd):
    """
    Applies to create a 2D  Depthwise convolution layer
    Depthwise convolution performs just the first step of a depthwise spatial convolution (which acts on each input channel separately).

     Args:
         kernel_size :(int or tupleof ints)
             shape (spatial extent) of the receptive field

         depth_multiplier:(int , decimal or None, default to None)
             The number of depthwise convolution output filters for each input filters.
             The total number of depthwise convolution output filters will be equal to input_filters * depth_multiplier

         strides:(int or tupleof ints ,default to 1)
              stride of the convolution (increment when sliding the filter over the input)

         auto_pad:bool
             if `False`, then the filter will be shifted over the "valid" area of input, that is,
             no value outside the area is used. If ``pad=True`` means 'same

         *padding (optional)
             auto_pad can help you calculate the pad you need.
             if you have special need , you still can use the paddding
             implicit paddings on both sides of the input. Can be a single number or a double tuple (padH, padW)
             or quadruple(pad_left, pad_right, pad_top, pad_btm )

         padding_mode:string (default is 'zero', available option are 'reflect', 'replicate','constant','circular')

         activation: (None, string, function or Layer)
             activation function after the convolution operation for apply non-linearity.

         use_bias:bool
             the layer will have no bias if `False` is passed here

         dilation:(int or tupleof ints)
             the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1

         groups
             split input into groups, \text{in\_channels}in_channels should be divisible by the number of groups.
             Default: 1


         name
             name of the layer

     Shape:
         - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
           additional dimensions and :math:`H_{in} = \text{in\_features}`
         - Output: :math:`(N, *, H_{out})` where all but the last dimension
           are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

     Attributes:
         weight: the learnable weights of the module of shape
             :math:`(\text{out\_features}, \text{in\_features})`. The values are
             initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
             :math:`k = \frac{1}{\text{in\_features}}`
         bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                 If :attr:`bias` is ``True``, the values are initialized from
                 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                 :math:`k = \frac{1}{\text{in\_features}}`

     Examples:
         >>> input = to_tensor(torch.randn(1,32,32,32))
         >>> conv1= DepthwiseConv2d((3,3),depth_multiplier=2,strides=2,activation='leaky_relu', auto_pad=True,use_bias=False)
         >>> output = conv1(input)
         >>> print(output.size())
         torch.Size([1, 64, 16, 16])
         >>> print(conv1.weight.size())
         torch.Size([64, 1, 3, 3])
         >>> print(conv1.padding)
         (1, 1, 1, 1)
         >>> print(conv1.num_filters)
         64

     """

    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, name=None, keep_output=False, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', kwargs.get('num_filters', None)))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _pair(padding)
        elif isinstance(padding, tuple):
            pass
        groups = 1
        super(DepthwiseConv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias, dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier, depthwise=True, separable=False,
                                              keep_output=keep_output, **kwargs)

        self.activation = get_activation(activation)

    def conv2d_forward(self, x, **kwargs):
        self.rank = 2
        if len(self.padding) == self.rank:
            self.padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        if self.padding_mode == 'circular':
            expanded_padding = (
            (self.padding[0] + 1) // 2, self.padding[1] // 2, (self.padding[2] + 1) // 2, self.padding[3] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, self.padding, mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv2d(x, self.weight, self.bias, self.strides, _pair(0), self.dilation, self.groups)

    def forward(self, x, **kwargs):

        x = self.conv2d_forward(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DepthwiseConv3d(_ConvNd):
    def __init__(self, kernel_size, depth_multiplier=1, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, groups=1, name=None, keep_output=False, **kwargs):
        rank = 3
        kernel_size = _triple(kernel_size)
        strides = _triple(kwargs.get('stride', strides))
        dilation = _triple(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', kwargs.get('num_filters', None)))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode

        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _triple(padding)
        elif isinstance(padding, tuple):
            pass
        super(DepthwiseConv3d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                              use_bias,
                                              dilation, groups, transposed=False, name=name,
                                              depth_multiplier=depth_multiplier,
                                              depthwise=False, separable=False, keep_output=False, **kwargs)

        self.activation = get_activation(activation)
        self._built = False

    def conv3d_forward(self, x, **kwargs):
        if self.padding_mode == 'circular':
            expanded_padding = (
                (self.padding[2] + 1) // 2, self.padding[2] // 2, (self.padding[1] + 1) // 2, self.padding[1] // 2,
                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            x = F.pad(x, expanded_padding, mode='circular')
        else:
            x = F.pad(x, (
            self.padding[2], self.padding[2], self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
                      mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        return F.conv3d(x, self.weight, self.bias, self.strides, _triple(0), self.dilation, self.groups)

    def forward(self, x, **kwargs):
        x = self.conv3d_forward(x)

        if self.activation is not None:
            x = self.activation(x)
        return x


# class MixConv2d(Layer):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
#     def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
#         super(MixConv2d, self).__init__()
#
#         groups = len(k)
#         if method == 'equal_ch':  # equal channels per group
#             i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
#             ch = [(i == g).sum() for g in range(groups)]
#         else:  # 'equal_params': equal parameter count per group
#             b = [out_ch] + [0] * groups
#             a = np.eye(groups + 1, groups, k=-1)
#             a -= np.roll(a, 1, axis=1)
#             a *= np.array(k) ** 2
#             a[0] = 1
#             ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b
#
#         self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch,
#                                           out_channels=ch[g],
#                                           kernel_size=k[g],
#                                           stride=stride,
#                                           padding=k[g] // 2,  # 'same' pad
#                                           dilation=dilation,
#                                           bias=bias) for g in range(groups)])
#
#     def forward(self, x, **kwargs):
#         return torch.cat([m(x) for m in self.m], 1)

class DeformConv2d(Layer):
    def __init__(self, kernel_size, num_filters=None, strides=1, offset_group=2, auto_pad=True, padding_mode='zero',
                 activation=None, use_bias=False, dilation=1, groups=1, name=None, depth_multiplier=None,
                 keep_output=False, **kwargs):
        super(DeformConv2d, self).__init__(keep_output=keep_output, name=name)
        self.rank = 2
        self.kernel_size = _pair(kernel_size)
        self.num_filters = kwargs.get('num_filters')
        if self.num_filters is None and depth_multiplier is not None:
            self.depth_multiplier = depth_multiplier

        self.dilation = _pair(dilation)
        self.strides = _pair(strides)
        self.use_bias = use_bias
        self.auto_pad = auto_pad
        self.padding_mode = padding_mode
        self.activation = get_activation(activation)
        self.padding = kwargs.get('padding', None)

        if self.padding is not None and isinstance(self.padding, int):
            if self.padding > 0:
                self.auto_pad = False
            self.padding = _pair(self.padding)
        else:
            self.padding = _pair(0)
        self.groups = groups
        if self.input_filters % self.groups != 0:
            raise ValueError('in_channels must be divisible by groups')

    def build(self, input_shape: TensorShape):
        if self._built == False:
            if self.num_filters % self.groups != 0:
                raise ValueError('out_channels must be divisible by groups')

            self.offset = Parameter(torch.Tensor(self.input_filters, 2 * self.input_filters, 3, 3))
            kaiming_uniform(self.offset, a=math.sqrt(5))
            self.weight = Parameter(
                torch.Tensor(self.num_filters, self.input_filters // self.groups, *self.kernel_size))
            kaiming_uniform(self.weight, a=math.sqrt(5))
            if self.use_bias:
                self.bias = Parameter(torch.empty(self.num_filters))
                init.zeros_(self.bias)
            self._built = True

    def offetconv2d_forward(self, x, **kwargs):
        if self.auto_pad == True:
            ih, iw = list(x.size())[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = self.dilation[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode=self.padding_mode)
        return F.conv2d(x, self.offset, None, (1, 1), (0, 0), (1, 1), (1, 1))

    def conv2d_forward(self, x, **kwargs):
        if self.auto_pad == True:
            ih, iw = list(x.size())[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = self.dilation[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
                          mode='constant' if self.padding_mode == 'zero' else self.padding_mode)
        return F.conv2d(x, self.weight, self.bias, self.strides, self.padding, self.dilation, self.groups)

    def forward(self, x, **kwargs):
        """
        Args:
            x (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
                out_height, out_width]): offsets to be applied for each position in the
                convolution kernel.
        """

        # B 2*input,H,W
        offset = self.offetconv2d_forward(x).round_()
        # 2,H,W-->B,2,H,W
        grid = meshgrid(x.shape[3], x.shape[2]).unsqueeze(0).repeat(x.size(0))

        offset = grid + offset

        deform_x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class GcdConv1d(Layer):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, divisor_rank=0, self_norm=True, is_shuffle=False, name=None,
                 depth_multiplier=None, keep_output=False, **kwargs):
        super(GcdConv1d, self).__init__(keep_output=keep_output, name=name)
        self.rank = 1
        self.kernel_size = _single(kernel_size)
        self.num_filters = num_filters
        if self.num_filters is None and depth_multiplier is not None:
            self.depth_multiplier = depth_multiplier
        self.strides = _single(strides)
        self.auto_pad = auto_pad
        self.padding = 0
        self.padding_mode = padding_mode

        self.activation = get_activation(activation)
        self.dilation = _single(dilation)
        self.self_norm = self_norm
        self.norm = None
        self.is_shuffle = is_shuffle
        self.use_bias = use_bias
        self.divisor_rank = divisor_rank
        self.crossgroup_fusion = False
        self.weight = None
        self.bias = None
        self.groups = 1
        self._built = False

    def calculate_gcd(self):
        if self.input_filters is None or not isinstance(self.input_filters, int):
            raise ValueError('in_channels must be integer ')
        gcd_list = gcd(self.input_filters, self.num_filters)
        if len(gcd_list) == 0:
            self.groups = self.input_filters
            self.num_filters_1 = self.input_filters
        else:
            self.gcd = gcd_list[0]
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]

        if self.input_filters == self.num_filters or self.input_filters == self.gcd or self.num_filters == self.gcd:
            self.groups = gcd_list[min(int(self.divisor_rank + 1), len(gcd_list))]

    def build(self, input_shape: TensorShape):
        if not self._built:
            self.calculate_gcd()
            # print('input:{0} -> output:{1}   {2}  {3}  gcd:{4} group:{5}   通道縮放倍數:{5} '.format(self.input_filters,
            #                                                                                    self.num_filters,
            #                                                                                    self.input_filters // self.groups,
            #                                                                                    self.num_filters // self.groups,
            #                                                                                    self.gcd, self.groups,
            #                                                                                    self.num_filters / self.num_filters))

            self.channel_kernal = 2 if self.crossgroup_fusion == True and self.groups > 3 else 1
            self.channel_dilation = 1
            if self.crossgroup_fusion == True and self.groups >= 4:
                self.channel_dilation = 2
            self.kernel_size = (self.channel_kernal,) + _pair(self.kernel_size)
            self.dilation = (self.channel_dilation,) + _pair(self.dilation)
            self.strides = (1,) + _pair(self.strides)
            reshape_input_shape = [-1, self.input_filters // self.groups, self.groups, self._input_shape[1]]

            self.weight = Parameter(torch.Tensor(self.num_filters // self.groups, self.input_filters // self.groups,
                                                 *self.kernel_size))  #
            kaiming_uniform(self.weight, mode='fan_in')
            self._parameters['weight'] = self.weight

            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.num_filters // self.groups))
                init.zeros_(self.bias)
                self._parameters['bias'] = self.bias

            if self.self_norm:
                self.norm = get_normalization('batch')
                init.ones_(self.norm.weight)
                init.zeros_(self.norm.bias)

            self.to(self.device)
            self._built = True

    def forward(self, x, **kwargs):

        if self.auto_pad:
            ih, iw = x.size()[-2:]
            kh, kw = self.kernel_size[-2:]
            sh, sw = self.strides[-2:]
            dh, dw = _pair(self.dilation)[-2:]
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], mode=self.padding_mode)

        x = x.view(x.size(0), x.size(1) // self.groups, self.groups, x.size(2))
        pad_g = max((self.groups - 1) * 1 + (self.channel_kernal - 1) * self.channel_dilation + 1 - self.groups, 0)
        x = F.pad(x, [0, 0, 0, 0, pad_g // 2, pad_g - pad_g // 2], mode='reflect')

        x = F.conv2d(x, self.weight, self.bias, self.strides, self.padding, self.dilation, 1)
        if self.is_shuffle:
            x = x.transpose(2, 1)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
        if self.self_norm:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation']).__repr__()
        s += ',auto_pad={auto_pad},use_bias={use_bias} ,dilation={dilation}}'
        if self.gcd != 1:
            s += ', gcd={gcd},divisor_rank={divisor_rank},self_norm={self_norm},crossgroup_fusion={' \
                 'crossgroup_fusion},is_shuffle={is_shuffle} '
        if self._input_shape is not None:
            s += ', input_shape={0}, input_filter={1}'.format(self._input_shape.clone().tolist(), self.input_filters)
        if self.output_shape is not None:
            s += ', output_shape={0}'.format(self.output_shape if isinstance(self.output_shape, (
                list, tuple)) else self.output_shape.clone().tolist())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class GatedConv2d(Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support zero-size tensor and more features.
    """

    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding=None, padding_mode='zero',
                 activation=None,
                 use_bias=False, dilation=1, groups=1, depth_multiplier=None, norm=l2_normalize, name=None,
                 keep_output=False, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        super(GatedConv2d, self).__init__(kernel_size=kernel_size, num_filters=num_filters, strides=strides,
                                          auto_pad=auto_pad, padding=padding, padding_mode=padding_mode,
                                          activation=activation,
                                          use_bias=use_bias, dilation=dilation, groups=groups, name=name,
                                          depth_multiplier=depth_multiplier,
                                          keep_output=False, **kwargs)

        self.norm = norm

    def forward(self, x, **kwargs):

        x = super().forward(x)
        if self.strides == (2, 2):
            x = x[:, :, 1::2, 1::2]

        x, g = split(x, axis=1, num_splits=2)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = x * g.sigmoid_()
        return x


class GcdConv2d(_ConvNd):
    def __init__(self, kernel_size, num_filters=None, strides=1, auto_pad=True, padding_mode='zero', activation=None,
                 use_bias=False, dilation=1, divisor_rank=0, self_norm=True, is_shuffle=False, crossgroup_fusion=True,
                 name=None, depth_multiplier=None, keep_output=False, **kwargs):
        rank = 2
        kernel_size = _pair(kernel_size)
        strides = _pair(kwargs.get('stride', strides))
        dilation = _pair(kwargs.get('dilation_rate', dilation))
        num_filters = kwargs.get('filters', kwargs.get('out_channels', kwargs.get('num_filters', None)))
        use_bias = kwargs.get('bias', use_bias)
        padding_mode = padding_mode.lower().replace('zeros', 'zero') if isinstance(padding_mode, str) else padding_mode
        padding = kwargs.get('padding', None)
        if isinstance(padding, str) and auto_pad == False:
            auto_pad = (padding.lower() == 'same')
        elif isinstance(padding, int):
            padding = _pair(padding)
        elif isinstance(padding, tuple):
            pass
        groups = 1
        super(GcdConv2d, self).__init__(rank, kernel_size, num_filters, strides, auto_pad, padding, padding_mode,
                                        use_bias, dilation, groups, transposed=False, name=name,
                                        depth_multiplier=depth_multiplier, depthwise=True, separable=False,
                                        keep_output=False, **kwargs)

        self.activation = get_activation(activation)

        self.norm = None
        if self_norm == True:
            self.self_norm = self_norm
            self.norm = get_normalization('instance')

        self.is_shuffle = is_shuffle
        self.divisor_rank = divisor_rank
        self.crossgroup_fusion = crossgroup_fusion

    def calculate_gcd(self):
        if self.input_filters is None or not isinstance(self.input_filters, int):
            raise ValueError('in_channels must be integer ')
        self.register_buffer('gcd', torch.zeros((1)))

        gcd_list = gcd(self.input_filters, self.num_filters)
        if len(gcd_list) == 0:
            self.groups = self.input_filters
            self.num_filters_1 = self.input_filters
        else:
            self.gcd = torch.tensor(gcd_list[0])
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]

        if self.input_filters == self.num_filters or self.input_filters == self.gcd or self.num_filters == self.gcd:
            self.groups = gcd_list[min(int(self.divisor_rank + 1), len(gcd_list))]

    def get_padding(self, input_shape):
        pad_w = 0
        pad_h = 0
        pad_z = 0
        if self.auto_pad == True:
            iz, ih, iw = list(input_shape)[-3:]
            kz, kh, kw = self.actual_kernel_size[-3:]
            sz, sh, sw = self.actual_strides[-3:]
            dz, dh, dw = self.actual_dilation[-3:]
            oz, oh, ow = math.ceil(iz / sz), math.ceil(ih / sh), math.ceil(iw / sw)
            pad_z = max((oz - 1) * sz + (kz - 1) * dz + 1 - iz, 0)
            pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
            pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
            if pad_h % 2 == 1 and sh > 1:
                pad_h += 1
            if pad_w % 2 == 1 and sw > 1:
                pad_w += 1

        elif len(self.padding) == 3:
            pad_z = self.padding[0]
            pad_h = self.padding[1] * 2
            pad_w = self.padding[2] * 2

        if self.padding_mode == 'circular':
            self.padding = (pad_z, 0, pad_h // 2, pad_h - (pad_h // 2), pad_w // 2, pad_w - (pad_w // 2))
        else:
            self.padding = (pad_z, pad_h // 2, pad_w // 2)

    def build(self, input_shape: TensorShape):
        if self._built == False:
            self.input_filters = input_shape[self.filter_index]
            self.calculate_gcd()
            if self.num_filters is None and self.depth_multiplier is not None:
                self.num_filters = int(round(self.input_filters * self.depth_multiplier, 0))
            if self.input_filters % self.groups != 0:
                raise ValueError('in_channels must be divisible by groups')
            print('{0} input:{1} -> output:{2}   {3}  {4}  gcd:{5} group:{6}   通道縮放倍數:{7} '.format(self.name,
                                                                                                         self.input_filters,
                                                                                                         self.num_filters,
                                                                                                         self.input_filters // self.groups,
                                                                                                         self.num_filters // self.groups,
                                                                                                         self.gcd,
                                                                                                         self.groups,
                                                                                                         self.num_filters / float(
                                                                                                             self.input_filters)))

            self.channel_kernal = torch.tensor(2 if self.crossgroup_fusion == True and self.groups > 4 else 1)
            self.channel_dilation = torch.tensor(1)
            # if self.crossgroup_fusion == True and self.groups > 6:
            #     self.channel_dilation = torch.tensor(2)

            self.actual_kernel_size = (self.channel_kernal,) + _pair(self.kernel_size)
            self.actual_dilation = (self.channel_dilation,) + _pair(self.dilation)
            self.actual_strides = (1,) + _pair(self.strides)

            self.get_padding([input_shape[1] // self.groups, self.groups, input_shape[2], input_shape[3]])

            self.weight = Parameter(torch.Tensor(self.num_filters // self.groups, self.input_filters // self.groups,
                                                 *self.actual_kernel_size))  #
            kaiming_uniform(self.weight, mode='fan_in')

            if self.use_bias:
                self.bias = Parameter(torch.Tensor(self.num_filters // self.groups))
                init.zeros_(self.bias)
            else:
                self.register_parameter('bias', None)

            self.to(self.device)
            self._built = True

    def forward(self, x, **kwargs):

        x = x.view(x.size(0), x.size(1) // self.groups, self.groups, x.size(2), x.size(3))
        x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0),
                  mode='constant' if self.padding_mode == 'zero' else self.padding_mode)

        if self.channel_kernal.item() == 2:
            x = torch.cat([x, x[:, :, 0:1, :, :]], dim=2)

        x = F.conv3d(x, self.weight, self.bias, self.actual_strides, _triple(0), self.actual_dilation, 1)

        if self.is_shuffle:
            x = x.transpose_(2, 1)
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4))
        if self.self_norm == True:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        s = 'kernel_size={kernel_size}, {num_filters},strides={strides}'
        if 'activation' in self.__dict__ and self.__dict__['activation'] is not None:
            if inspect.isfunction(self.__dict__['activation']):
                s += ', activation={0}'.format(self.__dict__['activation'].__name__)
            elif isinstance(self.__dict__['activation'], nn.Module):
                s += ', activation={0}'.format(self.__dict__['activation'].__repr__())
        s += ',auto_pad={auto_pad},use_bias={use_bias} ,dilation={dilation}'
        if self.gcd != 1:
            s += ', divisor_rank={divisor_rank},self_norm={self_norm},crossgroup_fusion={crossgroup_fusion},is_shuffle={is_shuffle} '
        # if self._input_shape is not None:
        #     s += ', input_shape={0}, input_filter={1}'.format(self._input_shape.clone().tolist(),
        #                                                       self.input_filters)
        # if self.output_shape is not None:
        #     s += ', output_shape={0}'.format(self.output_shape if isinstance(self.output_shape, (
        #     list, tuple)) else self.output_shape.clone().tolist())
        #     if self.bias is None:
        #         s += ', use_bias=False'
        return s.format(**self.__dict__)


class Lambda(Layer):
    def __init__(self, funcs):
        super(Lambda, self).__init__()
        self.funcs = funcs

    def forward(self, x):
        return self.funcs(x)


class Reshape(Layer):
    def __init__(self, target_shape, batch_size=None, keep_output=False, name=None):
        """
        Reshape the input volume
        Args:
            *target_shape (ints): new shape, WITHOUT specifying batch size as first
            dimension, as it will remain unchanged.
        """
        super(Reshape, self).__init__(keep_output=keep_output, name=name)
        super().__setattr__('_built', True)
        self.batch_size = batch_size
        if isinstance(target_shape, numbers.Integral):
            self.target_shape = (target_shape,)
        elif isinstance(target_shape, list):
            self.target_shape = tuple(target_shape)
        else:
            self.target_shape = target_shape

    def forward(self, x, **kwargs):
        x = enforce_singleton(x)
        shp = self.target_shape
        if -1 in shp:
            return torch.reshape(x, (int_shape(x)[0],) + tuple(shp)) if self.batch_size is None else torch.reshape(x, (
            self.batch_size,) + tuple(shp))
        else:
            return torch.reshape(x, (-1,) + tuple(shp)) if self.batch_size is None else torch.reshape(x, (
            self.batch_size,) + tuple(shp))

class Squeeze(Layer):
    """Squeeze Layer
        remove a length=1 axis

    """
    def __init__(self, axis=1,  name=None,**kwargs):
        """
        Squeeze the input tensor
        Args:
            axis (None, int,tuple, list): The axis need to be squeezed..
        """
        super(Squeeze, self).__init__(name=name)
        self.axis = axis
        super().__setattr__('_built', True)

    def forward(self, x, **kwargs):
        return torch.squeeze(x,dim=self.axis)

class Permute(Layer):
    """Permute Layer

    """

    def __init__(self, *args, name=None):
        """
        Permute the input tensor
        Args:
            *shape (ints): new shape, WITHOUT specifying batch size as first
            dimension, as it will remain unchanged.
        """
        super(Permute, self).__init__(name=name)
        self.pattern = args
        super().__setattr__('_built', True)

    def forward(self, x, **kwargs):
        return permute(x, self.pattern)


class SelfAttention(Layer):
    """ Self attention Laye"""

    def __init__(self, reduction_factor=8, keep_output=False, name=None):
        super(SelfAttention, self).__init__(keep_output=keep_output, name=name)
        self.rank = 2
        # self.activation = activation
        self.reduction_factor = reduction_factor
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.attention = None
        self.softmax = nn.Softmax(dim=-1)  #

    def build(self, input_shape: TensorShape):
        self.query_conv = Conv2d((1, 1), depth_multiplier=1.0 / self.reduction_factor)
        self.key_conv = Conv2d((1, 1), depth_multiplier=1.0 / self.reduction_factor)
        self.value_conv = Conv2d((1, 1), depth_multiplier=1)
        self.register_parameter('gamma', Parameter(torch.zeros(1)))
        # self.to(self.device)

    def forward(self, x, **kwargs):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        B, C, width, height = x.size()
        proj_query = self.query_conv(x).view(B, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(B, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        self.attention = self.softmax(energy).clone()  # BX (N) X (N)
        proj_value = self.value_conv(x).view(B, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, self.attention.permute(0, 2, 1))
        out = out.view(B, C, width, height)

        out = self.gamma * out.clone() + x
        return out


class CoordConv2d(Layer):
    """Implementation of the CoordConv modules from https://arxiv.org/abs/1807.03247

        Examples:
            >>> input = to_tensor(torch.randn(2,32,64,64))
            >>> conv= CoordConv2d((3,3),64,strides=1,activation='leaky_relu', auto_pad=True,use_bias=False)
            >>> output = conv(input)
            >>> print(output.size())
            torch.Size([2, 64, 64, 64])



    """

    def __init__(self, kernel_size, num_filters, strides, auto_pad=True, activation=None, use_bias=False, group=1,
                 dilation=1, with_r=False, image_size=None, name=None, keep_output=False, **kwargs):
        super().__init__(keep_output=keep_output, name=name)
        self.rank = 2
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.auto_pad = auto_pad
        self.use_bias = use_bias
        self.group = group
        self.dilation = dilation
        self.with_r = with_r
        self.conv = Conv2d(self.kernel_size, self.num_filters, self.strides, auto_pad=self.auto_pad,
                           activation=activation, use_bias=self.use_bias, group=self.group,
                           dilation=self.dilation)
        if image_size is not None and isinstance(image_size, tuple):
            self.comput_coord(image_size)

    def comput_coord(self, input_shape):
        _, _, y_dim, x_dim = input_shape.dims
        xs = torch.linspace(0, 1, x_dim, requires_grad=False)
        ys = torch.linspace(0, 1, y_dim, requires_grad=False)
        grid_x, grid_y = torch.meshgrid([xs, ys])
        grid = torch.stack([grid_y, grid_x], 0).unsqueeze(0)
        grid = grid * 2 - 1
        grid = grid.transpose(2, 3)
        self.register_buffer('coord', grid.float().detach())

    def build(self, input_shape: TensorShape):
        if not self._built:
            self.comput_coord(input_shape)

    def append_coords(self, x):
        """
        An alternative implementation for PyTorch with auto-infering the x-y dimensions.
        https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
        """
        batch_size, _, y_dim, x_dim = x.size()
        if y_dim != int_shape(self.coord)[2] or x_dim != int_shape(self.coord)[3]:
            self.comput_coord(int_shape(x)[1:])
        grid = self.coord.repeat(batch_size, 1, 1, 1).cast(x.dtype).to(x.device)
        ret = torch.cat([x, grid], dim=1)

        if self.with_r:
            rr = reduce_sum((grid - 0.5).square(), axis=1, keepdims=True)
            ret = torch.cat([ret, rr], dim=1)

        return ret

    def forward(self, x, **kwargs):

        ret = self.append_coords(x)
        ret = self.conv(ret)
        return ret


class Upsampling1d(Layer):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True, name=None, keep_output=False,
                 **kwargs):
        super(Upsampling1d, self).__init__(keep_output=keep_output, name=name)
        self.rank = 1
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, **kwargs):

        if self.mode == 'pixel_shuffle':
            return F.pixel_shuffle(x, int(self.scale_factor))
        elif self.mode == 'nearest':
            return F.interpolate(x, self.size, self.scale_factor, self.mode, None)
        else:
            return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class Upsampling2d(Layer):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, antialias: bool = False, name=None, keep_output=False,
                 **kwargs):
        """

        Args:
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
            scale_factor (float or Tuple[float]): multiplier for spatial size. If `scale_factor` is a tuple,
            its length has to match the number of spatial dimensions; `input.dim() - 2`.
            mode: algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'`` | ``'nearest-exact'`` | ``'pixel-shuffle'``. Default: ``'nearest'``
            align_corners:
            antialias:
            name:
            keep_output:
            **kwargs:
        """
        super(Upsampling2d, self).__init__(keep_output=keep_output, name=name)
        self.mode = mode
        if mode in ("nearest", "area", "nearest-exact"):
            if align_corners is not None:
                align_corners =None
        else:
            if align_corners is None:
                align_corners = False

        self.rank = 2
        self.size = size

        if isinstance(scale_factor, tuple):
            if self.mode == 'pixel_shuffle':
                self.scale_factor=int(scale_factor[0])
            else:
                self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            if self.mode == 'pixel_shuffle':
                self.scale_factor=int(scale_factor)
            else:
                self.scale_factor = [scale_factor for _ in range(self.rank)]

        self.align_corners = align_corners
        self.antialias=antialias


    def forward(self, x, **kwargs):

        if self.mode == 'pixel_shuffle':
            return F.pixel_shuffle(concate([x]*int(self.scale_factor)**2,axis=1), int(self.scale_factor))
        elif self.mode == 'nearest':
            return F.interpolate(x, self.size, self.scale_factor, self.mode, None)
        else:
            return F.interpolate(x, self.size, self.scale_factor, mode=self.mode, align_corners=self.align_corners,antialias=self.antialias)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class Upsampling3d(Layer):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True, name=None, keep_output=False,
                 **kwargs):
        super(Upsampling3d, self).__init__(keep_output=keep_output, name=name)
        self.rank = 3
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, **kwargs):

        if self.mode == 'pixel_shuffle':
            batch_size, channels, in_depth, in_height, in_width = x.size()
            channels //= self.upscale_factor ** 3

            out_depth = in_depth * self.upscale_factor
            out_height = in_height * self.upscale_factor
            out_width = in_width * self.upscale_factor
            input_view = x.contiguous().view(
                batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
                in_depth, in_height, in_width)
            shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)
        elif self.mode == 'nearest':
            return F.interpolate(x, self.size, self.scale_factor, self.mode, None)
        else:
            return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class Dropout(Layer):
    def __init__(self, dropout_rate=0, name=None, keep_output=False, **kwargs):
        super(Dropout, self).__init__(keep_output=keep_output, name=name)
        self.inplace = False
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, ""but got {}".format(dropout_rate))
        self.dropout_rate = dropout_rate
        self._built=True

    def forward(self, x, **kwargs):
        if not hasattr(self, 'inplace') and 'inplace' not in kwargs:
            self.inplace = False
        if self.inplace and not x.is_leaf:
            self.inplace = True
        return F.dropout(x, self.dropout_rate, self.training, self.inplace)

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.dropout_rate, self.inplace)


class AlphaDropout(Layer):
    """
     .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    def __init__(self, dropout_rate=0, name=None, keep_output=False, **kwargs):
        super(AlphaDropout, self).__init__(keep_output=keep_output, name=name)
        self.inplace = True
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, ""but got {}".format(dropout_rate))
        self.dropout_rate = dropout_rate

    def forward(self, x, **kwargs):
        if not hasattr(self, 'inplace') and 'inplace' not in kwargs:
            self.inplace = False
        if self.inplace and not x.is_leaf:
            self.inplace = True
        return F.alpha_dropout(x, self.dropout_rate, self.training, self.inplace)

    def extra_repr(self):
        return 'p={}, inplace={}'.format(self.dropout_rate, self.inplace)


class DropBlock2d(Layer):
    r"""Randomly zeroes spatial blocks of the input tensor.


    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, dropout_rate=0.1, block_size=7, name=None, keep_output=False, **kwargs):
        super(DropBlock2d, self).__init__(keep_output=keep_output, name=name)
        self.dropout_rate = dropout_rate
        self.block_size = block_size

    def forward(self, x, **kwargs):

        if not self.training or self.dropout_rate == 0:
            return x
        gamma = self.dropout_rate / self.block_size ** 2
        for sh in x.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(x) * gamma)
        Msum = F.conv2d(M,
                        torch.ones((x.shape[1], 1, self.block_size, self.block_size)).to(device=x.device,
                                                                                         dtype=x.dtype),
                        padding=self.block_size // 2,
                        groups=x.shape[1])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=x.device, dtype=x.dtype)
        return x * mask * mask.numel() / mask.sum()  # TODO x * mask * self.keep_prob ?


class SingleImageLayer(Layer):
    def __init__(self, image, is_recursive=False, name=None, keep_output=False, **kwargs):
        super(SingleImageLayer, self).__init__(keep_output=keep_output, name=name)
        self.rank = 2
        if isinstance(image, (np.ndarray, torch.Tensor)):
            self.origin_image = to_tensor(image, requires_grad=True).squeeze()
            self.input_shape = int_shape(self.origin_image)
            self.weight = Parameter(self.origin_image.clone(), trainable=True)
            self.input_filters = int_shape(self.origin_image)[0]
            self._built = True

    def forward(self):
        return self.weight.unsqueeze(0)

    def extra_repr(self):
        return 'is_recursive={0}'.format(self.is_recursive)
