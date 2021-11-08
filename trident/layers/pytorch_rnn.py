"""Pytorch recursive layers definition in trident"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers
import random
import warnings
from typing import Optional, Tuple, overload,Union

import torch
import torch.nn as nn
from torch import Tensor, _VF
from torch._jit_internal import List
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from trident.layers.pytorch_layers import Embedding, Dense, SoftMax

from trident.backend.pytorch_ops import *
from  trident.backend.common import *
from trident.backend.pytorch_backend import Layer, get_device

__all__ = ['RNNBase','RNN','LSTM','GRU','LSTMDecoder']
_rnn_impls = {
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
}


def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)



class RNNBase(Layer):
    __constants__ = ['mode', 'input_filters', 'hidden_size', 'num_layers', 'use_bias',
                     'batch_first', 'dropout_rate', 'bidirectional']

    mode: str
    input_filters: int
    hidden_size: int
    num_layers: int
    use_bias: bool
    batch_first: bool
    dropout_rate: float
    bidirectional: bool
    in_sequence: bool
    filter_index: int

    def __init__(self, mode: str, hidden_size: int,proj_size: int = 0,
                 num_layers: int = 1,stateful=False, use_bias: bool = True, batch_first: bool = False,
                 dropout_rate: float = 0., bidirectional: bool = False,keep_output=False,in_sequence=True,filter_index=-1,name=None) -> None:
        super(RNNBase, self).__init__(name=name,keep_output=keep_output)

        self.mode = mode
        self.hidden_size = hidden_size
        self.proj_size= proj_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.stateful=stateful
        self._batch_first = batch_first
        if not self._batch_first:
            self.batch_index =1
        else:
            self.batch_index = 0
        self.filter_index = -1
        self.dropout_rate = float(dropout_rate)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if not isinstance(dropout_rate, numbers.Number) or not 0 <= dropout_rate <= 1 or \
                isinstance(dropout_rate, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout_rate > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout_rate, num_layers))

        if mode == 'LSTM':
            self.gate_size = 4 * hidden_size
        elif mode == 'GRU':
            self.gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            self.gate_size = hidden_size
        elif mode == 'RNN_RELU':
            self.gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []

    @property
    def batch_first(self):
        return self._batch_first

    @batch_first.setter
    def batch_first(self, value: bool):
        if self._batch_first != value:
            self._batch_first = value
            if not self._batch_first:
                self.batch_index = 1
            else:
                self.batch_index = 0

    def initial_state(self,input) :
        pass

    def build(self, input_shape:TensorShape):
        if not self._built:
            for layer in range(self.num_layers):
                for direction in range(self.num_directions):
                    layer_input_size = input_shape[-1] if layer == 0 else self.hidden_size * self.num_directions

                    w_ih = Parameter(torch.Tensor(self.gate_size, layer_input_size).to(get_device()))
                    w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size).to(get_device()))
                    b_ih = Parameter(torch.Tensor(self.gate_size).to(get_device()))
                    # Second bias vector included for CuDNN compatibility. Only one
                    # bias vector is needed in standard definition.
                    b_hh = Parameter(torch.Tensor(self.gate_size).to(get_device()))
                    layer_params = (w_ih, w_hh, b_ih, b_hh)

                    suffix = '_reverse' if direction == 1 else ''
                    param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                    if self.use_bias:
                        param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                    param_names = [x.format(layer, suffix) for x in param_names]

                    for name, param in zip(param_names, layer_params):
                        if hasattr(self, "_flat_weights_names") and name in self._flat_weights_names:
                            # keep self._flat_weights up to date if you do self.weight = ...
                            idx = self._flat_weights_names.index(name)
                            self._flat_weights[idx] = param
                        self.register_parameter(name, param)

                    self._flat_weights_names.extend(param_names)
                    self._all_weights.append(param_names)

            self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
            self.flatten_parameters()
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                init.uniform_(weight, -stdv, stdv)
           # self.reset_parameters()


    # def __setattr__(self, attr, value):
    #     if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
    #         # keep self._flat_weights up to date if you do self.weight = ...
    #         self.register_parameter(attr, value)
    #         idx = self._flat_weights_names.index(attr)
    #         self._flat_weights[idx] = value
    #     #super(RNNBase, self).__setattr__(attr, value)

    def flatten_parameters(self) -> None:
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.use_bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, num_weights,
                        self.input_filters, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.proj_size,self.num_layers,self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super(RNNBase, self)._apply(fn)
        if self.built:
            # Resets _flat_weights
            # Note: be v. careful before removing this, as 3rd party device types
            # likely rely on this behavior to properly .to() modules like LSTM.
            self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
            # Flattens params (on CUDA)
            self.flatten_parameters()

        return ret

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_filters != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_filters. Expected {}, got {}'.format(
                    self.input_filters, input.size(-1)))

    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1

        if self.proj_size > 0:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.proj_size)
        else:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)

    def forward(self,
                input: Union[Tensor, PackedSequence],
                hx: Optional[Tensor] = None) -> Tuple[Union[Tensor, PackedSequence], Tensor]:
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            input = cast(Tensor, input)
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            input = cast(Tensor, input)
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        assert hx is not None
        input = cast(Tensor, input)
        self.check_forward_args(input, hx, batch_sizes)
        _impl = _rnn_impls[self.mode]
        if batch_sizes is None:
            result = _impl(input, hx, self._flat_weights, self.bias, self.num_layers,
                           self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _impl(input, batch_sizes, hx, self._flat_weights, self.bias,
                           self.num_layers, self.dropout, self.training, self.bidirectional)

        output: Union[Tensor, PackedSequence]
        output = result[0]
        hidden = result[1]

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    def extra_repr(self) -> str:
        s = '{input_filters}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.use_bias is not True:
            s += ', use_bias={use_bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout_rate != 0:
            s += ', dropout_rate={dropout_rate}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(RNNBase, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']

        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.use_bias:
                    self._all_weights += [weights]
                    self._flat_weights_names.extend(weights)
                else:
                    self._all_weights += [weights[:2]]
                    self._flat_weights_names.extend(weights[:2])
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]


    @property
    def all_weights(self) -> List[Parameter]:
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def _replicate_for_data_parallel(self):
        replica = super(RNNBase, self)._replicate_for_data_parallel()
        # Need to copy these caches, otherwise the replica will share the same
        # flat weights list.
        replica._flat_weights = replica._flat_weights[:]
        replica._flat_weights_names = replica._flat_weights_names[:]
        return replica


class RNN(RNNBase):
    r"""Applies a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_filters: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)`. Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_filters)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          or :func:`torch.nn.utils.rnn.pack_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided. If the RNN is bidirectional,
          num_directions should be 2, else it should be 1.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features (`h_t`) from the last layer of the RNN,
          for each `t`.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

    Shape:
        - Input1: :math:`(L, N, H_{in})` tensor containing input features where
          :math:`H_{in}=\text{input\_size}` and `L` represents a sequence length.
        - Input2: :math:`(S, N, H_{out})` tensor
          containing the initial hidden state for each element in the batch.
          :math:`H_{out}=\text{hidden\_size}`
          Defaults to zero if not provided. where :math:`S=\text{num\_layers} * \text{num\_directions}`
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\text{num\_directions} * \text{hidden\_size}`
        - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_filters)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(RNN, self).__init__(mode, *args, **kwargs)


# XXX: LSTM and GRU implementation is different from RNNBase, this is because:
# 1. we want to support nn.LSTM and nn.GRU in TorchScript and TorchScript in
#    its current state could not support the python Union Type or Any Type
# 2. TorchScript static typing does not allow a Function or Callable type in
#    Dict values, so we have to separately call _VF instead of using _rnn_impls
# 3. This is temporary only and in the transition state that we want to make it
#    on time for the release
#
# More discussion details in https://github.com/pytorch/pytorch/pull/23266
#
# TODO: remove the overriding implementations for LSTM and GRU when TorchScript
# support expressing these two modules generally.
from torch.nn.modules.rnn import LSTM

class LSTM(RNNBase):
    r"""Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_filters: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_filters)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the cell state for `t = seq_len`.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_filters)` for `k = 0`.
            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def __init__(self, hidden_size,proj_size=0,num_layers:int =2,activation=None,stateful=False,use_bias=False,use_attention=False,attention_size=16,batch_first=False,dropout_rate=0,bidirectional=False,keep_output=False,name=None, **kwargs):
        super(LSTM, self).__init__(mode='LSTM', hidden_size=hidden_size, proj_size=proj_size,
        num_layers=num_layers, stateful = stateful, use_bias=use_bias, batch_first = batch_first,
        dropout_rate= dropout_rate, bidirectional = bidirectional, keep_output =keep_output, in_sequence = True, filter_index =-1, name = name)

        self.use_attention=use_attention
        self.attention_size=attention_size




    def initial_state(self,input) :
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1

        h_zeros=torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device).to(get_device())
        c_zeros= torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device).to(get_device())
        hx = (h_zeros, c_zeros)
        return hx



    def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_forward_args(self, input: Tensor, hidden: Tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                               'Expected hidden[1] size {}, got {}')

    def permute_hidden(self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def attention(self, lstm_output):
        batch_size, sequence_length, channels = int_shape(lstm_output)
        if not hasattr(self, 'w_omega') or self.w_omega is None:
            self.w_omega = Parameter(torch.zeros(channels, self.attention_size).to(get_device()))
            self.u_omega = Parameter(torch.zeros(self.attention_size).to(get_device()))

        output_reshape = reshape(lstm_output, (-1, channels))
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(attn_tanh, reshape(self.u_omega, [-1, 1]))
        exps = reshape(torch.exp(attn_hidden_layer), [-1, sequence_length])
        alphas = exps / reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = reshape(alphas, [-1, sequence_length, 1])
        return lstm_output * alphas_reshape

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, x: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:  # noqa: F811
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self,  x:PackedSequence, hx: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:  # noqa: F811
        pass

    def forward(self, x, hx=None):
        orig_input = x
        is_packed_sequence=isinstance(orig_input, PackedSequence)
        self.flatten_parameters()
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if is_packed_sequence:
            x, batch_sizes, sorted_indices, unsorted_indices = x
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            if not self.batch_first:
                x = x.transpose(1,0)
            batch_sizes = None
            max_batch_size = x.size(0) if self.batch_first else x.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            #if self.hidden_state is None or self.cell_state is None or max_batch_size!=int_shape(self.hidden_state)[1]:
            hx=self.initial_state(x)
            hx = self.permute_hidden(hx, sorted_indices)
        else:
            if not self.stateful:
                hx=self.initial_state(x)
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(x, hx, batch_sizes)

        if not isinstance(x, PackedSequence):
            result = _VF.lstm(x,hx, self._flat_weights, self.use_bias, self.num_layers,
                              self.dropout_rate, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(x, batch_sizes, hx, self._flat_weights, self.use_bias,
                              self.num_layers, self.dropout_rate, self.training, self.bidirectional)


        output = result[0].permute(1, 0, 2) if self.batch_first == False else result[0]
        hidden = result[1:]
        # self.hidden_state=hidden[0]
        # self.cell_state=hidden[1]
        if self.use_attention:
            output = self.attention(output)

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if is_packed_sequence:
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:

            return output, self.permute_hidden(hidden, unsorted_indices)

class LSTMDecoder(Layer):
    def __init__(self, num_chars, embedding_dim, h_size=512, num_layers=2,sequence_length=128,stateful=True, dropout_rate=0.2,bidirectional=False,use_attention=False,attention_size=16,teacher_forcing_ratio=1):
        super().__init__()
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.num_chars = num_chars
        self.embedding_dim=embedding_dim
        self.h_size = h_size
        self.num_layers = num_layers
        self.sequence_length=sequence_length
        self.embedding = Embedding(embedding_dim=256, num_embeddings=num_chars, sparse=False, norm_type=2, add_noise=True, noise_intensity=0.12)
        self.lstm = LSTM(hidden_size=h_size, num_layers=num_layers, stateful=stateful, batch_first=False, dropout_rate=dropout_rate, bidirectional=bidirectional, use_attention=use_attention, attention_size=attention_size)
        self.fc_out =Dense(num_chars,use_bias=False,activation=leaky_relu)
        self.softmax=SoftMax(axis=-1)


    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
                ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:  # noqa: F811
        pass

    def forward(self, *x, **kwargs):  # noqa: F811
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        x,(self.hidden_state, self.cell_state) =unpack_singleton(x)
        B,N,C=int_shape(x)
        outputs =[]
        # input = [batch size,1]


        decoder_input =expand_dims(x[:,-1,:] ,1) # shape: (batch_size, input_size)
        decoder_hidden = (self.hidden_state, self.cell_state)

        # predict recursively
        for t in range(self.sequence_length):
            decoder_output, decoder_hidden =  self.lstm(decoder_input, decoder_hidden)
            outputs.append(self.softmax(self.fc_out (decoder_output.squeeze(1))))
            decoder_input = decoder_output
        return stack(outputs,1)




class GRU(RNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_filters: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_filters)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided. If the RNN is bidirectional,
          num_directions should be 2, else it should be 1.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features h_t from the last layer of the GRU,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.

          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

    Shape:
        - Input1: :math:`(L, N, H_{in})` tensor containing input features where
          :math:`H_{in}=\text{input\_size}` and `L` represents a sequence length.
        - Input2: :math:`(S, N, H_{out})` tensor
          containing the initial hidden state for each element in the batch.
          :math:`H_{out}=\text{hidden\_size}`
          Defaults to zero if not provided. where :math:`S=\text{num\_layers} * \text{num\_directions}`
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\text{num\_directions} * \text{hidden\_size}`
        - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_filters)` for `k = 0`.
            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__('GRU', *args, **kwargs)
        self.hidden_state = None
        self.stateful=kwargs.get('stateful',False)

    def initial_state(self,input) :
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1

        self.hidden_state = torch.zeros(self.num_layers * num_directions,
                                        max_batch_size, self.hidden_size,
                                        dtype=self.weights[0].dtype, device=self.weights[0].device, requires_grad=False)



    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:  # noqa: F811
        pass

    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(self, input: PackedSequence, hx: Optional[Tensor] = None) -> Tuple[PackedSequence, Tensor]:  # noqa: F811
        pass

    def forward(self,x):  # noqa: F811
        orig_input = x
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = x
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            if self.batch_first == False:
                x = x.permute(1, 0, 2)
            batch_sizes = None
            max_batch_size = x.size(0) if self.batch_first else x.size(1)
            sorted_indices = None
            unsorted_indices = None

        if self.stateful==False or self.hidden_state is None  or max_batch_size!=int_shape(self.hidden_state)[1]:
            self.initial_state(x)

        else:
            self.hidden_state= self.permute_hidden(self.hidden_state, sorted_indices)


        self.check_forward_args(x, self.hidden_state, batch_sizes)
        if batch_sizes is None:
            result = _VF.gru(x, self.hidden_state, self._flat_weights, self.use_bias, self.num_layers,
                             self.dropout_rate, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.gru(x, batch_sizes, self.hidden_state, self._flat_weights, self.use_bias,
                             self.num_layers, self.dropout_rate, self.training, self.bidirectional)
        output = result[0]
        self.hidden_state = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(self.hidden_state, unsorted_indices)
        else:
            if self.batch_first == False:
                x = x.permute(1, 0, 2)
            return output, self.permute_hidden(self.hidden_state, unsorted_indices)