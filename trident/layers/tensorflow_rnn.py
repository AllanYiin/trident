from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers
import warnings
from typing import Optional, Tuple, overload, List
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.ops.ragged import ragged_tensor
__all__ = ['RNNBase','RNN','LSTM']

def _rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None,
        batch_first=True,
        zero_output_for_mask=False):
  """Iterates over the time dimension of a tensor.

  Arguments:
      step_function: RNN step function.
          Args;
              input; Tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; List of tensors.
          Returns;
              output; Tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; List of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: Tensor of temporal data of shape `(samples, time, ...)`
          (at least 3D), or nested tensors, and each of which has shape
          `(samples, time, ...)`.
      initial_states: Tensor with shape `(samples, state_size)`
          (no time dimension), containing the initial values for the states used
          in the step function. In the case that state_size is in a nested
          shape, the shape of initial_states will also follow the nested
          structure.
      go_backwards: Boolean. If True, do the iteration over the time
          dimension in reverse order and return the reversed sequence.
      mask: Binary tensor with shape `(samples, time, 1)`,
          with a zero for every element that is masked.
      constants: List of constant values passed at each step.
      unroll: Whether to unroll the RNN or to use a symbolic `while_loop`.
      input_length: An integer or a 1-D Tensor, depending on whether
          the time dimension is fixed-length or not. In case of variable length
          input, it is used for masking in case there's no mask specified.
      time_major: Boolean. If true, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.
      zero_output_for_mask: Boolean. If True, the output for masked timestep
          will be zeros, whereas in the False case, output from previous
          timestep is returned.

  Returns:
      A tuple, `(last_output, outputs, new_states)`.
          last_output: the latest output of the rnn, of shape `(samples, ...)`
          outputs: tensor with shape `(samples, time, ...)` where each
              entry `outputs[s, t]` is the output of the step function
              at time `t` for sample `s`.
          new_states: list of tensors, latest states returned by
              the step function, of shape `(samples, ...)`.

  Raises:
      ValueError: if input dimension is less than 3.
      ValueError: if `unroll` is `True` but input timestep is not a fixed
      number.
      ValueError: if `mask` is provided (not `None`) but states is not provided
          (`len(states)` == 0).
  """

  def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return array_ops.transpose(input_t, axes)

  if batch_first:
    inputs = nest.map_structure(swap_batch_timestep, inputs)

  flatted_inputs = nest.flatten(inputs)
  time_steps = flatted_inputs[0].shape[0]
  batch = flatted_inputs[0].shape[1]
  time_steps_t = array_ops.shape(flatted_inputs[0])[0]

  for input_ in flatted_inputs:
    input_.shape.with_rank_at_least(3)

  if mask is not None:
    if mask.dtype != dtypes_module.bool:
      mask = math_ops.cast(mask, dtypes_module.bool)
    if len(mask.shape) == 2:
      mask = expand_dims(mask)
    if batch_first:
      mask = swap_batch_timestep(mask)

  if constants is None:
    constants = []

  # tf.where needs its condition tensor to be the same shape as its two
  # result tensors, but in our case the condition (mask) tensor is
  # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
  # So we need to broadcast the mask to match the shape of inputs.
  # That's what the tile call does, it just repeats the mask along its
  # second dimension n times.
  def _expand_mask(mask_t, input_t, fixed_dim=1):
    if nest.is_nested(mask_t):
      raise ValueError('mask_t is expected to be tensor, but got %s' % mask_t)
    if nest.is_nested(input_t):
      raise ValueError('input_t is expected to be tensor, but got %s' % input_t)
    rank_diff = len(input_t.shape) - len(mask_t.shape)
    for _ in range(rank_diff):
      mask_t = array_ops.expand_dims(mask_t, -1)
    multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
    return array_ops.tile(mask_t, multiples)

  if unroll:
    if not time_steps:
      raise ValueError('Unrolling requires a fixed number of timesteps.')
    states = tuple(initial_states)
    successive_states = []
    successive_outputs = []

    # Process the input tensors. The input tensor need to be split on the
    # time_step dim, and reverse if go_backwards is True. In the case of nested
    # input, the input is flattened and then transformed individually.
    # The result of this will be a tuple of lists, each of the item in tuple is
    # list of the tensor with shape (batch, feature)
    def _process_single_input_t(input_t):
      input_t = array_ops.unstack(input_t)  # unstack for time_step dim
      if go_backwards:
        input_t.reverse()
      return input_t

    if nest.is_nested(inputs):
      processed_input = nest.map_structure(_process_single_input_t, inputs)
    else:
      processed_input = (_process_single_input_t(inputs),)

    def _get_input_tensor(time):
      inp = [t_[time] for t_ in processed_input]
      return nest.pack_sequence_as(inputs, inp)

    if mask is not None:
      mask_list = array_ops.unstack(mask)
      if go_backwards:
        mask_list.reverse()

      for i in range(time_steps):
        inp = _get_input_tensor(i)
        mask_t = mask_list[i]
        output, new_states = step_function(inp,
                                           tuple(states) + tuple(constants))
        tiled_mask_t = _expand_mask(mask_t, output)

        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = array_ops.where_v2(tiled_mask_t, output, prev_output)

        flat_states = nest.flatten(states)
        flat_new_states = nest.flatten(new_states)
        tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_states)
        flat_final_states = tuple(
            array_ops.where_v2(m, s, ps)
            for m, s, ps in zip(tiled_mask_t, flat_new_states, flat_states))
        states = nest.pack_sequence_as(states, flat_final_states)

        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

      if zero_output_for_mask:
        last_output = array_ops.where_v2(
            _expand_mask(mask_list[-1], last_output), last_output,
            zeros_like(last_output))
        outputs = array_ops.where_v2(
            _expand_mask(mask, outputs, fixed_dim=2), outputs,
            zeros_like(outputs))

    else:  # mask is None
      for i in range(time_steps):
        inp = _get_input_tensor(i)
        output, states = step_function(inp, tuple(states) + tuple(constants))
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

  else:  # Unroll == False
    states = tuple(initial_states)

    # Create input tensor array, if the inputs is nested tensors, then it will
    # be flattened first, and tensor array will be created one per flattened
    # tensor.
    input_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='input_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))
    input_ta = tuple(
        ta.unstack(input_) if not go_backwards else ta
        .unstack(reverse(input_, 0))
        for ta, input_ in zip(input_ta, flatted_inputs))

    # Get the time(0) input and compute the output for that, the output will be
    # used to determine the dtype of output tensor array. Don't read from
    # input_ta due to TensorArray clear_after_read default to True.
    input_time_zero = nest.pack_sequence_as(inputs, [inp[0] for inp in flatted_inputs])
    # output_time_zero is used to determine the cell output shape and its dtype.
    output_time_zero, states = step_function(input_time_zero, tuple(initial_states) + tuple(constants))
    output_lists=[output_time_zero]
    # the value is discarded.
    #start
    for seq in range(1,time_steps):
        input_timet= nest.pack_sequence_as(inputs, [inp[seq] for inp in flatted_inputs])
        output_timet, states = step_function(input_timet, tuple(states) + tuple(constants))
        output_lists.append(output_timet)

    outputs=stack(output_lists,axis=0)
  return outputs, states[0],states[1]


def standard_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias,
                  mask, batch_first, go_backwards, sequence_lengths,
                  zero_output_for_mask):
  """LSTM with standard kernel implementation.

  This implementation can be run on all types for hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Note that the first half of the bias tensor should be ignored by this impl.
  The CuDNN impl need an extra set of input gate bias. In order to make the both
  function take same shape of parameter, that extra set of bias is also feed
  here.

  Args:
    inputs: input tensor of LSTM layer.
    init_h: initial state tensor for the cell output.
    init_c: initial state tensor for the cell hidden state.
    kernel: weights for cell kernel.
    recurrent_kernel: weights for cell recurrent kernel.
    bias: weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
    batch_first: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    state_1: the cell hidden state, which has same shape as init_c.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = int_shape(inputs)
  timesteps = input_shape[0] if not batch_first else input_shape[1]

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    # if ndim(cell_inputs)==2:
    #     cell_inputs=expand_dims(cell_inputs,0)
    hidden_list=array_ops.split(cell_states[0], int_shape(cell_states[0])[0], axis=0)
    cell_list=array_ops.split(cell_states[1], int_shape(cell_states[1])[0], axis=0)

    # if go_backwards:
    #     reverse_cell_inputs=reverse(cell_inputs,axis=0 if not batch_first else 1)



    for layer in range(len(kernel)//(2 if go_backwards else 1)):
        all_cell_inputs = [cell_inputs, reverse(cell_inputs, axis=0 if not batch_first else 1)] if go_backwards else [cell_inputs]
        for direction in range(2 if go_backwards else 1):
            cell_inputs=all_cell_inputs[direction]
            layer_idx=layer*(2 if go_backwards else 1)+direction

            h_tm1 =hidden_list[layer_idx][0].detach()  # previous memory state
            c_tm1 = cell_list[layer_idx][0].detach()   # previous carry state
            z = tf.matmul(cell_inputs, kernel[layer_idx],transpose_b=True)
            z=z+tf.matmul(h_tm1, recurrent_kernel[layer],transpose_b=True)
            # if self.use_bias:
            #     z +=bias

            z0, z1, z2, z3 = array_ops.split(z, 4, axis=-1)

            input_gate = sigmoid(z0)
            forget_gate = sigmoid(z1)
            cell_state = forget_gate * c_tm1 + input_gate * tanh(z2)
            output_gate = sigmoid(z3)
            h=output_gate * tanh(cell_state)
            hidden_list[layer_idx]=h
            cell_list[layer_idx]=cell_state
            all_cell_inputs[direction]=h

        if go_backwards:
            cell_inputs=concate(all_cell_inputs,axis=-1)
        else:
            cell_inputs=all_cell_inputs[0]
    return  cell_inputs , [stack(hidden_list,0), stack(cell_list,0)]

  outputs, hidden_state,cell_state= _rnn(
      step,
      inputs, [init_h, init_c],
      constants=None,
      unroll=False,
      batch_first=batch_first,
      mask=mask,
      go_backwards=go_backwards,
      input_length=(sequence_lengths
                    if sequence_lengths is not None else timesteps),
      zero_output_for_mask=zero_output_for_mask)
  return (outputs, hidden_state,cell_state)

#
# def gpu_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask,
#              time_major, go_backwards, sequence_lengths):
#   """LSTM with either CuDNN or ROCm implementation which is only available for GPU.
#
#   Note that currently only right padded data is supported, or the result will be
#   polluted by the unmasked data which should be filtered.
#
#   Args:
#     inputs: Input tensor of LSTM layer.
#     init_h: Initial state tensor for the cell output.
#     init_c: Initial state tensor for the cell hidden state.
#     kernel: Weights for cell kernel.
#     recurrent_kernel: Weights for cell recurrent kernel.
#     bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
#       is used in this case.
#     mask: Boolean tensor for mask out the steps within sequence.
#     time_major: Boolean, whether the inputs are in the format of [time, batch,
#       feature] or [batch, time, feature].
#     go_backwards: Boolean (default False). If True, process the input sequence
#       backwards and return the reversed sequence.
#     sequence_lengths: The lengths of all sequences coming from a variable length
#       input, such as ragged tensors. If the input has a fixed timestep size,
#       this should be None.
#
#   Returns:
#     last_output: Output tensor for the last timestep, which has shape
#       [batch, units].
#     outputs: Output tensor for all timesteps, which has shape
#       [batch, time, units].
#     state_0: The cell output, which has same shape as init_h.
#     state_1: The cell hidden state, which has same shape as init_c.
#     runtime: Constant string tensor which indicate real runtime hardware. This
#       value is for testing purpose and should not be used by user.
#   """
#   if not time_major and mask is None:
#     inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
#     seq_axis, batch_axis = (0, 1)
#   else:
#     seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
#   # For init_h and init_c, cuDNN expects one more dim of num_layers before or
#   # after batch dim for time major or batch major inputs respectively
#   init_h = array_ops.expand_dims(init_h, axis=seq_axis)
#   init_c = array_ops.expand_dims(init_c, axis=seq_axis)
#
#   weights = array_ops.split(kernel, 4, axis=1)
#   weights += array_ops.split(recurrent_kernel, 4, axis=1)
#   # CuDNN has an extra set of bias for inputs, we disable them (setting to 0),
#   # so that mathematically it is same as the canonical LSTM implementation.
#   full_bias = array_ops.concat((array_ops.zeros_like(bias), bias), 0)
#
#   if build_info.build_info['is_rocm_build']:
#     # ROCm MIOpen's weight sequence for LSTM is different from both canonical
#     # and Cudnn format
#     # MIOpen: [i, f, o, c] Cudnn/Canonical: [i, f, c, o]
#     # i is input gate weights.
#     # f is forget gate weights.
#     # o is output gate weights.
#     # c is cell gate weights.
#     weights = [weights[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
#     # full_bias is a tensor of shape (8*n,)
#     full_bias = array_ops.split(full_bias, 8, axis=0)
#     full_bias = [full_bias[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
#
#   params = _canonical_to_params(
#       weights=weights,
#       biases=array_ops.split(full_bias, 8),
#       shape=constant_op.constant([-1]),
#       transpose_weights=True)
#
#   if mask is not None:
#     sequence_lengths = calculate_sequence_by_mask(mask, time_major)
#
#   if sequence_lengths is not None:
#     if go_backwards:
#       # Three reversals are required. E.g.,
#       # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
#       # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
#       # output_from_cudnn = [6, 5, 4, 0, 0]
#       # expected_output = [0, 0, 6, 5 ,4]
#       inputs = array_ops.reverse_sequence_v2(
#           inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
#     outputs, h, c, _, _ = gen_cudnn_rnn_ops.cudnn_rnnv3(
#         inputs,
#         input_h=init_h,
#         input_c=init_c,
#         params=params,
#         is_training=True,
#         rnn_mode='lstm',
#         sequence_lengths=sequence_lengths,
#         time_major=time_major)
#     if go_backwards:
#       outputs = array_ops.reverse_sequence_v2(
#           outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
#       outputs = array_ops.reverse(outputs, axis=[seq_axis])
#   else:
#     # # Fill the array with shape [batch] with value of max timesteps.
#     # sequence_length = array_ops.fill([array_ops.shape(inputs)[1]],
#     #                                  array_ops.shape(inputs)[0])
#     if go_backwards:
#       # Reverse axis 0 since the input is already convert to time major.
#       inputs = array_ops.reverse(inputs, axis=[0])
#     outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
#         inputs, input_h=init_h, input_c=init_c, params=params, is_training=True,
#         rnn_mode='lstm')
#
#   last_output = outputs[-1]
#   if not time_major and mask is None:
#     outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
#   h = array_ops.squeeze(h, axis=seq_axis)
#   c = array_ops.squeeze(c, axis=seq_axis)
#
#   # In the case of variable length input, the cudnn kernel will fill zeros for
#   # the output, whereas the default keras behavior is to bring over the previous
#   # output for t-1, so that in the return_sequence=False case, user can quickly
#   # get the final effect output instead just 0s at the last timestep.
#   # In order to mimic the default keras behavior, we copy the final h state as
#   # the last_output, since it is numerically same as the output.
#   if mask is not None:
#     last_output = h
#   return last_output, outputs, h, c

_rnn_impls = {
    'RNN_TANH': standard_lstm,
    'RNN_RELU': standard_lstm,
}

from trident.backend.common import TensorShape, get_device
from trident.backend.tensorflow_ops import *
from trident.backend.tensorflow_backend import *

#[batch, timesteps, feature]
def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)





class RNNBase(Layer):
    mode: str
    input_filters: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout_rate: float
    bidirectional: bool
    def __init__(self, mode: str, hidden_size: int,
                 num_layers: int = 1,stateful=False, use_bias: bool = True, batch_first: bool = False,
                 dropout_rate: float = 0., bidirectional: bool = False,name=None) -> None:
        super(RNNBase, self).__init__(name=name)
        self.in_sequence=True
        self.mode = mode
        self.hidden_size = hidden_size
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
            self.kernals=[]
            self.recurrent_kernels=[]
            for layer in range(self.num_layers):
                for direction in range(self.num_directions):
                    layer_input_size = input_shape[-1] if layer == 0 else self.hidden_size*self.num_directions

                    w_ih = Parameter(random_normal((self.gate_size,layer_input_size)).to(get_device()),name='weight_ih_l{0}{1}'.format(layer,  '_reverse' if direction == 1 else ''))
                    w_hh = Parameter(random_normal((self.gate_size,self.hidden_size)).to(get_device()),name='weight_hh_l{0}{1}'.format(layer,  '_reverse' if direction == 1 else ''))
                    b_ih = Parameter(random_normal((self.gate_size)).to(get_device()),name='bias_ih_l{0}{1}'.format(layer,  '_reverse' if direction == 1 else ''))
                    # Second bias vector included for CuDNN compatibility. Only one
                    # bias vector is needed in standard definition.
                    b_hh = Parameter(random_normal(self.gate_size).to(get_device()),name='bias_hh_l{0}{1}'.format(layer,  '_reverse' if direction == 1 else ''))
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
                        if 'weight_ih' in name:
                            self.kernals.append(param)
                        elif 'weight_hh' in name:
                            self.recurrent_kernels.append(param)

                    self._flat_weights_names.extend(param_names)
                    self._all_weights.append(param_names)

            self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
            self.flatten_parameters()
            self.reset_parameters()




    def flatten_parameters(self) -> None:
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        pass
        # # Short-circuits if _flat_weights is only partially instantiated
        # if len(self._flat_weights) != len(self._flat_weights_names):
        #     return
        #
        # for w in self._flat_weights:
        #     if not isinstance(w, Tensor):
        #         return
        # # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # # or the tensors in _flat_weights are of different dtypes
        #
        # first_fw = self._flat_weights[0]
        # dtype = first_fw.dtype
        # for fw in self._flat_weights:
        #     if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
        #             not fw.data.is_cuda ):
        #         return
        #
        # # If any parameters alias, we fall back to the slower, copying code path. This is
        # # a sufficient check, because overlapping parameter buffers that don't completely
        # # alias would break the assumptions of the uniqueness check in
        # # Module.named_parameters().
        # unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        # if len(unique_data_ptrs) != len(self._flat_weights):
        #     return
        #
        # with torch.cuda.device_of(first_fw):
        #     import torch.backends.cudnn.rnn as rnn
        #
        #     # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
        #     # an inplace operation on self._flat_weights
        #     with torch.no_grad():
        #         if torch._use_cudnn_rnn_flatten_weight():
        #             torch._cudnn_rnn_flatten_weight(
        #                 self._flat_weights, (4 if self.use_bias else 2),
        #                 self.input_filters, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
        #                 self.batch_first, bool(self.bidirectional))

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
            weight.assign(random_uniform_like(weight,-stdv,stdv))

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if len(int_shape(input)) != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, len(int_shape(input))))
        if self.input_filters != int_shape(input)[-1]:
            raise RuntimeError(
                'int_shape(input)[-1] must be equal to input_filters. Expected {}, got {}'.format(
                    self.input_filters, int_shape(input)[-1]))

    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = batch_sizes[0]
            mini_batch = int(mini_batch)
        else:
            mini_batch = int_shape(input)[0] if self.batch_first else int_shape(input)[1]
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,  mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if int_shape(hx) != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)


    def forward(self, input: Tensor, hx: Optional[Tensor] = None, _rnn_impls=None) -> Tuple[Tensor, Tensor]:
        def step(inputs, states):
            return self.cell(inputs, states)
        batch_sizes = None
        max_batch_size = int_shape(input)[0] if self.batch_first else int_shape(input)[1]
        sorted_indices = None
        unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = zeros((self.num_layers * num_directions,max_batch_size, self.hidden_size), dtype=input.dtype).to(input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        _impl = _rnn_impls[self.mode]
        if batch_sizes is None:
            result = _impl(input, hx, self._flat_weights, self.use_bias, self.num_layers,
                           self.dropout_rate, self.training, self.bidirectional, self.batch_first)
        else:
            result = _impl(input, batch_sizes, hx, self._flat_weights, self.use_bias,
                           self.num_layers, self.dropout_rate, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1]


        return output, self.permute_hidden(hidden, unsorted_indices)

    def extra_repr(self) -> str:
        s = '{input_filters}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.use_bias is not True:
            s += ', bias={bias}'
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


    """

    def __init__(self, hidden_size,num_layers:int =2,activation=None,stateful=False,use_bias=False,batch_first=False,dropout_rate=0,bidirectional=False,name=None,  **kwargs):
        super(LSTM, self).__init__(mode='LSTM', hidden_size=hidden_size,
        num_layers= num_layers, stateful = stateful, use_bias= use_bias, batch_first= batch_first,
        dropout_rate= dropout_rate, bidirectional=bidirectional, name = name )

        self.filter_index = -1
        self.hidden_state=None
        self.cell_state=None

    def initial_state(self,input) :
        max_batch_size = int_shape(input)[0] if self.batch_first else int_shape(input)[1]
        num_directions = 2 if self.bidirectional else 1

        self.hidden_state= zeros((self.num_layers * num_directions, max_batch_size, self.hidden_size), dtype=self.weights[0].dtype, requires_grad=False).to(self.weights[0].device)
        self.cell_state =  zeros((self.num_layers * num_directions, max_batch_size, self.hidden_size), dtype=self.weights[0].dtype, requires_grad=False).to(self.weights[0].device)

    def clear_state(self):
        self.hidden_state= zeros_like(self.hidden_state,dtype=self.weights[0].dtype, requires_grad=False ).to(self.weights[0].device)
        self.cell_state= zeros_like(self.cell_state,dtype=self.weights[0].dtype, requires_grad=False).to(self.weights[0].device)

    def check_forward_args(self, input: Tensor, hidden: Tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(self.hidden_state, expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(self.cell_state, expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    def permute_hidden(self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return apply_permutation(self.hidden_state, permutation), apply_permutation(self.cell_state, permutation)

    def lstm_forward(self,x, init_h, init_c, mask=None , sequence_lengths=None, zero_output_for_mask=None):
        def step(x, hidden_state, cell_state,weight_ih, weight_hh,bias_ih,bias_hh,is_goback=False):
            # if is_goback:
            #     x=reverse(x,axis=1 if self.batch_first else 0)
            h_tm1 = hidden_state # previous memory state
            c_tm1 = cell_state# previous carry state
            proj = dot(x, weight_ih)
            if self.use_bias:
                proj += bias_ih
            proj += dot(h_tm1, weight_hh)
            if self.use_bias:
                proj += bias_hh

            proj0, proj1, proj2, proj3 = split(proj, 4, axis=-1)

            input_t = sigmoid(proj0)
            forget_t = sigmoid(proj1)
            cell_state= forget_t * c_tm1 + input_t * tanh(proj2)
            output_t = sigmoid(proj3)
            hidden_state = output_t * tanh(cell_state)
            return hidden_state, [hidden_state, cell_state]

        self.cell_state=init_c
        self.hidden_state=init_h
        for layer in range(self.num_layers):
            output_list=[]
            hidden_list=[]
            cell_list=[]
            for direction in range(self.num_directions):
                weight_ih=self._parameters['weight_ih_l{}{}'.format(layer,'_reverse' if direction == 1 else '')]
                weight_hh = self._parameters['weight_hh_l{}{}'.format(layer, '_reverse' if direction == 1 else '')]

                bias_ih = 0
                bias_hh = 0
                if self.use_bias:
                    bias_ih = self._parameters['bias_ih_l{}{}'.format(layer, '_reverse' if direction == 1 else '')]
                    bias_hh = self._parameters['bias_hh_l{}{}'.format(layer, '_reverse' if direction == 1 else '')]
                result=step(x,  self.hidden_state, self.cell_state, self.kernals, self.recurrent_kernels, bias_ih, bias_hh,is_goback=direction == 1)
                x = result[0]
                self.hidden_state = result[1:][0]
                self.cell_state = result[1:][1]
                output_list.append(x)
                hidden_list.append(self.hidden_state)
        #state num_layers * num_directions, batch, hidden_size
        return x, [self.hidden_state, self.cell_state]

    def forward(self, x, **kwargs):  # noqa: F811
        orig_input = x
        self.flatten_parameters()
        # xxx: isinstance check needs to be in conditional for TorchScript to compile

        if not self.batch_first:
            x = x.transpose([1,0,2])
        shp = int_shape(x)
        batch_sizes =None
        max_batch_size = shp[0] if self.batch_first else shp[1]
        sorted_indices = None
        unsorted_indices = None


        if self.hidden_state is None or self.cell_state is None or max_batch_size!=int_shape(self.hidden_state)[1]:
            self.initial_state(x)
        else:
            if not self.stateful:
                self.clear_state()
            self.hidden_state, self.cell_state = self.permute_hidden((self.hidden_state, self.cell_state), sorted_indices)

        self.check_forward_args(x, (self.hidden_state, self.cell_state), batch_sizes)
        if isinstance(x, ragged_tensor.RaggedTensor):
            return x.to_tensor()

        result =standard_lstm(x, init_h=self.hidden_state, init_c= self.cell_state, kernel= self.kernals, recurrent_kernel=self.recurrent_kernels, bias=self.use_bias,
                      mask=None, batch_first=self.batch_first, go_backwards=self.bidirectional, sequence_lengths=None,
                      zero_output_for_mask=None)


        output = result[0].permute([1, 0, 2]) if self.batch_first == False else result[0]
        #hidden = result[1:]
        self.hidden_state=result[1:][0].detach()
        self.cell_state=result[1:][1].detach()

        self.permute_hidden((self.hidden_state, self.cell_state), unsorted_indices)
        return output








# class GRU(RNNBase):
#     r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
#
#
#     For each element in the input sequence, each layer computes the following
#     function:
#
#     .. math::
#         \begin{array}{ll}
#             r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
#             z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
#             n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
#             h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
#         \end{array}
#
#     where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
#     at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
#     at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
#     :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
#     :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.
#
#     In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
#     (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
#     dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
#     variable which is :math:`0` with probability :attr:`dropout`.
#
#     Args:
#         input_filters: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two GRUs together to form a `stacked GRU`,
#             with the second GRU taking in outputs of the first GRU and
#             computing the final results. Default: 1
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             GRU layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``
#
#     Inputs: input, h_0
#         - **input** of shape `(seq_len, batch, input_filters)`: tensor containing the features
#           of the input sequence. The input can also be a packed variable length
#           sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
#           for details.
#         - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#           containing the initial hidden state for each element in the batch.
#           Defaults to zero if not provided. If the RNN is bidirectional,
#           num_directions should be 2, else it should be 1.
#
#     Outputs: output, h_n
#         - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
#           containing the output features h_t from the last layer of the GRU,
#           for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
#           given as the input, the output will also be a packed sequence.
#           For the unpacked case, the directions can be separated
#           using ``output.view(seq_len, batch, num_directions, hidden_size)``,
#           with forward and backward being direction `0` and `1` respectively.
#
#           Similarly, the directions can be separated in the packed case.
#         - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#           containing the hidden state for `t = seq_len`
#
#           Like *output*, the layers can be separated using
#           ``h_n.view(num_layers, num_directions, batch, hidden_size)``.
#
#     Shape:
#         - Input1: :math:`(L, N, H_{in})` tensor containing input features where
#           :math:`H_{in}=\text{input\_size}` and `L` represents a sequence length.
#         - Input2: :math:`(S, N, H_{out})` tensor
#           containing the initial hidden state for each element in the batch.
#           :math:`H_{out}=\text{hidden\_size}`
#           Defaults to zero if not provided. where :math:`S=\text{num\_layers} * \text{num\_directions}`
#           If the RNN is bidirectional, num_directions should be 2, else it should be 1.
#         - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\text{num\_directions} * \text{hidden\_size}`
#         - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
#           for each element in the batch
#
#     Attributes:
#         weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
#             (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_filters)` for `k = 0`.
#             Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
#         weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
#             (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
#         bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
#             (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
#         bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
#             (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`
#
#     .. note::
#         All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
#         where :math:`k = \frac{1}{\text{hidden\_size}}`
#
#     .. include:: ../cudnn_persistent_rnn.rst
#
#     Examples::
#
#         >>> rnn = nn.GRU(10, 20, 2)
#         >>> input = torch.randn(5, 3, 10)
#         >>> h0 = torch.randn(2, 3, 20)
#         >>> output, hn = rnn(input, h0)
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(GRU, self).__init__('GRU', *args, **kwargs)
#         self.hidden_state = None
#         self.stateful=kwargs.get('stateful',False)
#
#     def initial_state(self,input) :
#         max_batch_size = int_shape(input)[0] if self.batch_first else int_shape(input)[1]
#         num_directions = 2 if self.bidirectional else 1
#
#         self.hidden_state = zeros((self.num_layers * num_directions,max_batch_size, self.hidden_size),
#                                         dtype=self.weights[0].dtype).to(self.weights[0].device)
#
#
#
#
#
#     def forward(self, x, **kwargs):  # noqa: F811
#         orig_input = x
#         # xxx: isinstance check needs to be in conditional for TorchScript to compile
#
#         if self.batch_first == False:
#             x = x.permute(1, 0, 2)
#         batch_sizes = None
#         max_batch_size = x.size(0) if self.batch_first else x.size(1)
#         sorted_indices = None
#         unsorted_indices = None
#
#         if self.stateful==False or self.hidden_state is None  or max_batch_size!=int_shape(self.hidden_state)[1]:
#             self.initial_state(x)
#
#         else:
#             self.hidden_state= self.permute_hidden(self.hidden_state, sorted_indices)
#
#
#         self.check_forward_args(x, self.hidden_state, batch_sizes)
#         #if batch_sizes is None:
#         result =[]#gru(x, self.hidden_state, self._flat_weights, self.use_bias, self.num_layers,
#                          self.dropout_rate, self.training, self.bidirectional, self.batch_first)
#         # else:
#         #     result = _VF.gru(x, batch_sizes, self.hidden_state, self._flat_weights, self.use_bias,
#         #                      self.num_layers, self.dropout_rate, self.training, self.bidirectional)
#         output = result[0]
#         self.hidden_state = result[1].detach()
#
#         # xxx: isinstance check needs to be in conditional for TorchScript to compile
#
#         if self.batch_first == False:
#             x = x.permute(1, 0, 2)
#         return output, self.permute_hidden(self.hidden_state, unsorted_indices)
