import math
import numpy as np
import random

def conv_output_length1(input_length, filter_size,  stride,autopad=True, dilation=1):
    if autopad:
        return (input_length - 1) * stride + (filter_size - 1) * dilation + 1
    else:
        return (math.ceil(input_length / stride) - 1) * stride + (filter_size - 1) * dilation + 1



def conv_output_length(input_length, filter_size,  stride,autopad=True, dilation=1):
  """Determines output length of a convolution given input length.

  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full", "causal"
      stride: integer.
      dilation: dilation rate, integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  dilated_filter_size = (filter_size - 1) * dilation + 1
  if autopad:
    output_length = input_length
  else:
    output_length = input_length - dilated_filter_size + 1
  return (output_length + stride - 1) // stride


tt1=conv_output_length1(128,3,2,True,2)
tt2=conv_output_length(128,3,2,True,2)



def conv_input_length(output_length, filter_size,  stride,autopad=True):
  """Determines input length of a convolution given output length.

  Arguments:
      output_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.

  Returns:
      The input length (integer).
  """
  if output_length is None:
    return None

  if autopad:
    pad = filter_size // 2
  else:
    pad = 0
  return (output_length - 1) * stride - 2 * pad + filter_size

def deconv_output_length(input_length,
                         filter_size,
                         padding,
                         output_padding=None,
                         stride=0,
                         dilation=1):
  """Determines output length of a transposed convolution given input length.

  Arguments:
      input_length: Integer.
      filter_size: Integer.
      padding: one of `"same"`, `"valid"`, `"full"`.
      output_padding: Integer, amount of padding along the output dimension. Can
        be set to `None` in which case the output length is inferred.
      stride: Integer.
      dilation: Integer.

  Returns:
      The output length (integer).
  """
  assert padding in {'same', 'valid', 'full'}
  if input_length is None:
    return None

  # Get the dilated kernel size
  filter_size = filter_size + (filter_size - 1) * (dilation - 1)

  # Infer length if output padding is None, else compute the exact length
  if output_padding is None:
    if padding == 'valid':
      length = input_length * stride + max(filter_size - stride, 0)
    elif padding == 'full':
      length = input_length * stride - (stride + filter_size - 2)
    elif padding == 'same':
      length = input_length * stride

  else:
    if padding == 'same':
      pad = filter_size // 2
    elif padding == 'valid':
      pad = 0
    elif padding == 'full':
      pad = filter_size - 1

    length = ((input_length - 1) * stride + filter_size - 2 * pad +
              output_padding)
  return length

