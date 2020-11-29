from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import inspect
import os
import sys
import builtins
import random
import shutil
import string
import sys
import time
import uuid
import json
import numpy as np
from trident.backend.common import to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path,make_dir_if_need
from trident.backend.tensorspec import *
_session = get_session()
_backend = _session.backend
if _backend == 'pytorch':
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *

__all__ = ['Loss']

class Loss(object):
  """Loss base class.
  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
  Example subclass implementation:
  ```python
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
  ```
  When used with `tf.distribute.Strategy`, outside of built-in training loops
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
  'SUM_OVER_BATCH_SIZE' will raise an error.
  Please see this custom training [tutorial](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
  details on this.
  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
  ```python
  with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
            (1. / global_batch_size))
  ```
  """

  def __init__(self, reduction='mean', sample_weight=None,axis=None,name=None):
    """Initializes `Loss` class.
    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op.
    """
    assert  reduction.lower() in ['mean','sum','batch_sum','batch_mean','none']
    self.reduction = reduction.lower()
    self.name = name
    self.sample_weight=sample_weight
    self.axis=axis

  def __call__(self, output: Tensor, target: Tensor,**kwargs):
      result = self.forward(output, target, **kwargs)
      return result



  def forward(self, output: Tensor, target: Tensor,**kwargs):
    """Invokes the `Loss` instance.
    Args:
       output: The predicted values. shape = `[batch_size, d0, .. dN]`
      target: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`

    Returns:
      Loss values with the shape `[batch_size, d0, .. dN-1]`.
    """
    NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self, loss):
      if ndim(loss)==0 or self.reduction == 'none':
          return loss
      if ndim(loss)>=2 and self.reduction == 'batch_sum':
          loss=reshape(loss,(int_shape(loss)[0],-1))
          return loss.mean(1).sum()
      elif ndim(loss)>=2 and self.reduction == 'batch_mean':
          loss = reshape(loss, (int_shape(loss)[0], -1))
          return loss.mean(1).mean()
      elif self.reduction in ('mean', 'batch_mean'):
          return loss.mean()
      elif self.reduction in  ('sum', 'batch_sum'):
          return loss.sum()
      else:
          return loss.mean()

  def _handel_abnormal(self, loss):
      if any_abnormal_number(loss):
          sys.stderr.write('{0} has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.name))
          loss = where(is_nan(loss), zeros_like(loss,requires_grad=True), loss)
      return loss
