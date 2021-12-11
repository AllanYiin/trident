from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from typing import Callable, Any

import numpy as np
import torch

from trident.backend.common import get_session, OrderedDict
from trident.backend.tensorspec import *

_session = get_session()
_backend = _session.backend
if _backend == 'pytorch':
    import torch
    from trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    import tensorflow as tf
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

    def __init__(self, reduction='mean', sample_weight=None, axis=None, enable_ohem=False, ohem_ratio=3.5, input_names=None, output_names=None,name=None, **kwargs):
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
        assert reduction.lower() in ['mean', 'sum', 'batch_sum', 'batch_mean', 'none']
        self.reduction = reduction.lower()
        self.name = name
        self.sample_weight = sample_weight
        self.axis = axis
        self.enable_ohem = enable_ohem
        self.ohem_ratio = ohem_ratio
        self._signature = get_signature(self)
        self.input_names=input_names
        self.output_names=output_names
        self.update_signature(input_names,output_names)

    def __call__(self, output: Tensor, target: Tensor, **kwargs):
        target.to(output.device)
        result = self.forward(output, target, **kwargs)
        return result

    def _forward_unimplemented(self, *input: Any) -> None:
        r"""Defines the computation performed at every call.

        Should be overridden by all subclasses.

        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        raise NotImplementedError

    forward: Callable[..., Any] = _forward_unimplemented

    # def forward(self, output: Tensor, target: Tensor,**kwargs):
    #   """Invokes the `Loss` instance.
    #   Args:
    #      output: The predicted values. shape = `[batch_size, d0, .. dN]`
    #     target: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
    #       sparse loss functions such as sparse categorical crossentropy where
    #       shape = `[batch_size, d0, .. dN-1]`
    #
    #   Returns:
    #     Loss values with the shape `[batch_size, d0, .. dN-1]`.
    #   """
    #   NotImplementedError('Must be implemented in subclasses.')

    def _get_reduction(self, loss):
        reduction_axis = list(range(ndim(loss)))
        if ndim(loss) == 0 or self.reduction == 'none':
            return loss
        if ndim(loss) >= 2 and self.reduction == 'batch_sum':
            loss = reshape(loss, (int_shape(loss)[0], -1))
            return loss.sum(0).mean()
        elif ndim(loss) >= 2 and self.reduction == 'batch_mean':
            loss = reshape(loss, (int_shape(loss)[0], -1))
            return loss.mean(0).mean()
        elif self.reduction in ('mean', 'batch_mean'):
            return loss.mean()
        elif self.reduction in ('sum', 'batch_sum'):
            return loss.sum()
        else:
            return loss.mean()

    def _handel_abnormal(self, loss):
        if any_abnormal_number(loss):
            sys.stderr.write('{0} has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.name))
            loss = where(is_nan(loss), zeros_like(loss, requires_grad=True), loss)
        return loss

    def _do_ohem(self, output: Tensor, target: Tensor):
        pass

    @property
    def signature(self):
        if self._signature is None:
            self._signature = get_signature(self)
        return self._signature

    @signature.setter
    def signature(self,value):
        self._signature=value

    def update_signature(self,input_names,output_names):
        if self._signature is None:
            self._signature = get_signature(self)
        if isinstance(input_names,str):
            input_names=[input_names]
        if isinstance(output_names,str):
            output_names=[output_names]
        if input_names is not None and  len(input_names)==len(self._signature.inputs):

           new_input=OrderedDict()
           for i in range(len(input_names)):
               new_input[input_names[i]]=self._signature.inputs.value_list[i]
           print('update inputs {0}=>{1} '.format(self._signature.inputs.key_list,new_input.key_list))
           self._signature.inputs=new_input

        if output_names is not None and len(output_names)==len(self._signature.outputs):
           new_output=OrderedDict()
           for i in range(len(output_names)):
               new_output[output_names[i]]=self._signature.outputs.value_list[i]
           print('update outputs {0}=>{1} '.format(self._signature.outputs.key_list, new_output.key_list))
           self._signature.outputs=new_output


def _check_logsoftmax_logit(x:Tensor,axis=1):
    if isinstance(x, np.ndarray):
        if _backend == 'pytorch':
            if axis is None:
                axis = 1
        elif _backend == 'tensorflow':
            if axis is None:
                axis = -1
        if reduce_max(x) <= 0:
            output_exp = exp(x)
            return abs(1-reduce_mean(output_exp.sum(axis=axis)))<0.05
        return False
    elif _backend == 'pytorch':
        with torch.no_grad():
            if reduce_max(x)<=0:
                output_exp=exp(x)
                return abs(1-reduce_mean(output_exp.sum(axis=axis)))<0.05
            return False
    elif _backend == 'tensorflow':
        if reduce_max(x)<=0:
            output_exp=exp(x).copy().detach()
            return abs(1-reduce_mean(output_exp.sum(axis=axis)))<0.05
        return False

def _check_logit(x:Tensor,axis=None):
    if isinstance(x,np.ndarray):
        if _backend == 'pytorch':
            if axis is None:
                axis = 1
        elif _backend == 'tensorflow':
            if axis is None:
                axis = -1
        if reduce_max(x) <= 0:
            return abs(1 - reduce_mean(x.sum(axis=axis))) < 0.05
        return False
    elif _backend == 'pytorch':
        if axis is None:
            axis=1
        with torch.no_grad():
            if reduce_max(x)<=0:
                return abs(1-reduce_mean(x.sum(axis=axis)))<0.05
            return False
    elif _backend == 'tensorflow':
        if axis is None:
            axis = -1
        if reduce_max(x)<=0:
            return abs(1-reduce_mean(x.sum(axis=axis)))<0.05
        return False

