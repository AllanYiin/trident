from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Any

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


elif _backend == 'jax':
    import jax
    from trident.backend.jax_ops import *

from collections.abc import Mapping
from trident.optims.field_binding import FunctionObjective, normalize_names

__all__ = ['loss', 'FunctionLoss', 'Loss','_check_logit','_check_logsoftmax_logit','_check_softmax']


PUBLIC_LOSSES = (
    'ActiveContourLoss', 'AdaptiveWingLoss', 'BCELoss', 'BCEWithLogitsLoss',
    'CenterLoss', 'CosineSimilarityLoss', 'CrossEntropyLoss', 'DiceLoss',
    'EdgeLoss', 'ExponentialLoss', 'F1ScoreLoss', 'FocalLoss',
    'GPLoss', 'GeneralizedDiceLoss', 'HardTripletLoss', 'HuberLoss',
    'IoULoss', 'ItakuraSaitoLoss', 'KLDivergenceLoss', 'L1Loss',
    'L2Loss', 'LovaszSoftmax', 'MSELoss', 'MS_SSIMLoss',
    'NLLLoss', 'PerceptionLoss', 'SmoothL1Loss', 'SoftIoULoss',
    'StyleLoss', 'TransformInvariantLoss', 'TripletLoss', 'TripletMarginLoss',
    'TverskyLoss', 'WingLoss',
)


class Loss(object):
    """Base for Trident losses called as loss(output, target).

    Subclasses implement forward(). batch_sum sums over examples and averages
    the remaining coordinates. batch_mean averages all coordinates.
    """

    def __init__(self, reduction='mean', sample_weight=None, axis=None, enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,name=None, **kwargs):
        if not isinstance(reduction, str) or reduction.lower() not in ('mean', 'sum', 'batch_sum', 'batch_mean', 'batchmean', 'none'):
            raise ValueError("unknown loss reduction: {!r}".format(reduction))
        if enable_ohem and not 0 < ohem_thresh <= 1:
            raise ValueError("ohem_thresh must be in (0, 1]")
        self.reduction = reduction.lower()
        self.name = name
        self._built = False
        self.sample_weight = sample_weight
        self.axis = axis
        self.enable_ohem = enable_ohem
        self.ohem_thresh = ohem_thresh
        self._signature = get_signature(self)
        self.input_names=input_names
        self.output_names=output_names
        self.update_signature(input_names,output_names)


    @property
    def built(self):
        return self._built

    def build(self, output,target,**kwargs):
        self._built = True

    def __call__(self, output: Tensor, target: Tensor, **kwargs):
        if _backend == 'pytorch' and isinstance(target, torch.Tensor):
            target = target.to(output.device)
        return self.forward(output, target, **kwargs)

    def _forward_unimplemented(self, output: Any, target: Any, **kwargs) -> None:
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

    def _get_reduction(self, loss):
        if self.reduction == 'none':
            return loss
        if self.reduction == 'batchmean' and len(loss.shape):
            if _backend == 'tensorflow':
                return tf.reduce_sum(loss) / tf.cast(tf.shape(loss)[0], loss.dtype)
            return loss.sum() / loss.shape[0]
        if _backend == 'tensorflow':
            if self.reduction in ('mean', 'batch_mean'):
                return tf.reduce_mean(loss)
            if self.reduction == 'batch_sum' and loss.shape.rank != 0:
                return tf.reduce_mean(tf.reduce_sum(loss, axis=0))
            return tf.reduce_sum(loss)
        if self.reduction in ('mean', 'batch_mean'):
            return loss.mean()
        if self.reduction == 'batch_sum' and loss.ndim > 0:
            return loss.sum(axis=0).mean()
        return loss.sum()

    def _handel_abnormal(self, loss):
        if _backend == 'tensorflow':
            return tf.debugging.check_numerics(loss, "{} produced nonfinite loss".format(self.name))
        if _backend == 'jax':
            # Traced values cannot be inspected in Python. Preserve NaN/Inf for
            # the caller's compiled finite-gradient checks.
            if isinstance(loss, jax.core.Tracer):
                return loss
            finite = jax.numpy.all(jax.numpy.isfinite(loss))
        else:
            finite = torch.isfinite(loss).all()
        if not finite:
            raise FloatingPointError("{} produced nonfinite loss".format(self.name))
        return loss

    def _do_ohem(self, loss):
        if self.enable_ohem and len(loss.shape):
            from trident.optims.loss_kernels import hard_examples
            return hard_examples(loss, self.ohem_thresh)[0]
        return loss

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
        if isinstance(input_names, Mapping):
            self._field_names = normalize_names(input_names, self._signature.inputs)
            input_names = None
        elif input_names is not None:
            self._field_names = {}
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


class FunctionLoss(FunctionObjective, Loss):
    """A Python function registered as a loss with reusable field bindings."""

    forward = FunctionObjective.__call__

    @property
    def trainable_variables(self):
        return getattr(self.__wrapped__, 'trainable_variables', ())

    @property
    def gradient_denominator(self):
        return getattr(self.__wrapped__, 'gradient_denominator', None)


def loss(fn=None, *, input_names=None, name=None):
    def decorate(func):
        return FunctionLoss(func, input_names=input_names, name=name)
    return decorate if fn is None else decorate(fn)


def _class_axis(axis):
    return (1 if _backend == 'pytorch' else -1) if axis is None else axis


def _check_logsoftmax_logit(x: Tensor, axis=1):
    if _backend == 'pytorch':
        bounded = torch.clamp_max(x, 0)
    elif _backend == 'tensorflow':
        bounded = tf.minimum(x, tf.zeros_like(x))
    else:
        bounded = jax.numpy.minimum(x, 0)
    return (reduce_max(x) <= 0) & (reduce_mean(abs(1 - reduce_sum(exp(bounded), axis=_class_axis(axis)))) < .2)


def _check_softmax(x: Tensor, axis=1):
    return ((reduce_min(x) >= 0) & (reduce_max(x) <= 1)
            & (reduce_mean(abs(1 - reduce_sum(x, axis=_class_axis(axis)))) < .2))


def _check_logit(x: Tensor, axis=None):
    return ((reduce_min(x) >= 0) & (reduce_max(x) <= 1)
            & (reduce_mean(abs(1 - reduce_sum(x, axis=_class_axis(axis)))) < .05))
