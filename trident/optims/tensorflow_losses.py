from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
import math
import numbers
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.trackable.autotrackable import AutoTrackable
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tqdm import tqdm

from trident import context
from trident.backend import dtype as Dtype
from trident.backend.common import camel2snake, get_class, epsilon, PrintException, to_list, OrderedDict
from trident.backend.tensorspec import *
from trident.backend.tensorflow_backend import *
from trident.backend.tensorflow_ops import *
from trident.data.dataset import *

from trident.optims.loss_weights import dataset_class_weights
from trident.optims import loss_kernels as portable
from trident.optims.loss_kernels import (
    ActiveContourLoss, BCEWithLogitsLoss, CenterLoss, CosineSimilarityLoss, ExponentialLoss, GPLoss,
    HardTripletLoss, HuberLoss, ItakuraSaitoLoss, KLDivergenceLoss, LovaszSoftmax, MS_SSIMLoss,
    StyleLoss, TransformInvariantLoss, TripletLoss, TripletMarginLoss, TverskyLoss,
)
from trident.optims.losses import PUBLIC_LOSSES, Loss, _check_logsoftmax_logit,_check_logit

# def cosine_similarity(target, output):
#     assert target.ndim == 2
#     assert output.ndim == 2
#     target = l2_normalize(target, axis=1)
#     output = l2_normalize(output, axis=1)
#     return T.sum(target * output, axis=1, keepdims=False)
# def cosine_ranking_loss(target, output):
#     q = output[: ,:args.hidden_size]
#     a_correct = output[: ,args.hidden_size: 2 *args.hidden_size]
#     a_incorrect = output[: , 2 *args.hidden_size: 3 *args.hidden_size]
#
#     return mean \
#         (T.maximum(0., args.margin - cosine_similarity(q, a_correct) + cosine_similarity(q, a_incorrect)) - target
#             [0 ] *0, axis=-1)


__all__ = ['get_loss', '_ClassificationLoss', *PUBLIC_LOSSES]


class _ClassificationLoss(Loss, AutoTrackable):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name=None, **kwargs):
        """

        Args:
            axis (int): the position where the classes is.
            sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
            number of classes.
            from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
            ignore_index (int or list of int):
            cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
            less than 1..
            is_target_onehot (bool): Is the target tensor in onehot format?
            label_smooth (bool): Should use label smoothing?
            reduction (string): the method to aggrgate loss. None means no need to aggregate, 'mean' means average loss,
                'sum' means the summation of losses,'batch_mean' means average loss cross the batch axis then
                summation them.

        Attributes:
            need_target_onehot (bool): If True, means the before loss calculation , need to transform target as one-hot format, ex. label-smooth, default is False.
            is_multiselection (bool): If True, means the classification model is multi-selection, so cannot use  any softmax process, use sigmoid and binary_crosss_entropy
            insteaded.
            is_target_onehot (bool):  If True, means we have confirmed (not just declare) the target is transformed as  one-hot format
            reduction(str): The aggregation function for loss, available options are 'sum', 'mean 'and 'batch_mean', default is 'mean'
            axis (None or int): The axis we according with for loss calculation. Default is 1.
            from_logits (bool):If True, means  the sum of all probability will equal 1.
            is_logsoftmax (bool):If True, means model  use SoftMax as last layer or use any equivalent calculation.
            sample_weight(1D tensor):The loss weight for all classes.
            ignore_index(int , list, tuple): The classes we want to ignore in the loss calculation.
            cutoff(float): Means the decision boundary in this classification model, default=0.5.
            num_classes(int):number of  all the classes.
            label_smooth (bool):If True, mean we will apply label-smoothing in loss calculation.

        """
        super(_ClassificationLoss, self).__init__(reduction=reduction, sample_weight=sample_weight, axis=axis,
                                                  enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                                  input_names=input_names, output_names=output_names, name=name)
        self._set_name_scope()
        self._built = False
        self.label_statistics = None
        self.need_target_onehot = True
        self.is_multiselection = False
        self.is_target_onehot = False
        self.from_logits = from_logits
        self.is_logsoftmax = None
        self.ignore_index = [ignore_index] if not isinstance(ignore_index, (list, tuple)) else ignore_index
        self.ignore_index_weight = None

        if self.sample_weight is not None and auto_balance == True:
            ctx.print(magenta_color('auto_balance cannot be used with sample_weight together  at the same time'))
            self.auto_balance = None
        else:
            self.auto_balance = auto_balance

        if cutoff is not None and not 0 < cutoff < 1:
            raise ValueError('cutoff should between 0 and 1')
        self.cutoff = cutoff
        self.num_classes = None
        self.label_smooth = label_smooth
        self.reduction = reduction

        # initilize weight

    def _set_name_scope(self):
        """Creates a valid `name_scope` name."""
        if self.name is None:
            self.name = self.__class__.__name__
        elif self.name == '<lambda>':
            self.name = 'lambda'
        else:
            # E.g. '_my_loss' => 'my_loss'
            self.name = self.name.strip('_')
        with ops.name_scope_v2(self.name) as scope_name:
            self._name_scope = ops.name_scope_v2(scope_name)


    def flatten_check(self, output, target):
        "Check that `out` and `targ` have the same number of elements and flatten them."
        ndim_output = ndim(output)
        ndim_target = ndim(target)
        if ndim(output) > 2:
            if self.axis == -1:
                shp = int_shape(output)
                output = output.reshape((shp[0], -1, shp[-1]))

                if ndim_target == ndim_output - 1 and target.dtype == Dtype.long:
                    target = cast(target.reshape((shp[0], -1)), 'int64')

                elif ndim_target == ndim_output and target.dtype != Dtype.long:
                    target = target.reshape((shp[0], -1, shp[-1]))

                return output, target


        elif ndim(output) <= 2 and ndim(output) == ndim(target):
            return output, target
        elif ndim(output) <= 2 and ndim(output) == ndim(target) + 1:
            return output, target
        else:
            raise ValueError('output and target have diffent elements.')

    def build(self, output, target, **kwargs):

        if self.num_classes is None:
            self.num_classes = int_shape(output)[self.axis]

        if self.sample_weight is None:
            self.sample_weight = ones(self.num_classes).to(get_device())
        else:
            self.sample_weight = to_tensor(self.sample_weight).to(output.device)
        if len(self.sample_weight) != self.num_classes:
            raise ValueError('weight should be 1-D tensor and length equal to numbers of filters')

        ignore_index_weight = np.ones(self.num_classes, dtype=np.float32)
        if isinstance(self.ignore_index, int) and 0 <= self.ignore_index < self.num_classes:
            ignore_index_weight[self.ignore_index] = 0
        elif isinstance(self.ignore_index, (list, tuple)):
            for idx in self.ignore_index:
                if isinstance(idx, int) and 0 <= idx < self.num_classes:
                    ignore_index_weight[idx] = 0
        self.ignore_index_weight = to_tensor(ignore_index_weight, dtype=output.dtype, device=output.device)

        if self.label_smooth:
            self.need_target_onehot = True

        if self.auto_balance and self.label_statistics is None:
            self._calculate_label_statistics(target)
        self._built = True

    def _calculate_label_statistics(self, target):
        weights = dataset_class_weights(self, target, self.num_classes)
        if weights is not None:
            self.label_statistics = weights
            self.sample_weight = to_tensor(weights).detach()

    preprocess = portable.classification_preprocess

    def calculate_loss(self, output, target):
        """ Calculate the unaggregate loss.
        The loss function calculation logic should define here., please dont't aggregate the loss in this phase.

        Args:
            output (tf.Tensor):
            target (tf.Tensor):
        """
        ##dont do aggregation
        raise NotImplementedError


    def forward(self, output: Tensor, target: Tensor, **kwargs) -> 'loss':
        """

            Args:
                output (tf.Tensor):
                target (tf.Tensor):

            Returns:
                calculated loss

            """
        try:

            output, target = self.flatten_check(output, target)
            if not self.built:
                self.build(output, target)

            loss = self.calculate_loss(*self.preprocess(output, target.detach(), **kwargs))
            if self.enable_ohem:
                loss=self._do_ohem(loss)
            loss = self._handel_abnormal(loss)

            loss = self._get_reduction(loss)
            return loss

        except Exception as e:
            print(e)
            PrintException()
            raise e



class _AlignedClassificationLoss(_ClassificationLoss):
    forward = portable.ClassificationLoss.forward
    calculate_loss = portable.ClassificationLoss.calculate_loss


class _PairwiseLoss(Loss, AutoTrackable):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1, reduction='batch_mean', enable_ohem=False, ohem_thresh=0.7,
                 input_names=None, output_names=None, name=None, **kwargs):
        """

        Args:
            axis (int): the position where the classes is.
            sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
            number of classes.
            from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
            ignore_index (int or list of int):
            cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
            less than 1..
            is_target_onehot (bool): Is the target tensor in onehot format?
            label_smooth (bool): Should use label smoothing?
            reduction (string): the method to aggrgate loss. None means no need to aggregate, 'mean' means average loss,
                'sum' means the summation of losses,'batch_mean' means average loss cross the batch axis then
                summation them.

        Attributes:
            need_target_onehot (bool): If True, means the before loss calculation , need to transform target as one-hot format, ex. label-smooth, default is False.
            is_multiselection (bool): If True, means the classification model is multi-selection, so cannot use  any softmax process, use sigmoid and binary_crosss_entropy
            insteaded.
            is_target_onehot (bool):  If True, means we have confirmed (not just declare) the target is transformed as  one-hot format
            reduction(str): The aggregation function for loss, available options are 'sum', 'mean 'and 'batch_mean', default is 'mean'
            axis (None or int): The axis we according with for loss calculation. Default is 1.
            from_logits (bool):If True, means  the sum of all probability will equal 1.
            is_logsoftmax (bool):If True, means model  use SoftMax as last layer or use any equivalent calculation.
            sample_weight(1D tensor):The loss weight for all classes.
            ignore_index(int , list, tuple): The classes we want to ignore in the loss calculation.
            cutoff(float): Means the decision boundary in this classification model, default=0.5.
            num_classes(int):number of  all the classes.
            label_smooth (bool):If True, mean we will apply label-smoothing in loss calculation.

        """
        super(_PairwiseLoss, self).__init__(reduction=reduction, axis=axis, enable_ohem=enable_ohem,
                                            ohem_thresh=ohem_thresh, input_names=input_names,
                                            output_names=output_names, name=name)
        self._set_name_scope()

        # initilize weight

    def _set_name_scope(self):
        """Creates a valid `name_scope` name."""
        if self.name is None:
            self.name = self.__class__.__name__
        elif self.name == '<lambda>':
            self.name = 'lambda'
        else:
            # E.g. '_my_loss' => 'my_loss'
            self.name = self.name.strip('_')
        with ops.name_scope_v2(self.name) as scope_name:
            self._name_scope = ops.name_scope_v2(scope_name)


    def flatten_check(self, output, target):
        "Check that `out` and `targ` have the same number of elements and flatten them."
        out_shp = int_shape(output)
        tar_shp = int_shape(target)
        if ndim(output) > 2:
            output = reshape(output, (out_shp[0], -1, out_shp[-1]))
        if ndim(target) > 2 and len(tar_shp) - 1 == len(tar_shp):
            target = reshape(target, (tar_shp[0], -1))
        elif ndim(target) > 2 and len(tar_shp) == len(tar_shp):
            target = reshape(target, (tar_shp[0], -1, tar_shp[-1]))

        return output, target

    def preprocess(self, output, target, **kwargs):
        return output, portable.dense_pair_target(output, target, self.axis)

    def calculate_loss(self, output, target):
        """ Calculate the unaggregate loss.
        The loss function calculation logic should define here., please dont't aggregate the loss in this phase.

        Args:
            output (tf.Tensor):
            target (tf.Tensor):
        """
        ##dont do aggregation
        raise NotImplementedError


    def __call__(self, output: Tensor, target: Tensor, **kwargs):
        result = self.forward(output, target, **kwargs)
        return result

    forward = portable.PairwiseLoss.forward


class CrossEntropyLoss(_AlignedClassificationLoss):
    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False,
                 reduction='mean', enable_ohem=False, ohem_thresh=0.7,
                 input_names=None, output_names=None, name='CrossEntropyLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff, label_smooth=label_smooth, reduction=reduction,
                         enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, input_names=input_names,
                         output_names=output_names, name=name)

        self.need_target_onehot = False

    loss_terms = portable.CrossEntropyLoss.loss_terms



class NLLLoss(_AlignedClassificationLoss):
    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False,
                 reduction='mean', enable_ohem=False, ohem_thresh=0.7,
                 input_names=None, output_names=None, name='NLLLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff, label_smooth=label_smooth, reduction=reduction,
                         enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, input_names=input_names,
                         output_names=output_names, name=name)

        self.need_target_onehot = True

    loss_terms = portable.NLLLoss.loss_terms


class F1ScoreLoss(_AlignedClassificationLoss):
    def __init__(self, beta=1, axis=-1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None, name='F1ScoreLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff, label_smooth=label_smooth, reduction=reduction,
                         enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, input_names=input_names,
                         output_names=output_names, name=name)
        self.beta = beta

        self.need_target_onehot = True

    loss_terms = portable.F1ScoreLoss.loss_terms


class FocalLoss(_AlignedClassificationLoss):
    def __init__(self, alpha=0.25, gamma=2, normalized=False, threshold=None, axis=-1, sample_weight=None,
                 auto_balance=False,
                 from_logits=None, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None, name='FocalLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff, label_smooth=label_smooth, reduction=reduction,
                         enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, input_names=input_names,
                         output_names=output_names, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.normalized = normalized
        self.need_target_onehot = True

    loss_terms = portable.FocalLoss.loss_terms


class BCELoss(_AlignedClassificationLoss):
    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='BCELoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)

        self._input_from_logits = from_logits
        self.num_classes = None
        self.is_logsoftmax = False
        self.need_target_onehot = True
        self.is_target_onehot = False

    loss_terms = portable.BCELoss.loss_terms


class DiceLoss(_AlignedClassificationLoss):
    def __init__(self, smooth=1, axis=-1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None, name='DiceLoss'):
        """
        Args:
            axis (int): the axis where the class label is.
            sample_weight ():
            from_logits ():
            ignore_index ():
            cutoff ():
            label_smooth ():
            reduction (string):
            name (stringf):
        """

        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff, label_smooth=label_smooth, reduction=reduction,
                         enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, input_names=input_names,
                         output_names=output_names, name=name)
        self.smooth = smooth
        self.is_logsoftmax = False
        self.need_target_onehot = True
        self.is_multiselection = False

    loss_terms = portable.DiceLoss.loss_terms


class L1Loss(_PairwiseLoss):
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

     Function that takes the mean element-wise absolute value difference.

     See :class:`~torch.nn.L1Loss` for details.
     """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='L1Loss'):
        super(L1Loss, self).__init__(reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                     input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.reduction = reduction

    calculate_loss = portable.L1Loss.calculate_loss


class L2Loss(_PairwiseLoss):
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

        Measures the element-wise mean squared error.

        See :class:`~torch.nn.MSELoss` for details.
        """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='MSELoss'):
        super(L2Loss, self).__init__(reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                     input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.reduction = reduction

    calculate_loss = portable.L2Loss.calculate_loss


#
class MSELoss(_PairwiseLoss):
    """
    Calculate the MSE loss


    """

    def __init__(self, axis=-1, reduction='mean', name='MSELoss', input_names=None, output_names=None,
                 enable_ohem=False, ohem_thresh=.7):
        super().__init__(axis=axis, reduction=reduction, input_names=input_names, output_names=output_names, name=name,
                         enable_ohem=enable_ohem, ohem_thresh=ohem_thresh)

    calculate_loss = portable.MSELoss.calculate_loss
        # return math_ops.div_no_nan(tf.nn.l2_loss(output-target), math_ops.cast(tf.shape(output)[0], dtype=output.dtype), name='value')


class SmoothL1Loss(_PairwiseLoss):
    r"""Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='SmoothL1Loss'):
        super(SmoothL1Loss, self).__init__(enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, reduction=reduction,
                                           input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.reduction = reduction
        self.huber_delta = 0.5

    calculate_loss = portable.SmoothL1Loss.calculate_loss


class WingLoss(_PairwiseLoss):
    def __init__(self, omega=10, epsilon=2, reduction='mean', enable_ohem=False, ohem_thresh=0.7,
                 input_names=None, output_names=None, name='WingLoss'):
        super(WingLoss, self).__init__(enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, reduction=reduction,
                                       input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.omega = omega
        self.epsilon = epsilon

    calculate_loss = portable.WingLoss.calculate_loss


class AdaptiveWingLoss(_PairwiseLoss):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, enable_ohem=False, ohem_thresh=0.7,
                 input_names=None, output_names=None, name='AdaptiveWingLoss'):
        super(AdaptiveWingLoss, self).__init__(enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                               input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    calculate_loss = portable.AdaptiveWingLoss.calculate_loss


class PerceptionLoss(portable.PerceptionLoss):
    """Compare the frozen feature maps of predictions and targets."""


class EdgeLoss(_PairwiseLoss):
    def __init__(self, enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None, name='EdgeLoss'):
        super(EdgeLoss, self).__init__(enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                       input_names=input_names, output_names=output_names, name=name)
        self.name = name

    def first_order(self, x, axis=2):
        if ndim(x) == 3:
            shp = list(int_shape(x))
            x = x.reshape((shp[0], int(math.sqrt(float(shp[1]))), int(math.sqrt(float(shp[1]))), shp[-1]))
        h, w = x.shape[1:3]
        if axis == 1:
            return tf.math.abs((x[:, :h - 1, :w - 1, :] - x[:, 1:, :w - 1, :]))
        elif axis == 2:
            return tf.math.abs(x[:, :h - 1, :w - 1, :] - x[:, :h - 1, 1:, :])
        else:
            return None

    calculate_loss = portable.EdgeLoss.calculate_loss


class IoULoss(_AlignedClassificationLoss):
    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=0, cutoff=None,
                 label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='lou_loss'):
        super(IoULoss, self).__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance,
                                      from_logits=from_logits, ignore_index=ignore_index, cutoff=cutoff,
                                      label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem,
                                      ohem_thresh=ohem_thresh, input_names=input_names, output_names=output_names,
                                      name=name)
        self.need_target_onehot = True
        self.is_multiselection = False
        self.is_logsoftmax = False

    loss_terms = portable.IoULoss.loss_terms


class SoftIoULoss(_AlignedClassificationLoss):
    def __init__(self, n_classes=None, reduction='mean', reduced=False, input_names=None, output_names=None,
                 name='SoftIoULoss', axis=-1, ignore_index=0, **kwargs):
        super().__init__(axis=axis, reduction=reduction, name=name, input_names=input_names, output_names=output_names, ignore_index=ignore_index, **kwargs)
        self.n_classes = n_classes
        self.reduced = reduced

    loss_terms = portable.SoftIoULoss.loss_terms


class GeneralizedDiceLoss(DiceLoss):
    """Trident class-weighted Dice."""


def get_loss(loss_name):
    if loss_name is None:
        return None
    loss_modules = ['trident.optims.tensorflow_losses']
    if loss_name in __all__:
        loss_fn = get_class(loss_name, loss_modules)
    else:
        try:
            loss_fn = get_class(camel2snake(loss_name), loss_modules)
        except Exception:
            loss_fn = get_class(loss_name, loss_modules)
    return loss_fn

