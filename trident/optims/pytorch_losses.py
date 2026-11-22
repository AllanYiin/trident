from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import builtins
import math
import numbers
import sys
from math import *
from typing import Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from trident import context
from trident.backend import dtype as Dtype
from trident.backend.common import *
from trident.backend.model import ModelBase
from trident.backend.pytorch_backend import *
from trident.backend.pytorch_backend import Parameter as LayerParameter
from trident.backend.pytorch_ops import *
import trident.backend.pytorch_ops as ops
from trident.backend.tensorspec import ObjectType, object_type_inference
from trident.data.dataset import *
from trident.layers.pytorch_activations import sigmoid
from trident.layers.pytorch_layers import Dense
from trident.optims.loss_weights import dataset_class_weights
from trident.optims import loss_kernels as portable
from trident.optims.loss_kernels import HuberLoss, BCEWithLogitsLoss
from trident.optims.losses import PUBLIC_LOSSES, Loss, _check_logsoftmax_logit, _check_logit,_check_softmax

_device = get_device()
__all__ = ['get_loss', '_ClassificationLoss', *PUBLIC_LOSSES]

ctx = context._context()
_float_dtype = Dtype.float16 if ctx.amp_available == True and ctx.is_autocast_enabled == True and get_device() == 'cuda' else Dtype.float32


def _calculate_loss_unimplemented(self, output: Tensor, target: Tensor) -> None:
    raise NotImplementedError


class _ClassificationLoss(Loss):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=1, sample_weight=None,  class_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.1, input_names=None, output_names=None,
                 name=None, **kwargs):
        """

        Args:
            axis (int): the position where the classes are.
            sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
            number of classes.
            from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
            ignore_index (int or list of int):
            cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
            less than 1
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
            axis (None or int): The axis we according to for loss calculation. Default is 1.
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
        self.label_statistics = None
        self.valid_target_object_type = []
        self.need_target_onehot = False
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

    def _calculate_label_statistics(self, target):
        weights = dataset_class_weights(self, target, self.num_classes)
        if weights is not None:
            self.label_statistics = weights
            self.sample_weight = to_tensor(weights).detach()


    def flatten_check(self, output, target):
        "Check that `out` and `targ` have the same number of elements and flatten them."
        ndim_output = ndim(output)
        ndim_target = ndim(target)

        if ndim(output) > 2:
            if self.axis == 1:
                output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W => N,C,H*W
                if ndim_target == ndim_output - 1 and target.dtype == Dtype.long:
                    target = target.view(target.size(0), -1)
                elif ndim_target == ndim_output and target.dtype != Dtype.long:
                    target = target.reshape((target.size(0), target.size(1), -1))
            elif self.axis == -1:
                output = output.view(output.size(0), -1, output.size(-1))
                output = output.transpose(2, 1).contiguous()
                if ndim_target == ndim_output - 1 and target.dtype == Dtype.long:
                    target = target.view(target.size(0), -1)
                elif ndim_target == ndim_output and target.dtype != Dtype.long:
                    target = target.reshape((target.size(0), -1, target.size(-1)))
                    target = target.transpose(2, 1).contiguous()
            return output, target
        elif ndim(output) <= 2:
            return output, target
        else:

            return output, target
            # raise ValueError('output and target have diffent elements.')

    def build(self, output, target, **kwargs):
        if self.num_classes is None:
            self.num_classes = output.shape[self.axis % output.ndim]

        if self.sample_weight is None:
            self.sample_weight = torch.ones(self.num_classes, device=output.device)
        else:
            self.sample_weight = to_tensor(self.sample_weight).detach().to(output.device)
        if len(self.sample_weight) != self.num_classes:
            raise ValueError('weight should be 1-D tensor and length equal to numbers of filters')

        self.ignore_index_weight = torch.ones(self.num_classes, dtype=output.dtype, device=output.device)
        # ignore_index
        with torch.no_grad():
            if isinstance(self.ignore_index, int) and 0 <= self.ignore_index < self.num_classes:
                self.ignore_index_weight[self.ignore_index] = 0
            elif isinstance(self.ignore_index, (list, tuple)):
                for idx in self.ignore_index:
                    if isinstance(idx, int) and 0 <= idx < self.num_classes:
                        self.ignore_index_weight[idx] = 0

        if self.label_smooth:
            self.need_target_onehot = True

        if self.auto_balance :
            if self.label_statistics is None:
                self._calculate_label_statistics(target)

            else:
                self.sample_weight=to_tensor(self.label_statistics.copy()).detach()
        self._built = True

    preprocess = portable.classification_preprocess

    calculate_loss: Callable[..., Any] = _calculate_loss_unimplemented

    # def calculate_loss(self, output, target, **kwargs):
    #     """ Calculate the unaggregate loss.
    #     The loss function calculation logic should define here., please don't aggregate the loss in this phase.
    #
    #     Args:
    #         output (tf.Tensor):
    #         target (tf.Tensor):
    #     """
    #     ##dont do aggregation
    #     raise NotImplementedError


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
            if any_abnormal_number(loss):
                print('output shape:', output.shape, 'target shape:', target.shape)
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


class _PairwiseLoss(Loss):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=1, sample_weight=None, reduction='mean', enable_ohem=False, ohem_thresh=0.7,
                 input_names=None, output_names=None, name=None, **kwargs):
        """

        Args:
            axis (int): the position where the classes are.
            sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
            number of classes.
            from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
            ignore_index (int or list of int):
            cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
            less than 1
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
            axis (None or int): The axis we according to for loss calculation. Default is 1.
            from_logits (bool):If True, means  the sum of all probability will equal 1.
            is_logsoftmax (bool):If True, means model  use SoftMax as last layer or use any equivalent calculation.
            sample_weight(1D tensor):The loss weight for all classes.
            ignore_index(int , list, tuple): The classes we want to ignore in the loss calculation.
            cutoff(float): Means the decision boundary in this classification model, default=0.5.
            num_classes(int):number of  all the classes.
            label_smooth (bool):If True, mean we will apply label-smoothing in loss calculation.

        """
        super(_PairwiseLoss, self).__init__(reduction=reduction, axis=axis, enable_ohem=enable_ohem,
                                            ohem_thresh=ohem_thresh, input_names=input_names, output_names=output_names,
                                            name=name)
        self.sample_weight = sample_weight


        # initilize weight

    def flatten_check(self, output, target):
        "Check that `out` and `targ` have the same number of elements and flatten them."
        if int_shape(output) != int_shape(target):
            raise ValueError('Output and Target should have same shape in pairwise loss. Output:{0} Target: {1}'.format(
                int_shape(output), int_shape(target)))
        else:
            if ndim(output) == ndim(target) > 2:
                output = output.view(output.size(0), -1)  # N,C,H,W => N,C,H*W
                target = target.view(target.size(0), -1)
                return output, target
            else:
                return output, target


    def preprocess(self, output, target, **kwargs):
        return output, portable.stop_gradient(portable.as_like(portable.dense_pair_target(output, target, self.axis), output))


    def calculate_loss(self, output, target, **kwargs):
        """ Calculate the unaggregate loss.
        The loss function calculation logic should define here., please don't aggregate the loss in this phase.

        Args:
            output (tf.Tensor):
            target (tf.Tensor):
        """
        ##dont do aggregation
        raise NotImplementedError

    forward = portable.PairwiseLoss.forward


class CrossEntropyLoss(_ClassificationLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.

    `input` has to be a Tensor of size either :math:`(minibatch, C)` or
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    This criterion expects a class index in the range :math:`[0, C-1]` as the
    `target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`
    is specified, this criterion also accepts this class index (this index may not
    necessarily be in the class range).

    The loss can be described as:

    math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    or in the case of the :attr:`weight` argument being specified:

    math::
        \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

    The losses are averaged across observations for each minibatch.

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below).


    Args:
        axis (int): the position where the classes are.
        sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
        number of classes.
        from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
        ignore_index (int or list of int):
        cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
        less than 1ㄡ
        is_target_onehot (bool): Is the target tensor in onehot format?
        label_smooth (bool): Should use label smoothing?
        reduction (string): the method to aggrgate loss. None means no need to aggregate, 'mean' means average loss,
            'sum' means the summation of losses,'batch_mean' means average loss cross the batch axis then
            summation them.

    Examples:
    >>> output=to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]]).float()
    >>> print(output.shape)
    torch.Size([4, 3])
    >>> target=to_tensor([1,0,1,2]).long()
    >>> print(target.shape)
    torch.Size([4])
    >>> CrossEntropyLoss(reduction='mean')(output,target).cpu()
    tensor(1.1305)
    >>> CrossEntropyLoss(reduction='sum')(output,target).cpu()
    tensor(4.5221)
    >>> CrossEntropyLoss(label_smooth=True,reduction='mean')(output,target).cpu()
    tensor(1.0786)
    >>> CrossEntropyLoss(sample_weight=to_tensor([1.0,1.0,0.5]).float(),reduction='mean')(output,target).cpu()
    tensor(0.9889)
    >>> CrossEntropyLoss(ignore_index=2,reduction='mean')(output,target).cpu()
    tensor(0.8259)


    """

    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='CrossEntropyLoss'):
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
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)

        self.need_target_onehot = False

    def build(self, output, target, **kwargs):
        self.num_classes = output.shape[self.axis % output.ndim]
        super().build(output, target, **kwargs)

    def _loss_and_weight(self, output, target):
        if not self.built:
            self.build(output, target)
        axis = self.axis % output.ndim
        work_dtype = torch.float32 if output.dtype in (torch.float16, torch.bfloat16) else output.dtype
        values = output.to(work_dtype)
        if self.from_logits is True:
            log_probs = F.log_softmax(values, dim=axis)
        elif self.from_logits is False:
            log_probs = values.clamp_min(torch.finfo(work_dtype).tiny).log()
        elif _check_logsoftmax_logit(values.detach(), axis):
            log_probs = values
        elif _check_softmax(values.detach(), axis):
            log_probs = values.clamp_min(torch.finfo(work_dtype).tiny).log()
        else:
            log_probs = F.log_softmax(values, dim=axis)

        weights = (self.sample_weight.to(values) * self.ignore_index_weight.to(values)).detach()
        target = target.detach().to(values.device)
        shape = [1] * values.ndim
        shape[axis] = weights.numel()
        class_weights = weights.reshape(shape)
        smooth = 0.1 if self.label_smooth is True else float(self.label_smooth or 0)
        if not 0 <= smooth <= 1:
            raise ValueError("label_smooth must be in [0, 1]")
        if target.shape == output.shape:
            distribution = target.to(work_dtype)
            distribution = distribution * (1 - smooth) + smooth / weights.numel()
            active = (target * (class_weights != 0)).sum(dim=axis) != 0
            distribution_weights = distribution * class_weights
            active_weights = distribution_weights != 0
            safe_log_probs = log_probs.masked_fill(~active_weights, 0)
            raw = -(safe_log_probs * distribution_weights).sum(dim=axis)
            return raw.masked_fill(~active, 0), active.to(work_dtype)

        labels = target.long()
        valid = torch.ones_like(labels, dtype=torch.bool)
        for ignore in self.ignore_index:
            if ignore is not None:
                valid &= labels != ignore
        safe_labels = labels.masked_fill(~valid, 0)
        nll = -log_probs.gather(axis, safe_labels.unsqueeze(axis)).squeeze(axis)
        label_weights = weights[safe_labels].masked_fill(~valid, 0)
        nll = nll.masked_fill(label_weights == 0, 0) * label_weights
        if smooth:
            smooth_loss = -(log_probs.masked_fill(class_weights == 0, 0) * class_weights).mean(dim=axis)
            nll = (1 - smooth) * nll + smooth * smooth_loss.masked_fill(~valid, 0)
        return nll, label_weights

    def calculate_loss(self, output, target, **kwargs):
        return self._loss_and_weight(output, target)[0]

    def forward(self, output, target, **kwargs):
        self.gradient_denominator = None
        loss, weight = self._loss_and_weight(output, target)
        self._handel_abnormal(loss)
        if self.enable_ohem:
            count = max(1, int(loss.numel() * self.ohem_thresh))
            active = weight.flatten() != 0
            indices = loss.flatten().masked_fill(~active, -torch.inf).topk(count).indices
            loss, weight = loss.flatten()[indices], weight.flatten()[indices]
        if self.reduction in ('mean', 'batch_mean'):
            denominator = weight.sum()
            self.gradient_denominator = denominator.detach()
            return loss.sum() / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        return self._get_reduction(loss)


class NLLLoss(_AlignedClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='NllLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)
        self.need_target_onehot = False

    loss_terms = portable.NLLLoss.loss_terms


class F1ScoreLoss(_AlignedClassificationLoss):
    def __init__(self, beta=1, axis=1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None, name='F1ScoreLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)
        self.beta = beta

        self.need_target_onehot = True

    loss_terms = portable.F1ScoreLoss.loss_terms


class FocalLoss(_AlignedClassificationLoss):
    def __init__(self, alpha=0.25, gamma=2, normalized=False, threshold=None, axis=1, sample_weight=None,
                 auto_balance=False, from_logits=None, ignore_index=-100, cutoff=None,
                 label_smooth=False, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None,
                 output_names=None, name='FocalLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.normalized = normalized
        self.is_logsoftmax = None
        self.need_target_onehot = False
        self.is_target_onehot = None

    loss_terms = portable.FocalLoss.loss_terms


class BCELoss(_AlignedClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='BCELoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)

        self.num_classes = None
        self.is_logsoftmax = None
        self._input_from_logits = from_logits
        self.need_target_onehot = True
        self.is_target_onehot = False

    loss_terms = portable.BCELoss.loss_terms


class DiceLoss(_AlignedClassificationLoss):
    def __init__(self, smooth=1, axis=1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
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
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)
        self.smooth = smooth
        self.is_logsoftmax = None
        self.need_target_onehot = True
        self.is_multiselection = False

    loss_terms = portable.DiceLoss.loss_terms

        # dice = (2.0 * intersection.sum(1) + self.smooth)/(den1.sum(1) + den2.sum(1) + self.smooth)

        # if output.shape[1]==2:
        #     dice =  (2.0 * (intersection * sample_weight[:,1:].sum(1) )+ self.smooth) / clip(den1[:,1:].sum(1) + den2[:,1:].sum(1) + self.smooth,min=1e-7)
        # else:
        #
        #     dice =  torch.div((2.0 * (intersection * sample_weight).sum(1) + self.smooth) , clip(den1.sum(1) + den2.sum(1) + self.smooth,min=1e-7))

        # if any_abnormal_number(den2):
        #     print('{0} after dice {1}'.format(self.name,dice))

        # return 1.0 -dice


class TverskyLoss(portable.TverskyLoss):
    def __init__(self, alpha=.5, beta=.5, smooth=1., axis=1, **kwargs):
        super().__init__(alpha, beta, smooth, axis=axis, **kwargs)


class ActiveContourLoss(_AlignedClassificationLoss):
    def __init__(self, lambdaP=1, mu=1, axis=1, sample_weight=None, auto_balance=False, from_logits=None,
                 ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None, name='ActiveContourLoss'):
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
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)
        self.lambdaP = lambdaP
        self.mu = mu
        self.is_logsoftmax = None
        self.need_target_onehot = True
        self.is_multiselection = False

    loss_terms = portable.ActiveContourLoss.loss_terms


class GeneralizedDiceLoss(_AlignedClassificationLoss):
    def __init__(self, smooth=1., axis=1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None, name='GeneralizedDiceLoss'):
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
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)
        self.smooth = smooth
        self.is_logsoftmax = None
        self.need_target_onehot = True
        self.is_multiselection = False

    loss_terms = portable.GeneralizedDiceLoss.loss_terms


class KLDivergenceLoss(_PairwiseLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='KLDivergenceLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)

        self.num_classes = None
        self.is_logsoftmax = None
        self.need_target_onehot = True
        self.is_target_onehot = False

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """


        loss = nn.functional.kl_div(output, target, reduction='none')

        return loss


class L1Loss(_PairwiseLoss):
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

     Function that takes the mean element-wise absolute value difference.

     See :class:`~torch.nn.L1Loss` for details.
     """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='L1Loss'):
        super(L1Loss, self).__init__(axis=1,reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
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
        super(L2Loss, self).__init__(axis=1,reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                     input_names=input_names, output_names=output_names, name=name)

        self.name = name
        self.reduction = reduction

    calculate_loss = portable.L2Loss.calculate_loss


class SmoothL1Loss(_PairwiseLoss):
    r"""Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='SmoothL1Loss'):
        super(SmoothL1Loss, self).__init__(axis=1,enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, reduction=reduction,
                                           input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.reduction = reduction
        self.huber_delta = 0.5

    calculate_loss = portable.SmoothL1Loss.calculate_loss


class MSELoss(_PairwiseLoss):
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

        Measures the element-wise mean squared error.

        See :class:`~torch.nn.MSELoss` for details.
        """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='MSELoss'):
        super(MSELoss, self).__init__(axis=1,reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                      input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.reduction = reduction

    calculate_loss = portable.MSELoss.calculate_loss


class WingLoss(_PairwiseLoss):
    def __init__(self, omega=10, epsilon=2, input_names=None, output_names=None, name='WingLoss'):
        super(WingLoss, self).__init__(axis=1,input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.omega = omega
        self.epsilon = epsilon

    calculate_loss = portable.WingLoss.calculate_loss



class AdaptiveWingLoss(_PairwiseLoss):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, input_names=None, output_names=None,
                 name='AdaptiveWingLoss'):
        super(AdaptiveWingLoss, self).__init__(axis=1,input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    calculate_loss = portable.AdaptiveWingLoss.calculate_loss


class ExponentialLoss(_PairwiseLoss):
    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='ExponentialLoss'):
        super(ExponentialLoss, self).__init__(axis=1,reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                              input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.reduction = reduction

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        target = target.detach()
        if output.shape == target.shape:
            error = (output - target) ** 2
            loss = 1 - (-1 * error / np.pi).exp()
            return loss
        else:
            raise ValueError('output and target shape should the same in ExponentialLoss. ')


class ItakuraSaitoLoss(_PairwiseLoss):
    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='ItakuraSaitoLoss'):
        super(ItakuraSaitoLoss, self).__init__(axis=1,reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                               input_names=input_names, output_names=output_names, name=name)
        self.name = name
        self.reduction = reduction

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        #  y_true/(y_pred+1e-12) - log(y_true/(y_pred+1e-12)) - 1;
        target = target.detach()
        if output.shape == target.shape:
            if -1 <= output.min() < 0 and -1 <= target.min() < 0:
                output = output + 1
                target = target + 1
            loss = (target / (output + 1e-8)) - ((target + 1e-8) / (output + 1e-8)).log() - 1

            return loss
        else:
            raise ValueError('output and target shape should the same in ItakuraSaitoLoss. ')


class CosineSimilarityLoss(_PairwiseLoss):
    def __init__(self, enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='CosineSimilarityLoss'):
        super(CosineSimilarityLoss, self).__init__(axis=1,enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                                   input_names=input_names, output_names=output_names, name=name)

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        target = target.detach()
        return 1.0 - torch.cosine_similarity(output, target)


def gaussian(window_size, sigma):
    """Create 1-D gauss kernel
    Args:
        window_size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """

    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()

    return gauss.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(input, window_1d):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    assert all([ws == 1 for ws in window_1d.shape[1:-1]]), window_1d.shape
    window_1d = window_1d.to(input.device).to(input.dtype)
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= window_1d.shape[-1]:
            out = conv(out, weight=window_1d.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {window_1d.shape[-1]}"
            )

    return out


def ssim(X, Y, window, data_range: float):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    if data_range == 2:
        C1 = ((K1 * data_range) / data_range) ** 2
        C2 = ((K2 * data_range) / data_range) ** 2
    else:
        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window)
    mu2 = _gaussian_filter(Y, window)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (_gaussian_filter(X * X, window) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter(Y * Y, window) - mu2_sq)
    sigma12 = compensation * (_gaussian_filter(X * Y, window) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


class MS_SSIMLoss(_PairwiseLoss):
    def __init__(self, window_size=11, window_sigma=1.5, data_range=255., channel=3, use_padding=False, weights=None,
                 eps=1e-8, reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,
                 name='ms_ssim'):
        super(MS_SSIMLoss, self).__init__(reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=0.7, input_names=input_names,
                                          output_names=output_names, name=name)

        if not (window_size % 2 == 1):
            raise ValueError("Window size should be odd.")
        self.window_size = window_size
        self.channel = channel
        self.window_sigma = window_sigma
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

        spatial_dims = 2,

        self.data_range = data_range
        self.weights = weights

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=Dtype.float, requires_grad=False)

        self.levels = weights.shape[0]
        if self.levels is not None:
            weights = weights[:self.levels]
            weights = weights / weights.sum()

        self.weights = weights

    def build(self, output, target, **kwargs):
        self.win = gaussian(self.window_size, self.window_sigma)
        self.win = self.win.repeat([output.shape[1]] + [1] * (len(output.shape) - 1))
        self._built = True

    def ms_ssim(self, X, Y):
        '''
        interface of ms-ssim
        :param X: a batch of images, (N,C,H,W)
        :param Y: a batch of images, (N,C,H,W)
        :param window: 1-D gauss kernel
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param weights: weights for different levels
        :param use_padding: padding image before conv
        :param eps: use for avoid grad nan.
        :return:
        '''
        if not X.shape == Y.shape:
            raise ValueError("Input images should have the same dimensions.")

        for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)

        if not X.type() == Y.type():
            raise ValueError('Input images should have the same dtype. X:{0} Y:{1}'.format(X.type(), Y.type()))

        if len(X.shape) == 4:
            avg_pool = F.avg_pool2d
        elif len(X.shape) == 5:
            avg_pool = F.avg_pool3d
        else:
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

        smaller_side = min(X.shape[-2:])
        assert smaller_side > (self.window_size - 1) * (
                2 ** 3
        ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % (
                    (self.window_size - 1) * (2 ** 3))

        mcs = []
        for i in range(self.levels):
            ssim_per_channel, cs = ssim(X, Y, window=self.win, data_range=self.data_range)
            if i < self.levels - 1:
                mcs.append(torch.relu(cs))
                padding = [s % 2 for s in X.shape[2:]]
                X = avg_pool(X, kernel_size=2, padding=padding)
                Y = avg_pool(Y, kernel_size=2, padding=padding)

        ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
        ms_ssim_val = torch.prod(mcs_and_ssim ** self.weights.to(X).view(-1, 1, 1), dim=0)
        return ms_ssim_val

    def calculate_loss(self, output, target, **kwargs):
        target = target.to(output.dtype)
        return 1-self.ms_ssim(output, target)


class IoULoss(_AlignedClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=0, cutoff=None,
                 label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='lou_loss'):
        super(IoULoss, self).__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance,
                                      from_logits=from_logits, ignore_index=ignore_index, cutoff=cutoff,
                                      label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem,
                                      ohem_thresh=ohem_thresh, input_names=input_names, output_names=output_names,
                                      name=name)
        self.need_target_onehot = True
        self.is_multiselection = False
        self.is_logsoftmax = None

    loss_terms = portable.IoULoss.loss_terms


class SoftIoULoss(_AlignedClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=0, cutoff=None,
                 label_smooth=False, reduction="mean",  enable_ohem=False,ohem_thresh=0.7, input_names=None, output_names=None, name='softiou'):
        super(SoftIoULoss, self).__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance,
                                      from_logits=from_logits, ignore_index=ignore_index, cutoff=cutoff,
                                      label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem,
                                      ohem_thresh=ohem_thresh, input_names=input_names, output_names=output_names,
                                      name=name)

        self.reduction = reduction

    loss_terms = portable.SoftIoULoss.loss_terms


class LovaszSoftmax(_AlignedClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=None, ignore_index=0, cutoff=None,
                 label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='lovasz_loss'):
        super(LovaszSoftmax, self).__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance,
                                            from_logits=from_logits, ignore_index=ignore_index, cutoff=cutoff,
                                            label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem,
                                            ohem_thresh=ohem_thresh, input_names=input_names, output_names=output_names,
                                            name=name)
        self.need_target_onehot = True
        self.is_multiselection = False
        self.is_logsoftmax = None
        self.reduction = reduction
        self.num_classes = None

    loss_terms = portable.LovaszSoftmax.loss_terms


class TripletLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative
    examples` respectively). The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}


    where

     math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Args:
        margin (float, optional): Default: :math:`1`.
        p (int, optional): The norm degree for pairwise distance. Default: :math:`2`.
        swap (bool, optional): The distance swap is described in detail in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. Default: ``False``.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, D)` where :math:`D` is the vector dimension.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> anchor = torch.randn(100, 128, requires_grad=True)
    >>> positive = torch.randn(100, 128, requires_grad=True)
    >>> negative = torch.randn(100, 128, requires_grad=True)
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

     _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(self, margin: float = 1.0, p: float = 2., eps: float = 1e-6, swap: bool = False,
                 reduction: str = 'mean'):
        super(TripletLoss, self).__init__(reduction=reduction)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.margin, p=self.p,
                                     eps=self.eps, swap=self.swap, reduction=self.reduction)


TripletMarginLoss = TripletLoss


class HardTripletLoss(_Loss):
    def __init__(self, margin=.1, hardest=False, squared=False):
        super().__init__()
        self.margin, self.hardest, self.squared = margin, hardest, squared

    forward = portable.HardTripletLoss.forward


class CenterLoss(Layer):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, reduction="mean", reduced=False):
        super(CenterLoss, self).__init__()
        self.reduction = reduction
        self.reduced = reduced
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = LayerParameter(torch.randn(self.num_classes, self.feat_dim))
        init.kaiming_uniform_(self.centers, a=sqrt(5))
        self.to(_device)

    def forward(self, features, target) -> 'loss':
        """
        Args:
            features: feature matrix with shape (batch_size, feat_dim).
            target: ground truth labels with shape (num_classes).

        """

        centers = self.centers.to(features)
        selected = centers.index_select(0, target.long().to(features.device))
        distances = (features - selected).square().sum(-1).clamp(1e-8, 1e5)
        if self.reduction == 'none':
            return distances
        return distances.mean() if self.reduction in ('mean', 'batch_mean') else distances.sum()


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.to(_device)

    def forward(self, output, target) -> 'loss':
        target = target.detach()
        img_size = list(output.size())
        G = gram_matrix(output)
        Gt = gram_matrix(target)
        return F.mse_loss(G, Gt).div((img_size[1] * img_size[2] * img_size[3]))


class PerceptionLoss(portable.PerceptionLoss):
    """Compare the frozen feature maps of predictions and targets."""


class EdgeLoss(_PairwiseLoss):
    def __init__(self,  reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None,name='edge_loss'):
        super(EdgeLoss, self).__init__(axis=1, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                                     input_names=input_names, output_names=output_names, name=name)

        self.name = name
        self.reduction = reduction
    def first_order(self, x, axis=2):
        h, w = x.size(2), x.size(3)
        if axis == 2:
            return (x[:, :, :h - 1, :w - 1] - x[:, :, 1:, :w - 1]).abs()
        elif axis == 3:
            return (x[:, :, :h - 1, :w - 1] - x[:, :, :h - 1, 1:]).abs()
        else:
            return None

    def calculate_loss(self, output, target, **kwargs):
        if ndim(output)==3:
            output=output.unsqueeze(1)
        if ndim(target)==3:
            target=target.unsqueeze(1)

        loss = MSELoss(reduction="mean")(self.first_order(output, 2), self.first_order(target, 2)) + MSELoss(
            reduction="mean")(self.first_order(output, 3), self.first_order(target, 3))
        return loss


class TransformInvariantLoss(nn.Module):
    def __init__(self, loss: _Loss, embedded_func: Layer):
        super(TransformInvariantLoss, self).__init__()
        if loss is not None:
            self.loss=loss
        else:
            self.loss = MSELoss(reduction='mean')
        self.coverage = 110
        self.rotation_range = 20
        self.zoom_range = 0.1
        self.shift_range = 0.02
        self.random_flip = 0.3
        self.embedded_func = embedded_func
        self.to(_device)

    forward = portable.TransformInvariantLoss.forward


class GPLoss(nn.Module):
    def __init__(self, discriminator, l=10):
        super(GPLoss, self).__init__()
        self.discriminator = discriminator
        self.l = l

    def forward(self, real_data, fake_data, *, alpha=None):
        if alpha is None:
            alpha = real_data.new_empty((real_data.size(0),) + (1,) * (real_data.ndim - 1)).uniform_(0, 1)
        alpha = torch.as_tensor(alpha, dtype=real_data.dtype, device=real_data.device).expand_as(real_data)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.detach().requires_grad_(True)
        disc_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=real_data.new_ones(disc_interpolates.size()), create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean() * self.l


class AdaCos(_ClassificationLoss):
    def __init__(self, num_features, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.num_classes = None
        self.fc = None
        self.m = m
        self.need_target_onehot = True

    def calculate_loss(self, output, target, **kwargs):
        output, target = self.flatten_check(output, target)
        shp = int_shape(output)
        batch = shp[0]
        if self.num_classes is None or self.fc is None:
            self.num_classes = shp[1]
            self.s = math.sqrt(2) * math.log(self.num_classes - 1)
            self.fc = Dense(num_filters=self.num_classes, weights_norm='l2', use_bias=False)
            self.fc.build(tensor_to_shape(output, need_exclude_batch_axis=True, is_singleton=False))
            self.W = self.fc.weight
            nn.init.xavier_uniform_(self.fc.weight)

        # dot product
        logits = self.fc(F.normalize(output))
        if target is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / output.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output


def get_loss(loss_name):
    if loss_name is None:
        return None
    loss_modules = ['trident.optims.pytorch_losses']
    if loss_name in __all__:
        loss_fn = get_class(loss_name, loss_modules)
    else:
        try:
            loss_fn = get_class(camel2snake(loss_name), loss_modules)
        except Exception:
            loss_fn = get_class(loss_name, loss_modules)
    return loss_fn
