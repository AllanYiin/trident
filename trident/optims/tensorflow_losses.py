from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import loss_reduction
from tensorflow.python.ops.losses import util as tf_losses_utils
from trident.backend.common import camel2snake, get_class, epsilon, PrintException
from trident.backend.tensorspec import *
from trident.backend.tensorflow_backend import *
from trident.backend.tensorflow_ops import *
from trident.optims.losses import Loss
# def cosine_similarity(y_true, y_pred):
#     assert y_true.ndim == 2
#     assert y_pred.ndim == 2
#     y_true = l2_normalize(y_true, axis=1)
#     y_pred = l2_normalize(y_pred, axis=1)
#     return T.sum(y_true * y_pred, axis=1, keepdims=False)
# def cosine_ranking_loss(y_true, y_pred):
#     q = y_pred[: ,:args.hidden_size]
#     a_correct = y_pred[: ,args.hidden_size: 2 *args.hidden_size]
#     a_incorrect = y_pred[: , 2 *args.hidden_size: 3 *args.hidden_size]
#
#     return mean \
#         (T.maximum(0., args.margin - cosine_similarity(q, a_correct) + cosine_similarity(q, a_incorrect)) - y_true
#             [0 ] *0, axis=-1)


__all__ = ['get_loss','_ClassificationLoss', 'CrossEntropyLoss', 'MSELoss', 'EdgeLoss', 'NLLLoss', 'F1ScoreLoss', '_ClassificationLoss',
           'FocalLoss','DiceLoss','L1Loss','L2Loss','WingLoss','AdaptiveWingLoss']





class _ClassificationLoss(Loss):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1, sample_weight=None, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean', name=None, **kwargs):
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
        super(_ClassificationLoss, self).__init__(reduction=reduction, sample_weight=sample_weight,axis=axis,name=name)
        self._set_name_scope()
        self.need_target_onehot = True
        self.is_multiselection = False
        self.is_target_onehot = False
        self.from_logits = from_logits
        self.is_logsoftmax = False
        self.ignore_index = ignore_index
        self.ignore_index_weight=None
        if cutoff is not None and not 0 < cutoff < 1:
            raise ValueError('cutoff should between 0 and 1')
        self.cutoff = cutoff
        self.num_classes = None
        self.label_smooth = label_smooth

        # initilize weight

    def _set_name_scope(self):
        """Creates a valid `name_scope` name."""
        if self.name is None:
            name = self.__class__.__name__
        elif self.name == '<lambda>':
            name = 'lambda'
        else:
            # E.g. '_my_loss' => 'my_loss'
            name = self.name.strip('_')
        with ops.name_scope_v2(name) as scope_name:
            self._name_scope = ops.name_scope_v2(scope_name)

    def _get_reduction(self, loss):
        with self._name_scope:
            num_present=math_ops.cast(array_ops.size(loss, name='num_elements'), dtype=loss.dtype)
            if ndim(loss) <= 1 or self.reduction == 'none':
                return loss
            if ndim(loss) >= 2 and self.reduction == 'batch_sum':
                loss = reshape(loss, (int_shape(loss)[0], -1))
                return loss.mean(1).sum()
            elif ndim(loss) >= 2 and self.reduction == 'batch_mean':
                loss = reshape(loss, (int_shape(loss)[0], -1))
                return loss.mean(1).mean()
            elif self.reduction in ['mean', 'batch_mean']:
                total_loss = math_ops.reduce_sum(loss)
                return math_ops.div_no_nan(total_loss, num_present, name='value')
            elif self.reduction in ['sum', 'batch_sum']:
                return math_ops.reduce_sum(loss)
            else:
                total_loss = math_ops.reduce_sum(loss)
                return math_ops.div_no_nan(total_loss, num_present, name='value')

    def flatten_check(self, output, target):
        "Check that `out` and `targ` have the same number of elements and flatten them."
        if ndim(output) > 2 and len(output) == len(target)+1:
            shp = int_shape(output)
            output = output.reshape((shp[0], -1, shp[-1]))
            target = target.reshape((shp[0], -1))
            return output, target
        elif ndim(output) > 2 and len(output) == len(target):
            shp = int_shape(output)
            output = output.reshape((shp[0], -1, shp[-1]))
            target = target.reshape((shp[0], -1))
            return output, target
        else:
            raise ValueError('output and target have diffent elements.')

    def preprocess(self, output: Tensor, target: Tensor, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # check num_clases
        if self.num_classes is None:
            self.num_classes = int_shape(output)[self.axis]
        output, target = self.flatten_check(output, target)
        if self.sample_weight is None:
            self.sample_weight = ones(self.num_classes, requires_grad=False)
        elif len(self.sample_weight) != self.num_classes:
            raise ValueError('weight should be 1-D tensor and length equal to numbers of filters')
        else:
            pass

        output_exp = exp(output)

        if (ndim(output) >= 1 and 'float' in str(output.dtype) and output.min() >= 0 and output.max() <= 1 ):
            self.is_logsoftmax = False
            self.from_logits = True
            output = clip(output, min=1e-8, max=1 - 1e-8)

        elif (ndim(output) >=1 and 'float' in str(output.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1 ):
            self.is_logsoftmax = True
            self.from_logits = True
            output = clip(output, max=- 1e-8)
        else:
            self.is_logsoftmax = False
            self.from_logits = False

        self.ignore_index_weight=ones_like(self.sample_weight,requires_grad=False,dtype=output.dtype)
        # ignore_index

        if isinstance(self.ignore_index, int) and 0 <= self.ignore_index < self.num_classes:
            self.ignore_index_weight[self.ignore_index] = 0
        elif isinstance(self.ignore_index, (list, tuple)):
            for idx in self.ignore_index:
                if isinstance(idx, int) and 0 <= idx < int_shape(output)[self.axis]:
                    self.ignore_index_weight[idx] = 0
        if self.label_smooth:
            self.need_target_onehot = True
        if target.dtype == str2dtype('long'):
            self.is_target_onehot = False
        elif target.dtype != str2dtype('long') and (target.min() >= 0 and target.max() <= 1 and abs(output_exp.sum(-1).mean() - 1) < 1e-4):
            target = clip(target, min=1e-8, max=1 - 1e-8)
            self.is_target_onehot = True

        # need target onehot but currently not
        if  target.dtype==tf.int64 and self.need_target_onehot == True and self.is_target_onehot == False:
            target = make_onehot(target, num_classes=self.num_classes, axis=self.axis)
            if self.label_smooth:
                target = target #* (torch.Tensor(target.size()).uniform_(0.9, 1))
                self.is_target_onehot = True
            target.require_grads=False

        # setting cutoff
        # if self.cutoff is not None:
        #     mask = (output > self.cutoff).to(output.dtype)
        #     output = output * mask
        return output, target

    def calculate_loss(self, output, target, **kwargs):
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
            loss = self.calculate_loss(*self.preprocess(output, target,**kwargs))
            loss = self._get_reduction(loss)
            return loss
        except Exception as e:
            print(e)
            PrintException()
            raise e



class _PairwiseLoss(Loss):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1,  reduction='batch_mean', name=None, **kwargs):
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
        super(_PairwiseLoss, self).__init__(reduction=reduction,axis=axis,name=name)
        self._set_name_scope()


        # initilize weight

    def _set_name_scope(self):
        """Creates a valid `name_scope` name."""
        if self.name is None:
            name = self.__class__.__name__
        elif self.name == '<lambda>':
            name = 'lambda'
        else:
            # E.g. '_my_loss' => 'my_loss'
            name = self.name.strip('_')
        with ops.name_scope_v2(name) as scope_name:
            self._name_scope = ops.name_scope_v2(scope_name)

    def _get_reduction(self, loss):
        with self._name_scope:
            if ndim(loss) <= 1 or self.reduction == 'none':
                return loss
            num_present = math_ops.cast(array_ops.size(loss, name='num_elements'), dtype=loss.dtype)
            batch_size=math_ops.cast(tf.constant(array_ops.shape(loss,name='shape')[0]), dtype=loss.dtype)

            if ndim(loss) >= 2 and self.reduction == 'batch_sum':
                total_loss = math_ops.div_no_nan(math_ops.reduce_sum(loss),batch_size, name='value')
                return loss.mean(1).sum()
            elif ndim(loss) >= 2 and self.reduction == 'batch_mean':
                total_loss = math_ops.reduce_sum(loss)
                return math_ops.div_no_nan(total_loss, math_ops.div_no_nan(num_present,batch_size), name='value')
            elif self.reduction in ('mean', 'batch_mean'):
                total_loss = math_ops.reduce_sum(loss)
                return math_ops.div_no_nan(total_loss, num_present, name='value')
            elif self.reduction == ('sum', 'batch_sum'):
                return math_ops.reduce_sum(loss)
            else:
                total_loss = math_ops.reduce_sum(loss)
                return math_ops.div_no_nan(total_loss, num_present, name='value')


    def flatten_check(self, output, target):
        "Check that `out` and `targ` have the same number of elements and flatten them."
        if ndim(output) > 2:
            output = output.view(-1, output.size(-1))
        if ndim(target) > 2:

            target = target.view(-1, target.size(-1))

        if len(output) == len(target):
            return output, target
        else:
            raise ValueError('output and target have diffent elements.')

    def preprocess(self, output: Tensor, target: Tensor, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        output, target = self.flatten_check(output, target)
        if output.shape == target.shape:
            return output, target

        elif target.dtype == tf.int64 and ndim(output) == ndim(target) + 1:
            num_class = int_shape(output)[self.axis]
            target = make_onehot(target, num_class, self.axis).float()
        return output, target

    def calculate_loss(self, output, target, **kwargs):
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

    def forward(self, output: Tensor, target: Tensor, **kwargs) -> 'loss':
        """

            Args:
                output (tf.Tensor):
                target (tf.Tensor):

            Returns:
                calculated loss

            """
        try:
            loss = self.calculate_loss(*self.preprocess(output, target,**kwargs))
            loss=self._get_reduction(loss)
            return loss
        except Exception as e:
            print(e)
            PrintException()
            raise e





class CrossEntropyLoss(_ClassificationLoss):
    """
    Calculate the cross entropy loss

    Examples:
    >>> output=to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]])
    >>> print(output.shape)
    (4, 3)
    >>> target=make_onehot([1,0,1,2],3,axis=-1)
    >>> print(target.shape)
    (4, 3)
    >>> CrossEntropyLoss(reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.1305245>
    >>> CrossEntropyLoss(reduction='sum')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=4.522098>
    >>> CrossEntropyLoss(label_smooth=True,reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.1305245>
    >>> CrossEntropyLoss(loss_weights=to_tensor([1.0,1.0,0]),reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.84725726>
    >>> CrossEntropyLoss(ignore_index=2,reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.84725726>




    """

    def __init__(self, axis=-1, sample_weight=None, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False,
                 reduction='mean', name='CrossEntropyLoss'):
        super().__init__(axis, sample_weight, from_logits, ignore_index, cutoff, label_smooth, reduction, name)
        self._built = True
        self.need_target_onehot=True

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # if self.is_logsoftmax == False:
        #     if not self.from_logits:
        #         output=softmax(output,self.axis)
        #     loss =tf.nn.sparse_softmax_cross_entropy_with_logits(target,output,self.sample_weight,name='weighted_cross_entropy')
        # else:
        #     loss= -cast(target,output.dtype)*output*self.sample_weight
        # return loss


        sample_weight = self.sample_weight * self.ignore_index_weight
        if ndim(output) == 2:
            pass
        else:
            reshape_shape = [1] * ndim(output)
            reshape_shape[self.axis] = self.num_classes
            sample_weight = self.sample_weight.reshape(reshape_shape)
            if self.is_logsoftmax == True:
                sample_weight = self.sample_weight.reshape(reshape_shape) * self.ignore_index_weight.reshape(reshape_shape)
        # -sum([p[i] * log2(q[i]) for i in range(len(p))])
        if self.is_logsoftmax == False:
            loss = -(target * log_softmax(output, axis=self.axis) * sample_weight)
        else:
            loss = -(target * output * sample_weight)
        return loss


class NLLLoss(_ClassificationLoss):
    """
    Calculate the cross entropy loss

    Examples:
    >>> output=to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]])
    >>> print(output.shape)
    (4, 3)
    >>> target=make_onehot([1,0,1,2],3,axis=-1)
    >>> print(target.shape)
    (4, 3)
    >>> NLLLoss(reduction='mean')(output,target)
    tensor(1.1034)
    >>> NLLLoss(reduction='sum')(output,target)
    tensor(4.4136)
    >>> NLLLoss(label_smooth=True,reduction='mean')(output,target)
    tensor(1.1034)
    >>> NLLLoss(loss_weights=to_tensor([1.0,1.0,0]),reduction='mean')(output,target)
    tensor(0.8259)
    >>> NLLLoss(ignore_index=2,reduction='mean')(output,target)
    tensor(0.8259)




    """

    def __init__(self, axis=-1, loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False,
                 reduction='mean', name='NLLLoss'):
        super().__init__(axis, loss_weights, from_logits, ignore_index, cutoff, label_smooth, reduction, name)
        self._built = True

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        reshape_shape = [1] * ndim(output)
        reshape_shape[self.axis] = self.num_classes
        loss = -target * output * reshape(self.loss_weights, reshape_shape)
        return loss


class F1ScoreLoss(_ClassificationLoss):
    """
    Calculate the cross entropy loss

    Examples:
    >>> output=to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]])
    >>> print(output.shape)
    (4, 3)
    >>> target=make_onehot([1,0,1,2],3,axis=-1)
    >>> print(target.shape)
    (4, 3)
    >>> F1ScoreLoss(reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.6669905>
    >>> F1ScoreLoss(reduction='sum')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=2.667962>
    >>> F1ScoreLoss(label_smooth=True,reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.6669905>
    >>> F1ScoreLoss(loss_weights=to_tensor([1.0,1.0,0]),reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.6669905>
    >>> F1ScoreLoss(ignore_index=2,reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.6669905>




    """

    def __init__(self,num_classes=None, beta=1, axis=-1, loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None,
                 label_smooth=False, reduction='mean', name='CrossEntropyLoss'):
        super().__init__(axis, loss_weights, from_logits, ignore_index, cutoff, label_smooth, reduction, name)
        self.beta = beta
        self._built = True
        self.num_classes=num_classes

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        reshape_shape = [1] * ndim(output)
        reshape_shape[self.axis] = self.num_classes
        correct_predictions = reduce_sum(output * target, axis=self.axis, keepdims=True)
        precision = correct_predictions / reduce_sum(output, axis=self.axis, keepdims=True)
        recall = correct_predictions / reduce_sum(target, axis=self.axis, keepdims=True)
        return 1 - (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall)


class FocalLoss(_ClassificationLoss):
    """
    Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
    threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """

    def __init__(self, alpha=0.5, gamma=2, normalized=False, threshold=None, axis=-1, loss_weights=None,
                 from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean',
                 name='FocalLoss'):
        super().__init__(axis, loss_weights, from_logits, ignore_index, cutoff, label_smooth, reduction, name)
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.normalized = normalized

    def calculate_loss(self, output, target, **kwargs):
        """


        Args:
            output: Tensor of arbitrary shape
            target: Tensor of the same shape as input


        Returns:

            """

        logp = CrossEntropyLoss()(output, target)
        p = exp(-logp)

        # compute the loss
        if self.threshold is None:
            focal_term = pow((1 - p), self.gamma)
        else:
            focal_term = pow(((1.0 - p) / self.threshold), self.gamma)
            focal_term[p < self.threshold] = 1

        loss = focal_term * logp

        if self.alpha is not None:
            loss = loss * (self.alpha * target + (1 - self.alpha) * (1 - target))
        if self.normalized:
            norm_factor = sum(focal_term)
            loss = loss / norm_factor

        return loss

class DiceLoss(_ClassificationLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    Args:
        axis (int): the position where the classes is.
        sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
        number of classes.
        from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
        ignore_index (int or list of int):
        cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
        less than 1..
        label_smooth (bool): Should use label smoothing?
        reduction (string): the method to aggrgate loss. None means no need to aggregate, 'mean' means average loss,
            'sum' means the summation of losses,'batch_mean' means average loss cross the batch axis then
            summation them.

    Examples:
    >>> output=zeros((1,3,128,128))
    >>> output[0,1,32:64,48:92]=1
    >>> output[0,2,12:32,56:64]=1
    >>> target=zeros((1,128,128)).long()
    >>> target[0,33:63,50:9]=1
    >>> target[0,13:35,52:65]=2
    >>> DiceLoss(reduction='mean')(output,target).cpu()
    tensor(0.8271)
    >>> DiceLoss(ignore_index=0,reduction='mean')(output,target).cpu()
    tensor(0.9829)



    Reference:
        https://arxiv.org/abs/1707.03237


    """

    def __init__(self, smooth=1., axis=-1, sample_weight=None, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean', name='DiceLoss'):
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

        super().__init__(axis, sample_weight, from_logits, ignore_index, cutoff, label_smooth, reduction, name)
        self.smooth = smooth
        self.is_logsoftmax = False
        self.need_target_onehot = True
        self.is_multiselection = False
        self._built = True

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        if self.is_logsoftmax:
            output=exp(output)
        reduce_axes=list(range(target.ndim))
        axis=self.axis if self.axis>=0 else target.ndim+self.axis
        reduce_axes.remove(0)
        reduce_axes.remove(axis)
        loss_weights=self.sample_weight.copy()
        # for k in range(target.ndim-self.loss_weights.ndim):
        #     loss_weights=loss_weights.expand_dims(0)
        intersection = reduce_sum(target * output, axis=reduce_axes)*loss_weights
        den1 = reduce_sum(output, axis=reduce_axes)*loss_weights
        den2 = reduce_sum(target, axis=reduce_axes)*loss_weights
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (den1 + den2 + self.smooth))
        return dice



class L1Loss(_PairwiseLoss):
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

     Function that takes the mean element-wise absolute value difference.

     See :class:`~torch.nn.L1Loss` for details.
     """
    def __init__(self, reduction='mean', name='L1Loss'):
        super(L1Loss, self).__init__(reduction)
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
        with self._name_scope:
            return tf.math.abs(output-target,name='l1_loss')



class L2Loss(_PairwiseLoss):
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

        Measures the element-wise mean squared error.

        See :class:`~torch.nn.MSELoss` for details.
        """
    def __init__(self, reduction='mean', name='MSELoss'):
        super(L2Loss, self).__init__(reduction)
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
        with self._name_scope:
            return 0.5*tf.math.square(output-target,name='l2_loss')


#
# class SmoothL1Loss(_PairwiseLoss):
#     r"""Function that uses a squared term if the absolute
#     element-wise error falls below 1 and an L1 term otherwise.
#
#     See :class:`~torch.nn.SmoothL1Loss` for details.
#     """
#     def __init__(self, reduction='mean', name='SmoothL1Loss'):
#         super(SmoothL1Loss, self).__init__(reduction=reduction)
#         self.name = name
#         self.reduction = reduction
#         self.huber_delta = 0.5
#
#     def calculate_loss(self, output, target, **kwargs):
#         """
#
#         Args:
#             output ():
#             target ():
#             **kwargs ():
#
#         Returns:
#
#         """
#         return smooth_l1_loss(output, target, reduction='none')
#


class MSELoss(_PairwiseLoss):
    """
    Calculate the MSE loss


    """

    def __init__(self, axis=-1,reduction='sum', name='MSELoss'):
        super().__init__(axis, reduction, name)
        self._built = True

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        #with self._name_scope:

            #num_present = tf.reduce_sum(math_ops.cast(array_ops.size(output, name='num_elements'), dtype=output.dtype),name='reduce_sum')
        with self._name_scope:
            return tf.nn.l2_loss(output-target)
        #return math_ops.div_no_nan(tf.nn.l2_loss(output-target), math_ops.cast(tf.shape(output)[0], dtype=output.dtype), name='value')


class WingLoss(_PairwiseLoss):
    def __init__(self, omega=10, epsilon=2, name='WingLoss'):
        super(WingLoss, self).__init__()
        self.name = name
        self.omega = omega
        self.epsilon = epsilon

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with self._name_scope:
            delta_y = tf.math.abs(target - output)
            delta_y1 = delta_y[delta_y < self.omega]
            delta_y2 = delta_y[delta_y >= self.omega]
            loss1 = self.omega * tf.math.log(1 + delta_y1 / self.epsilon)
            C = self.omega - self.omega * tf.math.log(1 + self.omega / self.epsilon)
            loss2 = delta_y2 - C
            return (tf.reduce_sum(loss1) + tf.reduce_sum(loss2)) / (len(loss1) + len(loss2))


class AdaptiveWingLoss(_PairwiseLoss):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, name='AdaptiveWingLoss'):
        super(AdaptiveWingLoss, self).__init__()
        self.name = name
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with self._name_scope:
            y = target
            y_hat = output
            delta_y =tf.math.abs (y - y_hat)
            delta_y1 = delta_y[delta_y < self.theta]
            delta_y2 = delta_y[delta_y >= self.theta]
            y1 = y[delta_y < self.theta]
            y2 = y[delta_y >= self.theta]
            loss1 = self.omega * tf.math.log(1 + tf.math.pow(delta_y1 / self.omega, self.alpha - y1))
            A = self.omega * (1 / (1 + tf.math.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
                tf.math.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
            C = self.theta * A - self.omega * tf.math.log(1 + tf.math.pow(self.theta / self.epsilon, self.alpha - y2))
            loss2 = A * delta_y2 - C
            return (tf.reduce_sum(loss1) + tf.reduce_sum(loss2)) / (len(loss1) + len(loss2))





class EdgeLoss(object):
    def __init__(self, name='EdgeLoss'):
        self.name = name
        super(EdgeLoss, self).__init__()

    def first_order(self, x, axis=2):
        h, w = x.shape[1:3]
        if axis == 1:
            return tf.math.abs((x[:, :h - 1, :w - 1, :] - x[:, 1:, :w - 1, :]))
        elif axis == 2:
            return tf.math.abs(x[:, :h - 1, :w - 1, :] - x[:, :h - 1, 1:, :])
        else:
            return None

    def call(self, y_true, y_pred):
        loss1 = tf.reduce_mean(tf.math.square(self.first_order(y_pred, 1) - self.first_order(y_true, 1)))
        loss2 = tf.reduce_mean(tf.math.square(self.first_order(y_pred, 2) - self.first_order(y_true, 2)))
        return loss1 + loss2


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

