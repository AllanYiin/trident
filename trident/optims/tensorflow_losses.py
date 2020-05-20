from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

from trident.backend.common import camel2snake, get_class, epsilon
from trident.backend.tensorflow_backend import Layer
from trident.backend.tensorflow_ops import *

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


__all__ = ['get_loss', 'CrossEntropyLoss', 'MSELoss', 'EdgeLoss', 'NLLLoss', 'F1ScoreLoss', '_ClassificationLoss',
           'FocalLoss']


def make_onehot(labels, classes, axis=-1):
    """

    Args:
        labels ():
        classes ():
        axis ():

    Returns:

    """
    return tf.one_hot(indices=labels, depth=classes, on_value=1.0, off_value=0.0, axis=axis)


class _ClassificationLoss(Layer):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1, loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None,
                 label_smooth=False, reduction='mean', name=None, **kwargs):
        """
        Args:
            axis (int): the position where the classes is.
            loss_weights (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
            number of classes.
            from_logits (bool): wheather the output tensor is normalized as a probability (total equal to 1)
            ignore_index (int or list of int):
            cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
            less than 1..
            is_target_onehot (bool): Is the target tensor in onehot format?
            label_smooth (bool): Should use label smoothing?
            reduction (string): the method to aggrgate loss. None means no need to aggregate, 'mean' means average loss,
                'sum' means the summation of losses,'batch_mean' means average loss cross the batch axis then
                summation them.

        """
        super(_ClassificationLoss, self).__init__(name=name)
        self.need_target_onehot = True

        self.reduction = reduction
        self.axis = axis
        self.from_logits = from_logits
        self.is_logsoftmax = False
        self.loss_weights = loss_weights
        self.ignore_index = ignore_index
        if cutoff is not None and not 0 < cutoff < 1:
            raise ValueError('cutoff should between 0 and 1')
        self.cutoff = cutoff
        self.num_classes = None
        self.label_smooth = label_smooth

    def preprocess(self, output: tf.Tensor, target: tf.Tensor, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # check num_clases
        if self.num_classes is None:
            self.num_classes = output.shape[self.axis]

        output_exp = exp(output)

        if self.is_logsoftmax:
            output = clip(output, max=-1e-8)
        elif self.from_logits:
            output = clip(output, min=1e-8, max=1 - 1e-8)
        elif (reduce_min(output) >= 0 and reduce_max(output) <= 1 and reduce_mean(
                abs(reduce_sum(output, self.axis) - 1)) < 1e-4):
            self.from_logits = True
            output = clip(output, min=1e-8, max=1 - 1e-8)
        elif (reduce_min(output_exp) >= 0 and reduce_max(output_exp) <= 1 and reduce_mean(
                abs(reduce_sum(output_exp, self.axis) - 1)) < 1e-4):
            self.is_logsoftmax = True
            self.from_logits = True
            output = clip(output, max=-1e-8)
        else:
            output = clip(softmax(output, self.axis), epsilon(), 1.0 - epsilon())
            self.from_logits = True
            output = clip(output, min=1e-8, max=1 - 1e-8)

        # initilize weight
        if self.loss_weights is not None and len(self.loss_weights) != self.num_classes:
            raise ValueError('weight should be 1-D tensor and length equal to numbers of filters')
        if self.loss_weights is None:
            self.loss_weights = ones(self.num_classes)
        else:
            self.loss_weights = to_tensor(self.loss_weights)

        # ignore_index
        if isinstance(self.ignore_index, int) and 0 <= self.ignore_index < int_shape(output)[self.axis]:
            filter = np.ones(int_shape(self.loss_weights))
            filter[self.ignore_index] = 0
            self.loss_weights = self.loss_weights * to_tensor(filter)
        elif isinstance(self.ignore_index, (list, tuple)):
            for idx in self.ignore_index:
                if isinstance(idx, int) and 0 <= idx < int_shape(output)[self.axis]:
                    self.loss_weights[idx] = 0
        if self.label_smooth:
            self.need_target_onehot = True
        # need target onehot but currently not
        if self.need_target_onehot == True and reduce_sum(cast((target > 1), 'float')) > 0:
            target = make_onehot(target, classes=self.num_classes, axis=self.axis)
            if self.label_smooth:
                target = target * to_tensor(np.random.uniform(0.9, 1, int_shape(target)))

        # setting cutoff
        if self.cutoff is not None:
            mask = (output > self.cutoff)
            output = output * mask
        return output, target

    def calculate_loss(self, output: tf.Tensor, target: tf.Tensor, **kwargs):
        """ Calculate the unaggregate loss.
        The loss function calculation logic should define here., please dont't aggregate the loss in this phase.

        Args:
            output (tf.Tensor):
            target (tf.Tensor):
        """
        ##dont do aggregation
        raise NotImplementedError

    def postprocess(self, loss):
        """Process the final losss aggregation

        Args:
            loss (tf.Tensor): the unaggregate loss.

        Returns:
            aggregated loss.

        """
        if self.reduction == 'mean':
            return reduce_mean(loss)
        elif self.reduction == 'sum':
            return reduce_sum(loss)
        elif self.reduction == 'batch_mean':
            axes = range(0, len(loss))
            return reduce_sum(reduce_mean(loss, axes[1:]))

    def forward(self, output: tf.Tensor, target: tf.Tensor, **kwargs):
        """

        Args:
            output (tf.Tensor):
            target (tf.Tensor):

        Returns:
            calculated loss

        """
        loss = self.calculate_loss(*self.preprocess(output, target))
        loss = self.postprocess(loss)
        return loss


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

    def __init__(self, axis=1, loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False,
                 reduction='mean', name='CrossEntropyLoss'):
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
        if self.is_logsoftmax == False:
            loss = -reduce_sum(target * log(output) * reshape(self.loss_weights, reshape_shape), axis=self.axis)
        else:
            loss = -reduce_sum(target * output * reshape(self.loss_weights, reshape_shape), axis=self.axis)
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

    def __init__(self, axis=1, loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False,
                 reduction='mean', name='CrossEntropyLoss'):
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

    def __init__(self, beta=1, axis=1, loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None,
                 label_smooth=False, reduction='mean', name='CrossEntropyLoss'):
        super().__init__(axis, loss_weights, from_logits, ignore_index, cutoff, label_smooth, reduction, name)
        self.beta = beta
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

    def __init__(self, alpha=0.5, gamma=2, normalized=False, threshold=None, axis=1, loss_weights=None,
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




def MSELoss(output, target):
    return tf.reduce_mean((square(output - target)))


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

