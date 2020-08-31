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


__all__ = ['get_loss','_ClassificationLoss', 'CrossEntropyLoss', 'MSELoss', 'EdgeLoss', 'NLLLoss', 'F1ScoreLoss', '_ClassificationLoss',
           'FocalLoss']





class _ClassificationLoss(Layer):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1, loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None,
                 label_smooth=False, reduction='mean', name=None, **kwargs):
        """
        Args:
            axis (int): the position where the classes is.
            loss_weights (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
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
            is_multiselection (bool): If True, means the classification model is multi-selection, so cannot use  any softmax process, use sigmoid and binary_crosss_entropy insteaded.
            is_target_onehot (bool):  If True, means we have confirmed (not just declare) the target is transformed as  one-hot format
            reduction(str): The aggregation function for loss, available options are 'sum', 'mean 'and 'batch_mean', default is 'mean'
            axis (None or int): The axis we according with for loss calculation. Default is 1.
            from_logits (bool):If True, means  the sum of all probability will equal 1.
            is_logsoftmax (bool):If True, means model  use SoftMax as last layer or use any equivalent calculation.
            loss_weights(1D tensor):The loss weight for all classes.
            ignore_index(int , list, tuple): The classes we want to ignore in the loss calculation.
            cutoff(float): Means the decision boundary in this classification model, default=0.5.
            num_classes(int):number of  all the classes.
            label_smooth (bool):If True, mean we will apply label-smoothing in loss calculation.

        """
        super(_ClassificationLoss, self).__init__(name=name)
        self.need_target_onehot = True
        self.is_multiselection = False
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

        # if self.is_logsoftmax:
        #     output = clip(output,max=-1e-8)
        # elif self.from_logits:
        #     output = clip(output, min=1e-8, max=1 - 1e-8)

        if (output.min() >= 0 and output.max() <= 1 and abs(output.sum(-1).mean() - 1) < 1e-4):
            self.from_logits = True
            output = clip(output, min=1e-8, max=1 - 1e-8)

        elif (output_exp.min() >= 0 and output_exp.max() <= 1 and abs(output_exp.sum(-1).mean() - 1) < 1e-4):
            self.is_logsoftmax = True
            self.from_logits = True
            output = clip(output,  max= - 1e-8)
        else:
            output = clip(softmax(output, self.axis), epsilon(), 1.0 - epsilon())
            self.from_logits = True
            output = clip(output, min=1e-8, max=1 - 1e-8)

        # initilize weight
        if self.loss_weights is not None and len(self.loss_weights) != self.num_classes:
            raise ValueError('weight should be 1-D tensor and length equal to numbers of filters')
        if self.loss_weights is None:
            self.loss_weights = ones(self.num_classes,requires_grad=False)
        else:
            self.loss_weights = to_tensor(self.loss_weights,requires_grad=False)

        # ignore_index

        if isinstance(self.ignore_index, int) and 0 <= self.ignore_index < int_shape(output)[self.axis]:
            self.loss_weights[self.ignore_index] = 0
        elif isinstance(self.ignore_index, (list, tuple)):

            for idx in self.ignore_index:
                if isinstance(idx, int) and 0 <= idx < int_shape(output)[self.axis]:
                    self.loss_weights[idx] = 0
        if self.label_smooth:
            self.need_target_onehot = True

        if target.dtype != str2dtype('int') and (target.min() >= 0 and target.max() <= 1 and abs(output_exp.sum(-1).mean() - 1) < 1e-4):
            target = clip(target, min=1e-8, max=1 - 1e-8)
            self.is_target_onehot = True

        # need target onehot but currently not
        if self.need_target_onehot == True and cast((target > 1),'float32').sum() > 0:
            target = make_onehot(target, num_classes=self.num_classes, axis=self.axis)
            if self.label_smooth:
                target = target * to_tensor(np.random.uniform(0.9, 1,target.shape))
                self.need_target_onehot = True
                self.is_target_onehot = True

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

    def __init__(self, axis=-1, loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False,
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
        if self.is_logsoftmax == False:
            if not self.from_logits:
                output=softmax(output,self.axis)
            loss =tf.nn.weighted_cross_entropy_with_logits(target,output,self.loss_weights)
        else:

            loss= -reduce_sum(target * output * self.loss_weights.expand_dims(0), axis=self.axis, keepdims=True)
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

