from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

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
from trident.backend.common import dtype as Dtype

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


__all__ = ['get_loss','_ClassificationLoss', 'CrossEntropyLoss', 'MSELoss', 'EdgeLoss', 'NLLLoss', 'F1ScoreLoss', '_ClassificationLoss',
           'FocalLoss','DiceLoss','L1Loss','L2Loss','SmoothL1Loss','WingLoss','AdaptiveWingLoss']





class _ClassificationLoss(Loss):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1, sample_weight=None,auto_balance=False, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name=None, **kwargs):
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
        super(_ClassificationLoss, self).__init__(reduction=reduction, sample_weight=sample_weight,axis=axis,enable_ohem=enable_ohem, ohem_ratio=ohem_ratio, namename=name)
        self._set_name_scope()
        self.need_target_onehot = True
        self.is_multiselection = False
        self.is_target_onehot = False
        self.from_logits = from_logits
        self.is_logsoftmax = False
        self.ignore_index = ignore_index
        self.ignore_index_weight=None
        self.auto_balance = auto_balance
        if self.auto_balance:
            self.running_statistics = None

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
            if ndim(loss) ==0 or self.reduction == 'none':
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
        if ndim(output) > 2 and ndim(output) == ndim(target)+1 :
            shp = int_shape(output)
            output = output.reshape((shp[0], -1, shp[-1]))
            target = cast(target.reshape((shp[0], -1)),'int64')
            return output, target
        elif ndim(output) > 2 and ndim(output) == ndim(target):
            shp = int_shape(output)
            output = output.reshape((shp[0], -1, shp[-1]))
            if ndim(target) > 2 :
                target = target.reshape((shp[0], -1, shp[-1]))
            return output, target
        elif ndim(output) == 2 and ndim(output) == ndim(target):
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
            if self.auto_balance and self.label_statistics is not None:
                output=output*to_tensor(self.label_statistics.copy())
            output = clip(output, min=1e-8, max=1 - 1e-8)

        elif (ndim(output) >=1 and 'float' in str(output.dtype) and output_exp.min() >= 0 and output_exp.max() <= 1 ):
            self.is_logsoftmax = True
            self.from_logits = True
            if self.auto_balance and self.label_statistics is not None:
                output = output +to_tensor(np.log(self.label_statistics.copy()))
            output = clip(output, max=- 1e-8)
        else:
            self.is_logsoftmax = False
            self.from_logits = False

        self.ignore_index_weight=np.ones_like(self.sample_weight)
        # ignore_index

        if isinstance(self.ignore_index, int) and 0 <= self.ignore_index < self.num_classes:
            self.ignore_index_weight[self.ignore_index] = 0
        elif isinstance(self.ignore_index, (list, tuple)):
            for idx in self.ignore_index:
                if isinstance(idx, int) and 0 <= idx < int_shape(output)[self.axis]:
                    self.ignore_index_weight[idx] = 0
        self.ignore_index_weight=to_tensor(self.ignore_index_weight,dtype=output.dtype,device=output.device)
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

    def __init__(self, axis=-1,  reduction='batch_mean',enable_ohem=False, ohem_ratio=3.5,name=None, **kwargs):
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
        super(_PairwiseLoss, self).__init__(reduction=reduction,axis=axis,enable_ohem=enable_ohem, ohem_ratio=ohem_ratio,name=name)
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
            if ndim(loss) == 0 or self.reduction == 'none':
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
            output = reshape(output,(-1,int_shape(output)[-1]))
        if ndim(target) > 2:
            target =reshape(target,(-1,int_shape(target)[-1]))

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

    def _do_ohem(self, output: Tensor, target: Tensor):
        if self.enable_ohem:
            output_ = output.clone()
            target_ = target.clone()
            num_hard = 0
            num_easy = 0
            hard_mask = None
            reduce_axis = list(range(output_.ndim))[1:]
            base_losses = pow(output_ - target, 2).mean(axis=reduce_axis) if len(reduce_axis) > 0 else pow(output_ - target, 2)
            if target.dtype == Dtype.int64:
                hard_mask = target < 0
                num_hard = reduce_sum(hard_mask).numpy()
                num_easy = int(self.ohem_ratio * num_hard)
            elif target.shape == output.shape:
                hard_mask = target < 0
                num_hard =reduce_sum(hard_mask).numpy()
                num_easy = int(self.ohem_ratio * num_hard)

            if num_hard == 0:
                return output, target
            base_losses[hard_mask] = math.inf

            easy_cases = topk(base_losses, k=clip(int(num_easy + num_hard),1,len(base_losses)))
            idxs = easy_cases

            output_hn = output.index_select(0, idxs)
            target_hn = target.index_select(0, idxs)
            return output_hn, target_hn
        else:
            return output, target

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
    <tf.Tensor: shape=(), dtype=float32, numpy=1.1305244>
    >>> CrossEntropyLoss(reduction='sum')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=4.5220976>
    >>> CrossEntropyLoss(label_smooth=True,reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.1305244>
    >>> CrossEntropyLoss(sample_weight=to_tensor([1.0,1.0,0.5]),reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.9888908>
    >>> CrossEntropyLoss(ignore_index=2,reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.84725726>




    """

    def __init__(self, axis=-1, sample_weight=None, auto_balance=False,from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False,
                 reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name='CrossEntropyLoss'):
        super().__init__(axis, sample_weight,auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction, enable_ohem,ohem_ratio,name)
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
        # if self.sample_weight is not None and ndim(self.sample_weight)==1:
        #     self.sample_weight=expand_dims(self.sample_weight,0)
        # if self.ignore_index_weight is not None and ndim(self.ignore_index_weight)==1:
        #     self.ignore_index_weight=expand_dims(self.ignore_index_weight,0)
        with self._name_scope:
            sample_weight = cast(self.sample_weight,output.dtype)*cast( self.ignore_index_weight,output.dtype)
            if ndim(output) == 2:
                pass
            else:
                reshape_shape = [1] * ndim(output)
                reshape_shape[self.axis] = self.num_classes
                sample_weight = sample_weight.reshape(reshape_shape)

            # -sum([p[i] * log2(q[i]) for i in range(len(p))])
            if not self.is_logsoftmax:
                loss = -reduce_sum(target * log_softmax(output, axis=self.axis) * sample_weight,axis=-1)
            else:

                loss = -reduce_sum(target * output * sample_weight,axis=-1)
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

    def __init__(self, axis=-1, sample_weight=None,auto_balance=False, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False,
                 reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name='NLLLoss'):
        super().__init__(axis, sample_weight, auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction, enable_ohem,ohem_ratio,name)
        self._built = True

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with self._name_scope:
            reshape_shape = [1] * ndim(output)
            reshape_shape[self.axis] = self.num_classes
            loss = reduce_sum(-target * output * reshape(tf.stop_gradient(self.sample_weight), reshape_shape),axis=-1)
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

    def __init__(self, beta=1, axis=-1, sample_weight=None,auto_balance=False, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name='CrossEntropyLoss'):
        super().__init__(axis, sample_weight, auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction, enable_ohem,ohem_ratio,name)
        self.beta = beta

        self.need_target_onehot = True
        self._built = True

    def calculate_loss(self, output, target, **kwargs):

        if self.is_logsoftmax:
            output = clip(exp(output), 1e-8, 1 - 1e-8)
            self.from_logits = True
        if not self.from_logits:
            output = softmax(output, self.axis)
        if target.dtype == tf.int64 or self.is_target_onehot == False:
            target = cast(make_onehot(target, self.num_classes, axis=1),output.dtype)

        tp = (target * output*self.sample_weight * self.ignore_index_weight).sum(axis=self.axis)
        # tn = ((1 - target) * (1 - output))
        # fp = ((1 - target) * output)
        # fn = (target * (1 - output))
        precision =true_divide(tp ,reduce_sum(output,axis=self.axis))
        recall =true_divide(tp ,reduce_sum(target,axis=self.axis))
        return 1-(1 + self.beta ** 2) * precision * recall /( self.beta ** 2*precision + recall)


class FocalLoss(_ClassificationLoss):
    """
    Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
    threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """

    def __init__(self, alpha=0.5, gamma=2, normalized=False, threshold=None, axis=-1, sample_weight=None,auto_balance=False,
                 from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_ratio=3.5 , name='FocalLoss'):
        super().__init__(axis, sample_weight,auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction, enable_ohem,ohem_ratio,name)
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
        with self._name_scope:
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

    def __init__(self, smooth=1., axis=-1, sample_weight=None,auto_balance=False, from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name='DiceLoss'):
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

        super().__init__(axis, sample_weight,auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction, enable_ohem,ohem_ratio,name)
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
        with self._name_scope:
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
    def __init__(self, reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name='L1Loss'):
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
    def __init__(self, reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name='MSELoss'):
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
            return tf.nn.l2_loss(output-target,name='l2_loss')


#
# class SmoothL1Loss(_PairwiseLoss):
#     r"""Function that uses a squared term if the absolute
#     element-wise error falls below 1 and an L1 term otherwise.
#
#     See :class:`~torch.nn.SmoothL1Loss` for details.
#     """
#     def __init__(self, reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name='SmoothL1Loss'):
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



class SmoothL1Loss(_PairwiseLoss):
    r"""Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """
    def __init__(self, reduction='mean' , enable_ohem=False, ohem_ratio=3.5, name='SmoothL1Loss'):
        super(SmoothL1Loss, self).__init__(enable_ohem=enable_ohem, ohem_ratio=ohem_ratio,reduction=reduction,name=name)
        self.name = name
        self.reduction = reduction
        self.huber_delta = 0.5

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with self._name_scope:
            target = math_ops.cast(target, dtype=output.dtype)
            diff = abs(target - output)
            less_than_one = cast(less(diff, 1.0), tf.float32)  # Bool to float32
            smooth_l1_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)  # 同上图公式

        return smooth_l1_loss



class WingLoss(_PairwiseLoss):
    def __init__(self, omega=10, epsilon=2, reduction='mean',enable_ohem=False, ohem_ratio=3.5, name='WingLoss'):
        super(WingLoss, self).__init__(enable_ohem=enable_ohem, ohem_ratio=ohem_ratio,reduction=reduction,name=name)
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
        delta_y = (target - output).abs()
        c = self.omega * (1.0 - log(1.0 + self.omega / self.epsilon))

        losses = where(
            greater(delta_y, self.omega, dtype=Dtype.bool),
            self.omega * log(1.0 + delta_y / self.epsilon),
            delta_y- c
        )


        return reduce_mean(losses,[1,2])

class AdaptiveWingLoss(_PairwiseLoss):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1  , enable_ohem=False, ohem_ratio=3.5, name='AdaptiveWingLoss'):
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





class EdgeLoss(_PairwiseLoss):
    def __init__(self,  enable_ohem=False, ohem_ratio=3.5,name='EdgeLoss'):
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

    def call(self, target, output):
        with self._name_scope:
            loss1 = tf.reduce_mean(tf.math.square(self.first_order(output, 1) - self.first_order(target, 1)))
            loss2 = tf.reduce_mean(tf.math.square(self.first_order(output, 2) - self.first_order(target, 2)))
            return loss1 + loss2

class IoULoss(Loss):
    def __init__(self, ignore_index=-1000, reduction='mean'):
        super(IoULoss, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def calculate_loss(self, output, target, **kwargs):
        output = argmax(output, 1)
        output_flat = output.reshape(-1)
        target_flat = target.reshape(-1)
        output_flat = output_flat[target_flat != self.ignore_index]
        target_flat = target_flat[target_flat != self.ignore_index]
        output_flat = output_flat[target_flat != 0]
        target_flat = target_flat[target_flat != 0]
        intersection = (output_flat == target_flat).to(output.dtype).sum()
        union = ((output_flat + target_flat) > 0).to(output.dtype).sum().clamp(min=1)
        loss = -(intersection / union).log()
        return loss


class SoftIoULoss(Loss):
    def __init__(self, n_classes, reduction="mean", reduced=False):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes
        self.reduction = reduction
        self.reduced = reduced

    def calculate_loss(self, output, target, **kwargs):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(output)

        pred =tf.math.softmax(output, dim=1)
        target_onehot = make_onehot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()



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

