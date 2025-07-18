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
from trident.backend.pytorch_ops import *
import trident.backend.pytorch_ops as ops
from trident.backend.tensorspec import ObjectType, object_type_inference
from trident.data.dataset import *
from trident.layers.pytorch_activations import sigmoid
from trident.layers.pytorch_layers import Dense
from trident.optims.losses import Loss, _check_logsoftmax_logit, _check_logit,_check_softmax

_device = get_device()
__all__ = ['_ClassificationLoss', 'MSELoss', 'CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'F1ScoreLoss', 'L1Loss',
           'SmoothL1Loss', 'L2Loss', 'CosineSimilarityLoss',
           'ExponentialLoss', 'ItakuraSaitoLoss', 'MS_SSIMLoss', 'DiceLoss', 'GeneralizedDiceLoss','ActiveContourLoss', 'WingLoss',
           'AdaptiveWingLoss', 'KLDivergenceLoss',
           'IoULoss', 'FocalLoss', 'SoftIoULoss', 'CenterLoss', 'TripletLoss', 'TripletMarginLoss',
           'LovaszSoftmax', 'PerceptionLoss', 'EdgeLoss', 'TransformInvariantLoss', 'get_loss']

ctx = context._context()
_float_dtype = Dtype.float16 if ctx.amp_available == True and ctx.is_autocast_enabled == True and get_device() == 'cuda' else Dtype.float32


def _calculate_loss_unimplemented(self, output: Tensor, target: Tensor) -> None:
    raise NotImplementedError



def _nlp_vercabs_unique_value_process(uniques,counts):
    uniques_array = to_numpy(uniques)
    counts_array = to_numpy(counts)
    counts_array = np.array([builtins.max(counts_array[uniques.index(i)], 1) if i in uniques else 1 for i in
                             range(int(uniques_array.max()) + 1)]).astype(np.float32)
    uniques_array = np.array(list(range(int(uniques_array.max()) + 1)))


    order_index = np.argsort(-1 * counts_array)
    sorted_ratio = counts_array[order_index] / counts_array.sum()
    soreted_uniques = uniques_array[order_index]
    sorted_ratio_cumsum = np.cumsum(sorted_ratio)
    threshold_value=np.array([n for  n in  range(len(sorted_ratio_cumsum)) if (sorted_ratio_cumsum[n]>0.5 and sorted_ratio_cumsum[n-1]<0.5) or (sorted_ratio_cumsum[n]>0.995 and sorted_ratio_cumsum[n-1]<0.995)])
    threshold_uniques=soreted_uniques[threshold_value]
    threshold_counts=counts_array[order_index][threshold_value]
    threshold_ratio = sorted_ratio[threshold_value]
    threshold_cumsum = sorted_ratio_cumsum[threshold_value]

    reweights1 = np.sqrt(counts_array.mean() / (counts_array))
    reweights0 = counts_array.sum() / (counts_array * len(counts_array))
    reweights=reweights1.copy()
    reweights[counts_array >=threshold_counts[0]] = reweights0[counts_array >=threshold_counts[0]]
    reweights[counts_array <= 10] =1
    reweights[0]=0.01
    #的

    return reweights.astype(np.float32), OrderedDict(zip(uniques_array, counts_array))


def _class_unique_value_process(uniques,counts):
    uniques_array = to_numpy(uniques)
    counts_array = to_numpy(counts)
    counts_array=np.array([builtins.max(counts_array[uniques.index(i)],1) if i in uniques else 1  for i in range(int(uniques_array.max())+1)]).astype(np.float32)
    uniques_array=np.array(list(range(int(uniques_array.max())+1)))
    reweights=1/sqrt(counts_array)
    reweights=reweights/(reweights.sum())*len(reweights)
    return reweights,OrderedDict(zip(uniques_array, counts_array))




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
        ctx = context._context()
        inferred_target_object_type = object_type_inference(target)
        if hasattr(ctx._thread_local_info, 'data_providers') and len(ctx._thread_local_info.data_providers) > 0:
            with torch.no_grad():
                dp = list(ctx._thread_local_info.data_providers.values())[-1]
                ds = None
                if dp.traindata.label is None:
                    pass
                elif isinstance(dp.traindata.label, ZipDataset):
                    for dp_ds in dp.traindata.label.items:
                        if dp_ds.symbol in self.signature.inputs:
                            ds = dp_ds
                        # maybe duplicate
                        elif dp_ds.object_type in self.valid_target_object_type and dp_ds.object_type == inferred_target_object_type:
                            ds = dp_ds
                else:
                    ds = dp.traindata.label
                if ds is None:
                    pass
                elif ds is not None:
                    if hasattr(ds, '_label_statistics') and ds._label_statistics is not None:
                        if not isinstance(ds,TextSequenceDataset):
                            self.label_statistics ,_= _class_unique_value_process(list(ds._label_statistics.keys()), list(ds._label_statistics.values()))
                        else:
                            self.label_statistics, _ =  _nlp_vercabs_unique_value_process(list(ds._label_statistics.keys()),     list(ds._label_statistics.values()))
                        if  self.label_statistics is not None:
                            self.sample_weight = to_tensor(self.label_statistics.copy()).detach()
                    elif isinstance(ds, LabelDataset) or dp.traindata.label.object_type in [ObjectType.classification_label]:
                        print('Start retrive label class distribution for auto-balance in loss function.')
                        unique, counts = np.unique(np.array([dp.traindata.label.items[i] for i in tqdm(range(len(dp.traindata.label.items)))]),
                            return_counts=True)
                        ctx.print('')
                        reweights,label_statistics=_class_unique_value_process(unique.tolist(), counts.tolist())
                        self.label_statistics = reweights
                        self.sample_weight = to_tensor(reweights).detach()
                        ds._label_statistics= label_statistics

                        del unique
                        del counts

                    elif isinstance(ds, BboxDataset) or dp.traindata.label.object_type in [ObjectType.absolute_bbox,ObjectType.relative_bbox]:
                        ctx.print('Start retrive label class distribution for auto-balance in loss function.')
                        unique, counts = torch.unique(to_tensor(
                            np.concatenate([dp.traindata.label[i][:, 4] for i in tqdm(range(len(ds.items)))], axis=0),
                            dtype=Dtype.long, device='cpu'), return_counts=True)
                        ctx.print('')
                        reweights, label_statistics = _class_unique_value_process(unique, counts)
                        self.label_statistics = reweights
                        self.sample_weight =to_tensor(reweights).detach()
                        ds._label_statistics = label_statistics
                        del unique
                        del counts

                    elif isinstance(ds, MaskDataset) or dp.traindata.label.object_type in [ObjectType.label_mask,
                                                                                           ObjectType.color_mask,
                                                                                           ObjectType.binary_mask]:
                        ctx.print('Start retrive label class distribution for auto-balance in loss function.')
                        sample_base=list(range(len(ds)))
                        if len(sample_base)>1000:
                            np.random.shuffle(sample_base)
                            sample_base=sample_base[:1000]
                        overall_unique=OrderedDict()
                        unique_results =[dict(zip(*np.unique(ds[i], return_counts=True))) for i in tqdm(sample_base)]
                        for d in unique_results:
                            for k,v in d.items():
                                if k not in overall_unique:
                                    overall_unique[k]=v
                                else:
                                    overall_unique[k]+=v

                        unique=list(sorted(overall_unique.key_list))
                        counts=[overall_unique[k] for k in unique]

                        # unique = to_list(to_numpy(unique))
                        # counts = to_numpy(counts)

                        ctx.print('')
                        reweights, label_statistics = _class_unique_value_process(unique, counts)

                        self.label_statistics = reweights
                        self.sample_weight =to_tensor(reweights).detach()
                        ds._label_statistics = label_statistics
                        del unique
                        del counts

                    elif isinstance(ds, TextSequenceDataset):
                        keys = [ds.text2index[k] for k in list(ds.vocabs_frequency.keys()) if k in ds.vocabs]
                        reweights, label_statistics = _nlp_vercabs_unique_value_process(keys, ds.vocabs_frequency.value_list)
                        self.label_statistics = reweights
                        self.sample_weight = to_tensor(reweights).detach()
                        ds._label_statistics = label_statistics


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
            if len(output)==1 or (len(output)==2 and output.shape[1]==1):
                self.num_classes=2
            else:
                self.num_classes = int_shape(output)[self.axis]

        if self.sample_weight is None:
            self.sample_weight = ones(self.num_classes, requires_grad=False).to(get_device())
        else:
            self.sample_weight = to_tensor(self.sample_weight).detach().to(output.device)
        if len(self.sample_weight) != self.num_classes:
            raise ValueError('weight should be 1-D tensor and length equal to numbers of filters')

        self.ignore_index_weight = ones(self.num_classes, requires_grad=False, dtype=output.dtype).to(get_device())
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

    def preprocess(self, output: Tensor, target: Tensor, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # print(2, 'output', output.shape, 'abnormal:', any_abnormal_number(output))
        # check num_clases
        self.is_target_onehot = None
        self.from_logits = None
        if len(output) == 0:
            return to_tensor(0.0)

        if output.max() > 0 or self.is_logsoftmax is None or not self.is_logsoftmax:
            if reduce_max(output) >= 0:
                self.is_logsoftmax = False
                if _check_logit(output, self.axis):
                    self.from_logits = True
                    #output = clip(output, min=1e-7, max=1)
                else:
                    self.from_logits = False
            elif _check_logsoftmax_logit(output, self.axis):
                self.is_logsoftmax = True
                self.from_logits = True
                # output = clip(output,min=-12 ,max=-1e-7)

            else:
                self.is_logsoftmax = False
                self.from_logits = False

            if (ndim(target) == ndim(output) and 'float' in str(
                    target.dtype) and target.min() >= 0 and target.max() <= 1):
                self.is_target_onehot = True
        else:
            pass
            # output = clipoutput, min=-12, max=-1e-7)

        # need target onehot but currently not
        if target.dtype == str2dtype('long') or target.dtype == str2dtype('int64'):
            self.is_target_onehot = False
        elif target.dtype != str2dtype('long') and (target.min() >= 0 and target.max() <= 1 and abs(target.sum(self.axis) - 1).mean() < 1e-4):
            self.is_target_onehot = True

        if target.dtype == Dtype.long and self.need_target_onehot == True and (self.is_target_onehot == False or self.is_target_onehot is None):
            target = make_onehot(target, num_classes=self.num_classes, axis=self.axis).to(get_device())
            self.is_target_onehot = True
        if self.label_smooth:
            target = target * random_uniform_like(target, 0.8, 1).to(target.device)
        target = target.detach()



        # setting cutoff
        # if self.cutoff is not None:
        #     mask = (output > self.cutoff).to(output.dtype)
        #     output = output * mask
        # print(8, 'output', output.shape, 'abnormal:', any_abnormal_number(output))

        return output, target

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

    def _handel_abnormal(self, loss):
        if any_abnormal_number(loss):
            sys.stderr.write(
                '{0} has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(
                    self.name))
            loss = where(is_abnormal_number(loss), zeros_like(loss), loss)
        return loss

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

    def _do_ohem(self, loss: Tensor):
        if self.enable_ohem and ndim(loss)>0:
            loss=loss.view(-1)
            loss, _ = torch.sort(loss, descending=True)

            n_min=int(len(loss)*self.ohem_thresh)

            # if loss[self.n_min] > self.ohem_thresh:
            #     loss = loss[loss > self.ohem_thresh]
            # else:
            loss = loss[:n_min]
            return loss
        else:
            return loss


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


    def preprocess(self, output: Tensor, target: Tensor, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # check num_clases
        # output, target = self.flatten_check(output, target)
        if self.sample_weight is None or len(self.sample_weight) != int_shape(output)[0]:
            self.sample_weight = ones((int_shape(output)[0]), dtype=output.dtype)
        if isinstance(target, numbers.Number):
            target = ones_like(output) * target

        if output.shape == target.shape:
            return output, target
        elif target.dtype == Dtype.int64 and ndim(output) == ndim(target) + 1:
            num_class = int_shape(output)[self.axis]
            target = make_onehot(target, num_class, self.axis).float()


        return output, target


    def calculate_loss(self, output, target, **kwargs):
        """ Calculate the unaggregate loss.
        The loss function calculation logic should define here., please don't aggregate the loss in this phase.

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
            if not self.built:
                self.build(output, target)

            loss = self.calculate_loss(*self.preprocess(output, target, **kwargs))
            if self.enable_ohem:
                loss=self._do_ohem(loss)
            loss = self._handel_abnormal(loss)
            loss = self._get_reduction(loss)
            return loss
        except Exception as e:
            print(e)
            PrintException()
            raise e

    def _do_ohem(self, loss: Tensor):
        if self.enable_ohem and ndim(loss)>0:
            loss=loss.view(-1)
            loss, _ = torch.sort(loss, descending=True)

            n_min=int(len(loss)*self.ohem_thresh)

            # if loss[self.n_min] > self.ohem_thresh:
            #     loss = loss[loss > self.ohem_thresh]
            # else:
            loss = loss[:n_min]
            return loss
        else:
            return loss

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

    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
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

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        Examples:
        >>> output =  to_tensor([[0.3, 0.3, 0.2], [0, 1, 0.5], [1, 1, 0]],dtype=torch.float32)
        >>> target = to_tensor([[0, 1, 0], [0, 1, 0], [1, 0, 0]],dtype=torch.float32)
        >>> CrossEntropyLoss(reduction='mean')(output,target).cpu()
        tensor(0.8695)
        >>> CrossEntropyLoss(reduction='mean')(output,argmax(target,-1)).cpu()
        tensor(0.8695)
        >>> CrossEntropyLoss(reduction='mean')(log_softmax(output),target).cpu()
        tensor(0.8695)
        >>> CrossEntropyLoss(reduction='mean')(log_softmax(output),argmax(target,-1)).cpu()
        tensor(0.8695)
        >>> CrossEntropyLoss(reduction='mean', sample_weight=to_tensor([1,2,3]))(output,target).cpu()
        tensor(1.4518)
        """



        if not self.need_target_onehot:
            if self.is_target_onehot and target.dtype != Dtype.long:
                target = argmax(target, self.axis)

            sample_weight =self.sample_weight.copy()

            target_shape = target.shape

            # gather_unbalance_weight=unbalance_weight.gather(1,target.unsqueeze(-1))
            # gather_unbalance_weight=torch.index_select(sample_weight, -1, target.reshape((-1))).reshape(target_shape).unsqueeze(1).detach(

            if not self.is_logsoftmax :
                if _check_softmax(output):
                    output=log(output+1e-12)
                else:
                    output = log_softmax(output, axis=self.axis)
            target = target.detach()
            sample_weight = (cast(sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype)).detach()

            return nn.functional.nll_loss(output, target.long(), weight=sample_weight, reduction='none')

        else:
            sample_weight =self.sample_weight.copy()

            if not self.is_logsoftmax :
                if _check_softmax(output):
                    output = log(output + 1e-12)
                else:
                    output = log_softmax(output, axis=self.axis)


            new_shape = [1] * ndim(output)
            new_shape[1] = int_shape(output)[1]
            sample_weight = ( cast(sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype)).detach()
            sample_weight = reshape(sample_weight, new_shape)
            target = target.detach()
            return -reduce_sum(target * output * sample_weight, axis=self.axis)


class NLLLoss(_ClassificationLoss):
    r"""The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
    weight to each of the classes. This is particularly useful when you have an
    unbalanced training set.

    The `input` given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ㄡ., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
    where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
    this class index (this index may not necessarily be in the class range).

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\},

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
    :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    ㄡ math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{'sum'.}
        \end{cases}

    Can also be used for higher dimension inputs, such as 2D images, by providing
    an input of size :math:`(minibatch, C, d_1, d_2, ㄡ., d_K)` with :math:`K \geq 1`,
    where :math:`K` is the number of dimensions, and a target of appropriate shape
    (see below). In the case of images, it computes NLL loss per-pixel.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When
            :attr:`size_average` is ``True``, the loss is averaged over
            non-ignored targets.
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
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ㄡ., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ㄡ., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(N)`, or
          :math:`(N, d_1, d_2, ㄡ., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.

    Examples:
    >>> output=to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]]).float()
    >>> print(output.shape)
    torch.Size([4, 3])
    >>> target=to_tensor([1,0,1,2]).long()
    >>> print(target.shape)
    torch.Size([4])
    >>> NLLLoss(reduction='mean')(output,target).cpu()
    tensor(-0.3375)
    >>> NLLLoss(reduction='sum')(output,target).cpu()
    tensor(-1.3500)
    >>> NLLLoss(label_smooth=True,reduction='mean')(output,target).cpu()
    tensor(1.1034)
    >>> NLLLoss(sample_weight=to_tensor([1.0,1.0,0.5]).float(),reduction='mean')(output,target).cpu()
    tensor(-0.2625)
    >>> NLLLoss(ignore_index=2,reduction='mean')(output,target).cpu()
    tensor(-0.2625)
    >>> output2 = torch.tensor([[-0.1, 0.2, -0.3, 0.4],[0.5, -0.6, 0.7, -0.8],[-0.9, 0.1, -0.11, 0.12]])
    >>> target2= torch.tensor([1,2,3]).long()
    >>> NLLLoss(reduction='mean')(output2,target2).cpu()
    tensor(1.0847)
    >>> NLLLoss(reduction='mean')(log_softmax(output2),target2).cpu()
    tensor(1.0847)

    """

    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='NllLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)
        self.need_target_onehot = False

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
        sample_weight = reshape(self.sample_weight, tuple(reshape_shape))

        if self.is_target_onehot and ndim(target) == ndim(output):
            if not self.is_logsoftmax:
                output = log_softmax(output, axis=self.axis)
            loss = -reduce_sum(target * output * sample_weight, axis=1)
        else:
            loss = F.nll_loss(output, target, weight=sample_weight, ignore_index=self.ignore_index, reduction='none')
        return loss


class F1ScoreLoss(_ClassificationLoss):
    """
    This operation computes the f-measure between the output and target. If beta is set as one,
    it's called the f1-scorce or dice similarity coefficient. f1-scorce is monotonic in jaccard distance.

    f-measure = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    This loss function is frequently used in semantic segmentation of images. Works with imbalanced classes, for
    balanced classes you should prefer cross_entropy instead.
    This operation works with both binary and multiclass classification.

    Args:
        beta: greater than one weights recall higher than precision, less than one for the opposite.
        Commonly chosen values are 0.5, 1 or 2.
        axis (int): the position where the classes are.
        sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
        number of classes.
        from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
        ignore_index (int or list of int):
        cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
        less than 1
        label_smooth (bool): Should use label smoothing?
        reduction (string): the method to aggrgate loss. None means no need to aggregate, 'mean' means average loss,
            'sum' means the summation of losses,'batch_mean' means average loss cross the batch axis then
            summation them.

    Returns:
        :class:`~cntk.ops.functions.Function`

    Examples:
    >>> output=to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]]).float()
    >>> print(output.shape)
    torch.Size([4, 3])
    >>> target=to_tensor([1,0,1,2]).long()
    >>> print(target.shape)
    torch.Size([4])
    >>> F1ScoreLoss(reduction='mean')(output,target).cpu()
    tensor(0.6670)
    >>> F1ScoreLoss(reduction='sum')(output,target).cpu()
    tensor(2.6680)
    >>> F1ScoreLoss(label_smooth=True,reduction='mean')(output,target).cpu()
    tensor(0.6740)
    >>> F1ScoreLoss(loss_weights=to_tensor([1.0,1.0,0]).float(),reduction='mean')(output,target).cpu()
    tensor(0.6670)
    >>> F1ScoreLoss(ignore_index=2,reduction='mean')(output,target).cpu()
    tensor(0.6670)



    """

    def __init__(self, beta=1, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.7, input_names=None, output_names=None, name='F1ScoreLoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         input_names=input_names, output_names=output_names, name=name)
        self.beta = beta

        self.need_target_onehot = True

    def calculate_loss(self, output, target, **kwargs):



        sample_weight =self.sample_weight.copy()

        sample_weight = (self.sample_weight * self.ignore_index_weight).detach()


        sample_weight_shape=list(int_shape(output))
        for n in range(len(sample_weight_shape)):
            if n==self.axis and sample_weight_shape[n]==len(sample_weight):
                pass
            else:
                sample_weight_shape[n]=1
        sample_weight = (self.sample_weight.to(get_device()) * self.ignore_index_weight.to(get_device())).detach()
        sample_weight=sample_weight.reshape(sample_weight_shape)


        if self.is_logsoftmax:
            output = exp(output)

        elif not self.from_logits:
            output = softmax(output, self.axis)
        if target.dtype == Dtype.int64 or self.is_target_onehot == False:
            target = make_onehot(target, self.num_classes, axis=self.axis).to(output.dtype)
        target.detach()

        tp = reduce_sum(target * output * sample_weight, axis=self.axis)
        # tn = ((1 - target) * (1 - output))
        # fp = ((1 - target) * output)
        # fn = (target * (1 - output))
        precision = tp / reduce_sum(output, axis=self.axis)
        recall = tp / reduce_sum(target, axis=self.axis)
        return 1 - (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall)


class FocalLoss(_ClassificationLoss):
    """
    Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

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


    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py

        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

     Examples:

        >>> FocalLoss(reduction='mean',axis=-1)(to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]]).float(),to_tensor([1,0,1,2]).long()).cpu()
        tensor(1.1305)
    """

    def __init__(self, alpha=0.25, gamma=2, normalized=False, threshold=None, axis=1, sample_weight=None,
                 auto_balance=False, from_logits=False, ignore_index=-100, cutoff=None,
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

    def calculate_loss(self, output, target, **kwargs):
        """


        Args:
            output: Tensor of arbitrary shape
            target: Tensor of the same shape as input



        Returns:

            """
        num_classes=output.size(self.axis)
        alpha = ones((num_classes)) * (1 - self.alpha)
        alpha[0] = self.alpha

        if self.is_target_onehot and target.dtype != Dtype.long:
            target = argmax(target, self.axis)
            self.is_target_onehot = False

        if not self.is_logsoftmax:
            if _check_softmax(output):
                output = log(output + 1e-12)
            else:
                output = log_softmax(output, axis=self.axis)
            self.is_logsoftmax = True

        ce_loss = nn.functional.nll_loss(output, target.long(), reduction='none')
        pt = exp(-ce_loss)

        alpha = alpha.gather(0, target.view(-1)).reshape(target.size())
        focal_loss = alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss


class BCELoss(_ClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.7, input_names=None, output_names=None, name='BCELoss'):
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
        sample_weight = (cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype)).detach()

        unbalance_weight = ones(tuple([1] * (ndim(output) - 1) + [self.num_classes]))
        if self.auto_balance and self.label_statistics is not None:
            if self.num_classes == len(self.label_statistics):
                unbalance_weight = clip(to_tensor(
                    np.reshape(self.label_statistics.copy(), tuple([1] * (ndim(output) - 1) + [self.num_classes]))),
                                        min=1).to(_float_dtype).detach()

        if self.is_logsoftmax and reduce_max(output) <= 0:
            output = exp(output)
        elif _check_logit(output, axis=self.axis) or _check_softmax(output, axis=self.axis):
            pass
        else:
            output = torch.sigmoid(output)
        loss = binary_cross_entropy(output, target, from_logits=True)

        sample_weight = expand_as(sample_weight, loss)
        unbalance_weight = expand_as(unbalance_weight, loss)
        loss = loss * unbalance_weight * sample_weight
        if ndim(loss) >= 2:
            return loss.sum(1)
        else:
            return loss


class DiceLoss(_ClassificationLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    Args:
        axis (int): the position where the classes are.
        sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
        number of classes.
        from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
        ignore_index (int or list of int):
        cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
        less than 1
        label_smooth (bool): Should use label smoothing?
        reduction (string): the method to aggrgate loss. None means no need to aggregate, 'mean' means average loss,
            'sum' means the summation of losses,'batch_mean' means average loss cross the batch axis then
            summation them.

    Examples:
    >>> output=zeros((1,128,128))
    >>> output[0,32:64,48:92]=1
    >>> output[0,12:32,56:64]=2
    >>> output=make_onehot(output,3,axis=1)
    >>> target=zeros((1,128,128)).long()
    >>> target[0,33:63,50:94]=1
    >>> target[0,13:35,52:65]=2
    >>> DiceLoss(reduction='mean')(output,target).cpu()
    tensor(0.8271)
    >>> DiceLoss(ignore_index=0,reduction='mean')(output,target).cpu()
    tensor(0.9829)



    Reference:
        https://arxiv.org/abs/1707.03237


    """

    def __init__(self, smooth=1, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
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

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # if any_abnormal_number(output):
        #     print('{0} calculate starting'.format(self.name))

        if self.is_logsoftmax or reduce_max(output) <= 0 :
            output = exp(output)
            self.from_logits = True
        elif abs(output.sum(1).mean()-1)<1e-3:
            self.from_logits = True


        if int_shape(output) != int_shape(target):
            target = make_onehot(target, self.num_classes, self.axis)
        if not self.from_logits:
            output = sigmoid(output)

        sample_weight =self.sample_weight.copy()
        # reduce the batch axis and all spatial axes
        target = target.detach()
        #scale=numel(output)/output.size(1)
        reduce_axes = list(range(ndim(target)))
        axis = self.axis if self.axis >= 0 else ndim(target) + self.axis
        reduce_axes.remove(0)
        reduce_axes.remove(axis)

        sample_weight = (self.sample_weight.to(get_device()) * self.ignore_index_weight.to(get_device())).detach()

        if ndim(output)==2:
            intersection = (target * output) * sample_weight
            den1 = output * sample_weight
            den2 = target* sample_weight
        else:
            intersection = reduce_sum((target * output), axis=reduce_axes) * sample_weight
            den1 = reduce_sum(output, axis=reduce_axes) * sample_weight
            den2 = reduce_sum(target, axis=reduce_axes) * sample_weight

        dice = (2.0 * intersection.sum(1) + self.smooth) / (den1.sum(1) + den2.sum(1)+ self.smooth)
        return 1.0 - dice

        # dice = (2.0 * intersection.sum(1) + self.smooth)/(den1.sum(1) + den2.sum(1) + self.smooth)

        # if output.shape[1]==2:
        #     dice =  (2.0 * (intersection * sample_weight[:,1:].sum(1) )+ self.smooth) / clip(den1[:,1:].sum(1) + den2[:,1:].sum(1) + self.smooth,min=1e-7)
        # else:
        #
        #     dice =  torch.div((2.0 * (intersection * sample_weight).sum(1) + self.smooth) , clip(den1.sum(1) + den2.sum(1) + self.smooth,min=1e-7))

        # if any_abnormal_number(den2):
        #     print('{0} after dice {1}'.format(self.name,dice))

        # return 1.0 -dice


class ActiveContourLoss(_ClassificationLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    Args:
        axis (int): the position where the classes are.
        sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
        number of classes.
        from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
        ignore_index (int or list of int):
        cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
        less than 1.
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
        cvpr2019 paper "Learning Active Contour Models for Medical Image Segmentation" by Chen, Xu, et al.
        https://github.com/xuuuuuuchen/Active-Contour-Loss


    """

    def __init__(self, lambdaP=1, mu=1, axis=1, sample_weight=None, auto_balance=False, from_logits=False,
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

    def flatten_check(self, output, target):
        return output, target

    def calculate_loss(self, output, target, **kwargs):
        B, C, H, W = output.shape
        if self.is_logsoftmax:
            output = exp(output)
        sample_weight = (cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype)).detach()
        unbalance_weight = ones([self.num_classes]).to(_float_dtype)
        if self.auto_balance and self.label_statistics is not None:
            if self.num_classes == len(self.label_statistics):
                unbalance_weight = to_tensor(self.label_statistics.copy()).detach().to(_float_dtype)
                unbalance_weight = where(unbalance_weight < 1e-7, ones([self.num_classes]).to(_float_dtype),
                                         pow(1 - unbalance_weight, 3.5))
                sample_weight = unbalance_weight

        sample_weight = expand_dims(sample_weight, 0)

        target = target.detach().to(output.dtype)
        if not self.is_target_onehot:
            target = make_onehot(target, C, axis=1)


        """
        length term
        """
        x = output[:,:,1:,:] - output[:,:,:-1,:]    # horizontal gradient (B, C, H-1, W)
        y = output[:,:,:,1:] - output[:,:,:,:-1]    # vertical gradient   (B, C, H,   W-1)

        delta_x = x[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
        delta_y = y[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
        delta_u = torch.abs(delta_x + delta_y)

        epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
        length = torch.mean(torch.sqrt(delta_u + epsilon))   # eq.(11) in the paper, mean is used instead of sum.


        """
        region term
        """

        C_in = torch.ones_like(output)
        C_out = torch.zeros_like(target)
        # print('C_in',C_in.shape)
        # print('C_out', C_out.shape)
        # print('output[:,0,:,:]', output[:,0,:,:].shape)
        # print('target[:, 0, :, :] ', target[:, 0, :, :] .shape)

        region_in = torch.abs(torch.mean(output[:,0:1,:,:] * ((target[:, 0:1, :, :] - C_in) ** 2)))         # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.abs(torch.mean((1-output[:,0:1,:,:]) * ((target[:, 0:1, :, :] - C_out) ** 2)))   # equ.(12) in the paper

        return length + self.lambdaP * (self.mu * region_in + region_out)


class GeneralizedDiceLoss(_ClassificationLoss):
    r"""This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.

    Compute the generalised Dice loss defined in:

        Sudre, C. et al. (2017) Generalised Dice overlaps as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Reference:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279


    Args:
        axis (int): the position where the classes are.
        sample_weight (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as
        number of classes.
        from_logits (bool): whether the output tensor is normalized as a probability (total equal to 1)
        ignore_index (int or list of int):
        cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number
        less than 1
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





    """

    def __init__(self, smooth=1., axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
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

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # if any_abnormal_number(output):
        #     print('{0} calculate starting'.format(self.name))
        if self.is_logsoftmax and reduce_max(output) <= 0:
            output = exp(output)
            self.from_logits = True

        target = target.detach()
        sample_weight = (self.sample_weight.to(get_device()) * self.ignore_index_weight.to(get_device())).detach()

        if int_shape(output) != int_shape(target):
            target = make_onehot(target, self.num_classes, self.axis)
        if not self.from_logits:
            output = sigmoid(output)
        reduce_axes = list(range(ndim(target)))
        axis = self.axis if self.axis >= 0 else ndim(target) + self.axis
        reduce_axes.remove(0)
        reduce_axes.remove(axis)

        intersection = reduce_sum(target * output, axis=reduce_axes)
        den1 = reduce_sum(output, axis=reduce_axes)
        den2 = reduce_sum(target, axis=reduce_axes)
        sample_weight = expand_as(sample_weight, intersection).detach()

        unbalance_weight = ones(tuple([1] * (ndim(intersection) - 1) + [self.num_classes]))
        if self.auto_balance and self.label_statistics is not None:
            if self.num_classes == len(self.label_statistics):
                unbalance_weight = clip(to_tensor(np.reshape(self.label_statistics.copy(), tuple(
                    [1] * (ndim(intersection) - 1) + [self.num_classes]))), min=1).to(intersection.dtype).detach()
                sample_weight = unbalance_weight
        intersection = intersection * sample_weight
        den1 = den1 * sample_weight
        den2 = den2 * sample_weight

        dice = (2.0 * intersection.sum(1) + self.smooth) / (den1.sum(1) + den2.sum(1) + self.smooth)

        # if output.shape[1]==2:
        #     dice =  (2.0 * (intersection * sample_weight[:,1:].sum(1) )+ self.smooth) / clip(den1[:,1:].sum(1) + den2[:,1:].sum(1) + self.smooth,min=1e-7)
        # else:
        #
        #     dice =  torch.div((2.0 * (intersection * sample_weight).sum(1) + self.smooth) , clip(den1.sum(1) + den2.sum(1) + self.smooth,min=1e-7))

        # if any_abnormal_number(den2):
        #     print('{0} after dice {1}'.format(self.name,dice))

        return 1.0 - dice


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

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        batch = int_shape(output)[0]
        target = target.detach()
        return F.l1_loss(output.reshape((batch, -1)), target.reshape((batch, -1)), reduction='none')


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

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        batch = int_shape(output)[0]
        target = target.detach()
        return 0.5 * ((output.reshape((batch, -1)) - target.reshape((batch, -1))) ** 2)


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

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        batch = int_shape(output)[0]
        target = target.detach()
        return F.smooth_l1_loss(output.reshape((batch, -1)), target.reshape((batch, -1)), reduction='none')


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

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        batch = int_shape(output)[0]
        target = target.detach()
        return ((output.reshape((batch, -1)) - target.reshape((batch, -1))) ** 2)


class WingLoss(_PairwiseLoss):
    def __init__(self, omega=10, epsilon=2, input_names=None, output_names=None, name='WingLoss'):
        super(WingLoss, self).__init__(axis=1,input_names=input_names, output_names=output_names, name=name)
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

        if ndim(target)==2:
            target=target.reshape((target.size(0),-1,2))
        if ndim(output) == 2:
            output = output.reshape((output.size(0), -1, 2))
        target = target.detach()
        delta_y = (target - output).abs()
        c = self.omega * (1.0 - log(1.0 + self.omega / self.epsilon))
        losses = where(
            greater(self.omega,delta_y),
            self.omega * log(1.0 + delta_y / self.epsilon),
            delta_y - c
        )

        return losses



class AdaptiveWingLoss(_PairwiseLoss):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, input_names=None, output_names=None,
                 name='AdaptiveWingLoss'):
        super(AdaptiveWingLoss, self).__init__(axis=1,input_names=input_names, output_names=output_names, name=name)
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
        target = target.detach()
        lossMat = torch.zeros_like(output, dtype=output.dtype).to(output.device)
        A = self.omega * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - target))) * (self.alpha - target) * (
                    (self.theta / self.epsilon) ** (self.alpha - target - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + (self.theta / self.epsilon) ** (self.alpha - target))
        case1_ind = torch.abs(target - output) < self.theta
        case2_ind = torch.abs(target - output) >= self.theta
        lossMat[case1_ind] = self.omega * torch.log(
            1 + torch.abs((target[case1_ind] - output[case1_ind]) / self.epsilon) ** (self.alpha - target[case1_ind]))
        lossMat[case2_ind] = A[case2_ind] * torch.abs(target[case2_ind] - output[case2_ind]) - C[case2_ind]
        return lossMat


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
        super(MS_SSIMLoss, self).__init__(reduction='mean', enable_ohem=False, ohem_thresh=0.7, input_names=input_names,
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

        self.weights = weights.to(get_device())

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
        ms_ssim_val = torch.prod(mcs_and_ssim ** self.weights.view(-1, 1, 1), dim=0)
        return ms_ssim_val

    def calculate_loss(self, output, target, **kwargs):
        target = target.to(output.dtype)
        return 1-self.ms_ssim(output, target)


class IoULoss(_ClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=0, cutoff=None,
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

    def calculate_loss(self, output, target, **kwargs):
        B, C, _ = output.shape
        if self.is_logsoftmax and reduce_max(output) <= 0:
            output = exp(output)
        if not self.is_target_onehot:
            target = make_onehot(target, C, axis=1)

        sample_weight = (self.sample_weight.to(get_device()) * self.ignore_index_weight.to(get_device()))

        unbalance_weight = ones((self.num_classes)).to(_float_dtype)
        if self.auto_balance and self.label_statistics is not None:
            if self.num_classes == len(self.label_statistics):
                unbalance_weight = to_tensor(self.label_statistics.copy()).detach()
                unbalance_weight = where(unbalance_weight < 1e-7, ones([self.num_classes]).to(_float_dtype),
                                         pow(1 - unbalance_weight, 3.5))
        sample_weight = unbalance_weight.detach()
        target = target.detach()

        tp = output * target
        fp = output * (1 - target)
        fn = (1 - output) * target
        intersection = reduce_sum(tp, axis=-1)
        union = reduce_sum(tp, axis=-1) + reduce_sum(fp, axis=-1) + reduce_sum(fn, axis=-1)
        iou = (intersection * sample_weight).sum(-1) / (union * sample_weight).sum(-1)
        return 1 - iou


class SoftIoULoss(_ClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=0, cutoff=None,
                 label_smooth=False, reduction="mean",  enable_ohem=False,ohem_thresh=0.7, input_names=None, output_names=None, name='softiou'):
        super(SoftIoULoss, self).__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance,
                                      from_logits=from_logits, ignore_index=ignore_index, cutoff=cutoff,
                                      label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem,
                                      ohem_thresh=ohem_thresh, input_names=input_names, output_names=output_names,
                                      name=name)

        self.reduction = reduction


    def calculate_loss(self, output, target, **kwargs):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(output)
        self.n_classes = int_shape(output)[self.axis]
        pred = F.softmax(output, dim=1)
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


def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(_ClassificationLoss):
    def __init__(self, axis=1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=0, cutoff=None,
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

    def flatten_check(self, output, target):
        assert output.dim() in [4, 5]

        if output.dim() == 4:
            output = output.permute(0, 2, 3, 1).contiguous()
            output_flatten = output.view(-1, self.num_class)
        elif output.dim() == 5:
            output = output.permute(0, 2, 3, 4, 1).contiguous()
            output_flatten = output.view(-1, self.num_class)
        target_flatten = target.view(-1)
        return output_flatten, target_flatten

    def lovasz_softmax_flat(self, output, target):

        losses = []
        for c in range(self.num_classes):
            target_c = (target == c).float()
            if self.num_classes == 1:
                input_c = output[:, 0]
            else:
                input_c = output[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(_lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)
        return losses

    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id

        """
        if per_image:
            loss = (self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)) for
                    log, lab in zip(logits, labels)).mean()
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore

        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * torch.tensor(signs, requires_grad=True))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = _lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    def calculate_loss(self, output, target, **kwargs):
        # print(output.shape, target.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        # print(output.shape, target.shape)

        losses = self.lovasz_softmax_flat(output, target) if self.num_classes > 2 else self.lovasz_hinge_flat(output,
                                                                                                              target)
        return losses


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
    """Hard Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def _pairwise_distance(self, x, squared=False, eps=1e-16):
        # Compute the 2D matrix of distances between all the embeddings.

        cor_mat = torch.matmul(x, x.t())
        norm_mat = cor_mat.diag()
        distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
        distances = F.relu(distances)

        if not squared:
            mask = torch.eq(distances, 0.0).float()
            distances = distances + mask * eps
            distances = torch.sqrt(distances)
            distances = distances * (1.0 - mask)

        return distances

    def _get_anchor_positive_triplet_mask(self, labels):
        # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same labeled.

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

        # Check if labels[i] == labels[j]
        labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

        mask = indices_not_equal * labels_equal

        return mask

    def _get_anchor_negative_triplet_mask(self, labels):
        # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

        # Check if labels[i] != labels[k]
        labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
        mask = labels_equal ^ 1

        return mask

    def _get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Check that i, j and k are distinct
        indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
        i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
        i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
        j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
        distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
        i_equal_j = torch.unsqueeze(label_equal, 2)
        i_equal_k = torch.unsqueeze(label_equal, 1)
        valid_labels = i_equal_j * (i_equal_k ^ 1)

        mask = distinct_indices * valid_labels  # Combine the two masks

        return mask

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = self._pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + 0.1)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = self._get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


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
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        init.kaiming_uniform_(self.centers, a=sqrt(5))
        self.to(_device)

    def forward(self, features, target) -> 'loss':
        """
        Args:
            features: feature matrix with shape (batch_size, feat_dim).
            target: ground truth labels with shape (num_classes).

        """

        batch_size = features.size(0)
        self.centers.to(features.dtype).to(features.device)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(
            self.centers.to(features.dtype).to(features.device), 2).sum(dim=1, keepdim=True).expand(self.num_classes,
                                                                                                    batch_size).t()
        distmat.addmm_(1, -2, features, self.centers.t())
        classes = torch.arange(self.num_classes).long().to(_device)

        target = target.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = target.eq(classes.expand(batch_size, self.num_classes))
        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-8, max=1e+5)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        if self.reduction == 'mean':
            return dist.mean()  # / self.num_classes
        else:
            return dist.sum()  # / self.num_classes


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.to(_device)

    def forward(self, output, target) -> 'loss':
        target.detach()
        img_size = list(output.size())
        G = gram_matrix(output)
        Gt = gram_matrix(target)
        return F.mse_loss(G, Gt).div((img_size[1] * img_size[2] * img_size[3]))


class PerceptionLoss(_PairwiseLoss):
    def __init__(self, net, reduction="mean"):
        super(PerceptionLoss, self).__init__()
        if isinstance(net, ModelBase):
            net = net.model
        self.ref_model = net
        self.ref_model.eval()
        self.ref_model.trainable = False

        self.layer_name_mapping = ['block1_relu2', 'block2_relu2', 'block3_relu3', 'block4_relu3']
        for name, module in self.ref_model.named_modules():
            if name in self.layer_name_mapping:
                module.keep_output = True
        for k in self.ref_model._modules.keys():
            is_start_delete = False
            if k == 'block4_conv3':
                is_start_delete = True
            if is_start_delete and k != 'block4_conv3':
                del self.ref_model._modules[k]

        self.reduction = reduction
        self.ref_model.to(_device)

    def vgg_preprocess(self, img):
        return ((img + 1) / 2) * to_tensor([[0.485, 0.456, 0.406]]).unsqueeze(-1).unsqueeze(-1) + to_tensor(
            [[0.229, 0.224, 0.225]]).unsqueeze(-1).unsqueeze(-1)

    def calculate_loss(self, output, target, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        target_features = OrderedDict()
        output_features = OrderedDict()
        if output.shape[1]==1:
            output=concate([output,output,output],axis=1)
        if target.shape[1]==1:
            target=concate([target,target,target],axis=1)
        _ = self.ref_model(self.vgg_preprocess(output))

        # _ = self.ref_model(output)
        for item in self.layer_name_mapping:
            output_features[item] = getattr(self.ref_model, item).output

        _ = self.ref_model(self.vgg_preprocess(target))

        for item in self.layer_name_mapping:
            target_features[item] = getattr(self.ref_model, item).output.detach()

        loss = 0
        num_filters = 0
        for item in self.layer_name_mapping:
            b, c, h, w = output_features[item].shape
            loss = loss + ((output_features[item] - target_features[item]) ** 2).sum() / (b * c * h * w)

        return loss / len(self.layer_name_mapping)


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

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        angle = torch.tensor(
            np.random.uniform(-self.rotation_range, self.rotation_range, target.size(0)) * np.pi / 180).float().to(
            output.device)
        scale = torch.tensor(np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range, target.size(0))).float().to(
            output.device)
        height, width = target.size()[-2:]
        center_tensor = torch.tensor([(width - 1) / 2, (height - 1) / 2]).expand(target.shape[0], -1).float().to(
            output.device)
        mat = ops._get_rotation_matrix2d(center_tensor, angle, scale).float().to(output.device)
        rotated_target = ops.warp_affine(target, mat, target.size()[2:]).float().to(output.device)

        embedded_output = self.embedded_func(output)
        embedded_target = self.embedded_func(rotated_target)
        return self.loss(embedded_output, embedded_target)


class GPLoss(nn.Module):
    def __init__(self, discriminator, l=10):
        super(GPLoss, self).__init__()
        self.discriminator = discriminator
        self.l = l

    def forward(self, real_data, fake_data):
        alpha = real_data.new_empty((real_data.size(0), 1, 1, 1)).uniform_(0, 1)
        alpha = alpha.expand_as(real_data)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = torch.tensor(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=real_data.new_ones(disc_interpolates.size()), create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * self.l


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
