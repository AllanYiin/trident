from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
import math
import numbers
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.training.tracking import tracking
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

from trident.optims.losses import Loss, _check_logsoftmax_logit,_check_logit

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


__all__ = ['get_loss', '_ClassificationLoss', 'CrossEntropyLoss', 'BCELoss', 'MSELoss', 'EdgeLoss', 'NLLLoss',
           'F1ScoreLoss', '_ClassificationLoss',
           'FocalLoss', 'DiceLoss', 'L1Loss', 'L2Loss', 'SmoothL1Loss', 'WingLoss', 'AdaptiveWingLoss', 'IoULoss',
           'SoftIoULoss']



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
    reweights=counts_array.sum()/(counts_array*len(counts_array))
    return reweights,OrderedDict(zip(uniques_array, counts_array))



class _ClassificationLoss(Loss, tracking.AutoTrackable):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.1, name=None, **kwargs):
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
                                                  enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, namename=name)
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

    def _get_reduction(self, loss):
        with ops.name_scope(self.name + '_reduction', "reduction_loss", [loss]) as name:
            num_present = math_ops.cast(array_ops.size(loss, name='num_elements'), dtype=loss.dtype)
            if ndim(loss) == 0 or self.reduction == 'none':
                return loss
            if ndim(loss) >= 2 and self.reduction == 'batch_sum':
                loss = reshape(loss, (int_shape(loss)[0], -1))
                return loss.mean(1).sum()
            elif ndim(loss) >= 2 and self.reduction == 'batch_mean':
                loss = reshape(loss, (int_shape(loss)[0], -1))
                return loss.mean(1).mean()
            elif self.reduction in ['mean', 'batch_mean']:
                return math_ops.reduce_mean(loss, name='value')
            elif self.reduction in ['sum', 'batch_sum']:
                return math_ops.reduce_sum(loss)
            else:
                total_loss = math_ops.reduce_sum(loss)
                return math_ops.div_no_nan(total_loss, num_present, name='value')

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
                        if not isinstance(ds, TextSequenceDataset):
                            self.label_statistics, _ = _class_unique_value_process(list(ds._label_statistics.keys()),
                                                                                   list(ds._label_statistics.values()))
                        else:
                            self.label_statistics, _ = _nlp_vercabs_unique_value_process(
                                list(ds._label_statistics.keys()), list(ds._label_statistics.values()))

                    elif isinstance(ds, LabelDataset) or dp.traindata.label.object_type in [
                        ObjectType.classification_label]:
                        print('Start retrive label class distribution for auto-balance in loss function.')
                        unique, counts = np.unique(
                            np.array([dp.traindata.label.items[i] for i in tqdm(range(len(dp.traindata.label.items)))]),
                            return_counts=True)
                        ctx.print('')
                        reweights, label_statistics = _class_unique_value_process(unique, counts)
                        self.label_statistics = reweights
                        ds._label_statistics = label_statistics

                        del unique
                        del counts

                    elif isinstance(ds, BboxDataset) or dp.traindata.label.object_type in [ObjectType.absolute_bbox,
                                                                                           ObjectType.relative_bbox]:
                        ctx.print('Start retrive label class distribution for auto-balance in loss function.')
                        unique, counts = torch.unique(to_tensor(
                            np.concatenate([dp.traindata.label[i][:, 4] for i in tqdm(range(len(ds.items)))], axis=0),
                            dtype=Dtype.long, device='cpu'), return_counts=True)
                        ctx.print('')
                        reweights, label_statistics = _class_unique_value_process(unique, counts)
                        self.label_statistics = reweights
                        ds._label_statistics = label_statistics
                        del unique
                        del counts

                    elif isinstance(ds, MaskDataset) or dp.traindata.label.object_type in [ObjectType.label_mask,
                                                                                           ObjectType.color_mask,
                                                                                           ObjectType.binary_mask]:
                        ctx.print('Start retrive label class distribution for auto-balance in loss function.')
                        sample_base = list(range(len(ds)))
                        if len(sample_base) > 1000:
                            np.random.shuffle(sample_base)
                            sample_base = sample_base[:1000]
                        overall_unique = OrderedDict()
                        unique_results = [dict(zip(*np.unique(ds[i], return_counts=True))) for i in tqdm(sample_base)]
                        for d in unique_results:
                            for k, v in d.items():
                                if k not in overall_unique:
                                    overall_unique[k] = v
                                else:
                                    overall_unique[k] += v

                        unique = list(sorted(overall_unique.key_list))
                        counts = [overall_unique[k] for k in unique]
                        # unique = to_list(to_numpy(unique))
                        # counts = to_numpy(counts)

                        ctx.print('')
                        reweights, label_statistics = _class_unique_value_process(unique, counts)
                        self.label_statistics = reweights
                        ds._label_statistics = label_statistics
                        del unique
                        del counts

                    elif isinstance(ds, TextSequenceDataset):
                        keys = [ds.text2index[k] for k in list(ds.vocabs_frequency.keys()) if k in ds.vocabs]
                        reweights, label_statistics = _nlp_vercabs_unique_value_process(keys,
                                                                                        ds.vocabs_frequency.value_list)
                        self.label_statistics = reweights
                        ds._label_statistics = label_statistics

    def preprocess(self, output: Tensor, target: Tensor, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # check num_clases
        self.is_target_onehot = None
        self.from_logits = None
        if len(output) == 0:
            return to_tensor(0.0)

        if reduce_max(output) > 0 or self.is_logsoftmax is None or not self.is_logsoftmax:
            if reduce_min(output) >= 0:
                self.is_logsoftmax = False
                if _check_logit(output, self.axis):
                    self.from_logits = True
                    #output = clip(output, min=1e-7, max=1)
                else:
                    self.from_logits = False

                #output = clip(output, min=1e-7, max=1)

            elif _check_logsoftmax_logit(output, self.axis):
                self.is_logsoftmax = True
                self.from_logits = True
                #output = clip(output, min=-12, max=-1e-7)

            else:
                self.is_logsoftmax = False
                self.from_logits = False

            if (ndim(target) == ndim(output) and 'float' in str(
                    target.dtype) and target.min() >= 0 and reduce_max(target) <= 1):
                self.is_target_onehot = True
        else:
            pass
            #output = clip(output, min=-12, max=-1e-7)

        if target.dtype == str2dtype('long'):
            self.is_target_onehot = False
        elif target.dtype != str2dtype('long') and (
                target.min() >= 0 and reduce_max(target) <= 1 and abs(target.sum(-1).mean() - 1) < 1e-4):
           #target = clip(target, min=1e-8, max=1 - 1e-8)
            self.is_target_onehot = True

        # need target onehot but currently not
        if target.dtype == Dtype.int64 and self.need_target_onehot == True and self.is_target_onehot == False:
            target = make_onehot(target, num_classes=self.num_classes, axis=self.axis)
            self.is_target_onehot = True
            if self.label_smooth:
                target = target + random_normal_like(target)
        target = target.detach()

        return output, target

    def calculate_loss(self, output, target):
        """ Calculate the unaggregate loss.
        The loss function calculation logic should define here., please dont't aggregate the loss in this phase.

        Args:
            output (tf.Tensor):
            target (tf.Tensor):
        """
        ##dont do aggregation
        raise NotImplementedError

    def _handel_abnormal(self, loss):
        if any_abnormal_number(loss):
            sys.stderr.write(
                '{0} has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(
                    self.name))
            loss = where(is_abnormal_number(loss), ones_like(loss) * 1e-7, loss)
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


class _PairwiseLoss(Loss, tracking.AutoTrackable):
    """Calculate loss for  complex classification task."""

    def __init__(self, axis=-1, reduction='batch_mean', enable_ohem=False, ohem_thresh=0.1, name=None, **kwargs):
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
                                            ohem_thresh=ohem_thresh, name=name)
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

    def _get_reduction(self, loss):
        with ops.name_scope(self.name + '_reduction', "reduction_loss", [loss]) as name:
            if ndim(loss) == 0 or self.reduction == 'none':
                return loss
            num_present = math_ops.cast(array_ops.size(loss, name='num_elements'), dtype=loss.dtype)
            batch_size = math_ops.cast(tf.constant(array_ops.shape(loss, name='shape')[0]), dtype=loss.dtype)

            if ndim(loss) >= 2 and self.reduction == 'batch_sum':
                loss = math_ops.div_no_nan(math_ops.reduce_sum(loss, 0), batch_size, name='value')
                return loss.mean(1).sum()
            elif ndim(loss) >= 2 and self.reduction == 'batch_mean':
                total_loss = math_ops.reduce_sum(loss, 0)
                return math_ops.div_no_nan(total_loss, math_ops.div_no_nan(num_present, batch_size), name='value')
            elif self.reduction in ('mean', 'batch_mean'):
                return lmath_ops.reduce_mean(loss)
            elif self.reduction == ('sum', 'batch_sum'):
                return math_ops.reduce_sum(loss)
            else:
                total_loss = math_ops.reduce_sum(loss)
                return math_ops.div_no_nan(total_loss, num_present, name='value')

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

    def preprocess(self, output: Tensor, target: Tensor, **kwargs):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        if isinstance(target, numbers.Number):
            target = ones_like(output) * target

        output, target = self.flatten_check(output, target)

        if output.shape == target.shape:
            return output, target

        elif target.dtype == Dtype.int64 and ndim(output) == ndim(target) + 1:
            num_class = int_shape(output)[self.axis]
            target = make_onehot(target, num_class, self.axis).float()
        return output, target

    def calculate_loss(self, output, target):
        """ Calculate the unaggregate loss.
        The loss function calculation logic should define here., please dont't aggregate the loss in this phase.

        Args:
            output (tf.Tensor):
            target (tf.Tensor):
        """
        ##dont do aggregation
        raise NotImplementedError

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


class CrossEntropyLoss(_ClassificationLoss):
    """
    Calculate the cross entropy loss

    Examples:
    >>> output=to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]])
    >>> print(int_shape(output))
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
    >>> CrossEntropyLoss(sample_weight=to_tensor([1.0,1.0,0.5]),reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.98889095>
    >>> CrossEntropyLoss(ignore_index=2,reduction='mean')(output,target)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.8472573>
    >>> target = to_tensor([[1., 0., 0.],[ 0., 1., 0.],[ 0., 0., 1.]])
    >>> out = to_tensor([[.9, .05, .05],[ .05, .89, .06],[ .05, .01, .94]])
    >>> CrossEntropyLoss(reduction='mean',axis=-1)(out,target)
    tf.Tensor(0.33333334, shape=(), dtype=float32)
    >>> CrossEntropyLoss(reduction='mean',axis=-1)(out,argmax(target,-1))
    tf.Tensor(0.33333334, shape=(), dtype=float32)




    """

    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False,
                 reduction='mean', enable_ohem=False, ohem_thresh=0.1, name='CrossEntropyLoss'):
        super().__init__(axis, sample_weight, auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction,
                         enable_ohem, ohem_thresh, name)

        self.need_target_onehot = False


    def calculate_loss(self, output, target):
        """
        The usual cross-entropy cost is defined as:

              labels * -log(sigmoid(logits)) +
                  (1 - labels) * -log(1 - sigmoid(logits))

          A value `pos_weight > 1` decreases the false negative count, hence increasing
          the recall.
          Conversely setting `pos_weight < 1` decreases the false positive count and
          increases the precision.
          This can be seen from the fact that `pos_weight` is introduced as a
          multiplicative coefficient for the positive labels term
          in the loss expression:

              labels * -log(sigmoid(logits)) * pos_weight +
                  (1 - labels) * -log(1 - sigmoid(logits))

          For brevity, let `x = logits`, `z = labels`, `q = pos_weight`.
          The loss is:

                qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
              = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
              = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
              = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
              = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
              = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

          Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
          the implementation uses

              (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

          `logits` and `labels` must have the same type and shape.

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:


        Examples:
        >>> output = tf.constant(value=[[0.3, 0.3, 0.2], [0, 1, 0.5], [1, 1, 0]], dtype=tf.float32, shape=[3, 3])
        >>> target = to_tensor([[0, 1, 0], [0, 1, 0], [1, 0, 0]],dtype=tf.float32)
        >>> CrossEntropyLoss(reduction='mean')(output,target)
        <tf.Tensor: shape=(), dtype=float32, numpy=0.8695473>
        >>> CrossEntropyLoss(reduction='mean')(output,argmax(target,-1))
        <tf.Tensor: shape=(), dtype=float32, numpy=0.8695473>
        >>> CrossEntropyLoss(reduction='mean')(log_softmax(output),target)
        <tf.Tensor: shape=(), dtype=float32, numpy=0.8695473>
        >>> CrossEntropyLoss(reduction='mean')(log_softmax(output),argmax(target,-1))
        <tf.Tensor: shape=(), dtype=float32, numpy=0.8695473>
        >>> CrossEntropyLoss(reduction='mean', sample_weight=to_tensor([1,2,3]))(output,target)
        <tf.Tensor: shape=(), dtype=float32, numpy=1.451763>

        """

        with ops.name_scope(self.name, "cross_entropy_loss", [output, target]) :
            output = ops.convert_to_tensor(output, name="output")
            target = ops.convert_to_tensor(target, name="target")

            sample_weight = (cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype)).detach()

            n_ = ndim(target) - ndim(sample_weight)
            for n in range(n_):
                sample_weight = expand_dims(sample_weight, 0)

            if self.is_target_onehot:
                target=argmax(target,-1)

                    # return nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=output,axis=-1)*sample_weight
                #return -math_ops.reduce_sum(target * output*sample_weight, axis=1)

            if  ndim(target) == ndim(output) - 1:
                target = expand_dims(target, -1)
                gather_sample_weight = squeeze(tf.gather(sample_weight, target, batch_dims=0, axis=-1),0)
                if not self.is_logsoftmax:
                    output = nn.log_softmax_v2(output)

                target = cast(target, cast_dtype=Dtype.int64)
                gather_prob = tf.gather(output, target, batch_dims=1, axis=-1)

                return math_ops.negative(gather_prob * gather_sample_weight,name=self.name)

            #return tf.nn.sparse_softmax_cross_entropy_with_logits(target,output)


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

    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False,
                 reduction='mean', enable_ohem=False, ohem_thresh=0.1, name='NLLLoss'):
        super().__init__(axis, sample_weight, auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction,
                         enable_ohem, ohem_thresh, name)

        self.need_target_onehot = True

    def calculate_loss(self, output, target):
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
            sample_weight = reshape(
                cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype), reshape_shape)

            loss = reduce_sum(-target * output * self.sample_weight, axis=self.axis)
            if ndim(loss) > 1:
                reduce_axes = list(range(loss.ndim))
                reduce_axes.remove(0)
                if len(reduce_axes) == 0:
                    reduce_axes = None
                return reduce_mean(loss, axis=reduce_axes)
            else:
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

    def __init__(self, beta=1, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.1, name='CrossEntropyLoss'):
        super().__init__(axis, sample_weight, auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction,
                         enable_ohem, ohem_thresh, name)
        self.beta = beta

        self.need_target_onehot = True


    def calculate_loss(self, output, target):
        with ops.name_scope(self.name, "f1_score_loss", [output, target]) as name:
            output = ops.convert_to_tensor(output, name="output")
            target = array_ops.stop_gradient(ops.convert_to_tensor(target, name="target"))

            sample_weight = array_ops.stop_gradient(
                cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype),
                name="sample_weight")
            n_ = ndim(target) - ndim(sample_weight)
            for n in range(n_):
                sample_weight = expand_dims(sample_weight, 0)

            if self.is_logsoftmax:
                output = clip(exp(clip(output, max=0)), 1e-7, 1 - 1e-7)
                self.from_logits = True
            if not self.from_logits:
                output = softmax(output, self.axis)
            if target.dtype == Dtype.int64 or self.is_target_onehot == False:
                target = cast(make_onehot(target, self.num_classes, axis=1), output.dtype)

            tp = (target * output)
            # tn = ((1 - target) * (1 - output))
            # fp = ((1 - target) * output)
            # fn = (target * (1 - output))
            #
            #

            # percision与recall，这里的K.epsilon代表一个小正数，用来避免分母为零
            precision = tp / (output + epsilon())
            recall = tp / (target + epsilon())

            # 计算f1
            return 1 - (1 + self.beta ** 2) * (precision * recall * sample_weight).sum(-1) / (
                        self.beta ** 2 * precision + recall + epsilon()).sum(-1)


class FocalLoss(_ClassificationLoss):
    """
    Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
    threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """

    def __init__(self, alpha=0.5, gamma=2, normalized=False, threshold=None, axis=-1, sample_weight=None,
                 auto_balance=False,
                 from_logits=False, ignore_index=-100, cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.1, name='FocalLoss'):
        super().__init__(axis, sample_weight, auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction,
                         enable_ohem, ohem_thresh, name)
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.normalized = normalized
        self.need_target_onehot = True


    def calculate_loss(self, output, target):
        """


        Args:
            output: Tensor of arbitrary shape
            target: Tensor of the same shape as input


        Returns:

            """
        with ops.name_scope(self.name, "focal_loss", [output, target]) as name:
            output = ops.convert_to_tensor(output, name="output")
            target = array_ops.stop_gradient(ops.convert_to_tensor(target, name="target"))

            sample_weight = array_ops.stop_gradient(
                cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype),
                name="sample_weight")

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


class BCELoss(_ClassificationLoss):
    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.1, name='BCELoss'):
        super().__init__(axis=axis, sample_weight=sample_weight, auto_balance=auto_balance, from_logits=from_logits,
                         ignore_index=ignore_index, cutoff=cutoff,
                         label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem, ohem_thresh=ohem_thresh,
                         name=name)

        self.num_classes = None
        self.is_logsoftmax = False
        self.need_target_onehot = True
        self.is_target_onehot = False

    def calculate_loss(self, output, target):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with ops.name_scope(self.name, "bce_loss", [output, target]) as name:
            output = ops.convert_to_tensor(output, name="output")
            target = array_ops.stop_gradient(ops.convert_to_tensor(target, name="target"))

            sample_weight = array_ops.stop_gradient(
                cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype),
                name="sample_weight")

            if self.is_logsoftmax:
                output = clip(exp(clip(output, max=0)), 1e-7, 1 - 1e-7)
            loss = binary_cross_entropy(output, target, from_logits=self.from_logits)
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

    def __init__(self, smooth=1, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=-100,
                 cutoff=None, label_smooth=False, reduction='mean',
                 enable_ohem=False, ohem_thresh=0.1, name='DiceLoss'):
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

        super().__init__(axis, sample_weight, auto_balance, from_logits, ignore_index, cutoff, label_smooth, reduction,
                         enable_ohem, ohem_thresh, name)
        self.smooth = smooth
        self.is_logsoftmax = False
        self.need_target_onehot = True
        self.is_multiselection = False

    def calculate_loss(self, output, target):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with ops.name_scope(self.name, "dice_loss", [output, target]) as name:
            output = ops.convert_to_tensor(output, name="output")
            target = array_ops.stop_gradient(ops.convert_to_tensor(target, name="target"))

            sample_weight = array_ops.stop_gradient(
                cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype),
                name="sample_weight")

            if self.is_logsoftmax and reduce_max(output) <= 0:
                output = exp(output)
                output = clip(output, 1e-7, 1)
                self.from_logits = True

            if int_shape(output) != int_shape(target):
                target = make_onehot(target, self.num_classes, self.axis)
            if not self.from_logits:
                output = sigmoid(output)

            # unbalance_weight = ones( (self.num_classes))
            if self.auto_balance and self.label_statistics is not None:
                if self.num_classes == len(self.label_statistics):
                    unbalance_weight = clip(to_tensor(sqrt(self.label_statistics.copy())), min=1).detach()
                    sample_weight = unbalance_weight
            intersection = reduce_sum((target * output), axis=[0, 1])
            den1 = reduce_sum(output, axis=[0, 1])
            den2 = reduce_sum(target, axis=[0, 1])

            dice = (2.0 * intersection * sample_weight + self.smooth) / (
                        den1 * sample_weight + den2 * sample_weight + self.smooth)
            return 1.0 - dice


class L1Loss(_PairwiseLoss):
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

     Function that takes the mean element-wise absolute value difference.

     See :class:`~torch.nn.L1Loss` for details.
     """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.1, name='L1Loss'):
        super(L1Loss, self).__init__(reduction)
        self.name = name
        self.reduction = reduction

    def calculate_loss(self, output, target):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with ops.name_scope(self.name, "l1_loss", [output, target]) as name:
            output = ops.convert_to_tensor(output, name="output")
            target = array_ops.stop_gradient(ops.convert_to_tensor(target, dtype=output.dtype, name="target"))
            return tf.math.abs(output - target, name='l1_loss')


class L2Loss(_PairwiseLoss):
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

        Measures the element-wise mean squared error.

        See :class:`~torch.nn.MSELoss` for details.
        """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.1, name='MSELoss'):
        super(L2Loss, self).__init__(reduction)
        self.name = name
        self.reduction = reduction

    def calculate_loss(self, output, target):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with ops.name_scope(self.name, "l2_loss", [output, target]) as name:
            output = ops.convert_to_tensor(output, name="output")
            target = array_ops.stop_gradient(ops.convert_to_tensor(target, dtype=output.dtype, name="target"))
            return tf.nn.l2_loss(output - target, name='l2_loss')


#
# class SmoothL1Loss(_PairwiseLoss):
#     r"""Function that uses a squared term if the absolute
#     element-wise error falls below 1 and an L1 term otherwise.
#
#     See :class:`~torch.nn.SmoothL1Loss` for details.
#     """
#     def __init__(self, reduction='mean' , enable_ohem=False, ohem_thresh=ohem_thresh, name='SmoothL1Loss'):
#         super(SmoothL1Loss, self).__init__(reduction=reduction)
#         self.name = name
#         self.reduction = reduction
#         self.huber_delta = 0.5
#
#     def calculate_loss(self, output, target):
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

    def __init__(self, axis=-1, reduction='mean', name='MSELoss'):
        super().__init__(axis, reduction, name)

    def calculate_loss(self, output, target):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        # with self._name_scope:

        # num_present = tf.reduce_sum(math_ops.cast(array_ops.size(output, name='num_elements'), dtype=output.dtype),name='reduce_sum')
        with ops.name_scope(self.name, "mse_loss", [output, target]) as name:
            output = ops.convert_to_tensor(output, name="output")
            target = array_ops.stop_gradient(ops.convert_to_tensor(target, dtype=output.dtype, name="target"))
            return (output - target) ** 2
        # return math_ops.div_no_nan(tf.nn.l2_loss(output-target), math_ops.cast(tf.shape(output)[0], dtype=output.dtype), name='value')


class SmoothL1Loss(_PairwiseLoss):
    r"""Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """

    def __init__(self, reduction='mean', enable_ohem=False, ohem_thresh=0.1, name='SmoothL1Loss'):
        super(SmoothL1Loss, self).__init__(enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, reduction=reduction,
                                           name=name)
        self.name = name
        self.reduction = reduction
        self.huber_delta = 0.5

    def calculate_loss(self, output, target):
        """

        Args:
            output ():
            target ():
            **kwargs ():

        Returns:

        """
        with ops.name_scope(self.name, "smooth_l1_loss", [output, target]) as name:
            output = ops.convert_to_tensor(output, name="output")
            target = array_ops.stop_gradient(ops.convert_to_tensor(target, dtype=output.dtype, name="target"))
            diff = abs(target - output)
            less_than_one = cast(less(diff, 1.0), tf.float32)  # Bool to float32
            smooth_l1_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)  # 同上图公式

        return smooth_l1_loss


class WingLoss(_PairwiseLoss):
    def __init__(self, omega=10, epsilon=2, reduction='mean', enable_ohem=False, ohem_thresh=0.1, name='WingLoss'):
        super(WingLoss, self).__init__(enable_ohem=enable_ohem, ohem_thresh=ohem_thresh, reduction=reduction, name=name)
        self.name = name
        self.omega = omega
        self.epsilon = epsilon

    def calculate_loss(self, output, target):
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
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, enable_ohem=False, ohem_thresh=0.1,
                 name='AdaptiveWingLoss'):
        super(AdaptiveWingLoss, self).__init__()
        self.name = name
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def calculate_loss(self, output, target):
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
            delta_y = tf.math.abs(y - y_hat)
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


class PerceptionLoss(_PairwiseLoss):
    def __init__(self, net, reduction="mean"):
        super(PerceptionLoss, self).__init__()
        if isinstance(net, ModelBase):
            net = net.model
        self.ref_model = net
        self.ref_model.trainable = False
        self.ref_model.eval()
        self.layer_name_mapping = {'3': "block1_conv2", '8': "block2_conv2", '15': "block3_conv3", '22': "block4_conv3"}
        for name, module in self.ref_model.named_modules():
            if name in list(self.layer_name_mapping.values()):
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
        return (img * 127.5) + 127.5 - to_tensor([[103.939, 116.779, 123.68]]).unsqueeze(0).unsqueeze(0)

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
        _ = self.ref_model(self.vgg_preprocess(output))

        for item in self.layer_name_mapping.values():
            output_features[item] = getattr(self.ref_model, item).output

        _ = self.ref_model(self.vgg_preprocess(target))

        for item in self.layer_name_mapping.values():
            target_features[item] = getattr(self.ref_model, item).output.detach()

        loss = 0
        num_filters = 0
        for i in range(len(self.layer_name_mapping)):
            b, c, h, w = output_features.value_list[i].shape
            loss = loss + ((output_features.value_list[i] - target_features.value_list[i]) ** 2).sum() / (b * c * h * w)

        return loss / len(self.layer_name_mapping)


class EdgeLoss(_PairwiseLoss):
    def __init__(self, enable_ohem=False, ohem_thresh=0.1, name='EdgeLoss'):
        self.name = name
        super(EdgeLoss, self).__init__()

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

    def calculate_loss(self, output, target):
        with self._name_scope:
            loss1 = tf.reduce_mean(tf.math.square(self.first_order(output, 1) - self.first_order(target, 1)))
            loss2 = tf.reduce_mean(tf.math.square(self.first_order(output, 2) - self.first_order(target, 2)))
            return loss1 + loss2


class IoULoss(_ClassificationLoss):
    def __init__(self, axis=-1, sample_weight=None, auto_balance=False, from_logits=False, ignore_index=0, cutoff=None,
                 label_smooth=False, reduction='mean', enable_ohem=False,
                 ohem_thresh=0.1, name='lou_loss'):
        super(IoULoss, self).__init__(raxis=axis, sample_weight=sample_weight, auto_balance=auto_balance,
                                      from_logits=from_logits, ignore_index=ignore_index, cutoff=cutoff,
                                      label_smooth=label_smooth, reduction=reduction, enable_ohem=enable_ohem,
                                      ohem_thresh=ohem_thresh, name=name)
        self.need_target_onehot = True
        self.is_multiselection = False
        self.is_logsoftmax = False

    def calculate_loss(self, output, target):
        with self._name_scope:
            reshape_shape = [1] * ndim(output)
            reshape_shape[self.axis] = self.num_classes
            sample_weight = reshape(
                cast(self.sample_weight, output.dtype) * cast(self.ignore_index_weight, output.dtype), reshape_shape)

            _dtype = output.dtype
            if self.is_logsoftmax:
                output = clip(exp(output), 1e-8, 1 - 1e-8)
                self.from_logits = True
            if not self.from_logits:
                output = softmax(output, self.axis)

            intersection = reduce_sum(output * target * sample_weight)
            union = reduce_sum(output + target) - intersection
            loss = -log(intersection / union + 1e-7)
            return loss


class SoftIoULoss(Loss):
    def __init__(self, n_classes, reduction="mean", reduced=False):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes
        self.reduction = reduction
        self.reduced = reduced

    def calculate_loss(self, output, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(output)

        pred = softmax(output, dim=1)
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

