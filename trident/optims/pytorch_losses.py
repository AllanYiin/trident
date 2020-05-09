from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from math import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.nn import _reduction as _Reduction
from torch.nn import init
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torchvision import transforms
from torchvision.transforms import functional as tvf

from trident.backend.common import *
from trident.backend.pytorch_backend import *
from trident.backend.pytorch_ops import *
from trident.data.image_common import *
from trident.layers.pytorch_activations import sigmoid


_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__all__ = ['MSELoss', 'CrossEntropyLoss', 'NllLoss', 'BCELoss', 'F1ScoreLoss', 'L1Loss', 'SmoothL1Loss', 'L2Loss', 'CosineSimilarityLoss',
           'ExponentialLoss','ItakuraSaitoLoss', 'make_onehot', 'MS_SSIMLoss', 'CrossEntropyLabelSmooth', 'mixup_criterion', 'DiceLoss',
           'IouLoss', 'focal_loss_with_logits', 'FocalLoss', 'SoftIoULoss', 'CenterLoss', 'TripletLoss',
           'LovaszSoftmax', 'PerceptionLoss', 'EdgeLoss', 'TransformInvariantLoss', 'get_loss']


class _ClassificationLoss(Layer):
    '''Calculate loss for classification task.

    '''
    def __init__(self,axis=1,loss_weights=None, from_logits=False, ignore_index=-100,cutoff=None ,label_smooth=False, reduction='mean',name=None,**kwargs):
        '''

        Args:
            axis (int): the position where the classes is.
            loss_weights (Tensor): means the weights of  classes , it shoud be a 1D tensor and length the same as number of classes.
            from_logits (bool): wheather the output tensor is normalized as a probability (total equal to 1)
            ignore_index (int or list of int):
            cutoff (None or decimal): the cutoff point of probability for classification, should be None of a number less than 1..
            is_target_onehot (bool): Is the target tensor in onehot format?
            label_smooth (bool): Should use label smoothing?
            reduction (string): the method to aggrgate loss. None means no need to aggregate, 'mean' means average loss,
                'sum' means the summation of losses,'batch_mean' means average loss cross the batch axis then summation them.

        '''
        super(_ClassificationLoss, self).__init__(name=name)
        self.need_target_onehot = False
        self.reduction = reduction
        self.axis=axis
        self.from_logits=from_logits
        self.loss_weights=loss_weights
        self.ignore_index=ignore_index
        if cutoff is not None and not 0<cutoff<1:
            raise ValueError('cutoff should between 0 and 1')
        self.cutoff=cutoff
        self.num_classes=None
        self.label_smooth=label_smooth



    def preprocess(self, output:torch.Tensor, target:torch.Tensor,**kwargs):
        #check num_clases
        if self.num_classes is None:
            self.num_classes=output.shape[self.axis]


        #initilize weight
        if self.loss_weights is not None and len(self.loss_weights)!=self.num_classes:
            raise ValueError('weight should be 1-D tensor and length equal to numbers of filters')
        if self.loss_weights is None:
            self.loss_weights=ones(self.num_classes).to(output.device)
        else:
            self.loss_weights=to_tensor(self.loss_weights).to(output.device)

        #ignore_index
        if isinstance(self.ignore_index, int) and 0 <= self.ignore_index < output.size(self.axis):
            self.loss_weights[self.ignore_index] = 0
        elif isinstance(self.ignore_index, (list, tuple)):
            for idx in self.ignore_index:
                if isinstance(idx, int) and 0 <= idx < output.size(self.axis):
                    self.loss_weights[idx] = 0
        if self.label_smooth:
            self.need_target_onehot=True
        #need target onehot but currently not
        if self.need_target_onehot==True and (target>1).float().sum()>0:
            target = make_onehot(target, classes=self.num_classes, axis=self.axis).to(output.device)
            if self.label_smooth:
                target=target*(torch.Tensor(target.size()).uniform_(0.9,1).to(output.device))

        #check is logit
        if  self.from_logits==True or  (0<=output<=1).all() and np.abs(output.sum(self.axis)-1).mean()<1e-4:
            self.from_logits=True
        else:
            # avoid numerical instability with epsilon clipping
            output = clip(softmax(output,self.axis), epsilon(), 1.0 - epsilon())
            self.from_logits=False

            #setting cutoff
            if self.cutoff is not None:
                mask= (output > self.cutoff).float()
                output=output*mask
        return output, target

    def calculate_loss(self,output, target,**kwargs):
        ##dont do aggregation
        raise NotImplementedError
    def postprocess(self,loss):
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return  loss.sum()
        elif self.reduction == 'batch_mean':
            return loss.mean(0).sum()

    def forward(self, output, target,**kwargs):
        loss=self.calculate_loss(*self.preprocess(output, target))
        loss=self.postprocess(loss)
        return loss



class CrossEntropyLoss(_ClassificationLoss):
    '''
    Calculate the cross entropy loss

    Examples:
    >>> output=to_tensor([[0.1, 0.7 , 0.2],[0.3 , 0.6 , 0.1],[0.9 , 0.05 , 0.05],[0.3 , 0.4 , 0.3]]).float()
    >>> print(output.shape)
    torch.Size([4, 3])
    >>> target=to_tensor([1,0,1,2]).long()
    >>> print(target.shape)
    torch.Size([4])
    >>> CrossEntropyLoss(reduction='mean')(output,target).cpu()
    tensor(1.1034)
    >>> CrossEntropyLoss(reduction='sum')(output,target).cpu()
    tensor(4.4136)
    >>> CrossEntropyLoss(label_smooth=True,reduction='mean')(output,target).cpu()
    tensor(1.1034)
    >>> CrossEntropyLoss(loss_weights=to_tensor([1.0,1.0,0]).float(),reduction='mean')(output,target).cpu()
    tensor(0.8259)
    >>> CrossEntropyLoss(ignore_index=2,reduction='mean')(output,target).cpu()
    tensor(0.8259)




    '''
    def __init__(self, axis=1,loss_weights=None, from_logits=False, ignore_index=-100, cutoff=None,label_smooth=False,reduction='mean', name='CrossEntropyLoss'):
        super().__init__(axis,loss_weights, from_logits, ignore_index,cutoff ,label_smooth, reduction,name)
        self._built = True
    def calculate_loss(self, output, target, **kwargs):
        if not self.need_target_onehot:
            loss=torch.nn.functional.cross_entropy(output,target,self.loss_weights,reduction= 'none')
        else:
            reshape_shape=list(ones(len(output.shape)))
            reshape_shape[self.axis]=self.num_classes
            loss = -target *nn.LogSoftmax(dim=self.axis)(output) * reshape(self.loss_weights,reshape_shape)
        return loss




class NllLoss(_Loss):
    def __init__(self, weight=None, with_logits=True, ignore_index=-100, reduction='mean', name='CrossEntropyLoss'):
        super().__init__(reduction=reduction)
        self.name = name
        self.weight = weight
        self.ignore_index = ignore_index
        self.with_logits = with_logits

    def forward(self, output, target):
        if not self.with_logits:
            output = torch.log_softmax(output, dim=1)
        if len(output.size()) == len(target.size()):
            target = argmax(target, 1)
        return F.nll_loss(output, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

    #     if len(output.shape)==2 and len(target.shape)==1:  #         return F.cross_entropy(output, target.long(),
    #     ignore_index=self.ignore_index,reduction=self.reduction)  #     elif len(output.shape)==2 and len(
    #     target.shape)==2:  #         return super().forward(output, argmax(target,1))  #     elif len(
    #     output.shape)>2 and len(target.shape)==len(output.shape)-1:  #         return super().forward(output.view(
    #     -1,output.size(1)), target.long().view(-1))  #     elif len(output.shape)>2 and len(target.shape)==len(
    #     output.shape):  #         return super().forward(output.view(-1,output.size(1)), argmax(target,1).view(-1))
NullLoss=NllLoss

class BCELoss(_Loss):
    def __init__(self, weight=None, reduction='mean', with_logit=False, name='BCELoss'):
        super().__init__()
        self.name = name
        self.reduction = reduction
        self.loss_weight = weight
        self.with_logit = with_logit

    def forward(self, output, target):
        if not self.with_logits:
            output = torch.softmax(output, dim=1)
        target1 = to_numpy(target)
        if self.with_logit == False:
            output = sigmoid(output)
        target1[target1 == 0] = 1
        target1[target1 != 1] = 0
        target = target.float()
        if target1.astype(np.float32).max() == 1:
            if output.shape == target.shape or output.squeeze().shape == target.shape:
                return F.binary_cross_entropy(output, target, weight=self.loss_weight, reduction=self.reduction)
        elif output.shape[1] == int(target1.max()) and len(output.shape) == len(target.shape) + 1:
            max_int = target1.max()
            target = make_onehot(target, max_int + 1)
            return F.binary_cross_entropy(output, target, weight=self.loss_weight, reduction=self.reduction)
        return F.binary_cross_entropy(output, target, weight=self.loss_weight, reduction=self.reduction)


class L1Loss(_Loss):
    def __init__(self, reduction='mean', name='L1Loss'):
        super(L1Loss, self).__init__(reduction)
        self.name = name
        self.reduction = reduction

    def forward(self, output, target):
        if output.shape == target.shape:
            return  F.l1_loss(output, target, reduction=self.reduction)
        else:
            raise ValueError('output and target shape should the same in L2Loss. ')


class L2Loss(_Loss):
    def __init__(self, reduction='mean', name='MSELoss'):
        super(L2Loss, self).__init__(reduction)
        self.name = name
        self.reduction = reduction

    def forward(self, output, target):
        if output.shape == target.shape:
            return 0.5 * F.mse_loss(output, target,reduction=self.reduction)
        else:
            raise ValueError('output and target shape should the same in L2Loss. ')

class SmoothL1Loss(_Loss):
    def __init__(self, reduction='mean', name='SmoothL1Loss'):
        super(SmoothL1Loss, self).__init__(reduction=reduction)
        self.name = name
        self.reduction = reduction
        self.huber_delta = 0.5

    def forward(self, output, target):
        if output.shape == target.shape:
            return F.smooth_l1_loss(output, target, reduction=self.reduction)
        else:
            raise ValueError('output and target shape should the same in SmoothL1Loss. ')


class MSELoss(_Loss):
    def __init__(self, reduction='mean', name='MSELoss'):
        super(MSELoss, self).__init__(reduction=reduction)
        self.name = name
        self.reduction = reduction

    def forward(self, output, target):
        return F.mse_loss(output, target, reduction=self.reduction)


class ExponentialLoss(_Loss):
    def __init__(self, reduction='mean', name='ExponentialLoss'):
        super(ExponentialLoss, self).__init__(reduction=reduction)
        self.name = name
        self.reduction = reduction

    def forward(self, output, target):

        if output.shape == target.shape:
            error = (output - target) ** 2
            loss = 1 - (-1 * error / np.pi).exp()
            if self.reduction == "mean":
                loss = loss.mean()
            if self.reduction == "sum":
                loss = loss.sum()
            if self.reduction == "batchwise_mean":
                loss = loss.mean(0).sum()
            return loss
        else:
            raise ValueError('output and target shape should the same in ExponentialLoss. ')


class ItakuraSaitoLoss(_Loss):
    def __init__(self, reduction='mean', name='ItakuraSaitoLoss'):
        super(ItakuraSaitoLoss, self).__init__(reduction=reduction)
        self.name = name
        self.reduction = reduction

    def forward(self, output, target):
        #  y_true/(y_pred+1e-12) - log(y_true/(y_pred+1e-12)) - 1;
        if output.shape == target.shape:
            if -1<=output.min()<0 and -1<=target.min()<0:
                output=output+1
                target=target+1
            loss= (target/(output+1e-8))-((target+1e-8)/(output+1e-8)).log()-1
            if self.reduction == "mean":
                loss = loss.mean()
            if self.reduction == "sum":
                loss = loss.sum()
            if self.reduction == "batchwise_mean":
                loss = loss.mean(0).sum()
            return loss
        else:
            raise ValueError('output and target shape should the same in ItakuraSaitoLoss. ')

class CosineSimilarityLoss(_Loss):
    def __init__(self, ):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output, target):
        return 1 - torch.cosine_similarity(output, target)



def make_onehot(labels, classes,axis=1):
    one_hot_shape = list(labels.size())
    if axis==-1:
        one_hot_shape.append(classes)
    else:
        one_hot_shape.insert(axis, classes)
    one_hot = torch.zeros(tuple(one_hot_shape)).to(_device)
    target = one_hot.scatter_(axis, labels.unsqueeze(axis).data, 1)
    return target


def gaussian(window_size, sigma=1.5):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None,  full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)


    ret = ssim_map.mean()


    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11,  val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size,  full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class MS_SSIMLoss(_Loss):
    def __init__(self, reduction='mean', window_size=11, max_val=255):
        super(MS_SSIMLoss, self).__init__()
        self.reduction = reduction
        self.window_size = window_size
        self.channel = 3

    def forward(self, output, target):
        return 1-msssim(output, target, window_size=self.window_size,normalize=True)


class CrossEntropyLabelSmooth(_Loss):
    def __init__(self, num_classes,axis=1,weight=None, reduction='mean'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.axis = axis
        self.logsoftmax = nn.LogSoftmax(dim=self.axis)
        self.reduction=reduction

        if isinstance(weight, list):
            weight = to_tensor(np.array(weight))
        elif isinstance(weight, np.ndarray):
            weight = to_tensor(weight)
        if weight is None or isinstance(weight, torch.Tensor):
            self.weight = weight
        else:
            raise ValueError('weight should be 1-D tensor')


    def forward(self, output, target):
        log_probs = self.logsoftmax(output)
        target = make_onehot(target, classes=self.num_classes, axis=self.axis)

        smooth = np.random.uniform( 0, 0.12,target.shape)
        smooth[:,0]=1
        smooth=to_tensor(smooth)
        target = (1 - smooth) * target + smooth / self.num_classes
        if self.weight is not None and len(self.weight)==self.num_classes:
            if self.reduction=='sum':
                return (-target * log_probs*self.weight).sum()
            elif self.reduction=='mean':
                return (-target * log_probs * self.weight).mean()
        else:
            if self.reduction == 'sum':
                return (-target * log_probs ).sum()
            elif self.reduction == 'mean':
                return (-target * log_probs).mean()



def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DiceLoss(_Loss):
    def __init__(self, axis=1,weight=None, smooth=1., ignore_index=-100,cutoff=None, reduction='mean'):
        super(DiceLoss, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.axis=axis
        self.cutoff=cutoff
        if isinstance(weight, list):
            weight = to_tensor(np.array(weight))
        elif isinstance(weight, np.ndarray):
            weight = to_tensor(weight)
        if weight is None or isinstance(weight, torch.Tensor):
            self.weight = weight
        else:
            raise ValueError('weight should be 1-D tensor')

    def forward(self, output, target):
        if self.weight is not None and len(self.weight) != output.size(self.axis):
            raise ValueError('weight should be 1-D tensor and length equal to numbers of filters')

        target = make_onehot(target, classes=output.size(self.axis),axis=self.axis)
        probs = F.softmax(output, dim=self.axis)
        if self.cutoff is not None:
            mask=(probs>self.cutoff).float()
            probs=probs*mask

        if probs.ndim==3:
            if self.weight is None:
                self.weight=to_tensor(np.ones((output.size(self.axis))))
            if isinstance(self.ignore_index,int) and 0<=self.ignore_index<output.size(self.axis):
                self.weight[self.ignore_index]=0
            elif isinstance(self.ignore_index,(list,tuple)):
                for idx in self.ignore_index:
                    if isinstance(idx, int) and 0 <= idx < output.size(self.axis):
                        self.weight[idx] = 0


            # probs=output#*((output>0.5).float())
            if self.reduction == 'mean':

                intersection = (target * probs).sum(1 if self.axis==-1 else -1)
                den1 = probs.sum(1 if self.axis==-1 else -1)
                den2 = target.sum(1 if self.axis==-1 else -1)
                dice = 1 - ((2 * intersection + self.smooth) / (den1 + den2 + self.smooth))
                if self.weight is not None:
                    dice = dice * (self.weight.unsqueeze(0))
                dice1 = 1 - ((2 * intersection[:, 1:].sum() + self.smooth) / (
                            den1[:, 1:].sum() + den2[:, 1:].sum() + self.smooth))
                return dice[:, 1:].mean()  # +dice1#.sum()
            else:

                intersection = (target * probs).sum(1)
                den1 = probs.sum(1)
                den2 = target.sum(1)
                dice = 1 - ((2 * intersection + self.smooth) / (den1 + den2 + self.smooth))
                return (dice.mean(0)*self.weight).sum()

        elif probs.ndim==4:
            # probs=output#*((output>0.5).float())
            if self.reduction == 'mean':
                intersection = (target * probs).sum(-1).sum(-1)
                den1 = probs.sum(-1).sum(-1)
                den2 = target.sum(-1).sum(-1)
                dice = 1 - ((2 * intersection + self.smooth) / (den1 + den2 + self.smooth))
                if self.weight is not None:
                    dice = dice * (self.weight.unsqueeze(0))
                dice1 = 1 - ((2 * intersection[:, 1:].sum() + self.smooth) / ( den1[:, 1:].sum() + den2[:, 1:].sum() + self.smooth))
                return dice[:, 1:].mean()  # +dice1#.sum()
            else:
                intersection = (target * probs)[:, 1:].sum()
                den1 = probs[:, 1:].sum()
                den2 = target[:, 1:].sum()
                dice = 1 - ((2 * intersection + self.smooth) / (den1 + den2 + self.smooth))
                return dice


class IouLoss(_Loss):
    def __init__(self, ignore_index=-1000, reduction='mean'):
        super(IouLoss, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, output, target):
        output = argmax(output, 1)
        output_flat = output.reshape(-1)
        target_flat = target.reshape(-1)
        output_flat = output_flat[target_flat != self.ignore_index]
        target_flat = target_flat[target_flat != self.ignore_index]
        output_flat = output_flat[target_flat != 0]
        target_flat = target_flat[target_flat != 0]
        intersection = (output_flat == target_flat).float().sum()
        union = ((output_flat + target_flat) > 0).float().sum().clamp(min=1)
        loss = -(intersection / union).log()
        return loss


def focal_loss_with_logits(input: torch.Tensor, target: torch.Tensor, gamma=2.0, alpha=0.25, reduction="mean",
                           normalized=False, threshold=None, ) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.

    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    if len(target.shape) == len(input.shape):
        target = target.type(input.type())
    elif len(target.shape) + 1 == len(input.shape) and target.shape[0] == input.shape[0]:
        num_classes = None
        if input.shape[1:-1] == target.shape[1:]:
            num_classes = input.shape[-1]
        elif input.shape[2:] == target.shape[1:]:
            num_classes = input.shape[1]
        target = make_onehot(target, num_classes)
        if target.shape != input.shape:
            target = target.reshape(input.shape)

    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    if threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / threshold).pow(gamma)
        focal_term[pt < threshold] = 1

    loss = -focal_term * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if normalized:
        norm_factor = focal_term.sum()
        loss = loss / norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)
    return loss


class FocalLoss(_Loss):
    def __init__(self, with_logits=False,alpha=0.5, gamma=2, ignore_index=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.with_logits=with_logits


    def forward(self, output, target):
        if not self.with_logits:
            output=F.softmax(output,dim=1)

        num_classes = output.size(1)
        if target.dtype == torch.int64:
            target_tensor = make_onehot(target, num_classes)
        loss = 0

        # Filter anchors with -1 label from loss computation
        not_ignored = target
        if self.ignore_index is not None:
            not_ignored = target != self.ignore_index

        for cls in range(num_classes):
            cls_target = (target == cls).long()
            cls_input = output[:, cls, ...]

            if self.ignore_index is not None:
                cls_target = cls_target[not_ignored]
                cls_input = cls_input[not_ignored]
            loss += focal_loss_with_logits(cls_input, cls_target, gamma=self.gamma, alpha=self.alpha,
                                           reduction=self.reduction)

        return loss / output.size(0) if self.reduction == 'mean' else loss


class SoftIoULoss(_Loss):
    def __init__(self, n_classes, reduction="mean", reduced=False):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes
        self.reduction = reduction
        self.reduced = reduced

    def forward(self, output, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(output)

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


class LovaszSoftmax(_Loss):
    def __init__(self, reduction="mean", reduced=False):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
        self.reduced = reduced
        self.num_classes = None

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, output, target):
        num_classes = output.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            if num_classes == 1:
                input_c = output[:, 0]
            else:
                input_c = output[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(_lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

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
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    def forward(self, output, target):
        # print(output.shape, target.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        self.num_classes = output.size(1)
        output, target = self.prob_flatten(output, target)

        # print(output.shape, target.shape)

        losses = self.lovasz_softmax_flat(output, target) if self.num_classes > 2 else self.lovasz_hinge_flat(output,
                                                                                                              target)
        return losses


class TripletLoss(_Loss):
    """Triplet loss with hard positive/negative mining.
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, global_feat, labels, margin=0.3, reduction="mean", reduced=False):
        super(TripletLoss, self).__init__()
        self.reduction = reduction
        self.reduced = reduced
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            target (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = output.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(output, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, output, output.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        if self.reduction == 'mean':
            return self.ranking_loss(dist_an, dist_ap, y).mean()
        else:
            return self.ranking_loss(dist_an, dist_ap, y).sum()


class CenterLoss(_Loss):
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

    def forward(self, output, target):
        """
        Args:
            output: feature matrix with shape (batch_size, feat_dim).
            target: ground truth labels with shape (num_classes).

        """
        assert output.size(0) == target.size(0), "features.size(0) is not equal to labels.size(0)"
        batch_size = output.size(0)
        distmat = torch.pow(output, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + torch.pow(
            self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, output, self.centers.t())
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
            return dist.mean() / self.num_classes
        else:
            return dist.sum() / self.num_classes


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.to(_device)

    def forward(self, output, target):
        target.detach()
        img_size = list(output.size())
        G = gram_matrix(output)
        Gt = gram_matrix(target)
        return F.mse_loss(G, Gt).div((img_size[1] * img_size[2] * img_size[3]))


class PerceptionLoss(_Loss):
    def __init__(self, net, reduction="mean"):
        super(PerceptionLoss, self).__init__()
        self.ref_model = net
        self.ref_model.trainable = False
        self.ref_model.eval()
        self.layer_name_mapping = {'3': "block1_conv2", '8': "block2_conv2", '15': "block3_conv3", '22': "block4_conv3"}
        for name, module in self.ref_model.named_modules():
            if name in list(self.layer_name_mapping.values()):
                module.keep_output = True

        self.reduction = reduction
        self.to(_device)

    def preprocess(self,img):
        return ((img+1)/2)*to_tensor([[0.485, 0.456, 0.406]]).unsqueeze(-1).unsqueeze(-1)+to_tensor([[0.229, 0.224, 0.225]]).unsqueeze(-1).unsqueeze(-1)



    def forward(self, output, target):
        target_features = OrderedDict()
        output_features = OrderedDict()
        _ = self.ref_model(self.preprocess(output))
        for item in self.layer_name_mapping.values():
            output_features[item] = getattr(self.ref_model, item).output

        _ = self.ref_model(self.preprocess(target))
        for item in self.layer_name_mapping.values():
            target_features[item] = getattr(self.ref_model, item).output.detach()

        loss = 0
        num_filters=0
        for i in range(len(self.layer_name_mapping)):
            b,c,h,w=output_features.value_list[i].shape
            loss += ((output_features.value_list[i] - target_features.value_list[i]) ** 2).sum()/(h*w)
            num_filters+=c
        return loss/(output.size(0)*num_filters)


class EdgeLoss(_Loss):
    def __init__(self, reduction="mean"):
        super(EdgeLoss, self).__init__()

        self.reduction = reduction

        self.styleloss = StyleLoss()
        self.to(_device)

    def first_order(self, x, axis=2):
        h, w = x.size(2), x.size(3)
        if axis == 2:
            return (x[:, :, :h - 1, :w - 1] - x[:, :, 1:, :w - 1]).abs()
        elif axis == 3:
            return (x[:, :, :h - 1, :w - 1] - x[:, :, :h - 1, 1:]).abs()
        else:
            return None

    def forward(self, output, target):
        loss = MSELoss(reduction=self.reduction)(self.first_order(output, 2), self.first_order(target, 2)) + MSELoss(
            reduction=self.reduction)(self.first_order(output, 3), self.first_order(target, 3))
        return loss


class TransformInvariantLoss(nn.Module):
    def __init__(self, loss: _Loss, embedded_func: Layer):
        super(TransformInvariantLoss, self).__init__()
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
        mat = get_rotation_matrix2d(center_tensor, angle, scale).float().to(output.device)
        rotated_target = warp_affine(target, mat, target.size()[2:]).float().to(output.device)

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
        gradients = torch.autograd.grad(outputs=disc_interpolates, output=interpolates,
                                        grad_outputs=real_data.new_ones(disc_interpolates.size()), create_graph=True,
                                        retain_graph=True, only_output=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean() * self.l


class F1ScoreLoss(_Loss):
    """
    This operation computes the f-measure between the output and target. If beta is set as one,
    its called the f1-scorce or dice similarity coefficient. f1-scorce is monotonic in jaccard distance.

    f-measure = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    This loss function is frequently used in semantic segmentation of images. Works with imbalanced classes, for
    balanced classes you should prefer cross_entropy instead.
    This operation works with both binary and multiclass classification.

    Args:
        output: the output values from the network
        target: it is usually a one-hot vector where the hot bit corresponds to the label index
        beta: greater than one weights recall higher than precision, less than one for the opposite.
        Commonly chosen values are 0.5, 1 or 2.

    Returns:
        :class:`~cntk.ops.functions.Function`

    """

    def __init__(self, reduction='mean', num_class=2, ignore_index=-100, beta=1):
        super(F1ScoreLoss, self).__init__(reduction=reduction)
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.beta = beta

    def forward(self, output, target):
        num_class = self.num_class
        output_class = None
        if target.ndim == 2:
            target = target.view(-1)
            if output.ndim == 2:
                output = output.view(-1)
            else:
                output = output.view(-1, output.size(2)).argmax(dim=1)

        if (target.ndim == 1 and (output.ndim == 1 or output.ndim == 2)):
            if output.ndim == 2:
                output_class = output.size(1)
                output = output.argmax(dim=1)

            if output_class is not None and output_class != self.num_class:
                num_class = output_class
            f1 = 0
            n = 0
            for k in range(num_class):
                if k != self.ignore_index:
                    k_output = torch.eq(output, k).float()
                    k_target = torch.eq(target, k).float()
                    if num_class >= 3 and 0 <= self.ignore_index < num_class:
                        k_output = torch.eq(output[target != self.ignore_index], k).float()
                        k_target = torch.eq(target[target != self.ignore_index], k).float()
                    tp = (k_target * k_output).sum().to(torch.float32)
                    tn = ((1 - k_target) * (1 - k_output)).sum().to(torch.float32)
                    fp = ((1 - k_target) * k_output).sum().to(torch.float32)
                    fn = (k_target * (1 - k_output)).sum().to(torch.float32)
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)

                    f1 = f1 + (1 - (1 + self.beta ** 2) * (precision * recall) / (
                                self.beta ** 2 * precision + recall + 1e-8))
                    n += 1
            if self.reduction == 'mean':
                f1 = f1 / n
            return f1
        else:
            raise ValueError(
                'target.ndim:{0}  output.ndim:{1} is not valid for F1 score calculation'.format(target.ndim,
                                                                                                output.ndim))


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
