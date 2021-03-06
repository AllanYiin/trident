from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import warnings
import builtins
import numpy as np
import os

from trident.data.vision_transforms import Unnormalize

from trident.backend.opencv_backend import array2image

from trident.backend.common import *
from trident.backend.common import get_backend
from trident.callbacks.callback_base import CallbackBase
from trident.data.image_common import *
from trident.misc.ipython_utils import is_in_colab

if get_backend() == 'pytorch':
    import torch.nn as nn
    from trident.backend.pytorch_backend import get_device
    from trident.backend.pytorch_ops import to_numpy, to_tensor, arange, shuffle, cast, clip, sqrt, int_shape

elif get_backend() == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import get_device
    from trident.backend.tensorflow_ops import to_numpy, to_tensor, arange, shuffle, cast, clip, sqrt, int_shape, concate, zeros_like, ones_like

working_directory = get_session().working_directory
__all__ = ['RegularizationCallbacksBase', 'MixupCallback', 'CutMixCallback']


class RegularizationCallbacksBase(CallbackBase):
    """ The base callback class for regularization  """

    def __init__(self):
        super(RegularizationCallbacksBase, self).__init__()

    pass


class MixupCallback(RegularizationCallbacksBase):
    """ Implementation. of the mixup regularization
     Mixup - a neural network regularization technique based on linear interpolation
     of labeled sample pairs - has stood out by its capacity to improve model's robustness
     and generalizability through a surprisingly simple formalism.

    References:
        mixup: BEYOND EMPIRICAL RISK MINIMIZATION
        https://arxiv.org/pdf/1710.09412.pdf

    """

    def __init__(self, alpha=1, loss_criterion=None, loss_weight=1, save_path=None, **kwargs):
        super(MixupCallback, self).__init__()
        self.alpha = alpha
        if loss_criterion is None:
            loss_criterion = get_class('CrossEntropyLoss', 'trident.optims.pytorch_losses' if get_backend() == 'pytorch' else 'trident.optims.tensorflow_losses')

        self.loss_criterion = loss_criterion()
        self.loss_weight = loss_weight
        if save_path is None:
            self.save_path = os.path.join(working_directory, 'Results')
        else:
            self.save_path = save_path
        make_dir_if_need(self.save_path)

    def on_loss_calculation_end(self, training_context):
        """Returns mixed inputs, pairs of targets, and lambda"""
        model = training_context['current_model']
        train_data = training_context['train_data']
        x = None
        y = None
        x = train_data.value_list[0].copy().detach().to(model.device)  # input
        y = train_data.value_list[1].copy().detach().to(model.device)  # label

        lam = builtins.min(builtins.max(np.random.beta(self.alpha, self.alpha), 0.3), 0.7)

        batch_size = int_shape(x)[0]
        index = arange(batch_size)
        index = cast(shuffle(index), 'long')
        this_loss = None
        mixed_x = None
        if get_backend() == 'pytorch':
            mixed_x = lam * x + (1 - lam) * x[index, :]
            pred = model(to_tensor(mixed_x, requires_grad=True, device=model.device))
            y_a, y_b = y, y[index]
            this_loss = lam * self.loss_criterion(pred, y_a.long()) + (1 - lam) * self.loss_criterion(pred, y_b.long())
            training_context['current_loss'] = training_context['current_loss'] + this_loss * self.loss_weight
            if training_context['is_collect_data']:
                training_context['losses'].collect('mixup_loss', training_context['steps'], float(to_numpy(this_loss * self.loss_weight)))

        elif get_backend() == 'tensorflow':
            with tf.device(get_device()):
                x1 = tf.gather(x, index, axis=0)
                y1 = tf.gather(y, index, axis=0)
                mixed_x = lam * x + (1 - lam) * x1
                pred = model(to_tensor(mixed_x, requires_grad=True))
                y_a, y_b = y, y1

                this_loss = lam * self.loss_criterion(pred, y_a) + (1 - lam) * self.loss_criterion(pred, y_b)

                training_context['current_loss'] = training_context['current_loss'] + this_loss * self.loss_weight
                if training_context['is_collect_data']:
                    training_context['losses'].collect('mixup_loss', training_context['steps'], float(to_numpy(this_loss * self.loss_weight)))

        if training_context['current_batch'] == 0:
            for item in mixed_x:
                if self.save_path is None and not is_in_colab():
                    item = Unnormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(to_numpy(item))
                    item = Unnormalize(0, 255)(item)
                    array2image(item).save(os.path.join(self.save_path, 'mixup_{0}.jpg'.format(get_time_suffix())))
                elif self.save_path is not None:
                    item = Unnormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(to_numpy(item))
                    item = Unnormalize(0, 255)(item)
                    array2image(item).save(os.path.join(self.save_path, 'mixup_{0}.jpg'.format(get_time_suffix())))
        mixed_x = None
        x = None
        y = None


class CutMixCallback(RegularizationCallbacksBase):
    """Implementation. of the cutmix regularization
    CutMix is a way to combine two images. It comes from MixUp and Cutout. In this
    data augmentation technique:patches are cut and pasted among training images
    where the ground truth labels are also mixed proportionally to the area of the patches

    References:
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
        https://arxiv.org/abs/1905.04899



    """

    def __init__(self, alpha=1, loss_criterion=None, loss_weight=1, save_path=None, **kwargs):
        super(CutMixCallback, self).__init__()
        self.alpha = alpha
        if loss_criterion is None:
            loss_criterion = get_class('CrossEntropyLoss', 'trident.optims.pytorch_losses' if get_backend() == 'pytorch' else 'trident.optims.tensorflow_losses')

        self.loss_criterion = loss_criterion()
        self.loss_weight = loss_weight
        if save_path is None:
            self.save_path = os.path.join(working_directory, 'Results')
        else:
            self.save_path = save_path
        make_dir_if_need(self.save_path)

    def rand_bbox(self, width, height, lam):
        """

        Args:
            width ():
            height ():
            lam ():

        Returns:

        """
        W = width
        H = height
        cut_rat = math.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def on_loss_calculation_end(self, training_context):
        """Returns mixed inputs, pairs of targets, and lambda"""
        model = training_context['current_model']
        train_data = training_context['train_data']
        x = None
        y = None
        x = train_data.value_list[0].copy().detach().to(model.device)  # input
        y = train_data.value_list[1].copy().detach().to(model.device)  # label

        lam = builtins.min(builtins.max(np.random.beta(self.alpha, self.alpha), 0.1), 0.4)

        batch_size = int_shape(x)[0]
        index = cast(arange(batch_size), 'int64')
        index = shuffle(index)

        this_loss = None
        if get_backend() == 'pytorch':
            y_a, y_b = y, y[index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.shape[3], x.shape[2], lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[3] * x.shape[2]))
            pred = model(to_tensor(x, requires_grad=True, device=model.device))
            this_loss = lam * self.loss_criterion(pred, y_a.long()) + (1 - lam) * self.loss_criterion(pred, y_b.long())
            training_context['current_loss'] = training_context['current_loss'] + this_loss * self.loss_weight
            if training_context['is_collect_data']:
                training_context['losses'].collect('cutmix_loss', training_context['steps'], float(to_numpy(this_loss * self.loss_weight)))

        elif get_backend() == 'tensorflow':
            with tf.device(get_device()):
                y1 = tf.gather(y, index, axis=0)
                x1 = tf.gather(x, index, axis=0)
                y_a, y_b = y, y1
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.shape[2], x.shape[1], lam)
                filter = np.zeros(int_shape(x))
                filter[:, bbx1:bbx2, bby1:bby2, :] = 1
                filter = to_tensor(x)
                x = x * (1 - filter) + x1 * filter
                # x[:, bbx1:bbx2, bby1:bby2, :] = x1[:, bbx1:bbx2, bby1:bby2,:]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[2] * x.shape[1]))
                pred = model(to_tensor(x, requires_grad=True))
                loss1 = self.loss_criterion(pred, y_a)
                loss2 = self.loss_criterion(pred, y_b)
                this_loss = lam * loss1 + (1 - lam) * loss2
                training_context['current_loss'] = training_context['current_loss'] + this_loss * self.loss_weight
                if training_context['is_collect_data']:
                    training_context['losses'].collect('cutmix_loss', training_context['steps'], float(to_numpy(this_loss * self.loss_weight)))

        if training_context['current_batch'] == 0:
            if self.save_path is None and not is_in_colab():
                for item in x:
                    item = Unnormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(to_numpy(item))
                    item = Unnormalize(0, 255)(item)
                    array2image(item).save(os.path.join(self.save_path, 'cutmix_{0}.jpg'.format(get_time_suffix())))

            elif self.save_path is not None:
                for item in x:
                    item = Unnormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(to_numpy(item))
                    item = Unnormalize(0, 255)(item)
                    array2image(item).save(os.path.join(self.save_path, 'cutmix_{0}.jpg'.format(get_time_suffix())))
        x = None
        y = None


class GradientClippingCallback(RegularizationCallbacksBase):
    """

    """

    def __init__(self, max_norm=2, norm_type='l2', clip_value=None, **kwargs):
        super(GradientClippingCallback, self).__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_value = clip_value

    def on_optimization_step_start(self, training_context):
        if get_backend() == 'pytorch':
            if self.clip_value is not None:
                nn.utils.clip_grad_value_(training_context['optimizer'].param_groups['params'], self.clip_value)
            else:
                nn.utils.clip_grad_norm_(training_context['optimizer'].param_groups['params'], max_norm=self.max_norm, norm_type=2 if self.norm_type == 'l2' else 1)
        elif get_backend() == 'tensorflow':
            # training_context['grads']
            pass
        elif get_backend() == 'cntk':
            pass










