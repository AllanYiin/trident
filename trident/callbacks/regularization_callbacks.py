from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import warnings

import numpy as np

from trident.backend.common import *
from trident.backend.load_backend import get_backend
from trident.callbacks.callback_base import CallbackBase
from trident.data.image_common import *

if get_backend()=='pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_ops import to_numpy,to_tensor
    from trident.optims.pytorch_losses import CrossEntropyLoss
elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import  to_numpy,to_tensor
    from trident.optims.tensorflow_losses import CrossEntropyLoss


__all__ = ['RegularizationCallbacksBase', 'MixupCallback', 'CutMixCallback']

class RegularizationCallbacksBase(CallbackBase):
    def __init__(self):
        super(RegularizationCallbacksBase, self).__init__()

    pass



class MixupCallback(RegularizationCallbacksBase):

    def __init__(self, alpha=1,loss_criterion=CrossEntropyLoss,loss_weight=1,  **kwargs):
        super(MixupCallback, self).__init__()
        self.alpha=alpha
        self.loss_criterion=loss_criterion()
        self.loss_weight=loss_weight
    def on_loss_calculation_end(self, training_context):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        train_data = training_context['train_data']
        x = to_numpy(train_data.value_list[0])
        y = to_numpy(train_data.value_list[1])
        model=training_context['current_model']



        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = x.shape[0]
        index = np.arange(batch_size)
        np.random.shuffle(index)
        mixed_x = lam * x + (1 - lam) * x[index, :]


        if training_context['current_batch']==0:
            for item in mixed_x:
                item=unnormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(item)
                item=unnormalize(0, 255)(item)
                array2image(item).save('Results/mixup_{0}.jpg'.format(get_time_suffix()))

        y_a, y_b = y, y[index]
        pred=model(to_tensor(mixed_x,requires_grad=True))
        this_loss=lam * self.loss_criterion(pred, to_tensor(y_a)) + (1 - lam) * self.loss_criterion(pred,to_tensor(y_b))
        if 'mixup_loss' not in training_context['losses']:
            training_context['losses']['mixup_loss'] = []
        training_context['current_loss'] = training_context['current_loss'] + this_loss *self.loss_weight
        if training_context['is_collect_data']:
            training_context['losses']['mixup_loss'].append(float(to_numpy(this_loss) * self.loss_weight))

class CutMixCallback(RegularizationCallbacksBase):

    def __init__(self, alpha=1,loss_criterion=CrossEntropyLoss,loss_weight=1,  **kwargs):
        super(CutMixCallback, self).__init__()
        self.alpha=alpha
        self.loss_criterion=loss_criterion()
        self.loss_weight=loss_weight

    def rand_bbox(self, width, height, lam):
        W = width
        H = height
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def on_loss_calculation_end(self, training_context):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        train_data = training_context['train_data']
        x = to_numpy(train_data.value_list[0])
        y = to_numpy(train_data.value_list[1])
        model = training_context['current_model']
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.shape[0]
        index = np.arange(batch_size)
        np.random.shuffle(index)
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.shape[3], x.shape[2], lam)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[3] * x.shape[2]))
        if training_context['current_batch'] == 0:
            for item in x:
                item = unnormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(item)
                item = unnormalize(0, 255)(item)
                array2image(item).save('Results/cutmix_{0}.jpg'.format(get_time_suffix()))

        pred = model(to_tensor(x,requires_grad=True))
        this_loss = lam * self.loss_criterion(pred, to_tensor(y_a)) + (1 - lam) * self.loss_criterion(pred, to_tensor(y_b))
        if 'mixup_loss' not in training_context['losses']:
            training_context['losses']['cutmix_loss'] = []
        training_context['current_loss'] = training_context['current_loss'] + this_loss *self.loss_weight
        if training_context['is_collect_data']:
            training_context['losses']['cutmix_loss'].append(float(to_numpy(this_loss) * self.loss_weight))



class GradientClippingCallback(RegularizationCallbacksBase):

    def __init__(self, max_norm=2,norm_type='l2',clip_value=None,  **kwargs):
        super(GradientClippingCallback, self).__init__()
        self.max_norm=max_norm
        self.norm_type=norm_type
        self.clip_value=clip_value

    def on_optimization_step_start(self, training_context):
        if get_backend() == 'pytorch':
            if self.clip_value is not None:
                nn.utils.clip_grad_value_(training_context['optimizer'].param_groups['params'],self.clip_value)
            else:
                nn.utils.clip_grad_norm_(training_context['optimizer'].param_groups['params'],max_norm=self.max_norm,norm_type=2 if self.norm_type=='l2' else 1)
        elif get_backend() == 'tensorflow':
            #training_context['grads']
            pass
        elif get_backend() == 'cntk':
            pass










