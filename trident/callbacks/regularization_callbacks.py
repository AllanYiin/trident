#
  ttr (training time regulazation)
#

import warnings
import math
import numpy as np
from  ..callbacks import CallbackBase
from ..backend.common import *
from ..backend.load_backend import get_backend

if get_backend()=='pytorch':
    from ..backend.pytorch_backend import to_numpy,to_tensor
    from ..optims.pytorch_losses import CrossEntropyLoss
elif get_backend()=='tensorflow':
    from ..backend.tensorflow_backend import  to_numpy,to_tensor
    from ..optims.tensorflow_losses import CrossEntropyLoss
elif get_backend()=='cntk':
    from ..backend.cntk_backend import  to_numpy,to_tensor
    from ..optims.cntk_losses import CrossEntropyLoss


__all__ = ['TrainingTimeRegularizationCallbacksBase', 'MixupCallback', 'CutMixCallback']

class TrainingTimeRegularizationCallbacksBase(CallbackBase):
    def __init__(self):
        super(TrainingTimeRegularizationCallbacksBase, self).__init__()

    pass



class MixupCallback(TrainingTimeRegularizationCallbacksBase):

    def __init__(self, alpha=1,loss_criterion=CrossEntropyLoss,loss_weight=1,  **kwargs):
        super(MixupCallback, self).__init__()
        self.alpha=alpha
        self.loss_criterion=loss_criterion()
        self.loss_weight=loss_weight
    def post_loss_calculation(self, training_context):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        x=to_numpy(training_context['current_input'])
        y=to_numpy(training_context['current_target'])
        model=training_context['current_model']






        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = x.shape[0]
        index = np.arange(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        pred=model(to_tensor(mixed_x))
        this_loss=lam * self.loss_criterion(pred, to_tensor(y_a)) + (1 - lam) * self.loss_criterion(pred,to_tensor(y_b))
        if 'mixup_loss' not in training_context['losses']:
            training_context['losses']['mixup_loss'] = []
        training_context['current_loss'] = training_context['current_loss'] + this_loss *self.loss_weight
        if training_context['is_collect_data']:
            training_context['losses']['mixup_loss'].append(float(to_numpy(this_loss) * self.loss_weight))

class CutMixCallback(TrainingTimeRegularizationCallbacksBase):

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

    def post_loss_calculation(self, training_context):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        x = to_numpy(training_context['current_input'])
        y = to_numpy(training_context['current_target'])
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

        pred = model(to_tensor(x))
        this_loss = lam * self.loss_criterion(pred, to_tensor(y_a)) + (1 - lam) * self.loss_criterion(pred, to_tensor(y_b))
        if 'mixup_loss' not in training_context['losses']:
            training_context['losses']['cutmix_loss'] = []
        training_context['current_loss'] = training_context['current_loss'] + this_loss *self.loss_weight
        if training_context['is_collect_data']:
            training_context['losses']['cutmix_loss'].append(float(to_numpy(this_loss) * self.loss_weight))







