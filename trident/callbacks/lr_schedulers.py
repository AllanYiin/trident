"""Learning Rate Scheduler Callbacks"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import warnings
import types
import numpy as np

from trident.backend.common import *
from trident.backend.load_backend import get_backend
from trident.callbacks.callback_base import CallbackBase

_session = get_session()

if get_backend()=='pytorch':
    from trident.backend.pytorch_ops import to_numpy,to_tensor
elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import  to_numpy,to_tensor



__all__ = ['AdjustLRCallback','ReduceLROnPlateau','reduce_lr_on_plateau','LambdaLR','lambda_lr','RandomCosineLR','random_cosine_lr','CosineLR','cosine_lr']





class AdjustLRCallbackBase(CallbackBase):
    """Basic class for learning rate scheduler"""
    def __init__(self):
        super(AdjustLRCallbackBase, self).__init__()
        self.base_lr=1e-3
        self.base_lrs = [1e-3]
    def adjust_learning_rate(self,training_context,new_lr):
        if hasattr(training_context['optimizer'],'adjust_learning_rate' ) and isinstance(training_context['optimizer'].adjust_learning_rate, types.MethodType):
            training_context['optimizer'].adjust_learning_rate(new_lr,True)
        else:
            old_lr=training_context['optimizer'].lr
            training_context['optimizer'].param_groups[0]['lr'] = new_lr
            training_context['current_lr'] =new_lr
            print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))


class AdjustLRCallback(AdjustLRCallbackBase):
    def __init__(self, index: int, unit:str='epoch',new_lr:float=1e-3):
        super().__init__()
        self.unit=unit
        self.index=index
        self.new_lr=new_lr

    def on_batch_end(self, training_context):
        if self.unit == 'batch' and training_context['steps'] == self.index:
            self.adjust_learning_rate(training_context,self.new_lr)
    def on_epoch_end(self, training_context):
        if self.unit == 'epoch' and training_context['current_epoch'] == self.index and training_context['current_batch'] == 0:
            self.adjust_learning_rate(training_context,self.new_lr)



class ReduceLROnPlateau(AdjustLRCallbackBase):
    """
    Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    """

    def __init__(self, monitor='total_losses', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,unit_base='epoch',
                 **kwargs):
        """

        Args:
            monitor: quantity to be monitored.
            factor: factor by which the learning rate will
                be reduced. new_lr = lr * factor
            patience: number of epochs that produced the monitored
                quantity with no improvement after which training will
                be stopped.
                Validation quantities may not be produced for every
                epoch, if the validation frequency
                (`model.fit(validation_freq=5)`) is greater than one.
            verbose: int. 0: quiet, 1: update messages.
            mode: one of {auto, min, max}. In `min` mode,
                lr will be reduced when the quantity
                monitored has stopped decreasing; in `max`
                mode it will be reduced when the quantity
                monitored has stopped increasing; in `auto`
                mode, the direction is automatically inferred
                from the name of the monitored quantity.
            min_delta: threshold for measuring the new optimum,
                to only focus on significant changes.
            cooldown: number of epochs to wait before resuming
                normal operation after lr has been reduced.
            min_lr: lower bound on the learning rate.

        """
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best = 0
        if monitor=='total_losses':
            mode='min'
        self.mode = mode
        self.monitor_op = None
        self.unit_base=unit_base
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_epoch_end(self, training_context):
        training_context['current_lr']=training_context['optimizer'].lr
        if self.unit_base == 'epoch':
            steps=None
            history=None
            if self.monitor in training_context['losses']:
                steps, history = training_context['losses'].get_series(self.monitor)
            elif self.monitor in training_context['metrics']:
                steps, history = training_context['metrics'].get_series(self.monitor)
            else:
                steps, history = training_context['losses'].get_series('total_losses')


            current = to_numpy(history[-min(5, len(history)):]).mean()
            if current is None:
                warnings.warn(
                    'Reduce LR on plateau conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (self.monitor, ','.join(training_context['metrics'].keys_list)), RuntimeWarning
                )

            else:
                if self.in_cooldown():
                    self.cooldown_counter -= 1
                    self.wait = 0

                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                elif not self.in_cooldown():
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(training_context['optimizer'].lr)
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            self.adjust_learning_rate(training_context, new_lr)


                            if self.verbose > 0:
                                print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                      'learning rate to %s.' % (training_context['current_epoch'] + 1, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

    def on_batch_end(self, training_context):
        if self.unit_base is None:
            if training_context['total_batch']>_session.epoch_equivalent:
                self.unit_base='batch'
                print('one epoch have {0} batches, use {1} as epoch equivalent in long epoch. '.format(training_context['total_batch'],_session.epoch_equivalent))
            else:
                self.unit_base = 'epoch'
                print('ReduceLROnPlateau reseted.')

        num_batches = training_context['steps']
        if self.unit_base=='batch' and training_context['steps']>0 and training_context['steps']%_session.epoch_equivalent==0:
            training_context['current_lr']=training_context['optimizer'].lr
            history=training_context['losses'].get(self.monitor,training_context['metrics'].get(self.monitor,training_context['losses']['total_losses']))
            steps,values=zip(*history)
            current =to_numpy(values[-min(5,len(values)):]).mean()
            if current is None:
                warnings.warn(
                    'Reduce LR on plateau conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (self.monitor, ','.join(training_context['metrics'].keys_list)), RuntimeWarning
                )

            else:
                if self.in_cooldown():
                    self.cooldown_counter -= 1
                    self.wait = 0

                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                elif not self.in_cooldown():
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(training_context['optimizer'].lr)
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            self.adjust_learning_rate(training_context, new_lr)

                            if self.verbose > 0:
                                print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                      'learning rate to %s.' % (training_context['current_epoch'], new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0


    def in_cooldown(self):
        return self.cooldown_counter > 0

def reduce_lr_on_plateau(monitor='total_loss' ,verbose=True, mode='min', factor=0.5, patience=5, threshold=1e-4, cooldown=0, min_lr=1e-8, **kwargs):
    """
     The function to initialize ReduceLROnPlateau
    Args:

         monitor: quantity to be monitored.
         factor: factor by which the learning rate will
             be reduced. new_lr = lr * factor
         patience: number of epochs that produced the monitored
             quantity with no improvement after which training will
             be stopped.
             Validation quantities may not be produced for every
             epoch, if the validation frequency
             (`model.fit(validation_freq=5)`) is greater than one.
         verbose: int. 0: quiet, 1: update messages.
         mode: one of {auto, min, max}. In `min` mode,
             lr will be reduced when the quantity
             monitored has stopped decreasing; in `max`
             mode it will be reduced when the quantity
             monitored has stopped increasing; in `auto`
             mode, the direction is automatically inferred
             from the name of the monitored quantity.
         threshold: threshold for measuring the new optimum,
             to only focus on significant changes.

         cooldown: number of epochs to wait before resuming
             normal operation after lr has been reduced.
         min_lr: lower bound on the learning rate.

    Returns:

    """
    return ReduceLROnPlateau(monitor=monitor,
                            mode=mode,factor=factor,patience=patience,verbose=verbose,
                            min_delta=threshold,
                            cooldown=cooldown,min_lr=min_lr)


class LambdaLR(AdjustLRCallbackBase):
    def __init__(self, offset=0,decay_start_epoch=50, **kwargs):
        super(LambdaLR, self).__init__()
        self.offset=offset
        self.decay_start_epoch=decay_start_epoch

    def on_epoch_end(self, training_context):
        n_epochs = training_context['total_epoch']
        epoch = training_context['current_epoch']
        if epoch>=10:
            lr=1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (n_epochs - self.decay_start_epoch)
            training_context['optimizer'].adjust_learning_rate(lr,True)

def lambda_lr(offset=0,decay_start_epoch=50):
   return LambdaLR(offset=offset,decay_start_epoch=decay_start_epoch)


class RandomCosineLR(AdjustLRCallbackBase):
    def __init__(self, min_lr=1e-8,period=1000,noise_weight=0.1, random_start_epoch=3,**kwargs):
        super(RandomCosineLR, self).__init__()
        self.max_lr = None

        self.min_lr = min_lr
        self.period=period
        self.T_max=period/2
        self.T_current = 0
        self.noise_weight=noise_weight
        self.random_start_epoch=random_start_epoch

    def on_batch_end(self, training_context):
        if self.max_lr is None:
            self.max_lr = training_context['base_lr']

        self.T_current=training_context['current_epoch'] * training_context['total_batch'] + training_context[ 'current_batch']% (2 * self.T_max)

        if self.T_current == 0:
            training_context['optimizer'].adjust_learning_rate(self.max_lr,verbose=False)
        elif (self.T_current - 1 - self.T_max) % (2 * self.T_max) == 0:
            training_context['optimizer'].adjust_learning_rate((self.min_lr+ (self.max_lr - self.min_lr)) *(1 - math.cos(math.pi / self.T_max)) / 2+ self.noise_weight * (random.random() - 0.5),verbose=False)
        else:
            training_context['optimizer'].adjust_learning_rate((( 1 + math.cos(math.pi * (self.T_current / self.T_max)) )*(self.max_lr- self.min_lr)*0.5 + self.min_lr)*(1+self.noise_weight * (random.random()- 0.5)),verbose=False)



def random_cosine_lr( min_lr=1e-8,period=100,cosine_weight=0.2,noise_weight=0.3, random_start_epoch=3,**kwargs):
   return RandomCosineLR(period=period,cosine_weight=cosine_weight,noise_weight=noise_weight, random_start_epoch=random_start_epoch,**kwargs)


class CosineLR(AdjustLRCallbackBase):
    def __init__(self, min_lr=1e-8,period=1000,**kwargs):
        super(CosineLR, self).__init__()
        self.max_lr = None
        self.min_lr = min_lr
        self.period=period
        self.T_max=period/2
        self.T_current = 0
    def on_batch_end(self, training_context):
        if self.max_lr  is None:
            self.max_lr= training_context['base_lr']

        self.T_current=training_context['current_epoch'] * training_context['total_batch'] + training_context[ 'current_batch']% (2 * self.T_max)

        if self.T_current == 0:
            training_context['optimizer'].adjust_learning_rate(self.max_lr,verbose=False)
        elif (self.T_current - 1 - self.T_max) % (2 * self.T_max) == 0:
            training_context['optimizer'].adjust_learning_rate((self.min_lr+ (self.max_lr - self.min_lr)) *(1 - math.cos(math.pi / self.T_max)) / 2,verbose=False)
        else:
            training_context['optimizer'].adjust_learning_rate(
                ((1 + math.cos(math.pi * (self.T_current / self.T_max))) * (self.max_lr - self.min_lr) * 0.5 + self.min_lr),
                verbose=False)


def cosine_lr( min_lr=1e-8,period=1000,cosine_weight=0.2,**kwargs):
   return CosineLR(period=period,cosine_weight=cosine_weight,**kwargs)



def get_lr_scheduler(lr_scheduler_name):
    """
    Initialize a learning rate scheduler by name

    Args:
        lr_scheduler_name (str):

    Returns:

    """
    if lr_scheduler_name is None:
        return None
    lr_scheduler_modules = ['trident.callbacks.lr_schedulers']
    lr_scheduler_fn=None
    if isinstance(lr_scheduler_name,str):
        if lr_scheduler_name in __all__:
            lr_scheduler_fn=get_function(lr_scheduler_name,lr_scheduler_modules)
    else:
        try:
            lr_scheduler_fn = get_function(lr_scheduler_name, lr_scheduler_modules)
        except Exception :
            lr_scheduler_fn = get_function(snake2camel(lr_scheduler_name), lr_scheduler_modules)
    return lr_scheduler_fn


