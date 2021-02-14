"""Learning Rate Scheduler Callbacks"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import math
import random
import warnings
import types
import numpy as np
from trident.backend.pytorch_backend import save, load

from trident.backend.common import *
from trident.backend.common import get_backend
from trident.callbacks.callback_base import CallbackBase

_session = get_session()

if get_backend()=='pytorch':
    from trident.backend.pytorch_ops import to_numpy,to_tensor,pow,clip
elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import  to_numpy,to_tensor,pow,clip



__all__ = ['AdjustLRCallback','ReduceLROnPlateau','reduce_lr_on_plateau','LambdaLR','lambda_lr','RandomCosineLR','random_cosine_lr','CosineLR','cosine_lr']





class AdjustLRCallbackBase(CallbackBase):
    """Basic class for learning rate scheduler"""
    def __init__(self):
        super(AdjustLRCallbackBase, self).__init__()
        self.base_lr=1e-3
        self.base_lrs = [1e-3]
    def adjust_learning_rate(self,training_context,new_lr,verbose=True):
        old_lr = training_context['optimizer'].lr

        if old_lr!=new_lr:
            training_context['optimizer'].param_groups[0]['lr'] = new_lr
            training_context['current_lr'] = new_lr
            if verbose:
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
        if self.unit == 'epoch' and training_context['current_epoch'] == self.index :
            self.adjust_learning_rate(training_context,self.new_lr)

class LRFinder(AdjustLRCallbackBase):
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """
    def __init__(self, start_lr=1e-7,end_lr=100,n_skip_beginning=10, n_skip_end=5,sma=1):
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9
        self.start_lr=start_lr
        self.end_lr=end_lr
        self.n_skip_beginning=n_skip_beginning
        self.n_skip_end=n_skip_end
        self.sma=sma

    def on_training_start(self, training_context):
        num_batches = training_context['total_epoch'] * training_context['total_batch']
        self.lr_mult = (float(self.end_lr) / float(self.start_lr)) ** (float(1) / float(num_batches))

        # Remember the original learning rate
        self.base_lr =training_context['optimizer'].lr
        save(training_context['current_model'].state_dict(),'Models/state_dict.pth')
        # Set the initial learning rate
        self.adjust_learning_rate(training_context, self.start_lr)
    def on_batch_end(self, training_context):
        # Log the learning rate
        lr =training_context['optimizer'].lr
        self.lrs.append(lr)
        loss =training_context['current_loss'].item()
        self.losses.append(loss)
        training_context['current_model'].load_state_dict(load('Models/state_dict.pth'))

        # Check whether the loss got too large or NaN
        if math.isnan(loss) or loss > self.best_loss * 4:
            self.adjust_learning_rate(training_context, self.base_lr )

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        self.adjust_learning_rate(training_context, lr,)

    def on_epoch_end(self, training_context):
        self.plot_loss(training_context)
        self.sma=1
        self.plot_loss_change(training_context)
        self.sma = 5
        self.plot_loss_change(training_context)
        self.adjust_learning_rate(training_context, self.base_lr)

    def plot_loss(self, training_context):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[self.n_skip_beginning:-self.n_skip_end], self.losses[self.n_skip_beginning:-self.n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self,training_context):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """

        assert self.sma >= 1
        derivatives = [0] * self.sma
        for i in range(self.sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - self.sma]) / self.sma
            derivatives.append(derivative)

        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[self.n_skip_beginning:-self.n_skip_end], derivatives[self.n_skip_beginning:-self.n_skip_end])
        plt.xscale('log')
        y_lim = (-0.01, 0.01)
        plt.ylim(y_lim)

class OnceCycleLR(AdjustLRCallbackBase):
    def __init__(self, lr_range: tuple = (0.1, 1.), momentum_range: tuple = (0.85, 0.95), annihilation_frac: float = 0.1, reduce_factor: float = 0.01,
                 last_step: int = -1):
        super().__init__()

        self.min_lr, self.max_lr = lr_range[0], lr_range[1]
        assert self.min_lr < self.max_lr, \
            "Argument lr_range must be (min_lr, max_lr), where min_lr < max_lr"
        self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
        assert self.min_momentum < self.max_momentum, \
            "Argument momentum_range must be (min_momentum, max_momentum), where min_momentum < max_momentum"
        self.annihilation_frac=annihilation_frac
        self.reduce_factor=reduce_factor
        self.last_step = last_step

    def on_training_start(self, training_context):
        self.num_steps =training_context['total_epoch']*training_context['total_batch']
        self.num_cycle_steps = int(self.num_steps * (1. - self.annihilation_frac))  # Total number of steps in the cycle
        self.final_lr = self.min_lr * self.reduce_factor

    def on_batch_end(self, training_context):
        current_step =training_context['steps']
        self.last_step =training_context['steps']-1

        if current_step <= self.num_cycle_steps // 2:
            # Scale up phase
            scale = current_step / (self.num_cycle_steps // 2)
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_cycle_steps:
            # Scale down phase
            scale = (current_step - self.num_cycle_steps // 2) / (self.num_cycle_steps - self.num_cycle_steps // 2)
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_steps:
            # Annihilation phase: only change lr
            scale = (current_step - self.num_cycle_steps) / (self.num_steps - self.num_cycle_steps)
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None
        else:
            # Exceeded given num_steps: do nothing
            return

        training_context['optimizer'].param_groups[0]['lr'] = lr
        if momentum:
            training_context['optimizer'].param_groups[0]['momentum'] = momentum


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
        if self.unit_base == 'epoch'  :
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
                self.unit_base='epoch'
                print('one epoch have {0} batches, use {1} as epoch equivalent in long epoch. '.format(training_context['total_batch'],_session.epoch_equivalent))
            else:
                self.unit_base = 'epoch'
                print('ReduceLROnPlateau reseted.')

        num_batches = training_context['steps']
        if self.unit_base=='epoch' and training_context['steps']>0 and training_context['steps']%_session.epoch_equivalent==0 and  training_context['current_model'].training==True:
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

def reduce_lr_on_plateau(monitor='total_losses' ,verbose=True, mode='min', factor=0.5, patience=5, threshold=1e-4, cooldown=0, min_lr=1e-8,unit_base='epoch', **kwargs):
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
                            cooldown=cooldown,min_lr=min_lr,unit_base=unit_base)


class LambdaLR(AdjustLRCallbackBase):
    def __init__(self, offset=0,decay_start_epoch=50, power=0.9, **kwargs):
        super(LambdaLR, self).__init__()
        self.offset=offset
        self.power=power
        self.decay_start_epoch=decay_start_epoch

    def on_epoch_end(self, training_context):
        if  training_context['current_epoch']==0:
            self.base_lr = training_context['optimizer'].lr
        n_epochs = training_context['total_epoch']
        epoch = training_context['current_epoch']
        if epoch>=self.decay_start_epoch:
            lr= clip(self.base_lr *pow(1- (epoch + self.offset - self.decay_start_epoch) / (n_epochs - self.decay_start_epoch),self.power),1e-6,np.inf)
            self.adjust_learning_rate(training_context,lr,verbose=False)

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

        self.T_current=training_context['steps']% (2 * self.T_max)

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


