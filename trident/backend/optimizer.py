import numpy as np
from trident.backend.common import *

_session = get_session()
_backend = _session.backend
if _backend == 'pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *




__all__ = [ 'OptimizerBase']


class OptimizerBase(object):
    def adjust_learning_rate(self, new_lr, verbose=True):
        if _backend =='pytorch':
            old_lr = self.param_groups[0]['lr']
            if old_lr != new_lr:
                self.param_groups[0]['lr'] = new_lr
                if verbose:
                    print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))
        elif _backend == 'tensorflow':
            old_lr = self.lr
            if old_lr != new_lr:
                if hasattr(self, '_set_hyper'):
                    if hasattr(self, '_optimizer'):
                        self._optimizer._set_hyper('learning_rate', new_lr)
                    else:
                        self._set_hyper('learning_rate', new_lr)
                elif hasattr(self, 'set_value'):
                    self.lr.set_value(new_lr)
                if verbose:
                    print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))


    @property
    def default_setting(self):
        if _backend == 'pytorch':
            return self.defaults
        elif _backend == 'tensorflow':
            return self.__dict__


    @default_setting.setter
    def default_setting(self, value):
        if _backend == 'pytorch':
            self.defaults = value
        elif _backend == 'tensorflow':
            self.__dict__.update(value)


    @property
    def parameters(self):
        if _backend == 'pytorch':
            return self.param_groups['params']
        elif _backend == 'tensorflow':
            return self.get_weights()


    @parameters.setter
    def parameters(self, params):
        if _backend == 'pytorch':
            self.param_groups = [{'params': list(params)}]
        elif _backend == 'tensorflow':
            self.set_weights(params)

    @property
    def lr(self):
        if _backend ==  'pytorch':
            return self.param_groups[0]['lr']
        elif _backend == 'tensorflow':
            if hasattr(self, '_optimizer'):
                return self._optimizer._get_hyper('learning_rate').numpy()
            else:
                return self._get_hyper('learning_rate').numpy()

    @lr.setter
    def lr(self, value: float):
        if self.lr != value:
            old_lr = self.lr
            new_lr = value
            if _backend == 'pytorch':
                self.param_groups[0]['lr'] = new_lr
            elif _backend == 'tensorflow':
                if hasattr(self, '_optimizer'):
                    self._optimizer._set_hyper('learning_rate', new_lr)
                else:
                    self._set_hyper('learning_rate', new_lr)

            print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))

    @property
    def base_lr(self):
        return self._base_lr

    @base_lr.setter
    def base_lr(self, value):
        self._base_lr = value

    def get_gradients(self, loss, params=None):
        if _backend == 'pytorch':
            return loss.grad
        elif _backend == 'tensorflow':
            return self.get_gradients(loss, params)


    def updates(self, closure, training_context):

        if _backend == 'pytorch':
            try:
                if callable(closure):
                    loss = closure()
                    loss.backward()
            except Exception as e:
                PrintException()
        elif _backend == 'tensorflow':
            if callable(closure):
                loss = closure()
            self.get_updates(loss, training_context['current_model'].trainable_)


        for callback in training_context['callbacks']:
            callback.on_optimization_step_end(training_context)

    def before_batch_train(self):
        if _backend == 'pytorch':
            self.zero_grad()
        else:
            pass