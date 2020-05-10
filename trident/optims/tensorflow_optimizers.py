from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import math
import os
import sys
import time
import uuid
from collections import OrderedDict, defaultdict
from functools import partial
from shutil import copyfile

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow_addons as tfa
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops

from trident.backend.common import get_session, addindent, get_time_suffix, get_class, format_time, get_terminal_size, snake2camel, camel2snake
from trident.backend.tensorflow_backend import  Sequential
from trident.backend.tensorflow_ops import to_numpy, to_tensor
from trident.backend.optimizer import OptimizerBase

_session=get_session()
_epsilon=_session.epsilon
_backend=_session.backend

__all__ = ['Adam','RMSprop','SGD','Adagrad','Adadelta','RAdam','Lookahead','Ranger','get_optimizer']


class Adam(tf.keras.optimizers.Adam,OptimizerBase):
    def get_value(self,x):
        return tf.keras.backend.get_value(x)
    def set_value(self,x):
        return tf.keras.backend.set_value(x)
    pass

class RMSprop(tf.keras.optimizers.RMSprop,OptimizerBase):
    def get_value(self,x):
        return tf.keras.backend.get_value(x)
    def set_value(self,x):
        return tf.keras.backend.set_value(x)
    pass

class SGD(tf.keras.optimizers.SGD,OptimizerBase):
    def get_value(self,x):
        return tf.keras.backend.get_value(x)
    def set_value(self,x):
        return tf.keras.backend.set_value(x)
    pass

class Adagrad(tf.keras.optimizers.Adagrad,OptimizerBase):
    def get_value(self,x):
        return tf.keras.backend.get_value(x)
    def set_value(self,x):
        return tf.keras.backend.set_value(x)
    pass

class Adadelta(tf.keras.optimizers.Adadelta,OptimizerBase):
    def get_value(self,x):
        return tf.keras.backend.get_value(x)
    def set_value(self,x):
        return tf.keras.backend.set_value(x)
    pass



class RAdam(tfa.optimizers.RectifiedAdam,OptimizerBase):
    '''Variant of the Adam optimizer whose adaptive learning rate is rectified
    so as to have a consistent variance.

    It implements the Rectified Adam (a.k.a. RAdam) proposed by
    Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
    And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).

    Example of usage:
    >>> opt = RAdam(lr=1e-3)

    Note: `amsgrad` is not described in the original paper. Use it with
          caution.

    RAdam is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.
    You can enable warmup by setting `total_steps` and `warmup_proportion`:

    >>> opt = RAdam(lr=1e-3, total_steps=10000, warmup_proportion=0.1,min_lr=1e-5,)

    In the above example, the learning rate will increase linearly
    from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
    in 9000 steps.


    '''

    def get_value(self,x):
        return x
    def set_value(self,x):
        return x
    pass


class Lookahead(tf.keras.optimizers.Optimizer,OptimizerBase):
    '''This class allows to extend optims with the lookahead mechanism.
        The mechanism is proposed by Michael R. Zhang et.al in the paper
        [Lookahead Optimizer: k steps forward, 1 step back]
        (https://arxiv.org/abs/1907.08610v1). The optimizer iteratively updates two
        sets of weights: the search directions for weights are chosen by the inner
        optimizer, while the "slow weights" are updated each `k` steps based on the
        directions of the "fast weights" and the two sets of weights are
        synchronized. This method improves the learning stability and lowers the
        variance of its inner optimizer.

    Example:
    >>> opt = Lookahead(SGD(learning_rate=0.01))
    '''

    def __init__(self,
                 optimizer,
                 sync_period=6,
                 slow_step_size=0.5,
                 name="Lookahead",
                 **kwargs):
        '''Wrap optimizer with the lookahead mechanism.

        Args:
            optimizer: The original optimizer that will be used to compute
                and apply the gradients.
            sync_period: An integer. The synchronization period of lookahead.
                Enable lookahead mechanism by setting it with a positive value.
            slow_step_size: A floating point value.
                The ratio for updating the slow weights.
            name: Optional name for the operations created when applying
                gradients. Defaults to "Lookahead".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        '''
        super(Lookahead, self).__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optims.Optimizer")

        self._optimizer = optimizer
        self._set_hyper('sync_period', sync_period)
        self._set_hyper('slow_step_size', slow_step_size)
        self._initialized = False

    def get_value(self,x):
        return tf.keras.backend.get_value(x)
    def set_value(self,x):
        return tf.keras.backend.set_value(x)
    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access
        for var in var_list:
            self.add_slot(var, 'slow')

    def _create_hypers(self):
        self._optimizer._create_hypers()  # pylint: disable=protected-access

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)  # pylint: disable=protected-access

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations  # pylint: disable=protected-access
        return super(Lookahead, self).apply_gradients(grads_and_vars, name)

    def _init_op(self, var):
        slow_var = self.get_slot(var, 'slow')
        return slow_var.assign(
            tf.where(
                tf.equal(self.iterations,
                         tf.constant(0, dtype=self.iterations.dtype)),
                var,
                slow_var,
            ),
            use_locking=self._use_locking)

    def _look_ahead_op(self, var):
        var_dtype = var.dtype.base_dtype
        slow_var = self.get_slot(var, 'slow')
        local_step = tf.cast(self.iterations + 1, tf.dtypes.int64)
        sync_period = self._get_hyper('sync_period', tf.dtypes.int64)
        slow_step_size = self._get_hyper('slow_step_size', var_dtype)
        step_back = slow_var + slow_step_size * (var - slow_var)
        sync_cond = tf.equal(
            tf.math.floordiv(local_step, sync_period) * sync_period,
            local_step)
        with tf.control_dependencies([step_back]):
            slow_update = slow_var.assign(
                tf.where(
                    sync_cond,
                    step_back,
                    slow_var,
                ),
                use_locking=self._use_locking)
            var_update = var.assign(
                tf.where(
                    sync_cond,
                    step_back,
                    var,
                ),
                use_locking=self._use_locking)
        return tf.group(slow_update, var_update)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    def _resource_apply_dense(self, grad, var, **kwargs):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_dense(grad, var)  # pylint: disable=protected-access
            with tf.control_dependencies([train_op]):
                look_ahead_op = self._look_ahead_op(var)
        return tf.group(init_op, train_op, look_ahead_op)

    def _resource_apply_sparse(self, grad, var, indices, **kwargs):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_sparse(  # pylint: disable=protected-access
                grad, var, indices)
            with tf.control_dependencies([train_op]):
                look_ahead_op = self._look_ahead_op(var)
        return tf.group(init_op, train_op, look_ahead_op)

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self._optimizer),
            'sync_period': self._serialize_hyperparameter('sync_period'),
            'slow_step_size': self._serialize_hyperparameter('slow_step_size'),
        }
        base_config = super(Lookahead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop('optimizer'),
            custom_objects=custom_objects,
        )
        return cls(optimizer, **config)


class Ranger(tf.keras.optimizers.Optimizer,OptimizerBase):
    '''This class allows to extend optims with the Ranger mechanism.
        Range= RAdam+ Lookahead

        Lookahead, proposed by Michael R. Zhang et.al in the paper
        [Lookahead Optimizer: k steps forward, 1 step back]
        (https://arxiv.org/abs/1907.08610v1), can be integrated with RAdam,
        which is announced by Less Wright and the new combined optimizer can also
        be called "Ranger". The mechanism can be enabled by using the lookahead
        wrapper. For example:

    Examples:
        >>> radam = RAdam()
        >>> ranger = Lookahead(RAdam(), sync_period=6, slow_step_size=0.5)

    '''

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, weight_decay=0., amsgrad=False,
                 sma_threshold=5.0, total_steps=0, warmup_proportion=0.1, min_lr=0.,
                 sync_period=6,
                 slow_step_size=0.5,
                 name="Ranger",
                 **kwargs):
        r"""Wrap optimizer with the lookahead mechanism.

        Args:
            optimizer: The original optimizer that will be used to compute
                and apply the gradients.
            sync_period: An integer. The synchronization period of lookahead.
                Enable lookahead mechanism by setting it with a positive value.
            slow_step_size: A floating point value.
                The ratio for updating the slow weights.
            name: Optional name for the operations created when applying
                gradients. Defaults to "Lookahead".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super(Ranger, self).__init__(name, **kwargs)



        self._optimizer = RAdam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=_epsilon, weight_decay=weight_decay, amsgrad=False,
                 sma_threshold=sma_threshold, total_steps=total_steps, warmup_proportion=warmup_proportion, min_lr=1e-9)
        self._set_hyper('sync_period', sync_period)
        self._set_hyper('slow_step_size', slow_step_size)
        self._initialized = False

    def get_value(self,x):
        return tf.keras.backend.get_value(x)
    def set_value(self,x):
        return tf.keras.backend.set_value(x)



    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access
        for var in var_list:
            self.add_slot(var, 'slow')

    def _create_hypers(self):
        self._optimizer._create_hypers()  # pylint: disable=protected-access

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)  # pylint: disable=protected-access

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations  # pylint: disable=protected-access
        return super(Ranger, self).apply_gradients(grads_and_vars, name)

    def _init_op(self, var):
        slow_var = self.get_slot(var, 'slow')
        return slow_var.assign(
            tf.where(
                tf.equal(self.iterations,
                         tf.constant(0, dtype=self.iterations.dtype)),
                var,
                slow_var,
            ),
            use_locking=self._use_locking)

    def _look_ahead_op(self, var):
        var_dtype = var.dtype.base_dtype
        slow_var = self.get_slot(var, 'slow')
        local_step = tf.cast(self.iterations + 1, tf.dtypes.int64)
        sync_period = self._get_hyper('sync_period', tf.dtypes.int64)
        slow_step_size = self._get_hyper('slow_step_size', var_dtype)
        step_back = slow_var + slow_step_size * (var - slow_var)
        sync_cond = tf.equal(
            tf.math.floordiv(local_step, sync_period) * sync_period,
            local_step)
        with tf.control_dependencies([step_back]):
            slow_update = slow_var.assign(
                tf.where(
                    sync_cond,
                    step_back,
                    slow_var,
                ),
                use_locking=self._use_locking)
            var_update = var.assign(
                tf.where(
                    sync_cond,
                    step_back,
                    var,
                ),
                use_locking=self._use_locking)
        return tf.group(slow_update, var_update)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    def _resource_apply_dense(self, grad, var, **kwargs):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_dense(grad, var)  # pylint: disable=protected-access
            with tf.control_dependencies([train_op]):
                look_ahead_op = self._look_ahead_op(var)
        return tf.group(init_op, train_op, look_ahead_op)

    def _resource_apply_sparse(self, grad, var, indices, **kwargs):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_sparse(  # pylint: disable=protected-access
                grad, var, indices)
            with tf.control_dependencies([train_op]):
                look_ahead_op = self._look_ahead_op(var)
        return tf.group(init_op, train_op, look_ahead_op)

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self._optimizer),
            'sync_period': self._serialize_hyperparameter('sync_period'),
            'slow_step_size': self._serialize_hyperparameter('slow_step_size'),
        }
        base_config = super(Ranger, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop('optimizer'),
            custom_objects=custom_objects,
        )
        return cls(optimizer, **config)


def get_optimizer(optimizer_name):
    if optimizer_name is None:
        return None
    optimizer_modules = ['trident.optims.tensorflow_optimizers']
    if optimizer_name in __all__:
        optimizer_class = get_class(optimizer_name, optimizer_modules)
    else:
        try:
            optimizer_class = get_class(snake2camel(optimizer_name), optimizer_modules)
        except Exception :
            optimizer_class = get_class(optimizer_name, optimizer_modules)
    return optimizer_class