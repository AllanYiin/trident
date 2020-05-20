from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import collections
import itertools
import math
import copy
import os
import sys
import time
import uuid
from collections import OrderedDict, defaultdict
from functools import partial
from shutil import copyfile

import numpy as np
import tensorflow as tf

import tensorflow_addons as tfa
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging

from trident.backend.common import get_session, addindent, get_time_suffix, get_class, format_time, get_terminal_size, \
    snake2camel, camel2snake
from trident.backend.optimizer import OptimizerBase
from trident.backend.tensorflow_backend import Sequential
from trident.backend.tensorflow_ops import *

_session=get_session()
_epsilon=_session.epsilon
_backend=_session.backend

__all__ = ['Adam','RMSprop','SGD','Adagrad','Adadelta','RAdam','Lookahead','Ranger','get_optimizer']

from collections import defaultdict

from copy import deepcopy
from itertools import chain


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


class Optimizer(object):
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Arguments:
        params (iterable): an iterable of :class:`tf.Variable` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        self.defaults = defaults
        self.grad_tape=tf.GradientTape(watch_accessed_variables=False)
        if isinstance(params, tf.Variable):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            type(params).__name__)

        self.state = defaultdict(dict)
        self.param_groups = []
        self._base_lr=1e-3

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def _filter_grads(self,grads_and_vars):
        """Filter out iterable with grad equal to None."""
        grads_and_vars = tuple(grads_and_vars)
        if not grads_and_vars:
            return grads_and_vars
        filtered = []
        vars_with_empty_grads = []
        for grad, var in grads_and_vars:
            if grad is None or any_abnormal_number(grad):
                vars_with_empty_grads.append(var)
            else:
                filtered.append((grad, var))
        filtered = tuple(filtered)
        if not filtered:
            raise ValueError("No gradients provided for any variable: %s." % ([v.name for _, v in grads_and_vars],))
        if vars_with_empty_grads:
           logging.warning(("Gradients do not exist for variables %s when minimizing the loss."), ([v.name for v in vars_with_empty_grads]))
        return filtered

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save ids instead of Tensors
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [id(p) for p in group['params']]
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use ids as keys
        packed_state = {(id(k) if isinstance(k, tf.Variable) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain(*(g['params'] for g in saved_groups)),
                      chain(*(g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, tf.Variable):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, collections.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`tf.Variable` s."""
        self.grad_tape.reset()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, tf.Variable):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, tf.Variable):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + type(param).__name__)
            # if not param.is_leaf:
            #     raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set([p.ref() for p in param_group['params']]))

        if not param_set.isdisjoint(set([p.ref() for p in param_group['params']])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def adjust_learning_rate(self, new_lr, verbose=True):
        """

        Args:
            new_lr (float):  new learning rate value
            verbose (bool): if True, will print the learning rate change information.

        """

        old_lr = self.param_groups[0]['lr']
        if old_lr != new_lr:
            self.param_groups[0]['lr'] = new_lr
            if verbose:
                print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))
    @property
    def lr(self):
        """

        Returns: current learning rate

        """
        return self.param_groups[0]['lr']


    @lr.setter
    def lr(self, value: float):
        if self.lr != value:
            old_lr = self.lr
            new_lr = value
            self.param_groups[0]['lr'] = new_lr
            print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))

    @property
    def base_lr(self):
        """

        Returns: base learning rate means the starting learning rate (after warmup)

        """
        return self._base_lr

    @base_lr.setter
    def base_lr(self, value):
        self._base_lr = value

class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.amsgrad=amsgrad
        self.eps=eps
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    def step(self, grads_and_vars=None):
        """Performs a single optimization step.

        Arguments:
            grads_and_vars (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.grads_and_vars = self._filter_grads(grads_and_vars)
        #grads_and_vars=zip(new_grads, new_vars)

        group=self.param_groups[0]
        for grad,p in self.grads_and_vars:
            if grad is None or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            amsgrad = group['amsgrad']

            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0.0
                # Exponential moving average of gradient values
                state['m'] = zeros_like(p)
                # Exponential moving average of squared gradient values
                state['v'] = zeros_like(p)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['vhat'] = zeros_like(p)


            m, v = state['m'], state['v']


            beta1, beta2 = group['betas']

            state['step'] += 1
            beta_1_power = 1 - beta1 ** state['step']
            beta_2_power   = 1 - beta2 ** state['step']

            lr = group['lr']
            if group['weight_decay'] > 0:
                lr = lr * (1. / (1. + group['weight_decay'] * state['step']))

            lr_t = lr * (math.sqrt(1. - math.pow(beta2, state['step'])) / (1. - math.pow(beta1, state['step'])))

            if group['weight_decay'] != 0:
                grad = grad+p*group['weight_decay']

            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            m_t =beta1*m+(1 - beta1)*grad

            #exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v_t =beta2*v+(1 - beta2)*(grad*grad)

            #exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                vhat = state['vhat']
                # Maintains the maximum of all 2nd moment running avg. till now
                vhat_t=maximum(vhat, v_t)
                state['vhat'] = vhat_t
                # Use the max. for normalizing running avg. of gradient
                p.assign(tf.Variable(to_numpy(p - lr_t * m_t / (sqrt(vhat_t) +group['eps']))))

            else:
                p.assign(tf.Variable(to_numpy(p - lr_t * m_t / (sqrt(v_t) + group['eps']))))

            # if reduce_mean(abs(p))>0:
            #     print(reduce_mean(abs(-(self.lr / beta_1_power) *exp_avg/ (denom + self.eps)))/reduce_mean(abs(p)))
            state['m']=m_t
            state['v'] = v_t


        return True



class RMSprop(Optimizer):
    r"""Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        self.weight_decay=weight_decay
        self.eps=eps
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)


    def step(self, grads_and_vars=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.grads_and_vars = self._filter_grads(grads_and_vars)
        #grads_and_vars=zip(new_grads, new_vars)

        group=self.param_groups[0]
        for grad,p in self.grads_and_vars:
            if grad is None or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0.0
                state['square_avg'] = zeros_like(p)
                if group['momentum'] > 0:
                    state['momentum_buffer'] = zeros_like(p)
                if group['centered']:
                    state['grad_avg'] = zeros_like(p)

            state['step'] += 1
            lr = group['lr']
            if group['weight_decay'] > 0:
                lr = lr * (1. / (1. + group['weight_decay'] * state['step']))


            if group['weight_decay'] != 0:
                grad = grad+p*group['weight_decay']

            square_avg = state['square_avg']
            alpha = group['alpha']
            square_avg_t=square_avg*alpha+ (1. - alpha) * square(grad)
            if group['centered']:
                grad_avg = state['grad_avg']
                grad_avg_t=grad_avg*alpha+grad*(1 - alpha)

                avg = sqrt(square_avg_t-grad_avg_t**2)+group['eps']
                state['grad_avg']=grad_avg_t
            else:

                avg=sqrt(square_avg_t)+group['eps']

            if group['momentum'] > 0:
                buf = state['momentum_buffer']
                buf_t=buf*group['momentum']+(grad/(avg+ self.eps))
                p.assign(tf.Variable(to_numpy(p - lr * buf_t)))
                state['momentum_buffer'] = buf_t
            else:

                p.assign(tf.Variable(to_numpy(p - lr * grad / (avg+ self.eps))))

            state['square_avg']=square_avg_t




        return True


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """


    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    def step(self, grads_and_vars=None):
        """Performs a single optimization step.

        Arguments:
            grads_and_vars (): The zipped gradients and watched parameters parr from tf.GradientTape

        """
        self.grads_and_vars = self._filter_grads(grads_and_vars)
        #grads_and_vars=zip(new_grads, new_vars)

        group=self.param_groups[0]
        dampening = group['dampening']
        nesterov = group['nesterov']
        for grad,p in self.grads_and_vars:
            if grad is None or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p.ref()]
            state['step'] += 1
            lr = group['lr']
            if group['weight_decay'] > 0:
                lr = lr * (1. / (1. + group['weight_decay'] * state['step']))

            if group['weight_decay'] != 0:
                grad = grad+p*group['weight_decay']

            if group['momentum'] != 0:

                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] =copy.deepcopy(grad)
                else:
                    buf = state['momentum_buffer']
                    buf_t=buf*group['momentum']+grad*(1 - dampening)
                    state['momentum_buffer']=buf_t
                if nesterov:
                    grad=grad+buf*group['momentum']
                else:
                    grad = buf

            p.assign(tf.Variable(to_numpy(p - lr * grad)))


        return True




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



class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
    def step(self, grads_and_vars=None):
        """Performs a single optimization step.

        Arguments:
            grads_and_vars (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.grads_and_vars = self._filter_grads(grads_and_vars)
        #grads_and_vars=zip(new_grads, new_vars)

        group=self.param_groups[0]
        for grad,p in self.grads_and_vars:
            if grad is None or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p.ref()]
            p_data_fp32=cast(to_tensor(to_numpy(p)),'float32')
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['m'] = zeros_like(p_data_fp32)
                state['v'] = zeros_like(p_data_fp32)
            else:
                state['m'] = state['m'].type_as(p_data_fp32)
                state['v'] = state['v'].type_as(p_data_fp32)

            m, v = state['m'], state['v']
            beta1, beta2 = group['betas']

            # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            m_t = beta1 * m + (1 - beta1) * grad

            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v_t = beta2 * v + (1 - beta2) * (grad * grad)

            state['step'] += 1
            buffered = group['buffer'][int(state['step'] % 10)]
            if state['step'] == buffered[0]:
                N_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = state['step']
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                buffered[1] = N_sma

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                N_sma_max - 2)) / (1 - beta1 ** state['step']+group['eps'])
                elif self.degenerated_to_sgd:
                    step_size = 1.0 / (1 - beta1 ** state['step']+group['eps'])
                else:
                    step_size = -1
                buffered[2] = step_size

            # more conservative since it's an approximated value
            if N_sma >= 5:
                if group['weight_decay'] != 0:
                    p_data_fp32 = p_data_fp32 + -group['weight_decay'] * group['lr']*p_data_fp32

                denom = sqrt(v_t)+group['eps']
                p_data_fp32=p_data_fp32-(step_size * group['lr']*denom)/(m_t+group['eps'])
                p.assign(tf.Variable(p_data_fp32))
            elif step_size > 0:
                if group['weight_decay'] != 0:
                    p_data_fp32 = p_data_fp32 -group['weight_decay'] * group['lr']*p_data_fp32
                p_data_fp32=p_data_fp32-(step_size * group['lr'])*m_t
                p.assign(tf.Variable(p_data_fp32))
            state['m'] = m_t
            state['v'] = v_t
        return True



class PlainRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.grad_tape=self.optimizer.grad_tape
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast.ref()]
            fast_data=to_tensor(to_numpy(fast))
            if "slow_param" not in param_state:
                param_state["slow_param"] = zeros_like(fast)
                param_state["slow_param"]=fast_data
            slow = param_state["slow_param"]
            slow += (fast_data - slow) * self.alpha
            fast.assign(tf.Variable(slow))

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, grads_and_vars=None):
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return True

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {(id(k) if isinstance(k, tf.Variable) else k): v for k, v in self.state.items()}
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {"fast_state": fast_state, "slow_state": slow_state, "param_groups": param_groups, }

    def load_state_dict(self, state_dict):
        slow_state_dict = {"state": state_dict["slow_state"], "param_groups": state_dict["param_groups"], }
        fast_state_dict = {"state": state_dict["fast_state"], "param_groups": state_dict["param_groups"], }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

class Ranger(Lookahead):
    """
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
    """
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0,degenerated_to_sgd=True):
        #parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError('Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError('Invalid eps: {eps}')

        #parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        #N_sma_threshold of 5 seems better in testing than 4.
        #In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        #prep defaults and init tensorflow.optim base


        self.optimizer =RAdam( params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, degenerated_to_sgd=degenerated_to_sgd)

        # look ahead params
        # now we can get to work...
        # removed as we now use step from RAdam...no need for duplicate step counting
        # for group in self.param_groups:
        #    group["step_counter"] = 0
        # print("group step counter init")
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
        self.grad_tape=self.optimizer.grad_tape

        self.degenerated_to_sgd=degenerated_to_sgd
        #adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold




        #self.first_run_check=0

        #lookahead weights
        #9/2/19 - lookahead param tensors have been moved to state storage.
        #This should resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        #self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        #don't use grad for lookahead weights
        #for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False

    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)






def get_optimizer(optimizer_name):
    """

    Args:
        optimizer_name ():

    Returns:

    """
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