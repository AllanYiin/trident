from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from functools import reduce
import collections
import copy
import math
import re
import scipy.optimize as sciopt
import tensorflow as tf
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import tracking
from tensorflow.python.keras.optimizer_v2 import adam
from trident.backend.common import get_session, get_class, snake2camel,get_time_suffix,camel2snake,get_session_value
from trident.backend.tensorflow_ops import *



__all__ = ['Adam', 'RMSprop', 'SGD', 'RAdam', 'Lookahead', 'Ranger','AdaBelief', 'get_optimizer']

from collections import defaultdict

from copy import deepcopy
from itertools import chain


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def gc_grads(grads, gradient_centralization=None):
    """Filter out iterable with grad equal to None or abnormal grad and do the gradient centralization."""
    if gradient_centralization is None:
        pass
    elif gradient_centralization == 'all':
        if len(int_shape(grads)) > 1:
            grads+=(-reduce_mean(grads, axis=list(range(1, len(int_shape(grads)))), keepdims=True))
    elif gradient_centralization == 'gcc':
        if len(int_shape(grads)) > 3:
            grads+=(-reduce_mean(grads, axis=list(range(1, len(int_shape(grads)))), keepdims=True))
    elif gradient_centralization == 'gc':
        if len(int_shape(grads)) > 1:
            grads+=(-reduce_mean(grads, axis=list(range(1, len(int_shape(grads)))), keepdims=True))
    return grads



class OptimizerWrapper(object):
    """Wrapper class that provides proxy access to an instance of some
       internal instance."""

    __wraps__  =tf.keras.optimizers.Optimizer
    __ignore__ = "class mro new init setattr getattr getattribute"

    def __init__(self, tf_optimizer):
        if self.__wraps__ is None:
            raise TypeError("base class Wrapper may not be instantiated")
        elif isinstance(tf_optimizer, self.__wraps__):
            self._tf_optimizer = tf_optimizer
        else:
            raise ValueError("wrapped object must be of %s" % self.__wraps__)

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._tf_optimizer, name)


    def step(self, grads_and_vars=None, **kwargs):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            grads_and_vars (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        """
        self._tf_optimizer.apply_gradients(grads_and_vars)

        # create proxies for wrapped object's double-underscore attributes


    def adjust_learning_rate(self, new_lr, verbose=True):
        """

        Args:
            new_lr (float):  new learning rate value
            verbose (bool): if True, will print the learning rate change information.

        """

        old_lr = self._tf_optimizer._get_hyper('learning_rate').value().numpy()
        if old_lr != new_lr:
            self._tf_optimizer._set_hyper('learning_rate',new_lr)
            if verbose:
                print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))

    @property
    def lr(self):
        """str: The getter method of the 'learning rate' property."""
        return self._tf_optimizer._get_hyper('learning_rate').value().numpy()

    @lr.setter
    def lr(self, value: float):
        if self.lr != value:
            old_lr = self._tf_optimizer ._get_hyper('learning_rate').value().numpy()
            new_lr = value
            self._tf_optimizer._set_hyper('learning_rate',new_lr)
            print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))

    @property
    def base_lr(self):
        """str: The getter method of the 'base learning rate' property (mean the starting learning rate ,
        excluding warmup )."""
        return self._base_lr

    @base_lr.setter
    def base_lr(self, value):
        self._base_lr = value

    class __metaclass__(type):
        def __init__(cls, name, bases, dct):

            def make_proxy(name):
                def proxy(self, *args):
                    return getattr(self._tf_optimizer, name)
                return proxy

            type.__init__(cls, name, bases, dct)
            if cls.__wraps__:
                ignore = set("__%s__" % n for n in cls.__ignore__.split())
                for name in dir(cls.__wraps__):
                    if name.startswith("__"):
                        if name not in ignore and name not in dct:
                            setattr(cls, name, property(make_proxy(name)))


class Optimizer(trackable.Trackable):
    """Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`tf.Variable` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).


    """

    def __init__(self, params, defaults):
        self._name=camel2snake(self.__class__.__name__)+get_time_suffix()
        self.defaults = defaults
        self.gradient_centralization = None
        if isinstance(params, tf.Variable):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " + type(params).__name__)

        self.state = defaultdict(dict)
        self.param_groups = []
        self._base_lr = 1e-3

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        self.grad_tape = None

    def __getstate__(self):
        return {'defaults': self.defaults, 'state': self.state, 'param_groups': self.param_groups, }

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

    def _filter_grads(self, grads_and_vars, gradient_centralization=None):
        """Filter out iterable with grad equal to None or abnormal grad and do the gradient centralization."""
        grads_and_vars = tuple(grads_and_vars)
        if not grads_and_vars:
            return grads_and_vars
        filtered_grad = []
        filtered_var = []
        vars_with_empty_grads = []
        for grad, var in grads_and_vars:
            if grad is None or any_abnormal_number(grad) or var._trainable == False:
                vars_with_empty_grads.append(var)
            else:
                if gradient_centralization is None:
                    filtered_grad.append(grad)
                    filtered_var.append(var)
                elif gradient_centralization == 'gc':
                    if len(int_shape(grad)) > 1:
                        filtered_grad.append(grad)
                        filtered_var.append(var)
                    else:
                        filtered_grad.append(grad)
                        filtered_var.append(var)
                elif gradient_centralization == 'gcc':
                    if len(int_shape(grad)) > 3:
                        new_grad = grad - reduce_mean(grad, axis=list(range(1, len(int_shape(grad)))), keepdims=True)
                        filtered_grad.append(new_grad)
                        filtered_var.append(var)
                    else:
                        filtered_grad.append(grad)
                        filtered_var.append(var)
                else:
                    filtered_grad.append(grad)
                    filtered_var.append(var)

        filtered = zip(filtered_grad, filtered_var)
        if not filtered:
            raise ValueError("No gradients provided for any variable: %s." % ([v.name for _, v in grads_and_vars],))

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
        packed_state = {(id(k) if isinstance(k, tf.Variable) else k): v for k, v in self.state.items()}
        return {'state': packed_state, 'param_groups': param_groups, }

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Args:
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
                  zip(chain(*(g['params'] for g in saved_groups)), chain(*(g['params'] for g in groups)))}

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

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`tf.Variable` s."""
        self.grad_tape.reset()
        if hasattr(self, 'grads_and_vars') and self.grads_and_vars is not None:
            for g, p in self.grads_and_vars:
                g = zeros_like(g)


    def step(self, grads_and_vars=None, **kwargs):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            grads_and_vars (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        """Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
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
                                "but one of the params is " + type(
                    param).__name__)  # if not param.is_leaf:  #     raise ValueError("can't optimize a non-leaf
                # Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " + name)
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
        """str: The getter method of the 'learning rate' property."""
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
        """str: The getter method of the 'base learning rate' property (mean the starting learning rate ,
        excluding warmup )."""
        return self._base_lr

    @base_lr.setter
    def base_lr(self, value):
        self._base_lr = value





class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Args:
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

    References
        .. _Adam\: A Method for Stochastic Optimization:
            https://arxiv.org/abs/1412.6980
        .. _On the Convergence of Adam and Beyond:
            https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
                 gradient_centralization=None):

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
        self.amsgrad = amsgrad
        self.eps = eps
        self.gradient_centralization = 'gc' if gradient_centralization == True else gradient_centralization
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)



    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # grads_and_vars=zip(new_grads, new_vars)

        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
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
            beta_2_power = 1 - beta2 ** state['step']

            if group['weight_decay'] != 0:
                grad = grad + p.value() * group['weight_decay']

            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            m_t = beta1 * m + (1 - beta1) * grad

            # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v_t = beta2 * v + (1 - beta2) * (grad * grad)

            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                vhat = state['vhat']
                # Maintains the maximum of all 2nd moment running avg. till now
                vhat_t = maximum(vhat, v_t)
                state['vhat'] = vhat_t
                denom = (sqrt(vhat_t) / math.sqrt(beta_2_power)) + group['eps']


            else:
                denom = (sqrt(v_t) / math.sqrt(beta_2_power)) + group['eps']

            step_size = group['lr'] / beta_1_power
            p.assign(p.value() - step_size * m_t / denom)

            # if reduce_mean(abs(p))>0:
            #     print(reduce_mean(abs(-(self.lr / beta_1_power) *exp_avg/ (denom + self.eps)))/reduce_mean(abs(p)))
            state['m'] = m_t
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

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,
                 gradient_centralization=None):
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
        self.weight_decay = weight_decay
        self.eps = eps
        self.gradient_centralization = 'gc' if gradient_centralization == True else gradient_centralization
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)

        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
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
                grad = grad + p.value() * group['weight_decay']

            square_avg = state['square_avg']
            alpha = group['alpha']
            square_avg_t = square_avg * alpha + (1. - alpha) * square(grad)
            if group['centered']:
                grad_avg = state['grad_avg']
                grad_avg_t = grad_avg * alpha + grad * (1 - alpha)

                avg = sqrt(square_avg_t - grad_avg_t ** 2) + group['eps']
                state['grad_avg'] = grad_avg_t
            else:

                avg = sqrt(square_avg_t) + group['eps']

            if group['momentum'] > 0:
                buf = state['momentum_buffer']
                buf_t = buf * group['momentum'] + (grad / (avg + self.eps))
                p.assign(p.value() - lr * buf_t)
                state['momentum_buffer'] = buf_t
            else:

                p.assign(p.value() - lr * grad / (avg + self.eps))

            state['square_avg'] = square_avg_t

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

    Examples:
        >>> SGD(lr=0.1, momentum=0.9)


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

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 gradient_centralization=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.gradient_centralization = 'gc' if gradient_centralization == True else gradient_centralization
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)

        group = self.param_groups[0]
        dampening = group['dampening']
        nesterov = group['nesterov']
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0.0
            else:
                state['step'] += 1
            lr = group['lr']
            if group['weight_decay'] > 0:
                lr = lr * (1. / (1. + group['weight_decay'] * state['step']))

            if group['weight_decay'] != 0:
                grad = grad + p * group['weight_decay']

            if group['momentum'] != 0:

                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = copy.deepcopy(grad)
                else:
                    buf = state['momentum_buffer']
                    buf_t = buf * group['momentum'] + grad * (1 - dampening)
                    state['momentum_buffer'] = buf_t
                if nesterov:
                    grad = grad + buf * group['momentum']
                else:
                    grad = buf

            p.assign(p.value() - lr * grad)

        return True


#
# class Adagrad(Optimizer):
#     """Implements Adagrad algorithm.
#
#      It has been proposed in `Adaptive Subgradient Methods for Online Learning
#      and Stochastic Optimization`_.
#
#      Arguments:
#          params (iterable): iterable of parameters to optimize or dicts defining
#              parameter groups
#          lr (float, optional): learning rate (default: 1e-2)
#          lr_decay (float, optional): learning rate decay (default: 0)
#          weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#          eps (float, optional): term added to the denominator to improve
#              numerical stability (default: 1e-10)
#
#      .. _Adaptive Subgradient Methods for Online Learning and Stochastic
#          Optimization: http://jmlr.org/papers/v12/duchi11a.html
#      """
#     def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10,gradient_centralization=None):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= lr_decay:
#             raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if not 0.0 <= initial_accumulator_value:
#             raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         self.gradient_centralization=gradient_centralization
#         self.eps = eps
#
#         defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
#                         initial_accumulator_value=initial_accumulator_value)
#         super(Adagrad, self).__init__(params, defaults)
#
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['sum'] =ones_like(p)*initial_accumulator_value
#
#     def share_memory(self):
#         pass
#         # for group in self.param_groups:
#         #     for p in group['params']:
#         #         state = self.state[p]
#         #         state['sum'].share_memory_()
#
#     def __setstate__(self, state):
#         super(Adagrad, self).__setstate__(state)
#
#     def step(self, grads_and_vars=None,**kwargs):
#         """Performs a single optimization step.
#
#         Args:
#             grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.
#
#         """
#         self.grads_and_vars = self._filter_grads(grads_and_vars)
#         # grads_and_vars=zip(new_grads, new_vars)
#
#         group = self.param_groups[0]
#         dampening = group['dampening']
#         nesterov = group['nesterov']
#         for grad, p in grads_and_vars:
#             if grad is None or any_abnormal_number(p) or not p.trainable:
#                 continue
#
#             if is_sparse(grad):
#                 raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#
#             state = self.state[p.ref()]
#             state['step'] += 1
#             lr = group['lr']
#
#             if group['weight_decay'] != 0:
#                 grad = grad + p * group['weight_decay']
#
#             clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
#
#             if is_sparse(grad):
#                 grad = grad.coalesce()  # the update is non-linear so indices must be unique
#                 grad_indices = grad._indices()
#                 grad_values = grad._values()
#                 size = grad.size()
#
#                 def make_sparse(values):
#                     constructor = grad.new
#                     if grad_indices.dim() == 0 or values.dim() == 0:
#                         return constructor().resize_as_(grad)
#                     return constructor(grad_indices, values, size)
#
#                 state['sum'].add_(make_sparse(grad_values.pow(2)))
#                 std = state['sum'].sparse_mask(grad)
#                 std_values = std._values().sqrt_().add_(group['eps'])
#                 p.add_(make_sparse(grad_values / std_values), alpha=-clr)
#             else:
#                 state['sum'].addcmul_(grad, grad, value=1)
#                 std = state['sum'].sqrt().add_(group['eps'])
#                 p.addcdiv_(grad, std, value=-clr)
#
#             p.assign(tf.Variable(to_numpy(p - lr * grad)))
#
#         return True
#
#
# class Adadelta(Optimizer):
#     """Implements Adadelta algorithm.
#
#     It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.
#
#     Arguments:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         rho (float, optional): coefficient used for computing a running average
#             of squared gradients (default: 0.9)
#         eps (float, optional): term added to the denominator to improve
#             numerical stability (default: 1e-6)
#         lr (float, optional): coefficient that scale delta before it is applied
#             to the parameters (default: 1.0)
#         weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#
#     __ https://arxiv.org/abs/1212.5701
#     """
#     def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0,gradient_centralization=None):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= rho <= 1.0:
#             raise ValueError("Invalid rho value: {}".format(rho))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         self.gradient_centralization=gradient_centralization
#         self.eps=eps
#         defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
#         super(Adadelta, self).__init__(params, defaults)
#     def __setstate__(self, state):
#         super(Adadelta, self).__setstate__(state)
#
#     def step(self, grads_and_vars=None,**kwargs):
#         """Performs a single optimization step.
#
#         Args:
#             grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.
#
#         """
#         self.grads_and_vars = self._filter_grads(grads_and_vars)
#         # grads_and_vars=zip(new_grads, new_vars)
#
#         group = self.param_groups[0]
#         dampening = group['dampening']
#         nesterov = group['nesterov']
#         for grad, p in grads_and_vars:
#             if grad is None or any_abnormal_number(p) or not p.trainable:
#                 continue
#
#             if is_sparse(grad):
#                 raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#
#             state = self.state[p.ref()]
#             # State initialization
#             if len(state) == 0:
#                 state['step'] = 0
#                 state['square_avg'] = zeros_like(p)
#                 state['acc_delta'] = zeros_like(p)
#
#             square_avg, acc_delta = state['square_avg'], state['acc_delta']
#             rho, eps = group['rho'], group['eps']
#
#             state['step'] += 1
#
#             if group['weight_decay'] != 0:
#                 grad=grad+group['weight_decay']*p
#
#
#             square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
#             std = sqrt(square_avg+self.eps)
#             delta =true_divide( sqrt(acc_delta+self.eps),std).mul_(grad)
#
#             p_t=p-group['lr']*delta
#             #acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
#             state['acc_delta']=acc_delta*rho+(delta**2)*(1 - rho)
#             p.assign(tf.Variable(to_numpy(p_t)))
#
#         return True
#
#
# class LBFGS(Optimizer):
#     """The Limited-Memory BFGS minimization algorithm.
#
#     Limited-memory quasi-Newton methods are useful for solving large problems
#     whose Hessian matrices cannot be computed at a reasonable cost or are not
#     sparse. Instead of storing fully dense n x n approximations of Hessian
#     matrices, they only save a few vectors of length n that represent the
#     approximations implicitly.
#     This module implements the algorithm known as L-BFGS, which, as its name
#     suggests, is a limited-memory version of the BFGS algorithm.
#
#     Reference:
#         https://github.com/tensorflow/probability/blob/v0.10.0/tensorflow_probability/python/optimizer/lbfgs.py
#
#     """
#
#     def __init__(self, params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9,
#                  history_size=100, line_search_fn=None,gradient_centralization=None):
#         if max_eval is None:
#             max_eval = max_iter * 5 // 4
#         defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad,
#             tolerance_change=tolerance_change, history_size=history_size, line_search_fn=line_search_fn)
#         super(LBFGS, self).__init__(params, defaults)
#
#         if len(self.param_groups) != 1:
#             raise ValueError("LBFGS doesn't support per-parameter options "
#                              "(parameter groups)")
#
#         self._params = self.param_groups[0]['params']
#         self._numel_cache = None
#         self.gradient_centralization = gradient_centralization
#
#     def _numel(self):
#         if self._numel_cache is None:
#             self._numel_cache =functools.reduce(lambda total, p: total + reduce_prod(int_shape(p)), self._params, 0)
#         return self._numel_cache
#
#     def _add_grad(self, step_size, update):
#         offset = 0
#         for p in self._params:
#             numel = p.numel()
#             # view as to avoid deprecated pointwise semantics
#             p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
#             offset += numel
#         assert offset == self._numel()
#
#     def __setstate__(self, state):
#         super(RAdam, self).__setstate__(state)
#
#     def step(self, grads_and_vars=None,**kwargs):
#         """Performs a single optimization step.
#
#         Args:
#             grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.
#
#         """
#         self.grads_and_vars = self._filter_grads(grads_and_vars)
#         # grads_and_vars=zip(new_grads, new_vars)
#         group = self.param_groups[0]
#         lr = group['lr']
#         max_iter = group['max_iter']
#         max_eval = group['max_eval']
#         tolerance_grad = group['tolerance_grad']
#         tolerance_change = group['tolerance_change']
#         line_search_fn = group['line_search_fn']
#         history_size = group['history_size']
#
#         # NOTE: LBFGS has only global state, but we register it as state for
#         # the first param, because this helps with casting in load_state_dict
#         state = self.state[self.param_groups[0]['params'][0].ref()]
#
#         current_evals = 1
#         # State initialization
#         if len(state) == 0:
#             state['step'] = 0
#             state['func_evals'] = 0
#             state['func_evals'] += 1
#
#         flat_grad = []
#
#
#         for grad, p in grads_and_vars:
#             if grad is None or any_abnormal_number(p) or not p.trainable:
#                 continue
#             flat_grad.append(reshape(grad,(-1)))
#
#         flat_grad=concate(flat_grad,axis=0)
#         opt_cond = flat_grad.abs().max() <= tolerance_grad
#
#         # optimal condition
#         if opt_cond:
#             return orig_loss
#
#         # tensors cached in state (for tracing)
#         d = state.get('d')
#         t = state.get('t')
#         old_dirs = state.get('old_dirs')
#         old_stps = state.get('old_stps')
#         ro = state.get('ro')
#         H_diag = state.get('H_diag')
#         prev_flat_grad = state.get('prev_flat_grad')
#         prev_loss = state.get('prev_loss')
#
#         n_iter = 0
#         # optimize for a max of max_iter iterations
#         while n_iter < max_iter:
#             # keep track of nb of iterations
#             n_iter += 1
#             state['step'] += 1
#
#             ############################################################
#             # compute gradient descent direction
#             ############################################################
#             if state['step'] == 1:
#                 d = flat_grad.neg()
#                 old_dirs = []
#                 old_stps = []
#                 ro = []
#                 H_diag = 1
#             else:
#                 # do lbfgs update (update memory)
#                 y = flat_grad.sub(prev_flat_grad)
#                 s = d.mul(t)
#                 ys = y.dot(s)  # y*s
#                 if ys > 1e-10:
#                     # updating memory
#                     if len(old_dirs) == history_size:
#                         # shift history by one (limited-memory)
#                         old_dirs.pop(0)
#                         old_stps.pop(0)
#                         ro.pop(0)
#
#                     # store new direction/step
#                     old_dirs.append(y)
#                     old_stps.append(s)
#                     ro.append(1. / ys)
#
#                     # update scale of initial Hessian approximation
#                     H_diag = ys / y.dot(y)  # (y*y)
#
#                 # compute the approximate (L-BFGS) inverse Hessian
#                 # multiplied by the gradient
#                 num_old = len(old_dirs)
#
#                 if 'al' not in state:
#                     state['al'] = [None] * history_size
#                 al = state['al']
#
#                 # iteration in L-BFGS loop collapsed to use just one buffer
#                 q = flat_grad.neg()
#                 for i in range(num_old - 1, -1, -1):
#                     al[i] = old_stps[i].dot(q) * ro[i]
#                     q.add_(old_dirs[i], alpha=-al[i])
#
#                 # multiply by initial Hessian
#                 # r/d is the final direction
#                 d = r = matmul(q, H_diag)
#                 for i in range(num_old):
#                     be_i = old_dirs[i].dot(r) * ro[i]
#                     r.add_(old_stps[i], alpha=al[i] - be_i)
#
#             if prev_flat_grad is None:
#                 prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
#             else:
#                 prev_flat_grad.copy_(flat_grad)
#             prev_loss = loss
#
#             ############################################################
#             # compute step length
#             ############################################################
#             # reset initial guess for step size
#             if state['step'] == 1:
#                 t = min(1., 1. / flat_grad.abs().sum()) * lr
#             else:
#                 t = lr
#
#             # directional derivative
#             gtd = flat_grad.dot(d)  # g * d
#
#             # directional derivative is below tolerance
#             if gtd > -tolerance_change:
#                 break
#
#             # optional line search: user function
#             ls_func_evals = 0
#             if line_search_fn is not None:
#                 # perform line search, using user function
#                 if line_search_fn != "strong_wolfe":
#                     raise RuntimeError("only 'strong_wolfe' is supported")
#                 else:
#                     x_init = self._clone_param()
#
#                     def obj_func(x, t, d):
#                         return self._directional_evaluate(closure, x, t, d)
#
#                     loss, flat_grad, t, ls_func_evals = _strong_wolfe(obj_func, x_init, t, d, loss, flat_grad, gtd)
#                 self._add_grad(t, d)
#                 opt_cond = flat_grad.abs().max() <= tolerance_grad
#             else:
#                 # no line search, simply move with fixed-step
#                 self._add_grad(t, d)
#                 if n_iter != max_iter:
#                     # re-evaluate function only if not in last iteration
#                     # the reason we do this: in a stochastic setting,
#                     # no use to re-evaluate that function here
#                     with torch.enable_grad():
#                         loss = float(closure())
#                     flat_grad = self._gather_flat_grad()
#                     opt_cond = flat_grad.abs().max() <= tolerance_grad
#                     ls_func_evals = 1
#
#             # update func eval
#             current_evals += ls_func_evals
#             state['func_evals'] += ls_func_evals
#
#             ############################################################
#             # check conditions
#             ############################################################
#             if n_iter == max_iter:
#                 break
#
#             if current_evals >= max_eval:
#                 break
#
#             # optimal condition
#             if opt_cond:
#                 break
#
#             # lack of progress
#             if d.mul(t).abs().max() <= tolerance_change:
#                 break
#
#             if abs(loss - prev_loss) < tolerance_change:
#                 break
#
#         state['d'] = d
#         state['t'] = t
#         state['old_dirs'] = old_dirs
#         state['old_stps'] = old_stps
#         state['ro'] = ro
#         state['H_diag'] = H_diag
#         state['prev_flat_grad'] = prev_flat_grad
#         state['prev_loss'] = prev_loss
#
#
#
#         return True


class RAdam(Optimizer):
    """Variant of the Adam optimizer whose adaptive learning rate is rectified

    so as to have a consistent variance.
    It implements the Rectified Adam (a.k.a. RAdam) proposed by
    Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
    And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).

    Examples:
        >>> opt =RAdam(lr=1e-3)


    Note: `amsgrad` is not described in the original paper. Use it with
          caution.

    RAdam is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, N_sma_threshhold=5, weight_decay=0,
                 degenerated_to_sgd=True, gradient_centralization=None):
        """Construct a new RAdam optimizer.
         Args:
             params: trainable parameters from model

             lr (float): The learning rate.
             betas:  beta1 means the exponential decay rate for the 1st moment estimates.
                 beta_2 means he exponential decay rate for the 2nd moment estimates.
             eps (float): A small constant for numerical stability.
             weight_decay(float): A floating point value. Weight decay for each param.

             N_sma_threshhold (float). The threshold for simple mean average.

             degenerated_to_sgd(bool): If True will be degenerated as sgd.

             gradient_centralization (None,string):
                if None, do nothing.
                if 'gcc' , means only convolution layer will apply 'Gradient Centralization'
                if 'gc', means convolution layer  and dense layer will apply 'Gradient Centralization'

         References:
             Gradient Centralization: A New Optimization Technique for Deep Neural Networks
             https://arxiv.org/abs/2004.01461


         """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.gradient_centralization = 'gc' if gradient_centralization == True else gradient_centralization
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)
        self.buffer = self.param_groups[0]['buffer']
        self.N_sma_threshhold = N_sma_threshhold

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            **kwargs ():
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)

        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['m'] = zeros_like(p)
                state['v'] = zeros_like(p)
            else:
                state['m'] = cast(state['m'], p.dtype)
                state['v'] = cast(state['v'], p.dtype)

            m, v = state['m'], state['v']
            beta1, beta2 = group['betas']

            # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            m_t = beta1 * m + (1 - beta1) * grad

            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v_t = beta2 * v + (1 - beta2) * (grad * grad)
            state['m'] = m_t
            state['v'] = v_t

            grad = gc_grads(grad, self.gradient_centralization)
            state['step'] += 1

            buffered = self.buffer[int(state['step'] % 10)]
            if state['step'] == buffered[0]:
                N_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = state['step']
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                buffered[1] = N_sma

                step_size = 1.0 / (1 - beta1 ** state['step'])
                if N_sma >= 5:
                    step_size = math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                N_sma_max - 2)) / (1 - beta1 ** state['step'])
                elif self.degenerated_to_sgd:
                    step_size = 1.0 / (1 - beta1 ** state['step'])
                else:
                    step_size = 1.0

                buffered[2] = step_size

            p_data = p.value()
            if group['weight_decay'] != 0:
                p_data = p_data - group['weight_decay'] * group['lr'] * p_data

            p_t = where(N_sma > self.N_sma_threshhold,
                        p_data - group['lr'] * step_size * m_t / (sqrt(v_t) + group['eps']),
                        p_data - group['lr'] * step_size * m_t)
            p.assign(p_t)

        return True


class PlainRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True, gradient_centralization=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.gradient_centralization = gradient_centralization
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)

        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['m'] = zeros_like(p)
                state['v'] = zeros_like(p)
            else:
                state['m'] = cast(state['m'], p.dtype)
                state['v'] = cast(state['v'], p.dtype)

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

            beta2_t = beta2 ** state['step']
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
            p_data = p.value()
            # more conservative since it's an approximated value
            if N_sma >= 5:
                if group['weight_decay'] != 0:
                    p_data = p_data + group['weight_decay'] * group['lr'] * p_data

                step_size = group['lr'] * math.sqrt(
                    (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                            N_sma_max - 2)) / (1 - beta1 ** state['step'])
                denom = sqrt(v_t) + group['eps']
                p_t = p_data - group['lr'] * step_size * m_t / (sqrt(v_t) + group['eps'])
                p.assign(p_t)
            elif self.degenerated_to_sgd:

                if group['weight_decay'] != 0:
                    p_data = p_data + group['weight_decay'] * group['lr'] * p_data
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                p_t = p_data - step_size * m_t
                p.assign(p_t)
        return True


class AdamW(Optimizer):
    """Optimizer that implements the Adam algorithm with weight decay.

    This is an implementation of the AdamW optimizer described in "Decoupled
    Weight Decay Regularization" by Loshch ilov & Hutter
    (https://arxiv.org/abs/1711.05101)
    ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).

    It computes the update step of `tf.keras.optimizers.Adam` and additionally
    decays the variable. Note that this is different from adding L2
    regularization on the variables to the loss: it regularizes variables with
    large gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.
    For further information see the documentation of the Adam Optimizer.

    Examples:
        >>> AdamW(lr=0.001, betas=(0.9, 0.999))

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0, gradient_centralization=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.gradient_centralization = 'gc' if gradient_centralization == True else gradient_centralization
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)

        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p.ref()]
            p_data = p.value()
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['m'] = zeros_like(p)
                state['v'] = zeros_like(p)
            else:
                state['m'] = cast(state['m'], p.dtype)
                state['v'] = cast(state['v'], p.dtype)

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

            denom = sqrt(v_t) + group['eps']
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            if group['warmup'] > state['step']:
                scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
            else:
                scheduled_lr = group['lr']

            step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

            if group['weight_decay'] != 0:
                p_data = p_data + group['weight_decay'] * scheduled_lr * p_data
            p_data=p_data - step_size * m_t / denom
            p.assign(p_data)

        return True


class Lookahead(Optimizer):
    """This class allows to extend optimizers with the lookahead mechanism.

    The mechanism is proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1). The optimizer iteratively updates two
    sets of weights: the search directions for weights are chosen by the inner
    optimizer, while the "slow weights" are updated each `k` steps based on the
    directions of the "fast weights" and the two sets of weights are
    synchronized. This method improves the learning stability and lowers the
    variance of its inner optimizer.

    Examples:
        >>> opt = Lookahead(SGD(lr=0.001))

    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer

        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    #
    # @property
    # def grad_tape(self):
    #     return self.optimizer.grad_tape
    #
    # @grad_tape.setter
    # def grad_tape(self, value):
    #     self.optimizer.grad_tape = value

    @property
    def grads_and_vars(self):
        return self.optimizer.grads_and_vars

    @grads_and_vars.setter
    def grads_and_vars(self, value):
        self.optimizer.grads_and_vars = value

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast.ref()]
            fast_data = fast.value()
            if "slow_param" not in param_state:
                param_state["slow_param"] = zeros_like(fast)
                param_state["slow_param"] = fast_data
            slow = param_state["slow_param"]
            slow += (fast_data - slow) * self.alpha
            fast.assign(slow)
            param_state["slow_param"] = slow

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """

        _ = self.optimizer.step(grads_and_vars, )
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


class Ranger(Optimizer):
    """Variant of the Adam optimizer whose adaptive learning rate is rectified

    so as to have a consistent variance.
    It implements the Rectified Adam (a.k.a. RAdam) proposed by
    Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
    And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).

    Examples:
        >>> opt =RAdam(lr=1e-3)


    Note: `amsgrad` is not described in the original paper. Use it with
          caution.

    RAdam is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), alpha=0.5, k=6,eps=1e-8, N_sma_threshhold=5, weight_decay=0,
                 degenerated_to_sgd=False, gradient_centralization=None):
        """Construct a new RAdam optimizer.
         Args:
             params: trainable parameters from model

             lr (float): The learning rate.
             betas:  beta1 means the exponential decay rate for the 1st moment estimates.
                 beta_2 means he exponential decay rate for the 2nd moment estimates.
             eps (float): A small constant for numerical stability.
             weight_decay(float): A floating point value. Weight decay for each param.

             N_sma_threshhold (float). The threshold for simple mean average.

             degenerated_to_sgd(bool): If True will be degenerated as sgd.

             gradient_centralization (None,string):
                if None, do nothing.
                if 'gcc' , means only convolution layer will apply 'Gradient Centralization'
                if 'gc', means convolution layer  and dense layer will apply 'Gradient Centralization'

         References:
             Gradient Centralization: A New Optimization Technique for Deep Neural Networks
             https://arxiv.org/abs/2004.01461


         """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {k}')
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.gradient_centralization = 'gc' if gradient_centralization == True else gradient_centralization
        self.degenerated_to_sgd = degenerated_to_sgd

        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold,eps=eps, weight_decay=weight_decay)
        super(Ranger, self).__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # now we can get to work...
        # removed as we now use step from RAdam...no need for duplicate step counting
        # for group in self.param_groups:
        #    group["step_counter"] = 0
        # print("group step counter init")

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # self.first_run_check=0

        # lookahead weights  # 9/2/19 - lookahead param tensors have been moved to state storage.  # This should   #
        # resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        # self.slow_weights = [[p.clone().detach() for p in group['params']]  #                     for group in
        # self.param_groups]

        # don't use grad for lookahead weights  # for w in it.chain(*self.slow_weights):  #    w.requires_grad = False


    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            **kwargs ():
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)

        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['m'] = zeros_like(p)
                state['v'] = zeros_like(p)
                # look ahead weight storage now in state dict
                state['slow_buffer'] =tf.raw_ops.DeepCopy(x=p.value())
            else:
                state['m'] = cast(state['m'], p.dtype)
                state['v'] = cast(state['v'], p.dtype)

            m, v = state['m'], state['v']
            beta1, beta2 = group['betas']
            if self.gradient_centralization!='gc':
                grad = gc_grads(grad, self.gradient_centralization)
            # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            m_t = beta1 * m + (1 - beta1) * grad

            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v_t = beta2 * v + (1 - beta2) * (grad * grad)
            state['m'] = m_t
            state['v'] = v_t


            state['step'] += 1

            buffered = self.radam_buffer[int(state['step'] % 10)]
            if state['step'] == buffered[0]:
                N_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = state['step']
                beta2_t = beta2 ** state['step']
                N_sma_max = 2.0 / (1.0 - beta2) - 1.0
                N_sma = N_sma_max - 2.0 * state['step'] * beta2_t / (1.0 - beta2_t)
                buffered[1] = N_sma

                step_size = 1.0 / (1.0 - beta1 ** state['step'])
                if N_sma >= 5:
                    step_size = math.sqrt(
                        (1.0 - beta2_t) * (N_sma - 4.0) / (N_sma_max - 4.0) * (N_sma - 2.0) / N_sma * N_sma_max / (
                                N_sma_max - 2.0)) / (1.0 - beta1 ** state['step'])
                else:
                    step_size = 1.0 / (1 - beta1 ** state['step'])
                if self.degenerated_to_sgd:
                    step_size = 1.0

                buffered[2] = step_size

            p_data =tf.raw_ops.DeepCopy(x=p.value())
            if group['weight_decay'] != 0:
                p_data = p_data - group['weight_decay'] * group['lr'] * p_data

            p_t = where(N_sma > self.N_sma_threshhold,
                        p_data - group['lr'] * step_size * m_t / (sqrt(v_t) + group['eps']),
                        p_data - group['lr'] * step_size * m_t)
            p.assign(p_t)

            # integrated look ahead...
            # we do it at the param level instead of group level
            if state['step'] % group['k'] == 0:
                slow_p = state['slow_buffer']  # get access to slow param tensor
                slow_p+=(self.alpha*(p_t- slow_p))  # (fast weights - slow weights) * alpha
                state['slow_buffer']=slow_p
                if any_abnormal_number(slow_p):
                    sys.stderr.write('{0} p_data_fp32 has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                    slow_p = where(is_nan(slow_p), p_t, slow_p)
                p.assign(slow_p)  # copy interpolated weights to RAdam param tensor

        return True




class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
            self,
            params, lr=1e-2,
            momentum=0.9,
            use_nesterov=False,
            weight_decay=0.0,
            exclude_from_weight_decay=None,
            exclude_from_layer_adaptation=None,
            classic_momentum=True,
            eeta=0.001):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def __setstate__(self, state):
        print("set state called")
        super(LARS, self).__setstate__(state)

    def step(self, grads_and_vars=None, epoch=None):
        """Performs a single optimization step.

        Args:
            epoch (int): current epoch
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        loss = None
        self.grads_and_vars = self._filter_grads(grads_and_vars, self.gradient_centralization)

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
                continue

            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]
            param = p.data
            grad = p.grad.data

            param_state = self.state[p]

            # TODO: get param names
            # if self._use_weight_decay(param_name):
            grad += self.weight_decay * param

            if self.classic_momentum:
                trust_ratio = 1.0

                # TODO: get param names
                # if self._do_layer_adaptation(param_name):
                w_norm = tf.nn.l2_normalize(param)
                g_norm = tf.nn.l2_normalize(grad)

                device = g_norm.get_device()
                trust_ratio = tf.where(
                    w_norm.ge(0),
                    tf.where(
                        greater_equal(g_norm, 0),
                        (self.eeta * w_norm / g_norm),
                        to_tensor([1.0]),
                    ), to_tensor([1.0]),
                ).numpy()[0]

                scaled_lr = lr * trust_ratio
                if "momentum_buffer" not in param_state:
                    next_v = param_state["momentum_buffer"] = zeros_like(
                        p.data
                    )
                else:
                    next_v = param_state["momentum_buffer"]

                next_v.mul_(momentum).add_(scaled_lr, grad)
                if self.use_nesterov:
                    update = (self.momentum * next_v) + (scaled_lr * grad)
                else:
                    update = next_v

                p.data.add_(-update)
            else:
                raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True



class AdaBelief(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Args:
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

    References
        .. _Adam\: A Method for Stochastic Optimization:
            https://arxiv.org/abs/1412.6980
        .. _On the Convergence of Adam and Beyond:
            https://openreview.net/forum?id=ryQu7f-RZ

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
                 gradient_centralization=None):
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
        self.amsgrad = amsgrad
        self.eps = eps
        self.gradient_centralization = 'gc' if gradient_centralization == True else gradient_centralization
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # grads_and_vars=zip(new_grads, new_vars)

        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
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
            beta_2_power = 1 - beta2 ** state['step']

            if group['weight_decay'] != 0:
                grad = grad + p.value() * group['weight_decay']

            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            m_t = beta1 * m + (1 - beta1) * grad
            grad_residual=grad-m_t
            # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            v_t = beta2 * v + (1 - beta2) * (grad_residual * grad_residual)

            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                vhat = state['vhat']
                # Maintains the maximum of all 2nd moment running avg. till now
                vhat_t = maximum(vhat, v_t)
                state['vhat'] = vhat_t
                denom = (sqrt(vhat_t) / math.sqrt(beta_2_power)) + group['eps']


            else:
                denom = (sqrt(v_t) / math.sqrt(beta_2_power)) + group['eps']

            step_size = group['lr'] / beta_1_power
            p.assign(p.value() - step_size * m_t / denom)

            # if reduce_mean(abs(p))>0:
            #     print(reduce_mean(abs(-(self.lr / beta_1_power) *exp_avg/ (denom + self.eps)))/reduce_mean(abs(p)))
            state['m'] = m_t
            state['v'] = v_t

        return True



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
        except Exception:
            optimizer_class = get_class(optimizer_name, optimizer_modules)
    return optimizer_class
