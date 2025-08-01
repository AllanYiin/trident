from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import sys
from functools import reduce
import collections
import copy
import math
import re
import numpy as np
import scipy.optimize as sciopt
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from tensorflow.python.trackable.base import Trackable
from trident.backend.common import get_session, get_class, snake2camel,get_time_suffix,camel2snake,get_session_value
from trident.backend.tensorflow_ops import *


__all__ = ['Adam', 'RMSprop', 'SGD', 'RAdam', 'Lookahead', 'Ranger','LARS','RangerLars','AdaBelief','RangerBelief','DiffGrad', 'get_optimizer']

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




class Optimizer(Trackable):
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

    def _filter_grads(self, grads_and_vars):
        """Filter out iterable with grad equal to None or abnormal grad and do the gradient centralization."""
        grads_and_vars = tuple(grads_and_vars)
        if not grads_and_vars:
            return grads_and_vars

        filtered = []
        vars_with_empty_grads = []
        for grad, var in grads_and_vars:
            if grad is None:
                vars_with_empty_grads.append(var)
            else:
                filtered.append((grad, var))
        filtered = tuple(filtered)

        if not filtered:
            raise ValueError("No gradients provided for any variable: %s." %
                             ([v.name for _, v in grads_and_vars],))
        if vars_with_empty_grads:
            sys.stdout.writelines(
                ("Gradients do not exist for variables {0} when minimizing the loss.").format([v.name for v in vars_with_empty_grads]))
        return filtered

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, tf.Variable) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

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
            with tf.device(param.device):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if isinstance(value, tf.Tensor):
                    if value.dtype!=param.dtype:
                        value=tf.cast(value,param.dtype)
                    return value
                elif isinstance(value, dict):
                    return {k: cast(param, v) for k, v in value.items()}
                elif isinstance(value,   collections.abc.Iterable):
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
            self.grads_and_vars=None
            gc.collect()


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
            for g in self.param_groups:
                g['lr'] = new_lr
            if verbose:
                print('learning rate changed! ( form {0:.3e} to {1:.3e})'.format(old_lr, new_lr))


    @property
    def parameters(self):
        """

        Returns: the weights need to train

        """
        return [self.param_groups[i]['params'] for i in range(len(self.param_groups))]

    @parameters.setter
    def parameters(self,value):
        """

        Returns: the weights need to train

        """
        if isinstance(value, tf.Variable):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            value.__class__.__name__)
        if not hasattr(self,'param_groups') or self.param_groups is None or len(self.param_groups)==0:
            self.param_groups=[]

            param_groups = list(value)
            if len(param_groups) == 0:
                raise ValueError("optimizer got an empty parameter list")
            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]
                for param_group in param_groups:
                    self.add_param_group(param_group)
        else:
            self.param_groups[0]['params']=value

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

        & V_t = \beta_1*V_{t-1} + (1-\beta_1)*grad
        & S_t = \beta_2*S_{t-1} + (1-\beta_2)*{grad} \odot {grad}
        & \hat{V_t} = \frac{V_t}{1-\beta_1^t}
        & \hat{S_t} = \frac{S_t}{1-\beta_2^t}
        & \hat{g} = learning\_rate*\frac{\hat{V_t}}{\sqrt{\hat{S_t}}+\epsilon}
        & param_{new} = param_{old} - \hat{g}

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

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        self.gradient_centralization = gradient_centralization



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
        #grads_and_vars = self._filter_grads(grads_and_vars)

        group=self.param_groups[0]

        for grad,p in grads_and_vars:
            # np_grad=to_numpy(grad)
            # print(p.name,np_grad.shape,np_grad.mean(),np.abs(np_grad).mean())
            if grad is None or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            if any_abnormal_number(grad):
                grad = where(is_abnormal_number(grad), zeros_like(grad), grad)

            amsgrad = group['amsgrad']
            p_data=p.value()#.detach()
            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = zeros_like(p_data)
                state['exp_avg_sq'] =zeros_like(p_data)
                state['max_exp_avg_sq'] = zeros_like(p_data)


            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']



            if group['weight_decay'] != 0:
                grad = grad + p_data * group['weight_decay']

            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)

            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * square(grad)


            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                max_exp_avg_sq=maximum(max_exp_avg_sq, exp_avg_sq)
                denom =(sqrt(max_exp_avg_sq)/sqrt(bias_correction2)) +group['eps']
            else:
                denom = (sqrt(exp_avg_sq)/sqrt(bias_correction2))+ group['eps']




            step_size = group['lr'] / bias_correction1
            G_grad = exp_avg/denom

            # if self.gradient_centralization in ['all', 'gc']:
            #     if len(list(int_shape(G_grad))) > 1:
            #         G_grad = G_grad - reduce_mean(G_grad, axis=list(range(ndim(G_grad) - 1)), keepdims=True)
            #

            if any_abnormal_number(G_grad):
                sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                G_grad = where(is_abnormal_number(G_grad),grad, G_grad)

            p.assign_add(-step_size*G_grad)
            # state['exp_avg'] = exp_avg
            # state['exp_avg_sq'] = exp_avg_sq
            # state['max_exp_avg_sq'] = exp_avg_sq
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
        self.gradient_centralization = gradient_centralization
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
            p_data=p.value()
            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0.0
                state['square_avg'] = zeros_like(p_data)
                if group['momentum'] > 0:
                    state['momentum_buffer'] = zeros_like(p_data)
                if group['centered']:
                    state['grad_avg'] = zeros_like(p_data)

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
        self.gradient_centralization = gradient_centralization
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

            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)
            # if group['weight_decay'] > 0:
            #     lr = lr * (1. / (1. + group['weight_decay'] * state['step']))

            if group['weight_decay'] != 0:
                grad = grad +p.value() * group['weight_decay']



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
#                 size = int_shape(grad)
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, N_sma_threshhold=5, weight_decay=0,
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

        self.degenerated_to_sgd = degenerated_to_sgd
        self.N_sma_threshhold = N_sma_threshhold
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)
        self.gradient_centralization = gradient_centralization
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
            p_data=p.value()#.detach()

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = zeros_like(p)
                state['exp_avg_sq'] = zeros_like(p)
            else:
                state['exp_avg'] = cast(state['exp_avg'], p_data.dtype)
                state['exp_avg_sq'] = cast(state['exp_avg_sq'], p_data.dtype)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)


            # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad

            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)

            # state['exp_avg_sq'] = exp_avg_sq
            # state['exp_avg'] = exp_avg
            state['step'] += 1
            #grad = gc_grads(grad, self.gradient_centralization)

            buffered = group['buffer'][int(state['step'] % 10)]
            if state['step'] == buffered[0]:
                N_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = state['step']
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                buffered[1] = N_sma

                if N_sma >= self.N_sma_threshhold:
                    step_size = math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                N_sma_max - 2)) / (1 - beta1 ** state['step'])
                elif self.degenerated_to_sgd:
                    step_size = 1.0 / (1 - beta1 ** state['step'])
                else:
                    step_size = -1
                buffered[2] = step_size


            if group['weight_decay'] != 0:
                p_data = p_data - group['weight_decay'] * group['lr'] * p_data

            if N_sma >= self.N_sma_threshhold:
                denom = sqrt(exp_avg_sq) + group['eps']
                p_data = p_data - group['lr'] * step_size * exp_avg /denom
            elif step_size > 0:
                p_data = p_data - group['lr'] * step_size * exp_avg

            p.assign(p_data)

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

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)
        self.gradient_centralization = gradient_centralization

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

            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)

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
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
        super(AdamW, self).__init__(params, defaults)
        self.gradient_centralization = gradient_centralization

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)
        #grads_and_vars = self._filter_grads(grads_and_vars)
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

            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)

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

    def __init__(self, optimizer, params, defaults, k=5, alpha=0.5):
        super().__init__(params, defaults)
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

    def __init__(self, params, lr=1e-3, betas=(.9, 0.999), alpha=0.5, k=6,eps=1e-6, N_sma_threshhold=5, weight_decay=0,
                  gradient_centralization=None):
        """Construct a new RAdam optimizer.
         Args:
             params: trainable parameters from model

             lr (float): The learning rate.
             betas:  beta1 means the exponential decay rate for the 1st moment estimates.
                 beta_2 means he exponential decay rate for the 2nd moment estimates.
             eps (float): A small constant for numerical stability.
             weight_decay(float): A floating point value. Weight decay for each param.

             N_sma_threshhold (float). The threshold for simple mean average.


             gradient_centralization (None,string):
                if None, do nothing.
                if 'gcc' , means only convolution layer will apply 'Gradient Centralization'
                if 'gc', means convolution layer  and dense layer will apply 'Gradient Centralization'

         References:
             Gradient Centralization: A New Optimization Technique for Deep Neural Networks
             https://arxiv.org/abs/2004.01461


         """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {}'.format(alpha))
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {}'.format(k))
        if not lr > 0:
            raise ValueError('Invalid Learning Rate: {}'.format(lr))
        if not eps > 0:
            raise ValueError('Invalid eps: {}'.format(eps))

        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.gradient_centralization = gradient_centralization
        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # now we can get to work...
        # removed as we now use step from RAdam...no need for
        # duplicate step counting
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

        # self.slow_weights = [[p.copy().detach() for p in group['params']]  #                     for group in
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
       # grads_and_vars = self._filter_grads(grads_and_vars)
        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Ranger does not support sparse gradients, please consider SparseAdam instead')
            if any_abnormal_number(grad):
                grad = where(is_abnormal_number(grad), zeros_like(grad), grad)

            p_data_fp32 =p.value()#.detach()
            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] =zeros_like(p_data_fp32)
                state['exp_avg_sq'] = zeros_like(p_data_fp32)
                # look ahead weight storage now in state dict
                state['slow_buffer'] = p_data_fp32.copy()
            else:
                state['exp_avg'] = cast(state['exp_avg'],p_data_fp32.dtype)
                state['exp_avg_sq'] =  cast(state['exp_avg_sq'],p_data_fp32.dtype)


            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']
            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)

            state['step'] += 1

            exp_avg=beta1 * exp_avg + (1.0 - beta1) * grad
            exp_avg_sq=beta2 * exp_avg_sq + (1.0 - beta2) * (grad**2)

            buffered = self.radam_buffer[int(state['step'] % 10)]

            if state["step"] == buffered[0]:
                N_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = state["step"]
                beta2_t = beta2 ** state["step"]
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                buffered[1] = N_sma

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = math.sqrt(
                        (1 - beta2_t)
                        * (N_sma - 4)
                        / (N_sma_max - 4)
                        * (N_sma - 2)
                        / N_sma
                        * N_sma_max
                        / (N_sma_max - 2)
                    ) / (1 - beta1 ** state["step"])
                else:
                    step_size = 1.0 / (1 - beta1 ** state["step"])
                buffered[2] = step_size


            if N_sma> self.N_sma_threshhold:
                denom = sqrt(exp_avg_sq) + group["eps"]
                G_grad = exp_avg / denom
            else:
                G_grad=exp_avg

            if group['weight_decay'] != 0:
                G_grad=G_grad+p_data_fp32*group['weight_decay']

            if self.gradient_centralization in ['all', 'gc']:
                if ndim(G_grad)> 1:
                    G_grad = G_grad - reduce_mean(G_grad, axis=list(range(ndim(G_grad) - 1)), keepdims=True)


            p_data_fp32 = p_data_fp32 + (-step_size * group['lr']) * G_grad

            if any_abnormal_number(p_data_fp32):
                sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n\r'.format(self.__class__.__name__))
                p_data_fp32 = where(is_abnormal_number(p_data_fp32), p.value(), p_data_fp32)

            p.assign(p_data_fp32, use_locking=False)

            state['exp_avg'] =exp_avg
            state['exp_avg_sq'] = exp_avg_sq


            # integrated look ahead...
            # we do it at the param level instead of group level

            if int(state['step'] %group['k']) == 0:
                slow_p = state['slow_buffer']  # get access to slow param tensor
                slow_p=slow_p+((p.value()- slow_p)*self.alpha)  # (fast weights - slow weights) * alpha
                if any_abnormal_number(slow_p):
                    sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                    slow_p = where(is_abnormal_number(slow_p), p.value(), slow_p)
                p.assign(slow_p, use_locking=False)  # copy interpolated weights to RAdam param tensor
                state['slow_buffer']=slow_p

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
            eeta=0.001,gradient_centralization=None):
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
        self.gradient_centralization=gradient_centralization
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

        #grads_and_vars=self._filter_grads(grads_and_vars)

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        group = self.param_groups[0]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        eeta = group["eeta"]
        lr = group["lr"]

        for grad, p in grads_and_vars:
            if grad is None or any_abnormal_number(p) or not p.trainable:
                continue


            param = p.value()

            param_state = self.state[p.ref()]

            # TODO: get param names
            # if self._use_weight_decay(param_name):
            grad =grad+ self.weight_decay * param


            if self.classic_momentum:
                trust_ratio = 1.0

                # TODO: get param names
                # if self._do_layer_adaptation(param_name):
                w_norm = norm(param)
                g_norm =norm(grad)

                device = g_norm.get_device()
                trust_ratio = tf.where(
                    greater_equal(w_norm,0),
                    tf.where(
                        greater_equal(g_norm, 0),
                        (self.eeta * true_divide(w_norm ,g_norm)),
                        to_tensor([1.0]),
                    ), to_tensor([1.0]),
                ).numpy()[0]

                scaled_lr = lr * trust_ratio
                if "momentum_buffer" not in param_state:
                    next_v = param_state["momentum_buffer"] = zeros_like(
                        param
                    )
                else:
                    next_v = param_state["momentum_buffer"]

                next_v=next_v*momentum+scaled_lr*grad
                if self.use_nesterov:
                    update = (self.momentum * next_v) + (scaled_lr * grad)
                else:
                    update = next_v

                p.assign(p.value()-update)
            else:
                raise NotImplementedError

        return True

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

class RangerLars(Optimizer):
    """
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
    """

    def __init__(self, params, lr=1e-3,alpha=0.5, k=6,N_sma_threshhold=5,betas=(0.9, 0.999),eeta=0.001, eps=1e-8, weight_decay=0,gradient_centralization=None):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {k}')

        if not lr > 0:
            raise ValueError('Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError('Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, N_sma_threshhold=N_sma_threshhold, betas=betas, eps=eps, weight_decay=weight_decay,eeta=eeta)
        super().__init__(params, defaults)
        self.gradient_centralization = gradient_centralization
        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]
        self.N_sma_threshhold=N_sma_threshhold
        self.alpha = alpha
        self.k = k

        # self.first_run_check=0

        # lookahead weights  # 9/2/19 - lookahead param tensors have been moved to state storage.  # This should   #
        # resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        # self.slow_weights = [[p.copy().detach() for p in group['params']]  #                     for group in
        # self.param_groups]

        # don't use grad for lookahead weights  # for w in it.chain(*self.slow_weights):  #    w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(RangerLars, self).__setstate__(state)


    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            **kwargs ():
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)
        #grads_and_vars = self._filter_grads(grads_and_vars)
        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None  or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            if any_abnormal_number(grad):
                grad = where(is_abnormal_number(grad), zeros_like(grad), grad)

            p_data = p.value()
            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0.0
                state['exp_avg'] =zeros_like(p_data)
                state['exp_avg_sq'] =zeros_like(p_data)
                # look ahead weight storage now in state dict
                state['slow_buffer'] =p_data.copy().detach()
            else:
                state['exp_avg'] = cast(state['exp_avg'], p.dtype)
                state['exp_avg_sq'] = cast(state['exp_avg_sq'], p.dtype)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1.0

            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)

            exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * square(grad)

            buffered = self.radam_buffer[int(state['step'] % 10)]

            if state["step"] == buffered[0]:
                N_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = state["step"]
                beta2_t = pow(beta2, state["step"])
                N_sma_max = 2.0 / (1 - beta2) - 1.0
                N_sma = N_sma_max - 2.0 * state["step"] * beta2_t / (1 - beta2_t)
                buffered[1] = N_sma

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = sqrt(
                        (1 - beta2_t)
                        * (N_sma - 4)
                        / (N_sma_max - 4)
                        * (N_sma - 2)
                        / N_sma
                        * N_sma_max
                        / (N_sma_max - 2)
                    ) / (1 - beta1 ** state["step"])
                else:
                    step_size = 1.0 / (1 - beta1 ** state["step"])
                buffered[2] = step_size

            var_t = zeros_like(p)

            if N_sma >= 5:
                denom = sqrt(exp_avg_sq) + group["eps"]
                var_t = (exp_avg / denom)
            else:
                var_t = exp_avg

            if group["weight_decay"] != 0:
                var_t += (-group['weight_decay'] * group['lr']) * p_data


            if self.gradient_centralization in ['all', 'gc']:
                if var_t.ndim > 1:
                    var_t+=(-var_t.mean(axis=tuple(range(1, var_t.ndim)), keepdims=True))

            radam_norm =  norm(var_t,axis=None)
            weight_norm = norm(p.value(),axis=None)
            if weight_norm == 0 or radam_norm == 0:
                trust_ratio = 1.0
            else:
                trust_ratio = clip(true_divide(weight_norm, radam_norm), 0.0, 10.0)

            state['weight_norm'] = weight_norm
            state['adam_norm'] = radam_norm
            state['trust_ratio'] = trust_ratio

            if any_abnormal_number(var_t):
                sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                var_t=(where(is_abnormal_number(var_t), p.value(), var_t))

            p.assign_add(var_t*trust_ratio* (-step_size * group['lr']), use_locking=False)
            state['exp_avg'] = exp_avg
            state['exp_avg_sq'] = exp_avg_sq
            # state['exp_avg'] = tf.Variable(initial_value=exp_avg_t)
            # state['exp_avg_sq'] = tf.Variable(initial_value=exp_avg_sq_t)

            # integrated look ahead...
            # we do it at the param level instead of group level
            if math_ops.floor_mod(state['step'] ,group['k']) == 0:
                slow_p = state['slow_buffer']  # get access to slow param tensor
                slow_p+= ((p.value()- slow_p)*self.alpha)  # (fast weights - slow weights) * alpha
                if any_abnormal_number(slow_p):
                    sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                    slow_p = where(is_abnormal_number(slow_p), p.value(), slow_p)
                p.assign(slow_p)  # copy interpolated weights to RAdam param tensor
                state['slow_buffer']=slow_p
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, amsgrad=False,
                 gradient_centralization=None, degenerated_to_sgd=True,weight_decouple=True, fixed_decay=False, rectify=True):
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
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, buffer=[[None, None, None] for _ in range(10)])
        super(AdaBelief, self).__init__(params, defaults)
        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        self.gradient_centralization = gradient_centralization

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
        #grads_and_vars = self._filter_grads(grads_and_vars)
        group = self.param_groups[0]
        for grad,p in grads_and_vars:
            if grad is None or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            if any_abnormal_number(grad):
                grad = where(is_abnormal_number(grad), zeros_like(grad), grad)

            amsgrad = group['amsgrad']
            p_data=p.value()#.detach()
            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = zeros_like(p_data)
                state['exp_avg_sq'] =zeros_like(p_data)
                state['max_exp_avg_sq'] = zeros_like(p_data)

            if self.weight_decouple:
                if not self.fixed_decay:
                    p_data=p_data*(1.0 - group['lr'] * group['weight_decay'])

                else:
                    p_data=p_data*(1.0 - group['weight_decay'])
            else:
                if group['weight_decay'] != 0:
                    grad=grad+p_data*group['weight_decay']


            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)

            # Decay the first and second moment running average coefficient
            # m_t = beta1 * m + (1 - beta1) * g_t
            exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
            grad_residual = grad - exp_avg
            exp_avg_sq=beta2*exp_avg_sq+ (1.0 - beta2) * (grad_residual*grad_residual)



            # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                max_exp_avg_sq=maximum(max_exp_avg_sq, exp_avg_sq)
                denom =(sqrt(max_exp_avg_sq)/sqrt(bias_correction2)) +group['eps']

            else:
                denom = (sqrt(exp_avg_sq)/sqrt(bias_correction2))+ group['eps']

            G_grad = exp_avg / denom
            if not self.rectify:
                # Default update
                step_size = group['lr'] / bias_correction1

            else:  # Rectified update, forked from RAdam
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
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                if N_sma >= 5:
                    denom = sqrt(exp_avg_sq)+group['eps']
                    G_grad = exp_avg / denom

                elif step_size > 0:
                    G_grad = exp_avg

            if self.gradient_centralization in ['all', 'gc']:
                if len(list(int_shape(G_grad))) > 1:
                    G_grad = G_grad - reduce_mean(G_grad, axis=list(range(ndim(G_grad) - 1)), keepdims=True)

            p_data = p_data - step_size * group['lr'] * G_grad
            if any_abnormal_number(p_data):
                sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n\r'.format(self.__class__.__name__))
                p_data = where(is_abnormal_number(p_data), p.value(), p_data)

            p.assign(p_data)
            state['exp_avg'] = exp_avg
            state['exp_avg_sq'] = exp_avg_sq
            state['max_exp_avg_sq'] = exp_avg_sq

        return True


class RangerBelief(Optimizer):
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), alpha=0.5, k=6,eps=1e-6, N_sma_threshhold=5, weight_decay=0,
                  gradient_centralization=None):
        """Construct a new RAdam optimizer.
         Args:
             params: trainable parameters from model

             lr (float): The learning rate.
             betas:  beta1 means the exponential decay rate for the 1st moment estimates.
                 beta_2 means he exponential decay rate for the 2nd moment estimates.
             eps (float): A small constant for numerical stability.
             weight_decay(float): A floating point value. Weight decay for each param.

             N_sma_threshhold (float). The threshold for simple mean average.


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



        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold,eps=eps, weight_decay=weight_decay)
        super(RangerBelief, self).__init__(params, defaults)
        self.gradient_centralization = gradient_centralization

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

        # self.slow_weights = [[p.copy().detach() for p in group['params']]  #                     for group in
        # self.param_groups]

        # don't use grad for lookahead weights  # for w in it.chain(*self.slow_weights):  #    w.requires_grad = False


    def __setstate__(self, state):
        super(RangerBelief, self).__setstate__(state)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Args:
            **kwargs ():
            grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        # self.grads_and_vars = self._filter_grads(grads_and_vars)
        # grads_and_vars=zip(new_grads, new_vars)
        #grads_and_vars = self._filter_grads(grads_and_vars)
        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None  or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            if any_abnormal_number(grad):
                grad = where(is_abnormal_number(grad), zeros_like(grad), grad)

            p_data = p.value()
            state = self.state[p.ref()]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] =zeros_like(p_data)
                state['exp_avg_sq'] =zeros_like(p_data)
                # look ahead weight storage now in state dict
                state['slow_buffer'] =p_data.copy()
                state['previous_grad'] =grad.copy()


            exp_avg, exp_avg_sq,previous_grad = state['exp_avg'], state['exp_avg_sq'],state['previous_grad']
            beta1, beta2 = group['betas']
            state['step']+=1
            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)

            exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
            grad_residual=grad-exp_avg
            exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * square(grad_residual)
            denom =sqrt(exp_avg_sq / (1.0 - beta2))+group['eps']
            # compute diffgrad coefficient (dfc)


            buffered = self.radam_buffer[int(state['step'] % 10)]

            if state["step"] == buffered[0]:
                N_sma, step_size = buffered[1], buffered[2]
            else:
                buffered[0] = state["step"]
                beta2_t = beta2 ** state["step"]
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                buffered[1] = N_sma

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = math.sqrt(
                        (1 - beta2_t)
                        * (N_sma - 4)
                        / (N_sma_max - 4)
                        * (N_sma - 2)
                        / N_sma
                        * N_sma_max
                        / (N_sma_max - 2)
                    ) / (1 - beta1 ** state["step"])
                else:
                    step_size = 1.0 / (1 - beta1 ** state["step"])
                buffered[2] = step_size





            diff = abs(previous_grad - grad_residual)
            dfc = 1. / (1. + exp(-diff))
            state['previous_grad'] = grad_residual

            # update momentum with dfc
            exp_avg1 = exp_avg * dfc

            if N_sma >= 5:

                G_grad = (exp_avg1 / denom)
            else:
                G_grad =  exp_avg1

            if group['weight_decay'] != 0:
                G_grad = G_grad + p_data* group['weight_decay']

            if self.gradient_centralization in ['all', 'gc']:
                if len(list(int_shape(G_grad))) > 1:
                    G_grad = G_grad - reduce_mean(G_grad, axis=list(range(ndim(G_grad) - 1)), keepdims=True)

            if any_abnormal_number(G_grad):
                sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                G_grad = where(is_abnormal_number(G_grad), p.value(), G_grad)

            p.assign_sub(G_grad *step_size* group['lr'], use_locking=False)
            state['exp_avg'] = exp_avg
            state['exp_avg_sq'] = exp_avg_sq


            # state['exp_avg'] = tf.Variable(initial_value=exp_avg1)
            # state['exp_avg_sq'] = tf.Variable(initial_value=exp_avg_sq_t)

            # integrated look ahead...
            # we do it at the param level instead of group level
            if math_ops.floor_mod(state['step'] ,group['k']) == 0:
                slow_p = state['slow_buffer']  # get access to slow param tensor
                slow_p+= ((p.value() - slow_p)*self.alpha)  # (fast weights - slow weights) * alpha
                if any_abnormal_number(slow_p):
                    sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                    slow_p = where(is_abnormal_number(slow_p), p.value(), slow_p)
                p.assign(slow_p)  # copy interpolated weights to RAdam param tensor
                state['slow_buffer']=slow_p
        return True

class DiffGrad(Optimizer):
    r"""Implements diffGrad algorithm. It is modified from the pytorch implementation of Adam.

    It has been proposed in `diffGrad: An Optimization Method for Convolutional Neural Networks`_.

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

    .. _diffGrad: An Optimization Method for Convolutional Neural Networks:
        https://arxiv.org/abs/1909.11015
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0,gradient_centralization=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DiffGrad, self).__init__(params, defaults)
        self.gradient_centralization=gradient_centralization

    def __setstate__(self, state):
        super(DiffGrad, self).__setstate__(state)

    def step(self, grads_and_vars=None, **kwargs):
        """Performs a single optimization step.

        Arguments:
          grads_and_vars (zipped tuple): A zipped gradients and parameters from gradient_tape.

        """
        group = self.param_groups[0]
        for grad, p in grads_and_vars:
            if grad is None or not p.trainable:
                continue

            if is_sparse(grad):
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            if any_abnormal_number(grad):
                grad = where(is_abnormal_number(grad), zeros_like(grad), grad)

            p_data = p.value()
            state = self.state[p.ref()]


            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = zeros_like(p_data)
                state['exp_avg_sq'] =zeros_like(p_data)
                # Previous gradient
                state['previous_grad'] =zeros_like(p_data)
            else:
                cast(state['exp_avg'] , p_data.dtype)
                cast(state['exp_avg_sq'], p_data.dtype)

            exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
            beta1, beta2 = group['betas']

            if self.gradient_centralization in ['all', 'gcc']:
                if ndim(grad) > 3:
                    grad = grad - reduce_mean(grad, axis=list(range(ndim(grad) - 1)), keepdims=True)

            state['step'] += 1
            bias_correction1 = 1 - pow(beta1, state['step'])
            bias_correction2 = 1 - pow(beta2, state['step'])

            if group['weight_decay'] != 0:
                grad+=(p.data*group['weight_decay'])

            # Decay the first and second moment running average coefficient
            exp_avg=beta1 * exp_avg + (1.0 - beta1) * grad
            exp_avg_sq=beta2 * exp_avg_sq + (1.0 - beta2) * square(grad)

            denom = sqrt(exp_avg_sq / bias_correction2) +group['eps']


            # compute diffgrad coefficient (dfc)
            diff = abs(previous_grad - grad)
            dfc = 1. / (1. + exp(-diff))

            # update momentum with dfc
            exp_avg1 = exp_avg * dfc
            state['previous_grad'] = grad

            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
            G_grad = true_divide(exp_avg1, denom)

            if self.gradient_centralization in ['all', 'gc']:
                if len(list(int_shape(G_grad))) > 1:
                    G_grad = G_grad - reduce_mean(G_grad, axis=list(range(ndim(G_grad) - 1)), keepdims=True)

            if any_abnormal_number(p_data):
                sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                G_grad = where(is_abnormal_number(G_grad), zeros_like(p_data), G_grad)
            p.assign_add(-step_size* G_grad)
            state['exp_avg'] = exp_avg
            state['exp_avg_sq'] = exp_avg_sq
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
