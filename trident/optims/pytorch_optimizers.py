from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import math
import sys
from collections import defaultdict
from functools import reduce
import torch
import torch.optim as optim
from torch.optim import optimizer,lbfgs,adagrad,adadelta,rmsprop
from trident.backend.common import get_class, snake2camel
from trident.backend.pytorch_ops import *

__all__ = ['Adam', 'SGD', 'LBFGS', 'Adadelta', 'Adagrad', 'RMSprop', 'RAdam', 'PlainRAdam', 'AdamW', 'Lookahead',
           'Ranger', 'RangerLars','AdaBelief','RangerBelief','DiffGrad','get_optimizer']





class Optimizer(optimizer.Optimizer):
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
        super().__init__(params, defaults)
        self._base_lr = 1e-3
        self.gradient_centralization=None


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
    def parameters(self):
        """

        Returns: the weights need to train

        """
        return self.param_groups['params']

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

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.999), eps=1e-7, weight_decay=0, amsgrad=False,
                 gradient_centralization=None, **kwargs):
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

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []


            for p in group['params'] :
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                if any_abnormal_number(grad):
                    grad = where(is_abnormal_number(grad), zeros_like(grad), grad)
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    #if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                G_grad = exp_avg / denom
                if self.gradient_centralization in ['all', 'gc']:
                    if len(list(G_grad.size())) > 1:
                        G_grad.add_(-G_grad.mean(dim=tuple(range(1, len(list(G_grad.size())))), keepdim=True))

                p.data.add_(G_grad, alpha=-step_size)

        return loss


class SGD(optim.SGD):
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
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

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
    def parameters(self):
        """

        Returns: the weights need to train

        """
        return self.param_groups['params']

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


class LBFGS(lbfgs.LBFGS):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self, params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None):
        super().__init__(params, lr=lr, max_iter=max_iter,max_eval=max_eval, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size,line_search_fn=line_search_fn)

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
    def parameters(self):
        """

        Returns: the weights need to train

        """
        return self.param_groups['params']

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


class Adadelta(adadelta.Adadelta):
    """Implements Adadelta algorithm.

    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1212.5701
    """
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-7, weight_decay=0):
        super().__init__(params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

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
    def parameters(self):
        """

        Returns: the weights need to train

        """
        return self.param_groups['params']

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


class Adagrad(adagrad.Adagrad):
    """Implements Adagrad algorithm.

     It has been proposed in `Adaptive Subgradient Methods for Online Learning
     and Stochastic Optimization`_.

     Arguments:
         params (iterable): iterable of parameters to optimize or dicts defining
             parameter groups
         lr (float, optional): learning rate (default: 1e-2)
         lr_decay (float, optional): learning rate decay (default: 0)
         weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
         eps (float, optional): term added to the denominator to improve
             numerical stability (default: 1e-10)

     .. _Adaptive Subgradient Methods for Online Learning and Stochastic
         Optimization: http://jmlr.org/papers/v12/duchi11a.html
     """
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-7):
        super().__init__(params, lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,initial_accumulator_value=initial_accumulator_value)

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
    def parameters(self):
        """

        Returns: the weights need to train

        """
        return self.param_groups['params']

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


class RMSprop(rmsprop.RMSprop):
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
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-7, weight_decay=0, momentum=0, centered=False,gradient_centralization=None):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,momentum=momentum,centered=centered)

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
    def parameters(self):
        """

        Returns: the weights need to train

        """
        return self.param_groups['params']

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


class RAdam(Optimizer):
    """Variant of the Adam optimizer whose adaptive learning rate is rectified
        so as to have a consistent variance.
        It implements the Rectified Adam (a.k.a. RAdam) proposed by
        Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
        And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).

        Example of usage:
        ```python
        opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
        ```

        Note: `amsgrad` is not described in the original paper. Use it with
              caution.
        RAdam is not a placement of the heuristic warmup, the settings should be
        kept if warmup has already been employed and tuned in the baseline method.
        You can enable warmup by setting `total_steps` and `warmup_proportion`:
        ```python
        opt = RAdam(lr=1e-3, betas=(0.9,0.999))

        ```
        In the above example, the learning rate will increase linearly
        from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
        in 9000 steps.
        Lookahead, proposed by Michael R. Zhang et.al in the paper
        [Lookahead Optimizer: k steps forward, 1 step back]
        (https://arxiv.org/abs/1907.08610v1), can be integrated with RAdam,
        which is announced by Less Wright and the new combined optimizer can also
        be called "Ranger". The mechanism can be enabled by using the lookahead
        wrapper. For example:

        ```python

        radam =RAdam()
        ranger = Lookahead(radam)

        ```
        """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0, N_sma_threshhold=5,
                 degenerated_to_sgd=True,gradient_centralization=None):
        """Construct a new RAdam optimizer.
        Args:
            params: trainable parameters from model

            lr (float): The learning rate.
            betas:  beta1 means the exponential decay rate for the 1st moment estimates.
                beta_2 means he exponential decay rate for the 2nd moment estimates.
            eps: A small constant for numerical stability.
            weight_decay: A floating point value. Weight decay for each param.

            N_sma_threshhold. A float value.
                The threshold for simple mean average.

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

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data)
                    state['exp_avg_sq'] = torch.zeros_like(p_data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad,value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

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
                    p_data.add_(p_data, alpha=-group['weight_decay'] * group['lr'])
                # more conservative since it's an approximated value
                if N_sma >= self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data.addcdiv_(exp_avg, denom,value=-step_size * group['lr'])

                elif step_size > 0:
                    p_data.add_(exp_avg, alpha=-step_size * group['lr'])
                p.data.copy_(p_data)

        return loss


class PlainRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True,gradient_centralization=None):
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

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data)
                    state['exp_avg_sq'] = torch.zeros_like(p_data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad,value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data.add_(p_data, alpha=-group['weight_decay'] * group['lr'])
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data.addcdiv_(grad, denom, value=-step_size)
                    p.data.copy_(p_data)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data.add_(p_data, alpha=-group['weight_decay'] * group['lr'])
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data.add_(grad,alpha=-step_size)
                    p.data.copy_(p_data)

        return loss


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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0, warmup=0,gradient_centralization=None):
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

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data = p.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data)
                    state['exp_avg_sq'] = torch.zeros_like(p_data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad,value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data.add_(p_data, alpha=-group['weight_decay'] * scheduled_lr)

                p_data.addcdiv_( grad, denom,value=-step_size)

                p.data.copy_(p_data)

        return loss


class Lookahead(Optimizer):
    def __init__(self, optimizer, params, defaults, k=5, alpha=0.5, gradient_centralization=None):
        super().__init__(params, defaults)
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
        self.gradient_centralization=gradient_centralization
        self.optimizer.gradient_centralization=gradient_centralization

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = self.optimizer.step(closure, )
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {(id(k) if isinstance(k, torch.Tensor) else k): v for k, v in self.state.items()}
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
    """
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
    """

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-7,
                 weight_decay=0,gradient_centralization=None):
        self.gradient_centralization=gradient_centralization
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {}'.format(alpha))
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {}'.format(k))
        if not lr > 0:
            raise ValueError('Invalid Learning Rate: {}'.format(lr))
        if not eps > 0:
            raise ValueError('Invalid eps: {}'.format(eps))

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to
        # make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

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

        # lookahead weights
        # 9/2/19 - lookahead param tensors have been moved to state storage.
        # This should resolve issues with load/save where weights were left in
        # GPU memory from first load, slowing down future runs.

        # self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        # don't use grad for lookahead weights
        # for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False
    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = None

        if closure is not None:
            loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data = p.data

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    state['step'] = 0.0
                    state['exp_avg'] = torch.zeros_like(p_data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p_data, memory_format=torch.preserve_format)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)


                state['step'] += 1.0
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])


                if N_sma > 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data.addcdiv_(exp_avg, denom,value=-step_size * group['lr'])
                else:
                    p_data.add_( exp_avg,alpha=-step_size * group['lr'])

                if any_abnormal_number(p_data):
                    sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n\r'.format(self.__class__.__name__))
                    p_data=where(is_abnormal_number(p_data),p.data,p_data)

                p.data.copy_(p_data)


                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']  # get access to slow param tensor
                    slow_p.add_( p.data - slow_p,alpha=self.alpha)  # (fast weights - slow weights) * alpha
                    if any_abnormal_number(slow_p):
                        sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                        slow_p = where(is_abnormal_number(slow_p), p.data, slow_p)
                    p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor

        return loss



class Ranger_new(Optimizer):
    """
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
    """

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-7,
                 weight_decay=0,gradient_centralization=None):
        self.gradient_centralization=gradient_centralization
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {}'.format(alpha))
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {}'.format(k))
        if not lr > 0:
            raise ValueError('Invalid Learning Rate: {}'.format(lr))
        if not eps > 0:
            raise ValueError('Invalid eps: {}'.format(eps))

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to
        # make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

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

        # lookahead weights
        # 9/2/19 - lookahead param tensors have been moved to state storage.
        # This should resolve issues with load/save where weights were left in
        # GPU memory from first load, slowing down future runs.

        # self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        # don't use grad for lookahead weights
        # for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False
    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = None

        if closure is not None:
            loss = closure()

        # Evaluate averages and grad, update param tensors
        n=0

        all_para=sum([len(group['params']) for group in self.param_groups])
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    state['step'] = 0.0
                    state['global_step'] = 0
                    state['layer_seq'] = n

                    state['is_train'] =n // (all_para / 20) <= 0

                    state['layer_lr'] = 1e-3 if state['is_train'] else 0
                    state['exp_avg'] = torch.zeros_like(p_data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p_data, memory_format=torch.preserve_format)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data)

                    state['global_step']+=1
                    state['is_train'] = n // (all_para / 20) <= state['global_step']//100
                    if state['global_step']<=2000:
                        state['layer_lr'] = 1e-3*pow(0.5, state['global_step']//100- n // (all_para / 20) ) if state['is_train'] else 0


                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)


                state['step'] += 1.0
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size


                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay']* state['layer_lr'])


                if N_sma > 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data.addcdiv_(exp_avg, denom,value=-step_size * state['layer_lr'])
                else:
                    p_data.add_( exp_avg,alpha=-step_size *state['layer_lr'])

                if any_abnormal_number(p_data):
                    sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n\r'.format(self.__class__.__name__))
                    p_data=where(is_abnormal_number(p_data),p.data,p_data)

                p.data.copy_(p_data)


                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']  # get access to slow param tensor
                    slow_p.add_( p.data - slow_p,alpha=self.alpha)  # (fast weights - slow weights) * alpha
                    if any_abnormal_number(slow_p):
                        sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                        slow_p = where(is_abnormal_number(slow_p), p.data, slow_p)
                    p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor
                n+=1

        return loss



class RangerLars(Optimizer):
    """
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
    """

    def __init__(self, params, lr=1e-3,betas=(0.9, 0.999),alpha=0.5, k=6,N_sma_threshhold=5,eps=1e-7, weight_decay=0,gradient_centralization=None):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {k}')

        if not lr > 0:
            raise ValueError('Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError('Invalid eps: {eps}')
        self.gradient_centralization=gradient_centralization
        self.N_sma_threshhold=N_sma_threshhold
        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k,  betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)
        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]
        self.alpha = alpha
        self.k = k

        # self.first_run_check=0

        # lookahead weights  # 9/2/19 - lookahead param tensors have been moved to state storage.  # This should   #
        # resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        # self.slow_weights = [[p.clone().detach() for p in group['params']]  #                     for group in
        # self.param_groups]

        # don't use grad for lookahead weights  # for w in it.chain(*self.slow_weights):  #    w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(RangerLars, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = None
        if closure is not None:
            loss = closure()

            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None or not p.requires_grad:
                        continue
                    grad = p.grad.data.float()

                    if grad.is_sparse:
                        raise RuntimeError('Ranger optimizer does not support sparse gradients')

                    p_data = p.data.float()

                    state = self.state[p]  # get state dict for this param

                    if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p_data, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p_data, memory_format=torch.preserve_format)

                        # look ahead weight storage now in state dict
                        state['slow_buffer'] = torch.empty_like(p.data)
                        state['slow_buffer'].copy_(p.data)
                    else:
                        state['exp_avg'] = state['exp_avg'].type_as(p_data)
                        state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data)

                    # begin computations
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']
                    if self.gradient_centralization in ['all', 'gcc']:
                        if len(list(grad.size())) > 3:
                            grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # grad=_filter_grads(grad,self.gradient_centralization)

                    state['step'] += 1
                    buffered = self.radam_buffer[int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= self.N_sma_threshhold:
                            step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        else:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        buffered[2] = step_size

                    update = zeros_like(p_data)
                    if N_sma >= 5:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                        update.addcdiv_(exp_avg, denom, value=step_size)
                    else:
                        update.add_(exp_avg, alpha=step_size)

                    # if group['weight_decay'] != 0:
                    #     update.add_(group['weight_decay'], p_data)
                    if group['weight_decay'] != 0:
                        grad = grad.add(p, alpha=group['weight_decay'] )

                    radam_norm = update.pow(2.0).sum().sqrt()
                    weight_norm = p.data.pow(2.0).sum().sqrt()
                    if weight_norm == 0 or radam_norm == 0:
                        trust_ratio = 1.0
                    else:
                        trust_ratio = weight_norm / radam_norm

                    trust_ratio = clip(to_tensor(trust_ratio),0.0,10.0)

                    state['weight_norm'] = weight_norm
                    state['adam_norm'] = radam_norm
                    state['trust_ratio'] = trust_ratio

                    p_data.add_(-update * trust_ratio * group['lr'])

                    if any_abnormal_number(p_data):
                        sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n\r'.format(self.__class__.__name__))
                    p_data.copy_(where(is_abnormal_number(p_data), p.data, p_data))

                    p.data.copy_(p_data)

                    # integrated look ahead...
                    # we do it at the param level instead of group level
                    if state['step'] % group['k'] == 0:
                        slow_p = state['slow_buffer']  # get access to slow param tensor
                        slow_p.add_(p.data - slow_p, alpha=self.alpha)  # (fast weights - slow weights) * alpha
                        if any_abnormal_number(slow_p):
                            sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                        slow_p = where(is_abnormal_number(slow_p), p.data, slow_p)
                        p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor


        return loss



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

    @torch.no_grad()
    def step(self, epoch=None, closure=None):
        """Performs a single optimization step.

        Args:
            epoch (int): current epoch
            closure (callable): call for get loss backward

        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

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
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.ge(0),
                        torch.where(
                            g_norm.ge(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(grad,alpha=scaled_lr)
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

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.999), eps=1e-7, weight_decay=0, amsgrad=False,
                 gradient_centralization=None, **kwargs):
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
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                p_data = p.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    #if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                #if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']


                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual=grad-exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class RangerBelief(Optimizer):
    """
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
    """

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-7,
                 weight_decay=0,gradient_centralization=None):
        # parameter checks
        self.first_run_check=0
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError('Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError('Invalid eps: {eps}')
        self.gradient_centralization=gradient_centralization

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold,
                        eps=eps, weight_decay=weight_decay)
        super(RangerBelief, self).__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # now we can get to work...
        # removed as we now use step from RAdam...no need for duplicate step counting
        for group in self.param_groups:
           group["step_counter"] = 0
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
        super(RangerBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = None

        if closure is not None:
            loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p_data, memory_format=torch.preserve_format)
                    state['previous_grad'] =torch.zeros_like(grad, memory_format=torch.preserve_format)
                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data)

                # begin computations
                exp_avg, exp_avg_sq,previous_grad = state['exp_avg'], state['exp_avg_sq'],state['previous_grad']
                beta1, beta2 = group['betas']
                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))


                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                #grad=_filter_grads(grad,self.gradient_centralization)

                state['step'] += 1
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= self.N_sma_threshhold:
                        step_size = math.sqrt( (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data.add_(p_data,alpha= -group['weight_decay'] * group['lr'])

                # compute diffgrad coefficient (dfc)
                diff = abs(previous_grad - grad)
                dfc = 1. / (1. + torch.exp(-diff))
                state['previous_grad'] = grad

                # update momentum with dfc
                exp_avg1 = exp_avg * dfc

                if N_sma >= self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data.addcdiv_(exp_avg1, denom, value=-step_size * group['lr'] )
                elif step_size > 0:
                    p_data.add_(exp_avg1, alpha=-step_size * group['lr'])

                if any_abnormal_number(p_data):
                    sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                    p_data.copy_(where(is_abnormal_number(p_data),p.data,p_data))

                p.data.copy_(p_data)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']  # get access to slow param tensor
                    slow_p.add_( p.data - slow_p,alpha=self.alpha)  # (fast weights - slow weights) * alpha
                    if any_abnormal_number(slow_p):
                        sys.stderr.write('{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(self.__class__.__name__))
                        slow_p = where(is_abnormal_number(slow_p), p.data, slow_p)
                    p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor


        return loss


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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0,gradient_centralization=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.gradient_centralization=gradient_centralization
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,gradient_centralization=gradient_centralization)
        super(DiffGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DiffGrad, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('diffGrad does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Previous gradient
                    state['previous_grad'] =  torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
                beta1, beta2 = group['betas']
                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # compute diffgrad coefficient (dfc)

                diff = abs(previous_grad - grad)
                dfc = 1. / (1. + torch.exp(-diff))
                state['previous_grad'] = grad  # used in paper but has the bug that previous grad is overwritten with grad and diff becomes always zero. Fixed in the next line.

                # update momentum with dfc
                exp_avg1 = exp_avg * dfc

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.add_(true_divide( exp_avg1, denom),alpha=-step_size)

        return loss




def get_optimizer(optimizer_name):
    if optimizer_name is None:
        return None
    optimizer_modules = ['trident.optims.pytorch_optimizers', 'torch.optim']
    if optimizer_name in __all__:
        optimizer_class = get_class(optimizer_name, optimizer_modules)
        return optimizer_class
    else:
        try:
            optimizer_class = get_class(snake2camel(optimizer_name), optimizer_modules)
            return optimizer_class
        except Exception:
            optimizer_class = get_class(optimizer_name, optimizer_modules)
        return optimizer_class
