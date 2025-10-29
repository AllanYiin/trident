from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import weakref
import math
import re
import gc
import sys
import warnings
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer, lbfgs, adagrad, adadelta, rmsprop, adam
from trident.backend.common import PrintException
from trident.backend.common import get_class, snake2camel
from trident.backend.pytorch_ops import *

__all__ = ['Adam', 'SGD', 'LBFGS', 'Adadelta', 'Adagrad', 'RMSprop', 'RAdam', 'PlainRAdam', 'AdamW', 'Lookahead',
           'Ranger', 'Ranger21', 'RangerLars', 'AdaBelief', 'RangerAdaBelief', 'DiffGrad', 'Lamb', 'Lion',
           'Sophia', 'Adan', 'Muon', 'AdamMini', 'GaLore', 'Apollo',
           'get_optimizer']


def cheb_steps(m, M, T):
    C, R = (M + m) / 2.0, (M - m) / 2.0
    thetas = (np.arange(T) + 0.5) / T * np.pi
    return 1.0 / (C - R * np.cos(thetas))


def cheb_perm(T):
    perm = np.array([0])
    while len(perm) < T:
        perm = np.vstack([perm, 2 * len(perm) - 1 - perm]).T.flatten()
    return perm


def get_chebs(num_epochs):
    num_epochs = num_epochs - 2
    steps = cheb_steps(0.1, 1, num_epochs)
    perm = cheb_perm(num_epochs)
    cheb_schedule = steps[perm]
    print(f"cheb schedule made with len {len(cheb_schedule)}")
    return cheb_schedule


def normalize_gradient(x, use_channels=False, epsilon=1e-8):
    """  use stdev to normalize gradients """
    size = x.dim()
    # print(f"size = {size}")

    if (size > 1) and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True) + epsilon
        # print(f"s = {s}")
        x.div_(s)  # , keepdim=True)

    elif torch.numel(x) > 2:
        s = x.std() + epsilon
        x.div_(s)  # , keepdim=True)
    return x


def centralize_gradient(x, gc_conv_only=False):
    """credit - https://github.com/Yonghongwei/Gradient-Centralization """

    size = x.dim()
    # print(f"size = {size}")

    if gc_conv_only:
        if size > 3:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    else:
        if size > 1:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    return x


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
        if 'lr' in defaults:
            self._base_lr = defaults['lr']
        self.use_adaptive_gradient_clipping = defaults.get('use_adaptive_gradient_clipping', False)
        self.agc_clip_val = 1e-2
        self.agc_eps = 1e-3
        self.gradient_centralization = defaults.get('gradient_centralization', None)
        # cautious optimizer settings
        self.enable_cautious = defaults.get('enable_cautious', False)
        self.cautious_eps = defaults.get('cautious_eps', 1e-8)

    def _apply_cautious(self, update, grad):
        mask = (update * grad > 0).to(grad.dtype)
        return update * mask / (mask.mean() + self.cautious_eps)

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
    def parameters(self, value):
        """

        Returns: the weights need to train

        """
        if isinstance(value, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(value))
        if not hasattr(self, 'param_groups') or self.param_groups is None or len(self.param_groups) == 0:
            self.param_groups = []

            param_groups = list(value)
            if len(param_groups) == 0:
                raise ValueError("optimizer got an empty parameter list")
            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]
                for param_group in param_groups:
                    self.add_param_group(param_group)
        else:
            self.param_groups[0]['params'] = value

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

    def unit_norm(self, x):
        """ axis-based Euclidean norm"""
        # verify shape
        keepdim = True
        dim = None

        xlen = len(x.shape)
        # print(f"xlen = {xlen}")

        if xlen <= 1:
            keepdim = False
        elif xlen in (2, 3):  # linear layers
            dim = 1
        elif xlen == 4:  # conv kernels
            dim = (1, 2, 3)
        else:
            dim = tuple(
                [x for x in range(1, xlen)]
            )  # create 1,..., xlen-1 tuple, while avoiding last dim ...

        return x.norm(dim=dim, keepdim=keepdim, p=2.0)

    def agc(self, p):
        """clip gradient values in excess of the unitwise norm.
        the hardcoded 1e-6 is simple stop from div by zero and no relation to standard optimizer eps
        """

        # params = [p for p in parameters if p.grad is not None]
        # if not params:
        #    return

        # for p in params:
        p_norm = self.unit_norm(p).clamp_(self.agc_eps)
        g_norm = self.unit_norm(p.grad)

        max_norm = p_norm * self.agc_clip_val

        clipped_grad = p.grad * (max_norm / g_norm.clamp(min=1e-6))

        new_grads = torch.where(g_norm > max_norm, clipped_grad, p.grad)
        p.grad.detach().copy_(new_grads)


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
                 gradient_centralization=None, enable_cautious=False, **kwargs):
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
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        enable_cautious=enable_cautious)

        super(Adam, self).__init__(params, defaults)
        self.gradient_centralization = gradient_centralization

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

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                state = self.state[p]
                half_precision = p.data.dtype == torch.float16
                if half_precision:
                    if 'fp32_param' not in state:
                        state['fp32_param'] = p.data.float().clone()
                    p_fp32 = state['fp32_param']
                    grad = p.grad.data.float()
                else:
                    p_fp32 = p.data
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                if any_abnormal_number(grad):
                    grad = where(is_abnormal_number(grad), zeros_like(grad), grad)
                amsgrad = group['amsgrad']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p_fp32, alpha=group['weight_decay'])

                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, len(list(grad.size())))), keepdim=True))

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
                if self.enable_cautious:
                    G_grad = self._apply_cautious(G_grad, grad)
                # if self.gradient_centralization in ['all', 'gc']:
                #     if ndim(G_grad) > 1:
                #         G_grad.add_(-G_grad.mean(axis=list(range(1, ndim(G_grad))), keepdims=True))

                p_fp32.add_(G_grad, alpha=-step_size)
                if half_precision:
                    p.data.copy_(p_fp32.half())
                    p.grad = p.grad.half()
                else:
                    p.data.copy_(p_fp32)
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
                 weight_decay=0, nesterov=False, **kwargs):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                         nesterov=nesterov)

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
        return self.param_groups[0]['params']

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
                 line_search_fn=None, **kwargs):
        super().__init__(params, lr=lr, max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad,
                         tolerance_change=tolerance_change, history_size=history_size,
                         line_search_fn=line_search_fn)

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
        return self.param_groups[0]['params']

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

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-7, weight_decay=0, **kwargs):
        super().__init__(params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

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
        return self.param_groups[0]['params']

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

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-7, **kwargs):
        super().__init__(params, lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                         initial_accumulator_value=initial_accumulator_value)

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
        return self.param_groups[0]['params']

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

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-7, weight_decay=0, momentum=0, centered=False,
                 gradient_centralization=None, **kwargs):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                         centered=centered)

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
        return self.param_groups[0]['params']

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
                 degenerated_to_sgd=True, gradient_centralization=None, enable_cautious=False, **kwargs):
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
                        buffer=[[None, None, None] for _ in range(10)],
                        enable_cautious=enable_cautious)
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
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data.float() if p.grad.data.dtype != torch.float32 else p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data = p.data.float() if p.data.dtype != torch.float32 else p.data

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

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
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
                    update = exp_avg / denom
                elif step_size > 0:
                    update = exp_avg
                if N_sma >= self.N_sma_threshhold or step_size > 0:
                    if self.enable_cautious:
                        update = self._apply_cautious(update, grad)
                    p_data.add_(update, alpha=-step_size * group['lr'])
                p.data.copy_(p_data)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss


class PlainRAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True,
                 gradient_centralization=None, enable_cautious=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        enable_cautious=enable_cautious)

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
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data.float() if p.grad.data.dtype != torch.float32 else p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data = p.data.float() if p.data.dtype != torch.float32 else p.data

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

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
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
                    update = grad / denom
                    if self.enable_cautious:
                        update = self._apply_cautious(update, grad)
                    p_data.add_(update, alpha=-step_size)
                    p.data.copy_(p_data)
                elif self.degenerated_to_sgd:
                    if group['weight_decay'] != 0:
                        p_data.add_(p_data, alpha=-group['weight_decay'] * group['lr'])
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    update = grad
                    if self.enable_cautious:
                        update = self._apply_cautious(update, grad)
                    p_data.add_(update, alpha=-step_size)
                    p.data.copy_(p_data)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()
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

    Args:
        enable_cautious (bool): If True, applies cautious masking to momentum
            updates as proposed in C-AdamW.

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0, warmup=0,
                 gradient_centralization=None, enable_cautious=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self._base_lr = lr
        self.gradient_centralization = gradient_centralization
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup,
                        enable_cautious=enable_cautious)
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
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data.float() if p.grad.data.dtype != torch.float32 else p.grad.data
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

                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, len(list(grad.size())))), keepdim=True))

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                scheduled_lr = group['lr']
                if group['warmup'] > state['step'] + 20:
                    scheduled_lr = 1e-8
                    self.adjust_learning_rate(scheduled_lr, verbose=False)
                    # scheduled_lr = group['lr']
                elif group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 * pow(pow(self.base_lr / 1e-8, 1 / 20), (group['warmup'] - state['step']))
                    self.adjust_learning_rate(scheduled_lr, verbose=False)
                elif group['warmup'] == state['step']:
                    self.adjust_learning_rate(self.base_lr, verbose=False)
                    scheduled_lr = self.base_lr
                else:
                    pass  # scheduled_lr =  group['lr']

                step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_data.add_(p_data, alpha=-group['weight_decay'] * scheduled_lr)

                G_grad = grad / denom
                if self.gradient_centralization in ['all', 'gc']:
                    if ndim(G_grad) > 1:
                        G_grad.add_(-G_grad.mean(axis=list(range(1, ndim(G_grad))), keepdims=True))
                if self.enable_cautious:
                    mask = (G_grad * grad > 0).to(grad.dtype)
                    p_data.add_(G_grad * mask / (mask.mean() + self.cautious_eps), alpha=-step_size)
                else:
                    p_data.add_(G_grad, alpha=-step_size)

                p.data.copy_(p_data)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss


class Lookahead(Optimizer):
    def __init__(self, optimizer, params, defaults, k=5, alpha=0.5, gradient_centralization=None, **kwargs):
        super().__init__(params, defaults)
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
        self.gradient_centralization = gradient_centralization
        self.optimizer.gradient_centralization = gradient_centralization

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
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()
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

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5,
                 weight_decay=0, gradient_centralization=None, use_adaptive_gradient_clipping=False,
                 enable_cautious=False, **kwargs):
        self.gradient_centralization = gradient_centralization
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
                        weight_decay=weight_decay, gradient_centralization=gradient_centralization,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        enable_cautious=enable_cautious)
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
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()
                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                # p_data = p.data
                p_data_fp32 = p.data.float() if p.data.dtype != torch.float32 else p.data

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    state['step'] = 0.0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p_data_fp32)
                    state['slow_buffer'].copy_(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                if group['weight_decay'] != 0:
                    grad.add_(p_data_fp32, alpha=group['weight_decay'])

                # if self.gradient_centralization in ['all', 'gcc']:
                #     if ndim(grad) > 3:
                #         grad.add_(-grad.mean(axis=tuple(range(1, ndim(grad) )), keepdims=True))

                state['step'] += 1.0

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                                N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if self.gradient_centralization in ['all', 'gc', 'gcc']:
                    if ndim(G_grad) > 1:
                        G_grad.add_(-G_grad.mean(axis=list(range(1, ndim(G_grad))), keepdims=True))

                if self.enable_cautious:
                    G_grad = self._apply_cautious(G_grad, grad)
                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])
                if any_abnormal_number(p_data_fp32):
                    sys.stderr.write(
                        '{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n\r'.format(
                            self.__class__.__name__))
                    fallback = p.data.float() if p.data.dtype != torch.float32 else p.data
                    p_data_fp32 = where(is_abnormal_number(p_data_fp32), fallback, p_data_fp32)

                p.data.copy_(p_data_fp32)

                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    if any_abnormal_number(slow_p):
                        sys.stderr.write(
                            '{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(
                                self.__class__.__name__))
                        fallback = p.data.float() if p.data.dtype != torch.float32 else p.data
                        slow_p = where(is_abnormal_number(slow_p), fallback, slow_p)
                    p.data.copy_(slow_p)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss


class Ranger21(Optimizer):
    """
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
    """

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.9, 0.999), eps=1e-8,
                 momentum_type="pnm", pnm_momentum_factor=1.0, momentum=0.9,
                 weight_decay=1e-4, gradient_centralization="all",
                 using_normgc=True,
                 use_madgrad=True,
                 use_adabelief=True,
                 softplus=True,
                 beta_softplus=50,
                 use_adaptive_gradient_clipping=True,
                 agc_clipping_value=1e-2,
                 agc_eps=1e-3,

                 normloss_active=True,
                 normloss_factor=1e-4, lookahead_active=True,
                 lookahead_mergetime=5,
                 lookahead_blending_alpha=0.5,
                 lookahead_load_at_validation=False,
                 use_cheb=False,
                 use_warmup=False,
                 num_warmup_iterations=None,
                 warmdown_active=True,
                 warmdown_start_pct=0.72,
                 warmdown_min_lr=3e-5,
                 decay_type="stable",
                 warmup_type="linear",
                 warmup_pct_default=0.22,
                 enable_cautious=False, **kwargs):
        self._base_lr = lr
        # momentum
        self.momentum_pnm = momentum_type == "pnm"
        self.pnm_momentum = pnm_momentum_factor
        self.gradient_centralization = gradient_centralization

        # decay
        self.decay = weight_decay
        self.decay_type = decay_type
        self.param_size = 0

        self.use_madgrad = use_madgrad
        if not self.use_madgrad:
            self.core_engine = "AdamW"
        else:
            self.core_engine = "madgrad"
        self.use_adabelief = use_adabelief
        # eps
        self.eps = eps

        # softplus for denom
        self.softplus = softplus
        self.beta_softplus = beta_softplus
        # norm loss
        self.normloss_active = normloss_active
        self.normloss_factor = normloss_factor
        self.gradient_centralization = gradient_centralization
        self.use_gc = True if gradient_centralization is not None else False
        self.use_gcnorm = using_normgc
        self.gc_conv_only = True if gradient_centralization == 'gcc' else False

        # lookahead
        self.lookahead_active = lookahead_active
        self.lookahead_mergetime = lookahead_mergetime
        self.lookahead_step = 0
        self.lookahead_alpha = lookahead_blending_alpha
        self.lookahead_validation_load = lookahead_load_at_validation

        # agc
        self.agc_active = use_adaptive_gradient_clipping
        self.agc_clip_val = agc_clipping_value
        self.agc_eps = agc_eps
        self.total_iterations = 5000
        # chebs
        self.use_cheb = use_cheb
        self.cheb_schedule = None
        if self.use_cheb:
            if self.total_iterations is None:
                raise ValueError(
                    "can't produce chebs without num epochs info being passed in"
                )
            self.cheb_schedule = get_chebs(self.total_iterations // 100)

        # self.total_iterations = num_epochs * num_batches_per_epoch
        # if not self.total_iterations:
        #     raise ValueError(
        #         "missing total iterations, which is calced from num epochs and num iters per epoch param"
        #     )

        # lr

        # warmup - we'll use default recommended in Ma/Yarats unless user specifies num iterations
        # -=-=-=-=-=-=-=-=-=-=-=-=-=--=-=--=-=-
        self.use_warmup = use_warmup
        self.warmup_complete = False
        self.warmup_type = warmup_type
        self.warmup_pct_default = warmup_pct_default

        if use_warmup == True and num_warmup_iterations is None:
            beta_warmup_iters = math.ceil(
                (2 / (1 - betas[1]))
            )  # default untuned linear warmup

            beta_pct = beta_warmup_iters / self.total_iterations
            # print(f"beta_warmup_pct = {beta_pct}")

            # this can be unreasonable for short runs...so let's compare vs warmup pct % of total epochs
            if beta_pct > 0.45:
                warmup_auto_pct = int(self.warmup_pct_default * self.total_iterations)
                self.num_warmup_iters = warmup_auto_pct
            else:
                self.num_warmup_iters = beta_warmup_iters

        else:  # user passed in specific num
            self.num_warmup_iters = num_warmup_iterations if num_warmup_iterations is not None else 0
            self.use_warmup = True if num_warmup_iterations is not None else False

        # warm down
        self.min_lr = warmdown_min_lr
        self.warmdown_active = warmdown_active
        self.warmup_curr_pct = 0.01  # used to verify warmup reaches full set point.

        if self.warmdown_active:
            if self.min_lr is None:
                self.min_lr = 0.0
            if self.min_lr > self._base_lr:
                warnings.warn(
                    (
                        "warmdown_min_lr ({:.3e}) is greater than the base learning rate ({:.3e}); "
                        "clamping warmdown minimum to the base learning rate to avoid increases during warmdown."
                    ).format(self.min_lr, self._base_lr),
                    RuntimeWarning,
                )
                self.min_lr = self._base_lr

            self.warmdown_lr_delta = self._base_lr - self.min_lr

            if self.warmdown_lr_delta <= 0:
                self.warmdown_active = False
            else:
                self.warm_down_start_pct = warmdown_start_pct
                self.start_warm_down = int(
                    self.warm_down_start_pct * self.total_iterations
                )
                self.warmdown_total_iterations = (
                    self.total_iterations - self.start_warm_down
                )
                self.warmdown_displayed = False  # print when warmdown begins...
        if not self.warmdown_active:
            self.warmdown_lr_delta = max(0.0, self._base_lr - (self.min_lr or 0.0))
            self.start_warm_down = self.total_iterations
            self.warmdown_total_iterations = 0
            self.warmdown_displayed = False

            """
            print(f"debug warmdown:\n")
            print(f"warm_down_start_pct = {self.warm_down_start_pct}")
            print(f"num_epochs = {self.num_epochs}, num_batches per epoch = {self.num_batches_per_epoch}")
            print(f" start warmdown at {self.start_warm_down}")
            print(f" total iterations of warmdown = {self.warmdown_total_iterations}")
            print(f" total lr delta = {self.warmdown_lr_delta}")
            """

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
        defaults = dict(
            lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay,
            gradient_centralization=gradient_centralization,
            enable_cautious=enable_cautious
        )

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
        super(Ranger21, self).__setstate__(state)


    def warmup_dampening(self, lr, step):

        style = self.warmup_type
        warmup = self.num_warmup_iters

        if style is None:
            style = "linear"

        if step > warmup:
            if not self.warmup_complete:

                if not self.warmup_curr_pct == 1.0:
                    print(
                        f"Error - lr did not achieve full set point from warmup, currently {self.warmup_curr_pct}"
                    )

                self.warmup_complete = True
                self.adjust_learning_rate(self.base_lr, verbose=False)
                print(f"\n** Ranger21 update = Warmup complete - lr set to {lr}\n")
            return self.base_lr

        if style == "linear":
            if step < warmup // 2:
                return 1e-8
            else:
                self.warmup_curr_pct = min(1.0, (step / warmup))
                new_lr = 1e-8 * math.pow(math.pow(self.base_lr / 1e-8, 1 / (warmup // 2)), step - warmup // 2)
            return new_lr

        # elif style == "exponential":
        # return lr * (1.0 - math.exp(-step / warmup))

        else:
            raise ValueError(f"warmup type {style} not implemented.")

    def get_warm_down(self, lr, iteration):
        """ linear style warmdown """
        if iteration < self.start_warm_down:
            return lr

        if iteration > self.start_warm_down - 1:
            # print when starting
            if not self.warmdown_displayed:
                print(
                    f"\n** Ranger21 update: Warmdown starting now.  Current iteration = {iteration}....\n"
                )
                self.warmdown_displayed = True

            warmdown_iteration = (
                                         iteration + 1
                                 ) - self.start_warm_down  # to force the first iteration to be 1 instead of 0

            if warmdown_iteration < 1:
                print(
                    f" warning - iteration started at {iteration} and {self.start_warm_down} with value {warmdown_iteration}"
                )
                warmdown_iteration = 1
            # print(f"warmdown iteration = {warmdown_iteration}")
            # linear start 3672  5650 total iterations 1972 iterations

            warmdown_pct = warmdown_iteration / (
                    self.warmdown_total_iterations + 1
            )  # +1 to offset that we have to include first as an iteration to support 1 index instead of 0 based.
            if warmdown_pct > 1.00:
                print(f"error in warmdown pct calc.  new pct = {warmdown_pct}")
                print(f"auto handled but please report issue")
                warmdown_pct = 1.00

            # .5
            lr_range = self.warmdown_lr_delta

            reduction = lr_range * warmdown_pct
            # print(f"lr reduction = {reduction} for {warmdown_pct} with iter {warmdown_iteration} and total iter {iteration}")
            new_lr = self.base_lr - reduction
            if new_lr < self.min_lr:
                print(f"error in warmdown - lr below min lr. current lr = {new_lr}")
                print(f"auto handling but please report issue!")
                new_lr = self.min_lr

            self.lr = new_lr
            self.adjust_learning_rate(new_lr=new_lr, verbose=False)
            return new_lr

            # new_lr = (
            #    self.min_lr
            #    + self.starting_lr
            #    * (1 + math.cos(math.pi * warmdown_iteration / self.warmdown_total_iterations))
            #    / 2
            # )
            # self.current_lr = new_lr
            # return new_lr

    #   Lookahead merge process
    def lookahead_process_step(self):
        """handles blending of params for lookahead step"""

        if not self.lookahead_active:
            return
        self.lookahead_step += 1

        if self.lookahead_step >= self.lookahead_mergetime:
            self.lookahead_step = 0
            # merge lookahead cached params and save current ones
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]

                    p.data.mul_(self.lookahead_alpha).add_(
                        param_state["lookahead_params"],
                        alpha=1.0 - self.lookahead_alpha,
                    )
                    # save for next merge
                    param_state["lookahead_params"].copy_(p.data)

    # def new_epoch_handler(self, iteration):

    #    self.epoch_count +=1

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        global variance_ma_belief
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size = 1
        variance_ma_sum = 0.0
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                if p.grad is None or not p.requires_grad:
                    continue
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                param_size += p.numel()
                # apply agc if enabled
                if self.agc_active:
                    self.agc(p)

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                # p_data = p.data
                p_data_fp32 = p.data.float() if p.data.dtype != torch.float32 else p.data

                state = self.state[p]  # get state dict for this param
                momentum = group["momentum"]

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries

                    state["step"] = 0

                    # Exponential moving average of gradient values
                    state["grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Exponential moving average of squared gradient values
                    state["variance_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if self.lookahead_active:
                        state["lookahead_params"] = torch.zeros_like(p.data)
                        state["lookahead_params"].copy_(p.data)

                    if self.use_adabelief:
                        state["variance_ma_belief"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if self.momentum_pnm:
                        state["neg_grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_variance_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # centralize gradients
                if self.use_gc:
                    grad = centralize_gradient(grad, gc_conv_only=self.gc_conv_only, )

                if self.use_gcnorm:
                    grad = normalize_gradient(grad)
                # else:
                #    grad = uncentralized_grad

                state["step"] += 1

                step = state["step"]
                lr = group["lr"]

                beta1, beta2 = group["betas"]
                grad_ma = state["grad_ma"]

                bias_correction2 = 1 - beta2 ** state["step"]
                # print(f"bias2 = {bias_correction2}")

                variance_ma = state["variance_ma"]
                if self.use_adabelief:
                    variance_ma_belief = state["variance_ma_belief"]

                # print(f"variance_ma, upper loop = {variance_ma}")

                # update the exp averages
                if self.use_adabelief:
                    grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)
                    grad_residual = grad - grad_ma
                    variance_ma_belief.mul_(beta2).addcmul(grad_residual, grad_residual, value=1 - beta2)

                # print(f"upper loop grad = {grad.shape}")
                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # print(f"variance_ma, grad adjusted")
                variance_ma_debiased = variance_ma / bias_correction2

                variance_ma_sum += variance_ma_debiased.sum()
                # print(f"variance_ma_sum = {variance_ma_sum}")
                # else: #madgrad

        # if not self.param_size:
        #     self.param_size = param_size
        #     print(f"params size saved")
        #     print(f"total param groups = {i + 1}")
        #     print(f"total params in groups = {j + 1}")
        #
        # if not self.param_size:
        #     raise ValueError("failed to set param size")

        # stable weight decay
        if self.use_madgrad:
            variance_normalized = torch.pow(variance_ma_sum / param_size, 1 / 3)
        else:
            variance_normalized = math.sqrt(variance_ma_sum / param_size)
            # variance_mean = variance_ma_sum / param_size

        if math.isnan(variance_normalized):
            print(variance_normalized)
            raise RuntimeError("hit nan for variance_normalized")

        # print(f"variance_mean = {variance_mean}")
        # print(f"variance_normalized = {variance_normalized}")
        # else:
        #    variance_normalized = math.pow((variance_ma / self.param_size), .3333)

        # print(f"variance mean sqrt = {variance_normalized}")

        # phase 2 - apply weight decay and step
        # ===========================================
        for group in self.param_groups:
            # print(f"In second phase loop")

            step = state["step"]

            # Perform stable weight decay
            decay = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]
            momentum = group["momentum"]

            beta1, beta2 = group["betas"]

            # warmup
            # ======================
            if self.use_warmup and not self.warmup_complete:
                lr = self.warmup_dampening(lr, step)
                self.adjust_learning_rate(new_lr=lr, verbose=False)
                # print(f"lr = {lr}")

            # chebyshev
            # ===================
            if self.use_cheb and self.warmup_complete:
                lr = self.get_cheb_lr(lr, step)
                self.adjust_learning_rate(new_lr=lr, verbose=False)

            # warmdown
            # ==========
            if self.warmdown_active:
                orig_lr = lr
                lr = self.get_warm_down(lr, step)
                self.adjust_learning_rate(new_lr=lr, verbose=False)
                assert lr > 0, "lr went negative"

            # madgrad outer
            if self.use_madgrad:
                ck = 1 - momentum
                lamb = lr * math.pow(step, 0.5)

            # stable decay and / or norm loss
            # ==================================
            if decay:
                if not self.use_madgrad:
                    # stable weight decay
                    p.data.mul_(1 - decay * lr / variance_normalized)
                else:
                    p.data.mul_(1 - decay * lamb / variance_normalized)

            if self.normloss_active:
                # apply norm loss
                unorm = self.unit_norm(p.data)
                correction = (2 * self.normloss_factor * (1 - torch.div(1, unorm + self.eps)))
                p.mul_(1 - lr * correction)

            # innner loop, params
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                inner_grad = p.grad

                if self.use_madgrad:
                    # ================== madgrad ============================
                    if "grad_sum_sq" not in state:
                        state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                        state["s"] = torch.zeros_like(p.data).detach()
                        if momentum != 0:
                            state["x0"] = torch.clone(p.data).detach()

                    if momentum != 0.0 and grad.is_sparse:
                        raise RuntimeError(
                            "momentum != 0 is not compatible with sparse gradients"
                        )

                    # centralize gradients
                    if self.use_gc:
                        inner_grad = centralize_gradient(
                            inner_grad,
                            gc_conv_only=self.gc_conv_only,
                        )

                    grad_sum_sq = state["grad_sum_sq"]
                    s = state["s"]
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3)
                        if self.softplus:
                            rms = F.softplus(rms, beta=self.beta_softplus)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments

                    # print(f" grad = {grad}")
                    # print(f"lamb = {lamb}")
                    # print(f"gsumsq = {grad_sum_sq}")

                    grad_sum_sq.addcmul_(inner_grad, inner_grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3)
                    if self.softplus:
                        rms = F.softplus(rms, beta=self.beta_softplus)

                    # Update s
                    s.data.add_(inner_grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)
                        ck = 1 - momentum
                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

                else:  # adam with pnm core
                    # ============= adamW with pnm option ========================

                    grad = p.grad

                    beta1, beta2 = group["betas"]

                    grad_ma = state["grad_ma"]
                    variance_ma = state["variance_ma"]
                    if self.use_adabelief:
                        variance_ma_belief = state["variance_ma_belief"]

                    if self.momentum_pnm:
                        max_variance_ma = state["max_variance_ma"]

                        if state["step"] % 2 == 1:
                            grad_ma, neg_grad_ma = (
                                state["grad_ma"],
                                state["neg_grad_ma"],
                            )
                        else:
                            grad_ma, neg_grad_ma = (
                                state["neg_grad_ma"],
                                state["grad_ma"],
                            )

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    if self.momentum_pnm:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_variance_ma, variance_ma, out=variance_ma)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (variance_ma.sqrt() / math.sqrt(bias_correction2)).add_(
                            group["eps"]
                        )

                    # centralize gradients
                    if self.use_gc: grad = centralize_gradient(grad, gc_conv_only=self.gc_conv_only, )
                    if self.use_gcnorm:
                        grad = normalize_gradient(grad)

                    if not self.use_adabelief:
                        grad_ma.mul_(beta1 ** 2).add_(grad, alpha=1 - beta1 ** 2)

                    noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)

                    step_size = lr / bias_correction1

                    # softplus the denom
                    if self.softplus:
                        denom = F.softplus(denom, beta=self.beta_softplus)

                    pnmomentum = (
                        grad_ma.mul(1 + self.momentum_pnm)
                        .add(neg_grad_ma, alpha=-self.momentum_pnm)
                        .mul(1 / noise_norm)
                    )

                    update = pnmomentum / denom
                    if self.enable_cautious:
                        update = self._apply_cautious(update, grad)
                    p.add_(update, alpha=-step_size)

                    # denom = variance_biased_ma.sqrt().add(eps)

                    # step_size = lr / bias_correction1

                    # update weights
                    # p.data.add_(weight_mod, alpha=-step_size)
                    # p.addcdiv_(grad_ma, denom, value=-step_size)
        # print(f"\n End optimizer step\n")

        # end of step processes....

        # lookahead
        # ---------------------
        if self.lookahead_active:
            self.lookahead_process_step()

        return loss


class RangerLars(Optimizer):
    """
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), alpha=0.5, k=6, N_sma_threshhold=5, eps=1e-7,
                 weight_decay=0, gradient_centralization=None, use_adaptive_gradient_clipping=False,
                 enable_cautious=False, **kwargs):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: {k}')

        if not lr > 0:
            raise ValueError('Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError('Invalid eps: {eps}')
        self.gradient_centralization = gradient_centralization
        self.N_sma_threshhold = N_sma_threshhold
        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, betas=betas, eps=eps, weight_decay=weight_decay,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        enable_cautious=enable_cautious)

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
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()
                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
                grad = p.grad.data.float() if p.grad.data.dtype != torch.float32 else p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data = p.data.float() if p.data.dtype != torch.float32 else p.data

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
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    G_grad = (exp_avg / denom) * step_size
                else:
                    G_grad = exp_avg * step_size
                update = G_grad

                # if group['weight_decay'] != 0:
                #     update.add_(group['weight_decay'], p_data)
                if group['weight_decay'] != 0:
                    G_grad = G_grad.add(p, alpha=group['weight_decay'])

                if self.gradient_centralization in ['all', 'gc']:
                    if ndim(G_grad) > 1:
                        G_grad = G_grad - G_grad.mean(axis=list(range(1, ndim(G_grad))), keepdims=True)

                if self.enable_cautious:
                    update = self._apply_cautious(update, grad)
                radam_norm = update.pow(2.0).sum().sqrt()
                weight_norm = p.data.pow(2.0).sum().sqrt()
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / radam_norm

                trust_ratio = clip(to_tensor(trust_ratio), 0.0, 10.0)

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                p_data.add_(-update * trust_ratio * group['lr'])

                if any_abnormal_number(p_data):
                    sys.stderr.write(
                        '{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n\r'.format(
                            self.__class__.__name__))
                p_data.copy_(where(is_abnormal_number(p_data), p.data, p_data))

                p.data.copy_(p_data)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']  # get access to slow param tensor
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)  # (fast weights - slow weights) * alpha
                    if any_abnormal_number(slow_p):
                        sys.stderr.write(
                            '{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n'.format(
                                self.__class__.__name__))
                    slow_p = where(is_abnormal_number(slow_p), p.data, slow_p)
                    p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()
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
            eeta=0.001, use_adaptive_gradient_clipping=False,
            enable_cautious=False, **kwargs):
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
            use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
            eeta=eeta,
            enable_cautious=enable_cautious,
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
            with torch.enable_grad():
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

                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                param = p.data
                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
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

                    if w_norm.item() > 0 and g_norm.item() > 0:
                        trust_ratio = (self.eeta * w_norm / g_norm).item()
                    else:
                        trust_ratio = 1.0

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(grad, alpha=scaled_lr)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v
                    if self.enable_cautious:
                        update = self._apply_cautious(update, grad)
                    p.data.add_(-update)
                    if half_precision:
                        p.data = p.data.half()
                        p.grad = p.grad.half()
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
    """Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high

    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7,
                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=True,
                 degenerated_to_sgd=True, use_adaptive_gradient_clipping=False, gradient_centralization=None,
                 enable_cautious=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, buffer=[[None, None, None] for _ in range(10)],
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        enable_cautious=enable_cautious)
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

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']
                version_higher = (torch.__version__ >= "1.5.0")
                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        version_higher = (torch.__version__ >= "1.5.0")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                p_data_fp32 = p.data.float() if p.data.dtype != torch.float32 else p.data
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data,
                                                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                        p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data,
                                                           memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                        p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data,
                                                                   memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                            p.data)

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p_data_fp32.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p_data_fp32.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p_data_fp32, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # if self.gradient_centralization in ['all', 'gcc']:
                #     if len(list(grad.size())) > 3:
                #         grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                #

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq.add_(group['eps']), out=max_exp_avg_sq)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # update
                if not self.rectify:
                    # Default update
                    step_size = 1 / bias_correction1
                    G_grad = exp_avg / denom
                

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

                    if N_sma > 5:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                        G_grad = exp_avg / denom
                    else:
                        G_grad = exp_avg

                #
                # if self.gradient_centralization is not None:
                #     if ndim(G_grad) > 1:
                #         G_grad = G_grad - G_grad.mean(axis=list(range(1, ndim(G_grad))), keepdims=True)

                if self.enable_cautious:
                    G_grad = self._apply_cautious(G_grad, grad)
                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])
                if any_abnormal_number(p_data_fp32):
                    sys.stderr.write(
                        '{0} p_data has abnormal value,trident automatically replace these abnormal value to zero.\n\r'.format(
                            self.__class__.__name__))
                    fallback = p.data.float() if p.data.dtype != torch.float32 else p.data
                    p_data_fp32 = where(is_abnormal_number(p_data_fp32), fallback, p_data_fp32)

                p.data.copy_(p_data_fp32)
                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()
        return loss


class RangerAdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch

    https://github.com/juntang-zhuang/Adabelief-Optimizer
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high
        print_change_log (boolean, optional) (default: True) If set as True, print the modifcation to
            default hyper-parameters
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020
    """

    def __init__(self, params, lr=1e-3,  # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 adabelief=True, weight_decouple=True, use_adaptive_gradient_clipping=False,
                 gradient_centralization=None, enable_cautious=False, **kwargs):

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        enable_cautious=enable_cautious)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.gradient_centralization = gradient_centralization
        # level of gradient centralization
        # self.gc_gradient_threshold = 3 if gc_conv_only else 1

        # Turn on AdaBelief or Not
        self.adabelief = adabelief

        # Turn on decoupled weight decay or not
        self.weight_decouple = weight_decouple

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_conv_only == False):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_conv_only == True):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(RangerAdaBelief, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        version_higher = (torch.__version__ >= "1.5.0")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
                grad = p.grad.data.float() if p.grad.data.dtype != torch.float32 else p.grad.data

                if not self.weight_decouple:  # if not decoupled weight decay, add weight decay to grad
                    grad.add_(p.data * group['weight_decay'])

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float() if p.data.dtype != torch.float32 else p.data

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                # if grad.dim() > self.gc_gradient_threshold:
                #    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                if self.gradient_centralization in ['all', 'gcc']:
                    if ndim(grad) > 3:
                        grad = grad - grad.mean(axis=list(range(1, ndim(grad))), keepdims=True)

                state['step'] += 1

                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # compute variance mov avg
                if self.adabelief:
                    exp_avg_sq.mul_(beta2).addcmul_(grad - exp_avg, grad - exp_avg, value=1 - beta2)
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                            state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                                N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # if group['weight_decay'] != 0:
                #    p_data_fp32.add_(-group['weight_decay']
                #                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    if self.adabelief:
                        denom = exp_avg_sq.add_(group['eps']).sqrt().add_(group['eps'])
                    else:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])

                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if self.weight_decouple and (group['weight_decay'] != 0):  # decoupled weight decay
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])

                # GC operation

                if self.gradient_centralization in ['all', 'gc']:
                    if ndim(G_grad) > 1:
                        G_grad = G_grad - G_grad.mean(axis=list(range(1, ndim(G_grad))), keepdims=True)

                if self.enable_cautious:
                    G_grad = self._apply_cautious(G_grad, grad)
                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)
                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()
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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0,
                 use_adaptive_gradient_clipping=False, gradient_centralization=None, enable_cautious=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.gradient_centralization = gradient_centralization
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        gradient_centralization=gradient_centralization,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        enable_cautious=enable_cautious)
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
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()
                p_data_fp32 = p.data.float() if p.data.dtype != torch.float32 else p.data

                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('diffGrad does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    # Previous gradient
                    state['previous_grad'] = torch.zeros_like(p_data_fp32)

                exp_avg, exp_avg_sq, previous_grad = state['exp_avg'], state['exp_avg_sq'], state['previous_grad']
                beta1, beta2 = group['betas']



                if self.gradient_centralization in ['all', 'gcc']:
                    if ndim(grad) > 3:
                        grad = grad - grad.mean(axis=list(range(1, ndim(grad))), keepdims=True)

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # compute diffgrad coefficient (dfc)
                diff = abs(previous_grad - grad)
                dfc = 1. / (1. + torch.exp(-diff))
                # state['previous_grad'] = grad %used in paper but has the bug that previous grad is overwritten with grad and diff becomes always zero. Fixed in the next line.
                state['previous_grad'] = grad.clone()

                # update momentum with dfc
                exp_avg1 = exp_avg * dfc
                G_grad = true_divide(exp_avg1, denom)

                if group['weight_decay'] != 0:
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])

                if self.gradient_centralization is not None:
                    if ndim(G_grad) > 1:
                        G_grad = G_grad - G_grad.mean(axis=list(range(1, ndim(G_grad))), keepdims=True)

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if self.enable_cautious:
                    G_grad = self._apply_cautious(G_grad, grad)
                p.data.add_(G_grad, alpha=-step_size)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()
        return loss


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False,
                 gradient_centralization=None, enable_cautious=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.gradient_centralization = gradient_centralization
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        gradient_centralization=gradient_centralization,
                        enable_cautious=enable_cautious)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

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
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                if self.gradient_centralization in ['all', 'gcc']:
                    if ndim(grad) > 3:
                        grad = grad - grad.mean(axis=list(range(1, ndim(grad))), keepdims=True)

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr']  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                update = adam_step
                if self.enable_cautious:
                    update = self._apply_cautious(update, grad)
                p.data.add_(update, alpha=-step_size * trust_ratio)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()
        return loss


class Lion(Optimizer):
    r"""Implements Lion, EvoLved Sign Momentum
     new optimizer discovered by Google Brain that is purportedly better than Adam(w).

    Args:
        enable_cautious (bool): If True, applies cautious masking to momentum
            updates similar to C-Lion.

    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-7, weight_decay=0,
                 use_adaptive_gradient_clipping=False, gradient_centralization=None,
                 enable_cautious=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        self.gradient_centralization = gradient_centralization
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        gradient_centralization=gradient_centralization,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        enable_cautious=enable_cautious)
        super(Lion, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Lion, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): call for get loss backward

        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
                grad = p.grad.data.float() if p.grad.data.dtype != torch.float32 else p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data = p.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if self.gradient_centralization in ['all', 'gcc']:
                    if len(list(grad.size())) > 3:
                        grad.add_(-grad.mean(dim=tuple(range(1, len(list(grad.size())))), keepdim=True))

                if group['weight_decay'] != 0:
                    grad.add_(p_data, alpha=group['weight_decay'])

                # weight update

                update = exp_avg.clone().lerp_(grad, 1 - beta1).sign_()
                if self.enable_cautious:
                    mask = (update * grad > 0).to(grad.dtype)
                    p.add_(update * mask / (mask.mean() + self.cautious_eps), alpha=- group['lr'])
                else:
                    p.add_(update, alpha=- group['lr'])

                # decay the momentum running average coefficient

                exp_avg.lerp_(grad, 1 - beta2)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss


class Sophia(Optimizer):
    """Implements Sophia optimizer for large language model pre-training.

    中文：Sophia 透過週期性地估計 Hessian 對角線並在更新時進行裁剪，
    在大型語言模型訓練中能加速收斂並穩定學習。

    This version follows the simplified algorithm described in the
    repository documentation. It uses an EMA of squared gradients as a
    Hessian diagonal approximation and clips the update for stability.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), rho=0.04,
                 eps=1e-8, weight_decay=0., hessian_interval=10,
                 use_adaptive_gradient_clipping=False,
                 gradient_centralization=None, enable_cautious=False):
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps,
                        weight_decay=weight_decay,
                        hessian_interval=hessian_interval,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        gradient_centralization=gradient_centralization,
                        enable_cautious=enable_cautious)
        super(Sophia, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            rho = group['rho']
            lr = group['lr']
            eps = group['eps']
            interval = group['hessian_interval']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                if self.use_adaptive_gradient_clipping:
                    self.agc(p)
                grad = p.grad.detach()
                if self.gradient_centralization in ['all', 'gcc']:
                    grad = centralize_gradient(grad, gc_conv_only=self.gradient_centralization == 'gcc')
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian_diag'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                h_diag = state['hessian_diag']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if state['step'] % interval == 0 or state['step'] == 1:
                    h_est = grad * grad
                    h_diag.mul_(beta2).add_(h_est, alpha=1 - beta2)

                denom = h_diag.add(eps)
                update = exp_avg / denom
                update.clamp_(min=-rho, max=rho)
                if self.enable_cautious:
                    update = self._apply_cautious(update, grad)
                p.data.add_(update, alpha=-lr)

        return loss


class Adan(Optimizer):
    """Implements Adan optimizer combining adaptive moments and Nesterov momentum.

    中文：Adan 結合自適應動量與 Nesterov 估計，能在各種任務上更快收斂。

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate.
        betas (Tuple[float, float, float] | Tuple[float, float]):
            coefficients for computing running averages. If two values are
            provided, they are interpreted as ``(beta1, beta3)`` and ``beta2``
            will be filled with the recommended default ``0.92``.
    """

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99),
                 eps=1e-8, weight_decay=0.,
                 use_adaptive_gradient_clipping=False,
                 gradient_centralization=None, enable_cautious=False):
        # allow specifying two betas (beta1, beta3) and insert the recommended
        # default for beta2. this keeps backward compatibility with optimizers
        # expecting only beta1 and beta2 while adhering to Adan's requirement
        # of three beta values.
        if len(betas) == 2:
            betas = (betas[0], 0.92, betas[1])
        elif len(betas) != 3:
            raise AssertionError('Adan requires two or three beta values')

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        gradient_centralization=gradient_centralization,
                        enable_cautious=enable_cautious)
        super(Adan, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad.detach()
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['n'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.zeros_like(p)

                m, v, n = state['m'], state['v'], state['n']
                prev_grad = state['prev_grad']

                grad_diff = grad - prev_grad

                m.mul_(1 - beta1).add_(grad, alpha=beta1)
                v.mul_(1 - beta2).add_(grad_diff, alpha=beta2)
                u = grad + (1 - beta2) * grad_diff
                n.mul_(1 - beta3).addcmul_(u, u, value=beta3)

                state['step'] += 1
                bias_correction1 = 1 - (1 - beta1) ** state['step']
                bias_correction3 = 1 - (1 - beta3) ** state['step']

                step_size = lr / bias_correction1
                denom = (n / bias_correction3).sqrt().add_(eps)
                update = m / denom
                if self.enable_cautious:
                    update = self._apply_cautious(update, grad)
                p.data.add_(update, alpha=-step_size)

                state['prev_grad'].copy_(grad)

        return loss


def _orthogonalize(mat: torch.Tensor, num_iters: int = 5):
    """
    Newton–Schulz approximation of the polar factor (orthogonal part) of `mat`.
    The in-place implementation avoids an unused `eye` tensor.
    """
    if mat.dim() < 2:
        return mat
    m, n = mat.shape
    for _ in range(num_iters):
        if m >= n:
            mat = 1.5 * mat - 0.5 * mat @ (mat.T @ mat)   # tall or square
        else:
            mat = 1.5 * mat - 0.5 * (mat @ mat.T) @ mat   # wide
    return mat


class Muon(Optimizer):
    """Momentum optimizer with orthogonalized updates for matrix parameters.

    中文：Muon 針對高維矩陣權重，先計算動量再將更新矩陣正交化，
    以提升大型模型訓練的穩定度與效率。
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.,
                 use_adaptive_gradient_clipping=False,
                 gradient_centralization=None, orthogonal_iters=5,
                 enable_cautious=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        gradient_centralization=gradient_centralization,
                        orthogonal_iters=orthogonal_iters,
                        enable_cautious=enable_cautious)
        super(Muon, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            k = group['orthogonal_iters']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad.detach()
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)

                update = buf.clone()
                if p.dim() >= 2:
                    update = _orthogonalize(update, num_iters=k)
                if self.enable_cautious:
                    update = self._apply_cautious(update, grad)
                p.add_(update, alpha=-lr)

        return loss


class AdamMini(Optimizer):
    """Memory-efficient AdamW variant using shared second moments.

    中文：AdamMini 透過多參數共享二階矩，大幅降低記憶體使用，
    適合在訓練大型模型時取代 AdamW。
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.,
                 use_adaptive_gradient_clipping=False,
                 gradient_centralization=None, enable_cautious=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        gradient_centralization=gradient_centralization,
                        enable_cautious=enable_cautious)
        super(AdamMini, self).__init__(params, defaults)
        for group in self.param_groups:
            group['shared_v'] = torch.zeros(1, device=next(iter(group['params'])).device)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            shared_v = group['shared_v']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad.detach()
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                shared_v.mul_(beta2).add_(grad.pow(2).mean(), alpha=1 - beta2)

                step_size = lr / (1 - beta1 ** state['step'])
                denom = shared_v.sqrt().add_(eps)
                update = exp_avg / denom
                if self.enable_cautious:
                    update = self._apply_cautious(update, grad)
                p.data.add_(update, alpha=-step_size)

        return loss


class GaLore(Optimizer):
    """Optimizer wrapper applying low-rank projection to gradients.

    中文：GaLore 在每次更新前將梯度以低秩形式近似，
    有效減少優化器狀態的記憶體佔用。
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 rank=8, weight_decay=0.,
                 use_adaptive_gradient_clipping=False,
                 gradient_centralization=None, enable_cautious=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, rank=rank,
                        weight_decay=weight_decay,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        gradient_centralization=gradient_centralization,
                        enable_cautious=enable_cautious)
        super(GaLore, self).__init__(params, defaults)

    def _low_rank(self, grad, rank):
        if grad.dim() < 2:
            return grad
        u, s, v = torch.linalg.svd(grad, full_matrices=False)
        s = s[:rank]
        u = u[:, :rank]
        v = v[:rank, :]
        return (u * s) @ v

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            rank = group['rank']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad.detach()
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                grad = self._low_rank(grad, rank)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom
                if self.enable_cautious:
                    update = self._apply_cautious(update, grad)
                p.data.add_(update, alpha=-step_size)

        return loss


class Apollo(Optimizer):
    """SGD-like memory footprint with AdamW-level performance.

    中文：Apollo 將通道級縮放與隨機投影結合，
    使優化器狀態接近 SGD 的輕量化，同時維持 AdamW 的效果。
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 rank=1, weight_decay=0.,
                 use_adaptive_gradient_clipping=False,
                 gradient_centralization=None, enable_cautious=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, rank=rank,
                        weight_decay=weight_decay,
                        use_adaptive_gradient_clipping=use_adaptive_gradient_clipping,
                        gradient_centralization=gradient_centralization,
                        enable_cautious=enable_cautious)
        super(Apollo, self).__init__(params, defaults)

        for group in self.param_groups:
            group['scale'] = torch.zeros(rank, device=next(iter(group['params'])).device)

    def _project(self, grad, rank):
        flat = grad.view(grad.size(0), -1)
        rand = torch.randn(flat.size(1), rank, device=grad.device)
        proj = flat @ rand
        scale = proj.norm(dim=0)
        return scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            rank = group['rank']
            scale = group['scale']

            for p in group['params']:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad.detach()
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                state['step'] += 1

                scale[:] = self._project(grad, rank)

                norm_grad = grad / (scale.view(-1, *([1] * (grad.dim() - 1))) + eps)
                exp_avg.mul_(beta1).add_(norm_grad, alpha=1 - beta1)

                step_size = lr / (1 - beta1 ** state['step'])
                update = exp_avg
                if self.enable_cautious:
                    update = self._apply_cautious(update, norm_grad)
                p.data.add_(update, alpha=-step_size)

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
