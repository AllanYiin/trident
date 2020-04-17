import itertools as it
import math
import os
import sys
import time
import uuid
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from shutil import copyfile

import cntk as C
import cntk.learners as optim
import numpy as np
from cntk import cntk_py, NDArrayView, asarray
from cntk.internal import typemap

from .trainers import OptimizerBase
from ..backend.cntk_backend import *
from ..backend.common import *

__all__ = ['RAdam','get_optimizer']



# class Adam(cntk_py.adam_learner,OptimizerMixin):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,**kwargs):
#         use_mean_gradient=kwargs.get('use_mean_gradient',True)
#         unit_gain =  C.learners.default_unit_gain_value()
#         momentum = kwargs.get('beta1',0)
#         variance_momentum = C.learners.momentum_schedule_per_sample(kwargs.get('beta2',0.9999986111120757))
#         gaussian_noise_injection_std_dev=0.0
#         minibatch_size=None
#         epoch_size=None
#         if betas is not None and isinstance(betas,(list,tuple)):
#             momentum=betas[0]
#             variance_momentum=C.learners.momentum_schedule_per_sample(betas[1])
#
#
#         lr, minibatch_size = optim._infer_learning_rate_schedule_and_ref_minibatch_size(use_mean_gradient, minibatch_size, lr,epoch_size)
#
#         momentum = optim._infer_learning_parameter_schedule(momentum, minibatch_size, epoch_size)
#         optim. _verify_momentum_type(momentum)
#         variance_momentum = optim._infer_learning_parameter_schedule(variance_momentum, minibatch_size, epoch_size)
#         optim._verify_momentum_type(variance_momentum)
#         gaussian_noise_injection_std_dev = optim.training_parameter_schedule(gaussian_noise_injection_std_dev)
#
#         additional_options = cntk_py.AdditionalLearningOptions()
#         additional_options.l1_regularization_weight =0.0
#         additional_options.l2_regularization_weight =0.0
#         additional_options.gaussian_noise_injection_std_dev = 0.0
#         additional_options.gradient_clipping_threshold_per_sample = np.inf
#         additional_options.gradient_clipping_with_truncation = True
#         if minibatch_size is not None:
#             additional_options.dict_options[cntk_py.Learner._MINIBATCH_SIZE] = cntk_py.SizeTWrapper(
#                 minibatch_size)  # need this to make proper typed DictionaryValue
#
#         super(Adam, self).__init__(params, lr, momentum, unit_gain,  variance_momentum, epsilon(), False, additional_options)



class Optimizer(object):
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self,  defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []


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
        packed_state = {(id(k) if isinstance(k, C.Parameter) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    # def load_state_dict(self, state_dict):
    #     r"""Loads the optimizer state.
    #
    #     Arguments:
    #         state_dict (dict): optimizer state. Should be an object returned
    #             from a call to :meth:`state_dict`.
    #     """
    #     # deepcopy, to be consistent with module API
    #     state_dict = deepcopy(state_dict)
    #     # Validate the state_dict
    #     groups = self.param_groups
    #     saved_groups = state_dict['param_groups']
    #
    #     if len(groups) != len(saved_groups):
    #         raise ValueError("loaded state dict has a different number of "
    #                          "parameter groups")
    #     param_lens = (len(g['params']) for g in groups)
    #     saved_lens = (len(g['params']) for g in saved_groups)
    #     if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
    #         raise ValueError("loaded state dict contains a parameter group "
    #                          "that doesn't match the size of optimizer's group")
    #
    #     # Update the state
    #     id_map = {old_id: p for old_id, p in
    #               zip(chain(*(g['params'] for g in saved_groups)),
    #                   chain(*(g['params'] for g in groups)))}
    #
    #     def cast(param, value):
    #         r"""Make a deep copy of value, casting all tensors to device of param."""
    #         if isinstance(value, torch.Tensor):
    #             # Floating-point types are a bit special here. They are the only ones
    #             # that are assumed to always match the type of params.
    #             if param.is_floating_point():
    #                 value = value.to(param.dtype)
    #             value = value.to(param.device)
    #             return value
    #         elif isinstance(value, dict):
    #             return {k: cast(param, v) for k, v in value.items()}
    #         elif isinstance(value, container_abcs.Iterable):
    #             return type(value)(cast(param, v) for v in value)
    #         else:
    #             return value
    #
    #     # Copy state assigned to params (and cast tensors to appropriate types).
    #     # State that is not assigned to params is copied as is (needed for
    #     # backward compatibility).
    #     state = defaultdict(dict)
    #     for k, v in state_dict['state'].items():
    #         if k in id_map:
    #             param = id_map[k]
    #             state[param] = cast(param, v)
    #         else:
    #             state[k] = v
    #
    #     # Update parameter groups, setting their 'params' value
    #     def update_group(group, new_group):
    #         new_group['params'] = group['params']
    #         return new_group
    #     param_groups = [
    #         update_group(g, ng) for g, ng in zip(groups, saved_groups)]
    #     self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self):
        pass
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             p.grad.detach_()
        #             p.grad.zero_()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
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
        if isinstance(params, C.Parameter):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, C.Parameter):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " +type(param))

        #
        # for name, default in self.defaults.items():
        #     if default is required and name not in param_group:
        #         raise ValueError("parameter group didn't specify a value of required optimization parameter " +
        #                          name)
        #     else:
        #         param_group.setdefault(name, default)

        # param_set = set()
        # for group in self.param_groups:
        #     param_set.update(set(group['params']))
        #
        # if not param_set.isdisjoint(set(param_group['params'])):
        #     raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)



# class RAdam(Optimizer,OptimizerMixin):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#
#         self.degenerated_to_sgd = degenerated_to_sgd
#         if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
#             for param in params:
#                 if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
#                     param['buffer'] = [[None, None, None] for _ in range(10)]
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
#                         buffer=[[None, None, None] for _ in range(10)])
#         super(RAdam, self).__init__(params, defaults)
#     def __setstate__(self, state):
#         super(RAdam, self).__setstate__(state)
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure()
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data.float()
#                 if grad.is_sparse:
#                     raise RuntimeError('RAdam does not support sparse gradients')
#
#                 p_data_fp32 = p.data.float()
#
#                 state = self.state[p]
#
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['exp_avg'] = C.zeros_like(p_data_fp32)
#                     state['exp_avg_sq'] = C.zeros_like(p_data_fp32)
#                 else:
#                     state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
#                     state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
#
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 beta1, beta2 = group['betas']
#
#                 exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
#                 exp_avg.mul_(beta1).add_(1 - beta1, grad)
#
#                 state['step'] += 1
#                 buffered = group['buffer'][int(state['step'] % 10)]
#                 if state['step'] == buffered[0]:
#                     N_sma, step_size = buffered[1], buffered[2]
#                 else:
#                     buffered[0] = state['step']
#                     beta2_t = beta2 ** state['step']
#                     N_sma_max = 2 / (1 - beta2) - 1
#                     N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
#                     buffered[1] = N_sma
#
#                     # more conservative since it's an approximated value
#                     if N_sma >= 5:
#                         step_size = math.sqrt(
#                             (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
#                                         N_sma_max - 2)) / (1 - beta1 ** state['step'])
#                     elif self.degenerated_to_sgd:
#                         step_size = 1.0 / (1 - beta1 ** state['step'])
#                     else:
#                         step_size = -1
#                     buffered[2] = step_size
#
#                 # more conservative since it's an approximated value
#                 if N_sma >= 5:
#                     if group['weight_decay'] != 0:
#                         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
#                     denom = exp_avg_sq.sqrt().add_(group['eps'])
#                     p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
#                     p.data.copy_(p_data_fp32)
#                 elif step_size > 0:
#                     if group['weight_decay'] != 0:
#                         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
#                     p_data_fp32.add_(-step_size * group['lr'], exp_avg)
#                     p.data.copy_(p_data_fp32)
#
#         return loss
#
# class PlainRAdam(Optimizer,OptimizerMixin):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#
#         self.degenerated_to_sgd = degenerated_to_sgd
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#
#         super(PlainRAdam, self).__init__(params, defaults)
#
#     def __setstate__(self, state):
#         super(PlainRAdam, self).__setstate__(state)
#
#     def step(self, closure=None):
#
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data.float()
#                 if grad.is_sparse:
#                     raise RuntimeError('RAdam does not support sparse gradients')
#
#                 p_data_fp32 = p.data.float()
#
#                 state = self.state[p]
#
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['exp_avg'] = C.zeros_like(p_data_fp32)
#                     state['exp_avg_sq'] = C.zeros_like(p_data_fp32)
#                 else:
#                     state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
#                     state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
#
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 beta1, beta2 = group['betas']
#
#                 exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
#                 exp_avg.mul_(beta1).add_(1 - beta1, grad)
#
#                 state['step'] += 1
#                 beta2_t = beta2 ** state['step']
#                 N_sma_max = 2 / (1 - beta2) - 1
#                 N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
#
#                 # more conservative since it's an approximated value
#                 if N_sma >= 5:
#                     if group['weight_decay'] != 0:
#                         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
#                     step_size = group['lr'] * math.sqrt(
#                         (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
#                                     N_sma_max - 2)) / (1 - beta1 ** state['step'])
#                     denom = exp_avg_sq.sqrt().add_(group['eps'])
#                     p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
#                     p.data.copy_(p_data_fp32)
#                 elif self.degenerated_to_sgd:
#                     if group['weight_decay'] != 0:
#                         p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
#                     step_size = group['lr'] / (1 - beta1 ** state['step'])
#                     p_data_fp32.add_(-step_size, exp_avg)
#                     p.data.copy_(p_data_fp32)
#
#         return loss
#
#
# class AdamW(Optimizer,OptimizerMixin):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup=0):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
#         super(AdamW, self).__init__(params, defaults)
#
#     def __setstate__(self, state):
#         super(AdamW, self).__setstate__(state)
#
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data.float()
#                 if grad.is_sparse:
#                     raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#
#                 p_data_fp32 = p.data.float()
#
#                 state = self.state[p]
#
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['exp_avg'] = C.zeros_like(p_data_fp32)
#                     state['exp_avg_sq'] = C.zeros_like(p_data_fp32)
#                 else:
#                     state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
#                     state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
#
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 beta1, beta2 = group['betas']
#
#                 state['step'] += 1
#
#                 exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
#                 exp_avg.mul_(beta1).add_(1 - beta1, grad)
#
#                 denom = exp_avg_sq.sqrt().add_(group['eps'])
#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']
#
#                 if group['warmup'] > state['step']:
#                     scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
#                 else:
#                     scheduled_lr = group['lr']
#
#                 step_size = scheduled_lr * math.sqrt(bias_correction2) / bias_correction1
#
#                 if group['weight_decay'] != 0:
#                     p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)
#
#                 p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
#
#                 p.data.copy_(p_data_fp32)
#
#         return loss
#
#
# class Lookahead(Optimizer,OptimizerMixin):
#     def __init__(self, optimizer, k=5, alpha=0.5):
#         self.optimizer = optimizer
#         self.k = k
#         self.alpha = alpha
#         self.param_groups = self.optimizer.param_groups
#         self.state = defaultdict(dict)
#         self.fast_state = self.optimizer.state
#         for group in self.param_groups:
#             group["counter"] = 0
#
#     def update(self, group):
#         for fast in group["params"]:
#             param_state = self.state[fast]
#             if "slow_param" not in param_state:
#                 param_state["slow_param"] = C.zeros_like(fast.data)
#                 param_state["slow_param"].copy_(fast.data)
#             slow = param_state["slow_param"]
#             update_add(slow ,(fast.data - slow) * self.alpha)
#             fast.data.copy_(slow)
#
#     def update_lookahead(self):
#         for group in self.param_groups:
#             self.update(group)
#
#     def step(self, closure=None):
#         loss = self.optimizer.step(closure)
#         for group in self.param_groups:
#             if group["counter"] == 0:
#                 self.update(group)
#             group["counter"] += 1
#             if group["counter"] >= self.k:
#                 group["counter"] = 0
#         return loss
#
#     def state_dict(self):
#         fast_state_dict = self.optimizer.state_dict()
#         slow_state = {(to_numpy(k) ): v for k, v in self.state.items()}
#         fast_state = fast_state_dict["state"]
#         param_groups = fast_state_dict["param_groups"]
#         return {"fast_state": fast_state, "slow_state": slow_state, "param_groups": param_groups, }
#
#     def load_state_dict(self, state_dict):
#         slow_state_dict = {"state": state_dict["slow_state"], "param_groups": state_dict["param_groups"], }
#         fast_state_dict = {"state": state_dict["fast_state"], "param_groups": state_dict["param_groups"], }
#         super(Lookahead, self).load_state_dict(slow_state_dict)
#         self.optimizer.load_state_dict(fast_state_dict)
#         self.fast_state = self.optimizer.state
#
#     def add_param_group(self, param_group):
#         param_group["counter"] = 0
#         self.optimizer.add_param_group(param_group)
#
# class Ranger(Optimizer,OptimizerMixin):
#     '''
#     https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger.py
#     '''
#     def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.999), eps=1e-5, weight_decay=0):
#         #parameter checks
#         if not 0.0 <= alpha <= 1.0:
#             raise ValueError('Invalid slow update rate: {alpha}')
#         if not 1 <= k:
#             raise ValueError('Invalid lookahead steps: {k}')
#         if not lr > 0:
#             raise ValueError('Invalid Learning Rate: {lr}')
#         if not eps > 0:
#             raise ValueError('Invalid eps: {eps}')
#
#         #parameter comments:
#         # beta1 (momentum) of .95 seems to work better than .90...
#         #N_sma_threshold of 5 seems better in testing than 4.
#         #In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.
#
#         #prep defaults and init torch.optim base
#         defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
#         super().__init__(params,defaults)
#
#         #adjustable threshold
#         self.N_sma_threshhold = N_sma_threshhold
#
#         #now we can get to work...
#         #removed as we now use step from RAdam...no need for duplicate step counting
#         #for group in self.param_groups:
#         #    group["step_counter"] = 0
#             #print("group step counter init")
#
#         #look ahead params
#         self.alpha = alpha
#         self.k = k
#
#         #radam buffer for state
#         self.radam_buffer = [[None,None,None] for ind in range(10)]
#
#         #self.first_run_check=0
#
#         #lookahead weights
#         #9/2/19 - lookahead param tensors have been moved to state storage.
#         #This should resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.
#
#         #self.slow_weights = [[p.clone().detach() for p in group['params']]
#         #                     for group in self.param_groups]
#
#         #don't use grad for lookahead weights
#         #for w in it.chain(*self.slow_weights):
#         #    w.requires_grad = False
#
#     def __setstate__(self, state):
#         print("set state called")
#         super(Ranger, self).__setstate__(state)
#
#
#     def step(self, closure=None):
#         loss = None
#         #note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
#         #Uncomment if you need to use the actual closure...
#
#         #if closure is not None:
#             #loss = closure()
#
#         #Evaluate averages and grad, update param tensors
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data.float()
#                 if grad.is_sparse:
#                     raise RuntimeError('Ranger optimizer does not support sparse gradients')
#
#                 p_data_fp32 = p.data.float()
#
#                 state = self.state[p]  #get state dict for this param
#
#                 if len(state) == 0:   #if first time to run...init dictionary with our desired entries
#                     #if self.first_run_check==0:
#                         #self.first_run_check=1
#                         #print("Initializing slow buffer...should not see this at load from saved model!")
#                     state['step'] = 0
#                     state['exp_avg'] = C.zeros_like(p_data_fp32)
#                     state['exp_avg_sq'] = C.zeros_like(p_data_fp32)
#
#                     #look ahead weight storage now in state dict
#                     state['slow_buffer'] = C.zeros_like(p.data)
#                     state['slow_buffer'].copy_(p.data)
#
#                 else:
#                     state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
#                     state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
#
#                 #begin computations
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 beta1, beta2 = group['betas']
#
#                 #compute variance mov avg
#                 C.assign(exp_avg_sq,exp_avg_sq*beta2+(1 - beta2)*C.square(grad))
#                 #compute mean moving avg
#                 C.assign(exp_avg,exp_avg*beta1+(1 - beta1)*grad)
#
#                 state['step'] += 1
#                 buffered = self.radam_buffer[int(state['step'] % 10)]
#                 if state['step'] == buffered[0]:
#                     N_sma, step_size = buffered[1], buffered[2]
#                 else:
#                     buffered[0] = state['step']
#                     beta2_t = beta2 ** state['step']
#                     N_sma_max = 2 / (1 - beta2) - 1
#                     N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
#                     buffered[1] = N_sma
#                     if N_sma > self.N_sma_threshhold:
#                         step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
#                     else:
#                         step_size = 1.0 / (1 - beta1 ** state['step'])
#                     buffered[2] = step_size
#
#                 if group['weight_decay'] != 0:
#                     C.assign(p_data_fp32,p_data_fp32+(-group['weight_decay'] * group['lr']*p_data_fp32))
#
#                 if N_sma > self.N_sma_threshhold:
#                     denom = C.sqrt(exp_avg_sq)+group['eps']
#                     C.assign(p_data_fp32,p_data_fp32+(-step_size * group['lr'])* (exp_avg/denom))
#                 else:
#                     C.assign(p_data_fp32,p_data_fp32+(-step_size * group['lr']* exp_avg))
#
#                 p.data.copy_(p_data_fp32)
#
#                 #integrated look ahead...
#                 #we do it at the param level instead of group level
#                 if state['step'] % group['k'] == 0:
#                     slow_p = state['slow_buffer'] #get access to slow param tensor
#                     slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
#                     p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor
#         return loss
#



class RAdam(Optimizer, OptimizerBase):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__( defaults)
        params_group={}
        params_group['params']=params
        params_group['weight_decay']=weight_decay
        params_group['lr']=lr
        params_group['eps'] = eps
        params_group['betas']=betas
        params_group['buffer'] = [[None, None, None] for _ in range(10)]
        self.add_param_group(params_group)
    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
    def step(self, grads_dict=None):
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p not in grads_dict:
                    continue
                grad =grads_dict[p]
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 =p.value

                state = self.state[p.uid]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = C.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = C.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].astype(np.float32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].astype(np.float32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq=exp_avg_sq*beta2+(1 - beta2)*C.square(grad)
                exp_avg=exp_avg*beta1+(1 - beta1)*grad

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
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32=p_data_fp32-group['weight_decay'] * group['lr']

                    denom = C.sqrt(exp_avg_sq)+group['eps']
                    p_data_fp32=p_data_fp32+(denom/exp_avg)*-step_size * group['lr']
                    C.assign(p,p_data_fp32)

                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32=p_data_fp32-group['weight_decay'] * group['lr']
                    p_data_fp32=exp_avg+-step_size * group['lr']
                    C.assign(p, p_data_fp32)

        return loss



def get_optimizer(optimizer_name):
    if optimizer_name is None:
        return None
    optimizer_modules = ['trident.optims.cntk_optimizers','cntk.learners']
    try:
        optimizer_class = get_class(snake2camel(optimizer_name), optimizer_modules)
    except Exception :
        optimizer_class = get_class(optimizer_name, optimizer_modules)
    return optimizer_class
