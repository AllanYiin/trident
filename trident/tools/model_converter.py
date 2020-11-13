from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import inspect
import itertools
import json
import numbers
import os
import shutil
import sys
import time
import uuid
from collections import namedtuple
from functools import partial

import numpy as np
from trident.backend import iteration_tools
from trident.data.dataset import ZipDataset
from trident.backend.common import to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path
from trident.backend.model import ModelBase, progress_bar
from trident.callbacks.visualization_callbacks import *
from trident.data.data_provider import *
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import tile_rgb_images, loss_metric_curve
from trident.backend.tensorspec import TensorSpec, assert_spec_compatibility

__all__ = ['load_torch_as_tf_model']

_session = get_session()
_backend = _session.backend
if _backend == 'pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *
    from trident.optims.pytorch_optimizers import *
elif _backend == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *
    from trident.optims.tensorflow_optimizers import *

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


def load_torch_as_tf_model(model:Layer,state_dict,strict=False):

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()


        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            load_as_tf_from_torch_state_dict(module,state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model)
        load = None  # break load->load reference cycle

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

        # if len(error_msgs) > 0:
        #     raise RuntimeError(
        #         'Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

def load_as_tf_from_torch_state_dict(model:Layer,state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Arguments:
        model ():
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in model._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    persistent_buffers = {k: v for k, v in model._buffers.items() if k not in model._non_persistent_buffers_set}
    local_name_params = itertools.chain(model._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}

    #torch   out-filters  in-filters,, kernel
    # tf    kernel,in-filters, out-filters
    #torch=>tf   permute(2,3,1,0)

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]

            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]

            if input_param.ndim>=3 and 'conv' in name:
                axis_order=list(range(ndim))[2:]
                axis_order.append(1)
                axis_order.append(0)
                permute_input_param=input_param.permute(*axis_order)
                if permute_input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, permute_input_param.shape, param.shape))
                    continue
                try:
                    param.assign(to_tensor(permute_input_param))

                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occured : {}.'
                                      .format(key, numel(param), numel(permute_input_param), ex.args))
            else:



                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                try:
                    param.assign(to_tensor(input_param))

                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occured : {}.'
                                      .format(key, numel(param), numel(input_param), ex.args))
        elif strict:
            missing_keys.append(key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix):
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in model._modules and input_name not in local_state:
                    unexpected_keys.append(key)