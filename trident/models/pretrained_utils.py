from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import inspect
import json
import numbers
import os
import shutil
import sys
import time
import uuid
from functools import partial
import builtins

import numpy as np
from trident.callbacks.lr_schedulers import AdjustLRCallback

from trident.backend import iteration_tools
from trident.data.dataset import ZipDataset
from trident.backend.common import get_backend, to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, Signature, TensorShape
from trident.backend.model import ModelBase, progress_bar
from trident.callbacks.visualization_callbacks import *
from trident.data.data_provider import *
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import tile_rgb_images, loss_metric_curve
from trident.backend.tensorspec import TensorSpec, assert_spec_compatibility, ObjectType
from trident.loggers.history import HistoryBase
from trident.context import split_path, make_dir_if_need, sanitize_path


_session = get_session()

working_directory=_session.working_directory


if get_backend() == 'pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *
    from trident.optims.pytorch_optimizers import *
    from trident.layers.pytorch_layers import *
elif get_backend() == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *
    from trident.optims.tensorflow_optimizers import *
    from trident.layers.tensorflow_layers import *


def _make_recovery_model_include_top(recovery_model:Layer,default_shape=None,input_shape=None, include_top=True, classes=1000, freeze_features=True):
    size_change=False
    if default_shape is None:
        if recovery_model.built:
            default_shape = tuple(recovery_model._input_shape.dims[1:] if isinstance(recovery_model._input_shape, TensorShape) else recovery_model._input_shape)
        else:
            default_shape = (3, 224, 224) if get_backend() == 'pytorch' else (224, 224, 3)
    if input_shape is not None and input_shape !=default_shape:
        size_change=True

    if freeze_features:
        recovery_model.trainable = False
        idx = -1
        is_last_dense=True
        while (len(recovery_model[idx]._parameters) == 0 or isinstance(recovery_model[idx], Dense)) and len(recovery_model[idx].output_shape) >= 2:
            layer = recovery_model[idx]
            if layer.output_shape.rank > 2:
                break
            elif len(recovery_model[idx]._parameters) > 0:
                if not include_top:
                    recovery_model.remove_at(idx)
                    idx+=1
                elif size_change or (is_last_dense and classes != 1000 and  isinstance(recovery_model[idx], Dense)):
                    if hasattr(recovery_model[idx],'num_filters') and recovery_model[idx].num_filters!=classes:
                        recovery_model[idx].num_filters=classes
                    recovery_model[idx]._built=False
                    recovery_model[idx]._parameters.clear()

                else:
                    recovery_model[idx].trainable = True
            else:
                if not include_top:
                    recovery_model.remove_at(idx)
                    idx+=1
            idx -= 1


    dims =list(default_shape)
    dims.insert(0, None)
    new_tensorshape =TensorShape(dims)
    if size_change:
        dims = list(input_shape)
        dims.insert(0, None)
        new_tensorshape=TensorShape(dims)
        for module in recovery_model.modules():
            module._input_shape=None
            module._output_shape = None

    recovery_model.to(get_device())
    dummy_input=to_tensor(new_tensorshape.get_dummy_tensor()).to(recovery_model.device)
    print(dummy_input.device)
    out = recovery_model(dummy_input)
    if isinstance(recovery_model.signature, Signature):
        recovery_model.signature.inputs.value_list[0].shape = TensorShape(dims)
        recovery_model.signature.inputs.value_list[0].object_type=ObjectType.rgb

    return recovery_model


