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
from trident.backend.common import get_backend,to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path,make_dir_if_need
from trident.backend.model import ModelBase, progress_bar
from trident.callbacks.visualization_callbacks import *
from trident.data.data_provider import *
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import tile_rgb_images, loss_metric_curve
from trident.backend.tensorspec import TensorSpec, assert_spec_compatibility
from trident.loggers.history import HistoryBase


_session = get_session()
_backend =get_backend()
working_directory=_session.working_directory


if _backend == 'pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *
    from trident.optims.pytorch_optimizers import *
    from trident.layers.pytorch_layers import *
elif _backend == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *
    from trident.optims.tensorflow_optimizers import *
    from trident.layers.tensorflow_layers import *


def _make_recovery_model_include_top(recovery_model:Layer,include_top=True, classes=1000, freeze_features=False):
    if freeze_features==True:
        recovery_model.trainable=False
        while len(recovery_model[-1]._parameters) == 0 or isinstance(recovery_model[-1], Dense) and len(recovery_model[-1].output_shape) >= 2:
            if  len(recovery_model[-1]._parameters) >0:
                recovery_model[-1].trainable=True

    if include_top==False:
        while  len(recovery_model[-1]._parameters)==0 or isinstance(recovery_model[-1],Dense) and len(recovery_model[-1].output_shape)>=2:
            recovery_model.remove_at(-1)
        recovery_model.class_names = []
    else:
        #include_top=True
        if classes != 1000:
            while  len(recovery_model[-1]._parameters)==0 or isinstance(recovery_model[-1],Dense) and len(recovery_model[-1].output_shape)>=2:
                m=recovery_model[-1]
                if isinstance(m,Dense):
                    recovery_model[-1]=Dense((classes))
                    recovery_model.add_module('softmax',SoftMax())
                    break
                else:
                    recovery_model.remove_at(-1)

            modules=list(recovery_model.modules()).reverse()

            recovery_model.class_names = []
    return recovery_model


