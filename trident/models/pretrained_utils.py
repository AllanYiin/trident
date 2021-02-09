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
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path, make_dir_if_need, Signature, TensorShape
from trident.backend.model import ModelBase, progress_bar
from trident.callbacks.visualization_callbacks import *
from trident.data.data_provider import *
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import tile_rgb_images, loss_metric_curve
from trident.backend.tensorspec import TensorSpec, assert_spec_compatibility, ObjectType
from trident.loggers.history import HistoryBase


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


def _make_recovery_model_include_top(recovery_model:Layer,input_shape=None, include_top=True, classes=1000, freeze_features=False):
    size_change=False
    default_shape=(3,224,224) if get_backend() == 'pytorch' else (224,224,3)
    if input_shape is not None and input_shape !=default_shape:
        size_change=True
        dims = list(input_shape)
        dims.insert(0, None)

        if isinstance(recovery_model.signature, Signature):
            recovery_model._input_shape = TensorShape(dims)
            recovery_model.input_spec.shape = TensorShape(dims)
            recovery_model.signature.inputs.value_list[0].object_type=ObjectType.rgb

    if freeze_features:
        recovery_model.trainable=False
        idx=-1
        while (len(recovery_model[idx]._parameters) == 0 or isinstance(recovery_model[idx], Dense))and len(recovery_model[idx].output_shape) >= 2:
            layer=recovery_model[idx]
            if layer.output_shape.rank>2:
                break
            if  len(recovery_model[idx]._parameters) >0:
                recovery_model[idx].trainable=True
            idx-=1

    if not include_top:
        while  len(recovery_model[-1]._parameters)==0 or isinstance(recovery_model[-1],Dense) and len(recovery_model[-1].output_shape)>=2:
            layer = recovery_model[-1]
            if layer.output_shape.rank > 2:
                break
            recovery_model.remove_at(-1)
        recovery_model.class_names = []
    elif size_change:
        new_layers=[]
        dims = list(input_shape)
        dims.insert(0, None)
        shp=TensorShape(dims)

        while len(recovery_model[-1]._parameters) == 0 or isinstance(recovery_model[-1], Dense) and len(recovery_model[-1].output_shape) >= 2:
            layer = recovery_model[-1]
            if layer.output_shape.rank > 2:
                break

            new_layer=copy.deepcopy(layer)
            if isinstance(layer,Dense) :
                if  layer.num_filters==1000 and classes != 1000:
                    new_layer=Dense((classes))
                    recovery_model.class_names = []
                else:
                    num_filters=new_layer.num_filters
                    new_layer=Dense((num_filters))
            new_layers.insert(0,new_layer)
            recovery_model.remove_at(-1)
        out=recovery_model(to_tensor(shp.get_dummy_tensor()))
        recovery_model[-1].output_shape=tensor_to_shape(out,need_exclude_batch_axis=True)
        fc_seq=0
        for ly in new_layers:
            if isinstance(ly, Dense):
                recovery_model.add_module('fc' if fc_seq==0 else 'fc{0}'.format(fc_seq),ly)
                fc_seq += 1
            else:
                recovery_model.add_module(ly.name, ly)

        if isinstance(recovery_model.signature, Signature):
            recovery_model.output_shape = TensorShape([None, classes])
            recovery_model.signature.outputs.value_list[0].shape = TensorShape([None, classes])
            recovery_model.signature.outputs.value_list[0].object_type = ObjectType.classification_label

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
            if isinstance(recovery_model.signature, Signature):
                recovery_model.output_shape= TensorShape([None,classes])
                recovery_model.signature.outputs.value_list[0].shape = TensorShape([None,classes])
                recovery_model.signature.outputs.value_list[0].object_type = ObjectType.classification_label
            recovery_model.class_names = []
    return recovery_model


