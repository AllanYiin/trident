from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import gc
import inspect
import numbers
import os
import random
import shutil
import sys
import tempfile
import time
import uuid
from functools import partial

import numpy as np
import tensorflow as tf

from trident import __version__
from trident import context
from trident.optims.field_binding import prepare_objective
from trident.optims.losses import FunctionLoss
from trident.optims.metrics import FunctionMetric
from trident.context import split_path, make_dir_if_need, sanitize_path
from trident.backend import dtype
from trident.backend.common import *
from trident.backend.model import ModelBase, progress_bar
from trident.backend.opencv_backend import array2image, image2array
from trident.backend.tensorflow_backend import Layer,Sequential,ModuleDict,ModuleList, Combine, summary, get_device, fix_layer, try_map_args_and_call, \
    DTYPE_MAPPING, fix_keras_module
from trident.backend.tensorflow_ops import *
from trident.backend.tensorflow_ops import is_tensor
from trident.backend.tensorflow_serialization import save, load_pthtar
from trident.backend.tensorspec import *
from trident.callbacks import LambdaCallback
from trident.callbacks.lr_schedulers import AdjustLRCallback
from trident.backend.checkpoint import staged_checkpoints
from trident.data.image_common import image_backend_adaption,reverse_image_backend_adaption
from trident.data.mask_common import color2label
from trident.data.transform import Transform
from trident.layers.tensorflow_layers import *
from trident.optims.tensorflow_constraints import get_constraint
from trident.optims.tensorflow_losses import get_loss
from trident.optims.tensorflow_metrics import get_metric
from trident.optims.tensorflow_optimizers import get_optimizer
from trident.optims.tensorflow_regularizers import *
from trident.optims.trainers import TrainingPlan

# from tensorflow.python.framework.ops import EagerTensor

__all__ = ['Model', 'ImageClassificationModel', 'ImageRegressionModel', 'ImageDetectionModel', 'ImageSegmentationModel',
           'FaceLandmarkModel', 'FaceRecognitionModel',
           'LanguageModel']
ctx = context._context()

working_directory = ctx.working_directory
_, term_width = get_terminal_size()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def _scale_gradient(gradient, factor):
    if isinstance(gradient, tf.IndexedSlices):
        return tf.IndexedSlices(gradient.values * tf.cast(factor, gradient.dtype),
                                gradient.indices, gradient.dense_shape)
    return gradient * tf.cast(factor, gradient.dtype)


def _merge_gradients(left, right):
    if isinstance(left, tf.IndexedSlices) and isinstance(right, tf.IndexedSlices):
        indices, segments = tf.unique(tf.concat((left.indices, right.indices), axis=0))
        values = tf.math.unsorted_segment_sum(tf.concat((left.values, right.values), axis=0),
                                              segments, tf.size(indices))
        return tf.IndexedSlices(values, indices, left.dense_shape)
    return tf.convert_to_tensor(left) + tf.convert_to_tensor(right)


class Model(ModelBase,Layer):
    def __init__(self, inputs=None, input_shape=None, output=None, name=None):
        super().__init__(inputs, input_shape, output, name)
        self.batch_index = 0
        self.filter_index = 1
        self._enable_tensorboard = False
        self.accumulate_grads_inteval = 1

    def _initial_graph(self, inputs=None, input_shape=None, output=None, initializer=None):
        if output is None:
            raise ValueError('There is at least one output')
        elif 'keras' in output.__module__:
            output._is_keras = True
            output = fix_keras_module(output)
            output.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self._model = output
            self.training_context['current_model'] = self._model
            if self.save_path is None:
                save_path = os.path.join('Models', '{0}_keras.h5'.format(self._model._name))
                self.save_path = sanitize_path(make_dir_if_need(save_path))
            else:
                self.save_path = sanitize_path(make_dir_if_need(self.save_path))
            self.training_context['save_path'] = self.save_path
            return self
        else:
            output._is_keras = False

        if isinstance(input_shape, numbers.Integral):
            input_shape = TensorShape([None] + [input_shape])
        elif isinstance(input_shape, (tuple, list)) and isinstance(input_shape[-1], numbers.Integral):
            input_shape = TensorShape([None] + list(input_shape))

        if isinstance(output, (np.ndarray, Tensor)):
            self._model = to_tensor(output, requires_grad=True)
            self._model._signature = Signature()
            self._model._signature.outputs['output'] = TensorSpec.tensor_to_spec(self._model)
            self._model._signature.outputs['output'].object_type = ObjectType.rgb
            if inputs is None:
                self._model._signature.inputs['input'] = TensorSpec.tensor_to_spec(self._model)
                self._model._signature.inputs['input'].object_type = ObjectType.rgb
            else:
                if inputs is not None and not isinstance(inputs, (tuple, list, dict)):
                    inputs = (inputs,)

                if not isinstance(inputs, dict):
                    for i in range(len(inputs)):
                        inp = to_tensor(inputs[i])
                        output._signature.inputs['input{0}'.format(i)] = TensorSpec.tensor_to_spec(inp,
                                                                                                   need_exclude_batch_axis=True,
                                                                                                   is_singleton=False,
                                                                                                   optional=False,
                                                                                                   name='input{0}'.format(
                                                                                                       i))
                else:
                    for k, v in inputs.items():
                        inp = to_tensor(v)
                        output._signature.inputs[k] = TensorSpec.tensor_to_spec(inp, need_exclude_batch_axis=True,
                                                                                is_singleton=False, optional=False,
                                                                                name=k)
            self._model.signature = self._model._signature

        elif isinstance(output, (Layer, tf.Module)):
            if inputs is not None and not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs,)
            if input_shape is not None and not isinstance(input_shape, (tuple, list, dict)):
                input_shape = (input_shape,)

            output._signature = get_signature(output, output.__class__.__name__)
            if inputs is not None and len(output._signature.inputs) >= len(inputs) > 0:
                if not isinstance(inputs, dict):
                    for i in range(len(inputs)):
                        k = output._signature.inputs.key_list[i]
                        inp = to_tensor(inputs[i])
                        output._signature.inputs[k] = TensorSpec.tensor_to_spec(inp, need_exclude_batch_axis=True,
                                                                                is_singleton=False,
                                                                                optional=output._signature.inputs[
                                                                                    k].optional, name=k)
                else:
                    available_items = output._signature.inputs.key_list.copy()
                    for k, v in inputs.items():
                        if k in output._signature.inputs.key_list:
                            output._signature.inputs[k] = TensorSpec.tensor_to_spec(v, need_exclude_batch_axis=True,
                                                                                    is_singleton=False,
                                                                                    optional=output._signature.inputs[
                                                                                        k].optional, name=k)
                            available_items.remove(k)
                        else:
                            for sk in available_items:
                                if sk.lower() == k.lower():
                                    output._signature.inputs[sk] = TensorSpec.tensor_to_spec(v,
                                                                                             need_exclude_batch_axis=True,
                                                                                             is_singleton=False,
                                                                                             optional=
                                                                                             output._signature.inputs[
                                                                                                 sk].optional, name=sk)
                                    available_items.remove(sk)
                                    break

            if input_shape is not None and len(output._signature.inputs) >= len(input_shape):
                if not isinstance(input_shape, dict):
                    for i in range(len(input_shape)):
                        k = output._signature.inputs.key_list[i]
                        output._signature.inputs[k].shape = input_shape[i] if isinstance(input_shape[i],
                                                                                         TensorShape) else TensorShape(
                            [None] + input_shape[i])
                        output._signature.inputs[k].dtype = dtype.float32
                        for module in output.modules():
                            if isinstance(module, Embedding):
                                output._signature.inputs[k].dtype = dtype.int64
                                break


                else:
                    available_items = output._signature.inputs.key_list.copy()
                    for k, v in input_shape.items():
                        if k in output._signature.inputs.key_list:
                            output._signature.inputs[k].shape = TensorShape([None] + v)
                            available_items.remove(k)
                        else:
                            for sk in available_items:
                                if sk.lower() == k.lower():
                                    output._signature.inputs[sk].shape = TensorShape([None] + v)
                                    available_items.remove(sk)
                                    break

            # input_shape = unpack_singleton(input_shape)
            #

        if isinstance(output, (Layer, tf.Module)):
            output.is_root = True
            output.nodes = OrderedDict()
            for name, mod in output.named_modules():
                if isinstance(mod, Layer):
                    mod.nodes = output.nodes
                    mod.relative_name = name

            out = None
            if inputs is not None:
                inputs = unpack_singleton(inputs)
                args = None
                if isinstance(inputs, dict):
                    out = output(*list(inputs.values()))
                elif isinstance(inputs, (list, tuple)):
                    out = output(*inputs)
                else:
                    out = output(inputs)

            else:

                # output.input_shape = input_shape
                dummay_input = [to_tensor(shp.get_dummy_tensor()).to(get_device()) for shp in input_shape]
                # prevent pytorch 'ValueError: Expected more than 1 value per channel when training, got input size ....
                output.to(get_device())
                output.eval()
                out = output(*dummay_input)
            if is_tensor(out):
                if  len(output.signature.outputs)==0:
                    output.signature.outputs['output']=TensorSpec.tensor_to_spec(out,need_exclude_batch_axis=True,is_singleton=False) if out is not None else TensorSpec(
                            shape=TensorShape([None]), optional=True, name=k)
                elif  len(output.signature.outputs)==1:
                    output.signature.outputs[output.signature.outputs.key_list[0]] = TensorSpec.tensor_to_spec(out,
                                                                                   need_exclude_batch_axis=True,
                                                                                   is_singleton=False) if out is not None else TensorSpec(
                            shape=TensorShape([None]), optional=True, name=k)
            elif isinstance(out, dict):
                output._signature.outputs.clear()
                for k, v in out.items():
                    if v is not None or k in output.signature.outputs:
                        spec = TensorSpec.tensor_to_spec(v, need_exclude_batch_axis=True,is_singleton=False,
                                                         name=k) if v is not None else TensorSpec(
                            shape=TensorShape([None]), optional=True, name=k)
                        if k in output.signature.outputs:
                            spec.optional = output.signature.outputs[k].optional
                            spec.default = output.signature.outputs[k].default
                        output.signature.outputs[k] = spec
            elif isinstance(out, (list, tuple)):
                specs = output._signature.outputs
                names = list(specs) if len(specs) == len(out) else [f'output{i}' for i in range(len(out))]
                specs.clear()
                for key, value in zip(names, out):
                    specs[key] = TensorSpec.tensor_to_spec(value, name=key)

            self._model = output
            # self._model.input_spec=TensorSpec(shape=self._model.input_shape,dtype=self._model.weights[0].data.dtype)


        elif isinstance(output, (list, tuple)) and all([isinstance(m, (tf.Module)) for m in output]):
            output_list = []
            model_list = []
            dummay_input = to_tensor(input_shape.get_dummy_tensor()).to(get_device()) if not is_tensor(
                inputs) else inputs.to(get_device())

            for op in output:
                if isinstance(op, (Layer, tf.Module)):
                    op.to(get_device())
                    # prevent pytorch 'ValueError: Expected more than 1 value per channel when training, got input size ....
                    op.eval()
                    out = fix_layer(op)(dummay_input)
                    model_list.append(op)
                    output_list.extend(*out)
            model = Combine(model_list)
            self._model = model
            self.name = model.name
            for i in range(len(output_list)):
                self._outputs['output_{0}'.format(i)] = TensorSpec(shape=tensor_to_shape(output_list[i]),
                                                                   name='output_{0}'.format(i))
                self._targets['target_{0}'.format(i)] = TensorSpec(shape=tensor_to_shape(output_list[i]),
                                                                   name='target_{0}'.format(i))

        elif isinstance(output, (np.ndarray, Tensor)):
            pass

        else:
            raise ValueError('Invalid output')

        self.training_context['current_model'] = self._model
        if callable(self._model):
            self.__call__ = self._model.__call__
        if not hasattr(self._model, 'name'):
            self._model._name = 'model_' + str(uuid.uuid4().node)
        if self.save_path is None:
            save_path = os.path.join('Models', '{0}.pth.tar_'.format(self._model._name))
            self.save_path = sanitize_path(make_dir_if_need(save_path))
        else:
            self.save_path = sanitize_path(make_dir_if_need(self.save_path))
        self.training_context['save_path'] = self.save_path

    @property
    def signature(self):
        return self._model.signature if self._model is not None else None

    @property
    def device(self):
        return get_device()

    def train(self, **kwargs):
        if self._model is not None and isinstance(self._model, Tensor):
            pass
        elif self._model is not None and isinstance(self._model, Layer) and self._model.built:
            self._model.train()
        elif self._model is not None and self._model._is_keras:
            self._model.train()
        else:
            raise ValueError('There is no built model ,nothing to learn')

    def eval(self):
        if self._model is not None and isinstance(self._model, Tensor):
            pass
        elif self._model is not None and isinstance(self._model, Layer) and self._model.built:
            self._model.eval()

        elif self._model is not None and self._model._is_keras:
            self._model.eval()
        else:
            raise ValueError('There is no built model ,nothing to evaluate')

    @property
    def layers(self):
        if getattr(self._model, '_is_keras', False):
            return self._model.layers
        elif self._model is not None and isinstance(self._model, Layer):
            return self._model.nodes
        elif self._model is not None and self._model._is_keras:
            return self._model.nodes
        else:
            return []

    def complie(self, optimizer="Adam",
                loss=None,
                metrics=None,
                loss_weights=None,
                **kwargs
                ):
        self.with_optimizer(optimizer, lr=2e-3, betas=(0.9, 0.999))
        if loss is not None:
            if isinstance(loss, str) or callable(loss) or inspect.isfunction(loss) or inspect.isclass(loss):
                loss_weights = 1.0 if loss_weights is None or not isinstance(loss, numbers.Number) else loss_weights
                self.with_loss(loss, loss_weight=loss_weights)
            elif isinstance(loss, list):
                if loss_weights is not None and isinstance(loss_weights, list) and len(loss_weights) == len(loss):
                    for k in range(len(loss)):
                        loss_item = loss[k]
                        weight = loss_weights[k] if isinstance(loss_weights[k], numbers.Number) else 1.0
                        self.with_loss(loss_item, loss_weight=weight)
                else:
                    for loss_item in loss:
                        self.with_loss(loss_item)

            elif isinstance(loss, dict):
                if loss_weights is not None and isinstance(loss_weights, dict):
                    for k, v in loss.items():
                        if k in loss_weights:
                            weight = loss_weights[k] if isinstance(loss_weights[k], numbers.Number) else 1.0
                            self.with_loss(v, loss_weight=weight, name=k)
                        else:
                            self.with_loss(v, loss_weight=1.0, name=k)
                else:
                    for k, v in loss.items():
                        self.with_loss(v, loss_weight=1., name=k)
        if metrics is not None:
            if isinstance(metrics, str) or callable(metrics) or inspect.isfunction(metrics) or inspect.isclass(metrics):
                self.with_metric(metrics)
            elif isinstance(metrics, list):
                for metric in metrics:
                    self.with_metric(metric)
            elif isinstance(metrics, dict):
                for k, v in metrics.items():
                    self.with_metric(v, name=k)
        return self

    def with_optimizer(self, optimizer, **kwargs):
        params = None
        if getattr(self._model, '_is_keras', False):
            params = self._model.weights
        else:
            params = [self._model] if is_tensor(self._model) else self._model.parameters()
        if isinstance(optimizer, str):
            optimizer_class = get_optimizer(optimizer)
            self.optimizer = optimizer_class(params, **kwargs)
        else:
            self.optimizer = optimizer(params, **kwargs)

        for loss in self._losses.values():
            self._register_loss_parameters(loss)
        self.base_lr = kwargs.get('lr', kwargs.get('learning_rate', 1e-3))
        self.training_context['current_lr'] = self.base_lr

        return self

    def _register_loss_parameters(self, loss):
        if self.optimizer is None:
            return
        owned = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        new = [v for v in getattr(loss, 'trainable_variables', ()) if id(v) not in owned]
        if new:
            self.optimizer.add_param_group({'params': new})

    def with_loss(self, loss, loss_weight=1, start_epoch=0, as_metric=False, name='', **kwargs):
        if isinstance(loss, str) and loss == 'output':
            self.use_output_as_loss = True
            return self
        alias, objective, bindings = prepare_objective(
            loss, name, kwargs.pop('input_names', None), kwargs, self._losses, get_loss, FunctionLoss)
        self._losses[alias] = objective
        self._field_bindings['loss', alias] = bindings

        # check whether the model is ended by SoftMax
        if isinstance(self._model,Layer):
            if isinstance(list(self._model.modules())[-1], SoftMax) and hasattr(self._losses[alias], 'is_logsoftmax'):
                self._losses[alias].is_logsoftmax = True
            elif isinstance(list(self._model.modules())[-1], Sequential) and isinstance(list(self._model.modules())[-1][-1],
                                                                                        SoftMax) and hasattr(
                self._losses[alias], 'is_logsoftmax'):
                self._losses[alias].is_logsoftmax = True
            elif isinstance(list(self._model.modules())[-1], Dense) and list(self._model.modules())[
                -1].activation == log_softmax and hasattr(self._losses[alias], 'is_logsoftmax'):
                self._losses[alias].is_logsoftmax = True
            elif isinstance(list(self._model.modules())[-1], ModuleDict) and self._losses[
                alias].signature is not None and hasattr(self._losses[alias], 'is_logsoftmax'):
                for k, v in list(self._model.modules())[-1].items():
                    if isinstance(list(v.modules())[-1], SoftMax) and k in self._losses[alias].signature.inputs:
                        self._losses[alias].is_logsoftmax = True
                    elif isinstance(list(v.modules())[-1], Sequential) and isinstance(list(v.modules())[-1][-1],
                                                                                      SoftMax) and k in self._losses[
                        alias].signature.inputs:
                        self._losses[alias].is_logsoftmax = True
                    elif isinstance(list(v.modules())[-1], Dense) and list(v.modules())[
                        -1].activation == log_softmax and k in self._losses[alias].signature.inputs:
                        self._losses[alias].is_logsoftmax = True

        self._register_loss_parameters(self._losses[alias])
        self.loss_weights[alias] = float(loss_weight)
        self._losses[alias].__name__ = alias
        # self._losses[alias].signature = argnames
        self._loss_logging[alias] = (start_epoch, as_metric)
        self._losses[alias].start_epoch = start_epoch
        self._losses[alias].as_metric = as_metric

        return self

    def with_metric(self, metric, print_only=False, name='', **kwargs):
        alias, objective, bindings = prepare_objective(
            metric, name, kwargs.pop('input_names', None), kwargs, self._metrics, get_metric, FunctionMetric)
        objective.print_only = print_only
        self._metrics[alias] = objective
        self._field_bindings['metric', alias] = bindings
        return self

    def with_regularizer(self, reg, reg_weight=None, **kwargs):
        if reg is None:
            return self
        reg_fn = None
        if isinstance(reg, str):
            reg_fn = get_reg(reg)
        elif reg is callable or inspect.isfunction(reg):
            reg_fn = reg

        reg_weight = kwargs.get('reg_weight', reg_weight)
        if 'reg_weight' in kwargs:
            kwargs.pop('reg_weight')

        args = get_signature(reg_fn)
        if reg_weight is not None:
            args.inputs['reg_weight'].default = reg_weight

        self._regs[reg_fn.__name__] = reg_fn
        self._regs[reg_fn.__name__].signature = args
        return self

    def with_constraint(self, constraint, **kwargs):
        if constraint is None:
            return self
        constraint_fn = None
        if isinstance(constraint, str):
            constraint_fn = get_constraint(constraint)

        if hasattr(constraint_fn, 'forward') and constraint_fn.__name__[-4:] == 'norm':
            self._constraints[constraint_fn.__name__] = constraint_fn(**kwargs)

        elif callable(constraint_fn) and constraint_fn.__name__[-4:] == 'norm':
            self._constraints[constraint_fn.__name__] = partial(constraint_fn, **kwargs)

        return self

    def with_model_save_path(self, save_path, **kwargs):
        if save_path is None or len(save_path) == 0:
            save_path = os.path.join('Models', '{0}.pth.tar_'.format(self.name))
        self.save_path = make_dir_if_need(save_path)
        self.training_context['save_path'] = self.save_path
        return self

    def with_automatic_mixed_precision_training(self, **kwargs):
        """Enable automatic mixed precision training
            only enable when using pytorch 1.6 (or higher) as backend and cuda is available.

        Args:
            **kwargs ():

        Returns:
            the model self

        """
        sys.stderr.write(
            'automatic mixed precision training only enable when using pytorch 1.6 (or higher) as backend and cuda is available.')

        # if ctx.amp_available:
        # self.is_autocast_enabled = True
        # self.gradscaler = torch.cuda.amp.GradScaler()
        # sys.stdout.write('Automatic Mixed Precision:{0}.\n'.format('Turn On'))
        # else:
        #     print(
        #         'automatic mixed precision training only enable when using pytorch 1.6 (or higher) as backend and cuda is available.')

        return self

    def with_grad_clipping(self, clipping_threshold=3.0, **kwargs):
        """Enable grad clipping


        Args:
            clipping_threshold ():
            **kwargs ():

        Returns:
            the model self

        """
        self.grad_clipping_by_norm = True
        self.grad_clipping_threshold = clipping_threshold
        return self

    def adjust_learning_rate_scheduling(self, index: int, unit='batch', new_value: float = None):
        callback = AdjustLRCallback(index, unit, new_value)
        callback.is_shared = False
        self.callbacks.append(callback)
        return self

    def adjust_learning_rate(self, lr):
        if self.optimizer is not None:
            self.optimizer.param_groups[0]['lr'] = lr
            self.training_context['current_lr'] = lr
        else:
            raise ValueError('There is no optimizer yet.')

    def do_on_epoch_start(self):
        super().do_on_epoch_start()
        if self.training_context['steps'] == 0:
            self.training_context['time_epoch_start'] = time.time()
            self.training_context['time_epoch_progress'] = 0

    def do_on_epoch_end(self):
        self._finish_gradient_window()
        super().do_on_epoch_end()

    def do_on_batch_start(self):
        super().do_on_batch_start()
        if self.training_context['steps'] == 0:
            self.training_context['time_batch_progress'] = 0
            self.training_context['time_epoch_progress'] = 0
        self.training_context['time_batch_start'] = time.time()
        if (self.training_context['steps'] + 1) % 100 == 0:
            if 'gpu' in self.model.device:
                gc.collect()

    def do_on_batch_end(self):
        super().do_on_batch_end()
        self.training_context['time_batch_end'] = time.time()
        self.training_context['time_batch_progress'] += (
                self.training_context['time_batch_end'] - self.training_context['time_batch_start'])
        self.training_context['time_epoch_progress'] += (
                self.training_context['time_batch_end'] - self.training_context['time_batch_start'])

        if self.training_context['current_batch'] == 0:
            temp = OrderedDict()
            for k in self.training_context['tmp_losses'].key_list:
                if k != 'epoch' and len(self.training_context['tmp_losses'][k]) > 0:
                    temp[k] = self.training_context['tmp_losses'][k][-1][-1]
            if 'total_losses' not in temp:
                temp['total_losses'] = to_numpy(self.training_context['current_loss']).mean()
            ctx.print('{ ' + ', '.join(
                ['{0}: {1}'.format(k, adaptive_format(v, value_type='loss')) for k, v in temp.items()]) + ' }')

    def do_on_data_received(self, train_data, test_data):

        # fields=train_data._fields
        # for i in range(len(fields)):
        if train_data is None and test_data is None:
            return self.training_context['train_data'], self.training_context['test_data']
        feed = self._prepare_model_fields(train_data)
        input_list = {feed[k] for k in self.signature.inputs if k in feed}
        for item in train_data:
            if getattr(getattr(train_data[item], 'dtype', None), 'kind', None) in {'U', 'S'}:
                train_data[item] = train_data[item].astype(str).tolist()
            else:
                train_data[item] = to_tensor(train_data[item], device=get_device())

            if item in input_list and 'float' in str(getattr(train_data[item], 'dtype', '')):
                train_data[item].require_grads = True

            if test_data is not None and item in test_data:
                if getattr(getattr(test_data[item], 'dtype', None), 'kind', None) in {'U', 'S'}:
                    test_data[item] = test_data[item].astype(str).tolist()
                else:
                    test_data[item] = to_tensor(test_data[item], device=get_device())

        self.training_context['train_data'] = train_data
        self.training_context['test_data'] = test_data


        train_data, test_data = super().do_on_data_received(self.training_context['train_data'],
                                                            self.training_context['test_data'])
        self.training_context['train_data'] = train_data
        self.training_context['test_data'] = test_data
        return train_data, test_data

    def do_calculate_forward(self, is_training=True):
        data = self.training_context['train_data'] if is_training or self.training_context['test_data'] is None or len(
            self.training_context['test_data']) == 0 else \
            self.training_context['test_data']
        data_feed = self._prepare_model_fields(data, validation=not is_training)
        model = self.training_context['current_model']
        if is_tensor(model):
            data['output'] = self._model
            if is_training and self.use_output_as_loss:
                this_loss = data['output'].sum()
                self.training_context['losses'].collect(model.signature.outputs.key_list[0],
                                                        self.training_context['steps'], this_loss)
                self._add_loss_term(this_loss, 'sum')
        else:
            if not is_training:
                model.eval()
            else:
                model.train()

            signature = model.signature

            output = try_map_args_and_call(model, data, data_feed, self.is_autocast_enabled)
            self._store_forward_outputs(data, output)
            if is_training and self.use_output_as_loss and is_tensor(output):
                this_loss = output.sum()
                self._add_loss_term(this_loss, 'sum')
                self.training_context['losses'].collect(signature.outputs.key_list[0],
                                                        self.training_context['steps'], to_scalar(this_loss))

    def do_on_loss_calculation_start(self):
        super().do_on_loss_calculation_start()

    def do_on_loss_calculation_end(self):
        super().do_on_loss_calculation_end()

    def do_calculate_losses(self):
        return super().do_calculate_losses()

    def get_current_loss(self):
        return self.training_context['current_loss']

    def do_calculate_regularizations(self):
        pass

    def do_calculate_constraints(self):
        super().do_calculate_constraints()

    def on_optimization_step_start(self):
        super().on_optimization_step_start()

    def _clear_gradient_window(self):
        self.training_context['gradient_terms'] = {}
        self.training_context['gradient_batches'] = 0
        self.training_context['grads_and_vars'] = ()

    def _finish_gradient_window(self, log_gradients=False):
        context = self.training_context
        merged = {}
        for weight, gradients in context.get('gradient_terms', {}).values():
            if weight == 0:
                continue
            for key, (gradient, variable) in gradients.items():
                scaled = gradient if weight is None else _scale_gradient(gradient, 1 / weight)
                if key in merged:
                    scaled = _merge_gradients(merged[key][0], scaled)
                merged[key] = scaled, variable
        pairs = tuple(merged.values())
        try:
            if not pairs:
                return False
            gradients, variables = zip(*pairs)
            for gradient in gradients:
                values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
                tf.debugging.assert_all_finite(values, "Nonfinite gradients; optimizer was not updated")
            if self.grad_clipping_by_norm:
                gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clipping_threshold)
            pairs = tuple(zip(gradients, variables))
            context['grads_and_vars'] = pairs
            super().on_optimization_step_start()
            if log_gradients:
                self.log_gradient(pairs)
            self.optimizer.step(pairs)
            for _, variable in pairs:
                tf.debugging.assert_all_finite(variable, "Optimizer produced nonfinite parameters")
            self.do_calculate_constraints()
            self.do_on_optimization_step_end()
            return True
        finally:
            self._clear_gradient_window()

    def do_gradient_update(self, log_gradients=False):
        context = self.training_context
        if self.accumulation_steps > 1 and context.get('gradient_batches', 0):
            context['tmp_losses'].collect('total_losses', context['steps'],
                                          to_scalar(context['current_loss']))
        ending = context['total_batch'] is not None and context['current_batch'] + 1 == context['total_batch']
        if ending or context.get('gradient_batches', 0) >= self.accumulation_steps:
            self._finish_gradient_window(log_gradients)

    def do_on_optimization_step_end(self):
        super().do_on_optimization_step_end()
        if self.accumulation_steps == 1:
            self.training_context['tmp_losses'].collect('total_losses', self.training_context['steps'],
                                                        to_scalar(self.training_context['current_loss']))
            if self.training_context['is_collect_data']:
                steps, values = self.training_context['tmp_losses'].get_series('total_losses')
                self.training_context['losses'].collect('total_losses', self.training_context['steps'],
                                                        to_scalar(to_numpy(values).mean()))
                if self.training_context['current_batch'] > 0:
                    self.training_context['tmp_losses'].reset()
        else:
            steps, values = self.training_context['tmp_losses'].get_series('total_losses')
            self.training_context['losses'].collect('total_losses', self.training_context['steps'],
                                                    to_scalar(to_numpy(values).mean()))
            if self.training_context['current_batch'] > 0:
                self.training_context['tmp_losses'].reset()

    def do_on_excution_exception(self):
        self._clear_gradient_window()
        super().do_on_excution_exception()

    def log_gradient(self, grads=None):
        grads = list(grads)
        grad_dict = OrderedDict()
        for i, (g, v) in enumerate(grads):
            grad_dict[str(i)] = to_numpy(g)
        self.gradients_history.append(grad_dict)

    def log_weight(self, weghts=None):
        weight_dict = OrderedDict()
        if isinstance(self._model, Layer):
            weights = self._model.get_weights()
            for i in range(len(weights)):
                w = weights[i]
                weight_dict[str(i)] = w
            self.weights_history.append(weight_dict)

    def save_model(self, save_path=None, **kwargs):
        if save_path is not None:
            self.training_context['save_path'] = os.fspath(save_path)
        folder, name, _ = split_path(self.training_context['save_path'])
        name = name or self.name
        for callback in self.callbacks:
            callback.on_model_saving_start(self.training_context)
        try:
            if (isinstance(self._model, tf.Module) or is_tensor(self._model)) and any_abnormal_number(self._model):
                sys.stderr.write('nan/inf detected in model weights. Skip saving.\n')
                return False
            if isinstance(self._model, Layer):
                stem = os.path.join(folder, name if name.endswith('_tf') else name + '_tf')
                state = dict(state_dict=self._model.state_dict(), backend='tensorflow',
                             trident_version=__version__, tensorflow_version=tf.version.VERSION,
                             signature=self._model.signature)
                if hasattr(self, 'vocabs'):
                    state['vocabs'] = self.vocabs
                with staged_checkpoints(stem + '.pth', stem + '.pth.tar') as (whole, archive):
                    with tf.device('/cpu:0'), archive.open('wb') as stream:
                        save(state, stream, is_compressed=True)
                    with whole.open('wb') as stream:
                        save(self._model, stream)
            elif getattr(self._model, '_is_keras', False):
                stem = name if name.endswith('keras') else name + '_keras'
                with staged_checkpoints(os.path.join(folder, stem + '.h5')) as (draft,):
                    self._model.save(str(draft))
            elif is_tensor(self._model):
                with staged_checkpoints(os.path.join(folder, name + '.npy')) as (draft,):
                    np.save(draft, to_numpy(self._model))
            else:
                raise TypeError("Expected a TensorFlow model or tensor")
        finally:
            for callback in self.callbacks:
                callback.on_model_saving_end(self.training_context)

    def save_onnx(self, save_path=None, dynamic_axes=None, **kwargs):
        pass

    def save_weights(self, save_path=None, **kwargs):
        for callback in self.training_context['callbacks']:
            callback.on_model_saving_start(self.training_context)

        if save_path is not None:
            self._model.save_weights(save_path)
        elif 'save_path' in self.training_context and self.training_context['save_path'] is not None:
            self._model.save_weights(self.training_context['save_path'])

        else:
            if 'Models' is not None and len('Models') > 1 and not os.path.exists('Models'):
                try:
                    os.makedirs('Models')
                except Exception as e:
                    pass
            save_full_path = os.path.join('Models/', 'model_{0}_epoch{1}.h5'.format(self._model.__name__,
                                                                                    self.training_context[
                                                                                        'current_epoch']))
            self._model.save_weights(self.training_context['save_full_path'])

        self._model.train()

    def load_model(self, file_path, **kwargs):
        ctx.print('Loading pretrained model from {}'.format(file_path))
        folder, filename, ext = split_path(file_path)
        if filename == '':
            filename = self.name
        state_dict = None
        pretrained_dict = None
        if ext == '.pth.tar':
            state_dict = load_pthtar(file_path)
        elif ext == '.pth':
            load_path = file_path
            if not os.path.exists(file_path):
                if os.path.exists(file_path.replace(ext, '.pth.tar')):
                    load_path = file_path.replace(ext, '.pth.tar')
                elif os.path.exists(os.path.join(working_directory, filename + ext)):
                    load_path = os.path.join(working_directory, filename + ext)
            recovery_pth = load_pthtar(load_path)

            if isinstance(recovery_pth, dict):
                state_dict = recovery_pth

            elif isinstance(recovery_pth, Layer):
                state_dict = recovery_pth.state_dict()

        if 'backend' in state_dict and state_dict['backend'] != 'tensorflow':
            raise RuntimeError(
                'The model archive {0} is a {1}-based model, but current backend is Tensorflow, so cannot load model properly.'.format(
                    file_path, state_dict['backend']))

        if "state_dict" in state_dict.keys():
            pretrained_dict = state_dict['state_dict']
        else:
            pretrained_dict = state_dict

        if check_keys(self._model, pretrained_dict):
            has_abnormal = False
            for key in pretrained_dict.keys():
                value = pretrained_dict[key]
                if is_tensor(value) and any_abnormal_number(value):
                    has_abnormal = True
                    pretrained_dict[key] = where(is_nan(value),
                                                 random_normal_like(value, mean=0, std=0.02).to(get_device()).cast(
                                                     value.dtype), value)
                if is_tensor(value) and ndim(value) == 0:
                    pretrained_dict[key] = to_tensor(value.item())

            if has_abnormal:
                sys.stderr.write(self._model._name + '  has_abnormal detected and  fixed!!\n')
            self._model.load_state_dict(pretrained_dict, strict=False)
            ctx.print('Model loaded!')
            # must switch to evluate first beforeinference or training
            # Dropout and Batch normalization will behavior change!!!

            self._model.eval()
        if "signature" in state_dict.keys() and (
                self._model.signature is None or state_dict['signature'] != self._model.signature):
            self._model.signature = state_dict['signature']
        self._model.to(get_device())

    def merge_grads(self, old_grads, new_grades):
        if isinstance(old_grads, list) and isinstance(new_grades, list) and len(old_grads) == len(new_grades):
            result = []
            for i in range(len(old_grads)):
                old_grad = old_grads[i]
                new_grade = new_grades[i]
                result.append(old_grad + new_grade)
            return result
        else:
            raise ValueError('In tensorflow ,grads should be list in eager mode')

    def train_model(self, train_data, test_data, current_epoch, current_batch, total_epoch, total_batch=None,
                    done=False,
                    is_collect_data=True, is_print_batch_progress=True, is_print_epoch_progress=True,
                    is_print_batch_gradients=True, log_gradients=False, log_weights=False, accumulate_grads=False,
                    is_out_sample_evaluation=False, **kwargs):
        with tf.device(get_device()):
            self.training_context['current_epoch'] = current_epoch
            self.training_context['current_batch'] = current_batch
            self.training_context['total_epoch'] = total_epoch
            self.training_context['total_batch'] = total_batch
            self.training_context['done'] = done
            self.training_context['is_collect_data'] = is_collect_data
            self.training_context['log_gradients'] = log_gradients
            self.training_context['log_weights'] = log_weights
            self.training_context['current_model'] = self._model
            self.training_context['accumulate_grads'] = None

            if self.training_context['current_batch'] == 0:
                if self.training_context['current_epoch'] == 0:
                    # epoch is not the logical inteval for us to control the flow
                    self.training_context['steps'] = 0
                    self._clear_gradient_window()

                    self.training_context['grads_state'] = OrderedDict()
                    self.training_context['grads_state']['first_layer'] = []
                    self.training_context['grads_state']['last_layer'] = []
                self.training_context['is_print_batch_progress'] = is_print_batch_progress
                self.training_context['is_print_epoch_progress'] = is_print_epoch_progress
                self.training_context['print_batch_progress_frequency'] = 1
                self.training_context['print_epoch_progress_frequency'] = 1

            self.training_context['train_data'] = train_data
            self.training_context['test_data'] = test_data
            self.training_context['is_out_sample_evaluation'] = is_out_sample_evaluation

            self.training_context['optimizer'] = self.optimizer
            self.training_context['current_lr'] = self.optimizer.lr
            self.batch_loss_history.collect('epoch', self.training_context['steps'], current_epoch)
            self.batch_metric_history.collect('epoch', self.training_context['steps'], current_epoch)
            is_epoch_end = (self.training_context['current_batch'] == self.training_context['total_batch'] - 1) if \
                self.training_context['total_batch'] is not None else done

            self.do_on_batch_start()
            for callback in self.callbacks:
                callback.on_batch_start(self.training_context)

            train_data, test_data = self.do_on_data_received(train_data, test_data)

            self._reset_loss_terms()
            for callback in self.callbacks:
                callback.on_data_received(self.training_context)

            if (not self.training_context[
                'skip_reset_total_loss']):  # and (not (self.training_context['steps'] + 1) % self.accumulation_steps != 0):
                self.training_context['current_loss'] = to_tensor(0.0, requires_grad=True)

            need_backward = self._should_update_gradients()
            with tf.GradientTape(persistent=self.accumulation_steps > 1) as grad_tape:

                if 'skip_generate_output' not in self.training_context or self.training_context[
                    'skip_generate_output'] == False:
                    if self.output_fn is not None and callable(self.output_fn):
                        self.output_fn(self, self.training_context, is_training=True,
                                       is_autocast_enabled=self.is_autocast_enabled)
                    else:
                        self.do_calculate_forward(is_training=True)


                self.do_calculate_losses()

                for k, v in self._regs.items():
                    this_loss = to_tensor(0.0, requires_grad=True)
                    if 'model' in v.signature.inputs:
                        this_loss = v(self._model) if self.training_context['stop_update'] < 1 else to_tensor(0.0,
                                                                                                              requires_grad=True)
                    elif 'output' in v.signature.inputs:
                        this_loss = try_map_args_and_call(v, self.train_data, self.training_context['data_feed'],
                                                          self.is_autocast_enabled) if self.training_context[
                                                                                           'stop_update'] < 1 else to_tensor(
                            0.0)
                    if any_abnormal_number(this_loss):
                        raise FloatingPointError('Nonfinite regularizer: {}'.format(k))
                    self._add_loss_term(this_loss, 'mean')

                    self.training_context['tmp_losses'].collect(k + '_Loss', self.training_context['steps'],
                                                                this_loss)

                context = self.training_context
                objective = context['current_loss']
                terms = []
                if self.accumulation_steps > 1:
                    summed = context.get('sum_reduced_loss')
                    if summed is not None:
                        terms.append(((None, 'sum'), summed, None))
                        objective -= summed
                    for name, (term, weight) in context.get('normalized_losses', {}).items():
                        terms.append(((name, 'mean'), term, float(weight)))
                        objective -= term
                weight = self._gradient_batch_weight() if self.accumulation_steps > 1 else 1
                terms.append(((None, 'mean'), objective, weight))

            if need_backward:
                owned = {id(variable if isinstance(variable, tf.Variable) else variable.value)
                         for variable in self._model.trainable_variables}
                for loss in self._losses.values():
                    owned.update(id(v) for v in getattr(loss, 'trainable_variables', ()))
                variables = tuple(variable for variable in grad_tape.watched_variables()
                                  if id(variable) in owned)
                accumulated = context.setdefault('gradient_terms', {})
                for name, term, weight in terms:
                    if weight == 0:
                        continue
                    previous, buffer = accumulated.get(name, (0, {}))
                    for gradient, variable in zip(grad_tape.gradient(term, variables), variables):
                        if gradient is None or not variable.trainable:
                            continue
                        value = gradient if weight is None else _scale_gradient(gradient, weight)
                        key = id(variable)
                        if key in buffer:
                            value = _merge_gradients(buffer[key][0], value)
                        buffer[key] = value, variable
                    accumulated[name] = (None if weight is None else previous + weight), buffer
                context['gradient_batches'] = context.get('gradient_batches', 0) + 1
            del grad_tape
            self.do_gradient_update(log_gradients and is_collect_data)
            self.training_context['current_lr'] = self.optimizer.lr

            if log_weights and is_collect_data:
                if isinstance(self._model, Layer):
                    self.log_weight(weghts=self._model.weights)
                elif is_tensor(self._model):
                    self.log_weight(weghts=self._model)

            if is_out_sample_evaluation == True and self.test_data is not None and len(self.test_data) > 0 and \
                    self.training_context['stop_update'] < 1:
                if self.output_fn is not None and callable(self.output_fn):
                    self.output_fn(self, self.training_context, is_training=False,
                                   is_autocast_enabled=self.is_autocast_enabled)
                else:
                    self.do_calculate_forward(is_training=False)
            self.do_calculate_metrics()

            if is_print_batch_progress or is_print_epoch_progress or is_epoch_end:
                for k, v in self.training_context['tmp_metrics'].items():
                    steps, values = self.training_context['tmp_metrics'].get_series(k)

                    # check
                    if len(steps) > 0:
                        self.training_context['metrics'].collect(k, self.training_context['steps'],
                                                                 to_numpy(values).mean())
                self.training_context['tmp_metrics'].reset()

            # ON_BATCH_END
            self.do_on_batch_end()
            # print batch progresss
            if is_print_batch_progress:
                self.print_batch_progress(self.training_context['print_batch_progress_frequency'])
                self.training_context['print_batch_progress_frequency'] = 1

            else:
                self.training_context['print_batch_progress_frequency'] += 1
            if is_out_sample_evaluation == True and self.test_data is not None and len(self.test_data) > 0:
                verbose = []
                for k in self.training_context['out_sample_metrics'].get_keys():
                    test_steps, test_values = self.training_context['out_sample_metrics'].get_series(k)
                    metric_value = test_values[-1]
                    history_metric_value = to_scalar(test_values[-1])
                    verbose.append('{0}: {1}'.format(k,
                                                     adaptive_format(history_metric_value, prev_value=test_values,
                                                                     value_type='metric', name=k)))
                out_sample_evaluation_str = cyan_color(
                    self.training_context['model_name'] + ' ' + 'out-of-sample evaluation: ' + ','.join(verbose))
                ctx.print(out_sample_evaluation_str)
            # self.training_context['steps'] += 1

    def summary(self,inputs=None):
        # self.rebinding_input_output(self._model.input_shape)
        if not hasattr(self._model, '_signature') or self._model._signature is None or self._model._signature.inputs is None or len(self._model._signature.outputs) == 0:
            self._model._signature = get_signature(self._model, self._model._name)
        summary(self._model, input_specs=self.signature.inputs.value_list,inputs=inputs)
        return self

    @property
    def preprocess_flow(self):
        return self._preprocess_flow

    @preprocess_flow.setter
    def preprocess_flow(self, value):
        self._preprocess_flow = value
        if isinstance(self.input_spec, TensorSpec):
            self.input_spec = None

    @property
    def reverse_preprocess_flow(self):
        return_list = []
        return_list.append(reverse_image_backend_adaption)
        for i in range(len(self._preprocess_flow)):
            fn = self._preprocess_flow[-1 - i]
            if fn.__qualname__ == 'normalize.<locals>.img_op':
                return_list.append(unnormalize(fn.mean, fn.std))
        return_list.append(array2image)
        return return_list

    def data_preprocess(self, img_data):
        if not hasattr(self, '_preprocess_flow') or self._preprocess_flow is None:
            self._preprocess_flow = []
        if img_data.ndim == 4:
            return to_numpy([self.data_preprocess(im) for im in img_data])
        if len(self._preprocess_flow) == 0:
            return np.expand_dims(image_backend_adaption(img_data))
        if isinstance(img_data, np.ndarray):
            for fc in self._preprocess_flow:
                if self._model is not None and self.signature is not None and len(
                        self.signature) > 1 and self.input_spec is not None:
                    img_data = fc(img_data, spec=self.input_spec)
                else:
                    img_data = fc(img_data)
            img_data = np.expand_dims(image_backend_adaption(img_data))
            if self.input_spec is None:
                self.input_spec = TensorSpec(shape=tensor_to_shape(to_tensor(img_data), need_exclude_batch_axis=False),
                                             object_type=ObjectType.rgb, name='input')

            return img_data
        else:
            return img_data

    def reverse_data_preprocess(self, img_data: np.ndarray):
        if img_data.ndim == 4:
            return to_numpy([self.reverse_data_preprocess(im) for im in img_data])
        if len(self.reverse_preprocess_flow) == 0:
            return reverse_image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_preprocess_flow:
                img_data = fc(img_data)
            img_data = reverse_image_backend_adaption(img_data)
        return img_data

    def extra_repr(self):
        return ''

    def __str__(self):
        self.__repr__()

    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        # We treat the extra repr like the submodule, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, value in self.__dict__.items():
            if isinstance(value, OrderedDict):
                for subkey, subvalue in value.items():
                    mod_str = repr(subvalue)
                    mod_str = addindent(mod_str, 2)
                    child_lines.append('(' + key + '): ' + mod_str)
            else:
                pass
                # mod_str = repr(value)
                # mod_str = addindent(mod_str, 2)
                # child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self._model.__class__)
        optimizer_attrs = dir(self.optimizer.__class__)
        attrs = list(self.__dict__.keys())+ list(super().__dict__.keys())

        losses = list(self._losses.keys())
        metrics = list(self._metrics.keys())
        regs = list(self._regs.keys())

        constraints = list(self._constraints.keys())
        keys = module_attrs + optimizer_attrs + attrs + losses + metrics + regs + constraints
        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    @property
    def enable_tensorboard(self):
        return self._enable_tensorboard

    @enable_tensorboard.setter
    def enable_tensorboard(self, value):
        self._enable_tensorboard = value
        if value == True:
            if get_backend() == 'pytorch':
                try:
                    from trident.loggers.pytorch_tensorboard import SummaryWriter
                    ctx.try_enable_tensorboard(SummaryWriter(os.path.join(working_directory, 'Logs')))

                except Exception as e:
                    ctx.print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                    ctx.print(e)
                    PrintException()
            elif get_backend() == 'tensorflow':
                try:
                    from trident.loggers.tensorflow_tensorboard import SummaryWriter
                    ctx.try_enable_tensorboard(SummaryWriter(os.path.join(working_directory, 'Logs')))
                except Exception as e:
                    ctx.print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                    ctx.print(e)
                    PrintException()


class MuiltiNetwork(Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self._networks = OrderedDict()

    def add_network(self, name, inputs=None, input_shape=None, output=None, initializer=None):
        if output is not None:
            output.name = name
            if isinstance(output.input_shape, TensorShape):
                input_shape = output.input_shape
            if isinstance(output, Model):
                self._networks[name] = output
            else:
                self._networks[name] = Model(inputs=inputs, input_shape=input_shape, output=output)

    def __getattr__(self, name):
        if name in self.__dict__['_networks']:
            return self.__dict__['_networks'][name]
        elif name in ['steps', 'current_epoch', 'current_batch', 'frame_steps', 'train_steps']:
            if name not in self.__dict__['training_context']:
                self.__dict__['training_context'][name] = -1
            return self.__dict__['training_context'][name]
        elif name in ['_input_shape', '_output_shape', '_class_names', 'class_names', 'output_fn']:
            return self.__dict__[name]

        elif name in ['reverse_preprocess_flow']:
            return self.__getattribute__(name)

        if name == 'signature' or name == '_signature':
            _model = self.__dict__['_model']
            if _model is not None and isinstance(_model, Layer):
                return _model.signature
            else:
                return Signature()
        if 'training_context' in self.__dict__:
            if name in self.__dict__['training_context']:
                return self.__dict__['training_context'][name]
        if '_model' in self.__dict__:
            _model = self.__dict__['_model']

            if _model is not None and name in _model.__dict__['_parameters']:
                return _model.__dict__['_parameters'][name]
            elif _model is not None and name in _model.__dict__['_buffers']:
                return _model.__dict__['_buffers'][name]
            elif _model is not None and name in _model.__dict__['_modules']:
                return _model.__dict__['_modules'][name]
            elif _model is not None and name in _model.__dict__:
                return _model.__dict__[name]
            elif _model is not None and "_" + name in _model.__dict__:
                return _model.__dict__["_" + name]

        if name in self.__dict__:
            return self.__dict__[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name, value):
        if name in ['_networks']:
            object.__setattr__(self, '_networks', value)
        elif name in ['steps', 'current_epoch', 'current_batch', 'frame_steps', 'train_steps']:
            self.__dict__['training_context'][name] = value
            for k, v in self.__dict__['_networks'].items():
                v.training_context[name] = value
        elif name in ['_input_shape', '_output_shape', '_class_names', 'class_names', 'output_fn']:
            object.__setattr__(self, name, value)

        elif name in ['_model']:
            object.__setattr__(self, '_model', value)


        else:

            if name == 'signature' or name == '_signature':
                _model = self.__dict__['_model']
                if _model is not None and isinstance(_model, Layer):
                    object.__setattr__(_model, "_" + 'signature', value)
            if 'training_context' in self.__dict__ and name in self.__dict__['training_context']:
                self.__dict__['training_context'][name] = value
            elif '_model' in self.__dict__ and self.__dict__['_model']:
                _model = self.__dict__['_model']
                if _model is not None and name in _model.__dict__['_parameters']:
                    _model.__dict__['_parameters'][name] = value
                elif _model is not None and name in _model.__dict__['_buffers']:
                    _model.__dict__['_buffers'][name] = value
                elif _model is not None and name in _model.__dict__['_modules']:
                    _model.__dict__['_modules'][name] = value

                elif _model is not None and name in _model.__dict__:
                    object.__setattr__(self.__dict__['_model'], name, value)
                elif _model is not None and "_" + name in _model.__dict__:
                    object.__setattr__(self.__dict__['_model'], "_" + name, value)
                else:
                    object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, value)

    def with_optimizer(self, optimizer, **kwargs):
        for k in self._networks.keys():
            self._networks[k].with_optimizer(optimizer=optimizer, **kwargs)
        return self

    def with_loss(self, loss, loss_weight=1, start_epoch=0, as_metric=False, name='', **kwargs):
        for k in self._networks.keys():
            self._networks[k].with_loss(loss, loss_weight=loss_weight, start_epoch=start_epoch, as_metric=as_metric,
                                        name=name, **kwargs)

        return self

    def with_metric(self, metric, print_only=False, name='', **kwargs):
        for k in self._networks.keys():
            self._networks[k].with_metric(metric, print_only=print_only, name=name, **kwargs)
        return self

    def with_regularizer(self, reg, **kwargs):
        for k in self._networks.keys():
            self._networks[k].with_regularizer(reg, **kwargs)
        return self

    def with_constraint(self, constraint, **kwargs):
        for k in self._networks.keys():
            self._networks[k].with_constraint(constraint, **kwargs)
        return self

    def with_initializer(self, initializer, **kwargs):
        for k in self._networks.keys():
            self._networks[k].with_initializer(initializer, **kwargs)
        return self

    def with_callbacks(self, *callbacks):
        for k in self._networks.keys():
            self._networks[k].with_callbacks(*callbacks)
        return self

    def with_model_save_path(self, save_path, **kwargs):
        for k in self._networks.keys():
            folder, filename, ext = split_path(save_path)
            current_save_path = os.path.join(folder, filename + '_' + k + ext)
            self._networks[k].with_model_save_path(current_save_path, **kwargs)
            self._networks[k].training_context['save_path'] = current_save_path
        return self

    def with_learning_rate_scheduler(self, lr_schedule=None, warmup=0, **kwargs):
        from copy import deepcopy
        for k in self._networks.keys():
            schedule = deepcopy(lr_schedule) if warmup else lr_schedule
            self._networks[k].with_learning_rate_scheduler(schedule, warmup, **kwargs)
        return self

    def with_automatic_mixed_precision_training(self, **kwargs):

        for k in self._networks.keys():
            self._networks[k].with_automatic_mixed_precision_training(**kwargs)
        return self

    def with_grad_clipping(self, clipping_threshold=3.0, **kwargs):
        for k in self._networks.keys():
            self._networks[k].with_grad_clipping(clipping_threshold, **kwargs)
        return self

    def adjust_learning_rate_scheduling(self, index: int, unit='batch', new_value: float = None):
        for k in self._networks.keys():
            self._networks[k].adjust_learning_rate_scheduling(index, unit, new_value)
        return self

    def adjust_learning_rate(self, lr):
        for k in self._networks.keys():
            self._networks[k].adjust_learning_rate(lr)
        return self

    def trigger_when(self, when='on_batch_end', frequency=None, unit='batch', action=None):
        new_callbacks = LambdaCallback(when, frequency=frequency, unit=unit, action=action)
        self.callbacks.append(new_callbacks)
        return self

    def unfreeze_model_scheduling(self, frequency: int, unit='epoch', slice_from=None, slice_to=None, module_name=None):
        for k in self._networks.keys():
            self._networks[k].unfreeze_model_scheduling(frequency, unit, slice_from, slice_to, module_name)
        return self

    def save_model(self, save_path=None, **kwargs):
        for k in self._networks.keys():
            self._networks[k].save_model(self._networks[k].training_context['save_path'], )
        return self

    def do_on_epoch_start(self):
        self.training_context['time_epoch_progress'] = 0
        self.training_context['time_batch_progress'] = 0
        for k in self._networks.keys():
            self._networks[k].do_on_epoch_start()

    def do_on_epoch_end(self):
        self.training_context['time_epoch_progress'] = 0

        total_loss = 0
        for k in self._networks.keys():
            self._networks[k].do_on_epoch_end()
            self.training_context['time_epoch_progress'] += self._networks[k].training_context['time_epoch_progress']
            steps, value = self._networks[k].epoch_loss_history.get_series('total_losses')
            for s, v in zip(steps, value):
                if s == self.training_context['current_epoch']:
                    total_loss += v
        self.epoch_loss_history.collect('total_losses', self.training_context['current_epoch'], total_loss)
        for callback in self.training_context['callbacks']:
            callback.on_epoch_end(self.training_context)

    def do_on_batch_start(self):
        if self.training_context['steps'] == 0:
            self.training_context['time_batch_progress'] = 0
        self.training_context['time_epoch_start'] = time.time()
        self.training_context['time_batch_start'] = time.time()
        # sub-model will do batch start automaticly
        # for k in self._networks.keys():
        #     # self._networks[k].do_on_batch_start()
        #     self._networks[k].training_context['time_epoch_start'] = time.time()
        #     self._networks[k].training_context['time_batch_start'] = time.time()

    def do_on_batch_end(self):
        self.training_context['time_batch_progress'] += (time.time() - self.training_context['time_batch_start'])
        self.training_context['time_epoch_progress'] += (time.time() - self.training_context['time_batch_start'])

    def do_post_gradient_update(self):
        pass

    def print_batch_progress(self, print_batch_progress_frequency):
        if 'max_name_length' not in self.training_context:
            self.training_context['max_name_length'] = len(self.name) + 1
        metric_strings = []
        slice_length = print_batch_progress_frequency // self.training_context['collect_data_inteval']
        for k in self.batch_metric_history.key_list:
            if k != 'epoch':
                metric_value = None
                batch_steps, batch_values = self.batch_metric_history.get_series(k)
                if len(batch_values) == 0:
                    batch_steps, batch_values = self.tmp_metrics.get_series(k)
                    metric_value = np.array(batch_values).mean()
                else:
                    if len(batch_values) > slice_length:
                        metric_value = np.array(batch_values[-1 * slice_length:]).mean()
                    else:
                        metric_value = np.array(batch_values).mean()

                metric_strings.append(
                    '{0}: {1} '.format(k, adaptive_format(metric_value, batch_values, value_type='metric', name=k)))

        loss_value = None
        loss_steps, loss_values = self.batch_loss_history.get_series('total_losses')
        if len(loss_values) == 0:
            loss_value = None
        else:
            if len(loss_values) > slice_length:
                loss_value = to_numpy(loss_values[-1 * slice_length:]).astype(np.float32).mean()
            else:
                loss_value = to_numpy(loss_values).astype(np.float32).mean()
        step_time = self.training_context['time_batch_progress']
        progress_bar(step_time, self.training_context['current_batch'],
                     self.training_context['total_batch'] if self.training_context['total_batch'] is not None else '*',
                     'Loss: {0} | {1} | lr: {2:<10.3e} | epoch: {3}'.format(
                         adaptive_format(loss_value, value_type='loss'), ', '.join(metric_strings),
                         self.training_context['current_lr'],
                         self.training_context['current_epoch']),
                     name=self.name.ljust(self.training_context['max_name_length'] + 1, ' '))
        self.training_context['time_batch_progress'] = 0

    def print_epoch_progress(self, *args, **kwargs):
        if 'max_name_length' not in self.training_context:
            self.training_context['max_name_length'] = len(self.name) + 1
        metric_strings = []
        for net in self._networks.value_list:
            for k in net.epoch_metric_history.key_list:
                if k != 'epoch':
                    metric_strings.append('{0}: {1}'.format(k, adaptive_format(net.epoch_metric_history.get_last(k)[-1],
                                                                               net.batch_metric_history.get_series(k)[
                                                                                   -1], 'metric', k)))

        for k in self.epoch_metric_history.key_list:
            if k != 'epoch':
                metric_strings.append(
                    '{0}: {1}'.format(k, adaptive_format(self.epoch_metric_history.get_last(k)[-1], None, 'metric', k)))

        step_time = self.training_context['time_epoch_progress']
        total_losses = 0
        if 'total_losses' in self.epoch_loss_history:
            total_losses = self.epoch_loss_history['total_losses'][-1][-1]
        progress_bar(step_time, self.training_context['current_epoch'] + 1, self.training_context['total_epoch'],
                     'Loss: {0}| {1} | lr: {2:<10.3e}'.format(adaptive_format(total_losses, value_type='loss'),
                                                              ', '.join(metric_strings),
                                                              self._networks.value_list[0].training_context[
                                                                  'current_lr']),
                     name=self.name.ljust(self.training_context['max_name_length'] + 1, ' '))
        self.training_context['time_epoch_progress'] = 0


class ImageClassificationModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(ImageClassificationModel, self).__init__(inputs, input_shape, output)

        self._class_names = []
        self._idx2lab = {}
        self._lab2idx = {}
        if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
            self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.inputs.value_list) > 0 and \
                self._model.signature.inputs.value_list[0].object_type is None:
            self._model.signature.inputs.value_list[0].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.outputs.value_list) > 0 and \
                self._model.signature.outputs.value_list[0].object_type is None:
            self._model.signature.outputs.value_list[0].object_type = ObjectType.classification_label

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, value):
        if self._class_names != value:
            self._class_names = value
            self._lab2idx = {v: k for k, v in enumerate(self._class_names)}
            self._idx2lab = {k: v for k, v in enumerate(self._class_names)}

    def index2label(self, idx: int):
        if self._idx2lab is None or len(self._idx2lab.items()) == 0:
            raise ValueError('You dont have proper mapping class names')
        elif idx not in self._idx2lab:
            raise ValueError('Index :{0} is not exist in class names'.format(idx))
        else:
            return self._idx2lab[idx]

    def label2index(self, label):
        if self._lab2idx is None or len(self._lab2idx.items()) == 0:
            raise ValueError('You dont have proper mapping class names')
        elif label not in self._lab2idx:
            raise ValueError('label :{0} is not exist in class names'.format(label))
        else:
            return self._lab2idx[label]

    def infer_single_image(self, img, topk=1):
        if self._model.built:
            if isinstance(self._model, Layer):
                self._model.eval()

            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            for func in self.preprocess_flow:
                if (inspect.isfunction(func) or isinstance(func, Transform)) and func is not image_backend_adaption:
                    img = func(img, spec=self._model.input_spec)
            img = image_backend_adaption(img)
            inp = to_tensor(np.expand_dims(img, 0)).to(self._model.device)
            result = self._model(inp)
            result = to_numpy(result)[0]
            if self.class_names is None or len(self.class_names) == 0:
                return result
            else:
                # argresult = np.argsort(result)
                # argresult1 =argresult[::-1]
                answer = OrderedDict()
                idxs = list(np.argsort(result)[::-1][:topk])
                for idx in idxs:
                    prob = result[idx]
                    answer[self.index2label(idx)] = (idx, prob)
                # idx=int(np.argmax(result,-1)[0])

                return answer
        else:
            raise ValueError('the model is not built yet.')


class ImageRegressionModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(ImageRegressionModel, self).__init__(inputs, input_shape, output)
        if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
            self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.inputs.value_list) > 0 and \
                self._model.signature.inputs.value_list[0].object_type is None:
            self._model.signature.inputs.value_list[0].object_type = ObjectType.rgb

    def infer_single_image(self, img):
        if self._model.built:
            self._model.eval()
            img = image2array(img)
            img_shp = img.shape
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            rescale_scale = 1.0
            for func in self.preprocess_flow:
                if (inspect.isfunction(func) or isinstance(func, Transform)) and func is not image_backend_adaption:
                    img = func(img, spec=self._model.input_spec)
                    if (inspect.isfunction(func) and func.__qualname__ == 'resize.<locals>.img_op') or (
                            isinstance(func, Transform) and func.name == 'resize'):
                        rescale_scale = func.scale
            img = image_backend_adaption(img)
            if isinstance(self._model, Layer):
                inp = to_tensor(np.expand_dims(img, 0)).to(self._model.device).to(self._model.weights[0].data.dtype)
                result = self._model(inp)
                result = to_numpy(result)
                return result.astype(np.int32)
            else:

                raise ValueError('the model is not layer.')

        else:
            raise ValueError('the model is not built yet.')


class ImageDetectionModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(ImageDetectionModel, self).__init__(inputs, input_shape, output)
        object.__setattr__(self, 'detection_threshold', 0.5)
        object.__setattr__(self, 'nms_threshold', 0.3)
        if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
            self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.inputs.value_list) > 0 and \
                self._model.signature.inputs.value_list[0].object_type is None:
            self._model.signature.inputs.value_list[0].object_type = ObjectType.rgb

    def infer_single_image(self, img, scale=1):
        if isinstance(self._model, Layer) and self._model.built:
            self._model.to(self.device)
            self._model.eval()

            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            rescale_scale = 1
            for func in self.preprocess_flow:
                if (inspect.isfunction(func) or isinstance(func, Transform)) and func is not image_backend_adaption:
                    img = func(img, spec=self._model.input_spec)
                    if (inspect.isfunction(func) and func.__qualname__ == 'resize.<locals>.img_op') or (
                            isinstance(func, Transform) and func.name == 'resize'):
                        rescale_scale = func.scale
            img = image_backend_adaption(img)
            inp = to_tensor(np.expand_dims(img, 0)).to(get_device()).to(self._model.weights[0].value().dtype)
            result = self._model(inp)
            self._model._set_inputs(to_tensor(np.expand_dims(img, 0)))
            bboxes = self.generate_bboxes(*result, threshold=self.detection_threshold, scale=scale)
            bboxes = self.nms(bboxes)
            # idx=int(np.argmax(result,-1)[0])
            return bboxes
        else:
            raise ValueError('the model is not built yet.')

    def generate_bboxes(self, *outputs, threshold=0.5, scale=1):
        raise NotImplementedError

    def nms(self, bboxes):
        raise NotImplementedError


class ImageSegmentationModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(ImageSegmentationModel, self).__init__(inputs, input_shape, output)
        self.palette = OrderedDict()
        self._class_names = []
        self._idx2lab = {}
        self._lab2idx = {}
        if self.input_spec is not None and self.input_spec.object_type is None:
            self.input_spec.object_type = ObjectType.rgb

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, value):
        if self._class_names is not None and self._class_names != value:
            self._class_names = value
            self._lab2idx = {v: k for k, v in enumerate(self._class_names)}
            self._idx2lab = {k: v for k, v in enumerate(self._class_names)}

    def index2label(self, idx: int):
        if self._idx2lab is None or len(self._idx2lab.items()) == 0:
            raise ValueError('You dont have proper mapping class names')
        elif idx not in self._idx2lab:
            raise ValueError('Index :{0} is not exist in class names'.format(idx))
        else:
            return self._idx2lab[idx]

    def label2index(self, label):
        if self._lab2idx is None or len(self._lab2idx.items()) == 0:
            raise ValueError('You dont have proper mapping class names')
        elif label not in self._lab2idx:
            raise ValueError('label :{0} is not exist in class names'.format(label))
        else:
            return self._lab2idx[label]

    def infer_single_image(self, img):
        if isinstance(self._model, Layer) and self._model.built:
            self._model.eval()
            if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
                self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            rescale_scale = 1.0
            for func in self.preprocess_flow:
                if (inspect.isfunction(func) or isinstance(func, Transform)) and not isinstance(func,
                                                                                                image_backend_adaption):
                    img = func(img, spec=self._model.input_spec)

                    if (inspect.isfunction(func) and func.__qualname__ == 'resize.<locals>.img_op') or (
                            isinstance(func, Transform) and func.name == 'resize'):
                        rescale_scale = func.scale

            img = image_backend_adaption(img)
            inp = cast(to_tensor(np.expand_dims(img, 0)).to(self._model.get_root().device),
                       self._model.weights[0].data.dtype)
            result = argmax(self._model(inp)[0], axis=-1)
            result = to_numpy(result)
            if self.class_names is None or len(self.class_names) == 0:
                return result
            else:
                if len(self.palette) > 0:
                    color_result = color2label(result, self.palette)
                    return color_result
                else:
                    return result
        else:
            raise ValueError('the model is not built yet.')


class ImageGenerationModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(ImageGenerationModel, self).__init__(inputs, input_shape, output)
        if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
            self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.inputs.value_list) > 0 and \
                self._model.signature.inputs.value_list[0].object_type is None:
            self._model.signature.inputs.value_list[0].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.outputs.value_list) > 0 and \
                self._model.signature.outputs.value_list[0].object_type is None:
            self._model.signature.outputs.value_list[0].object_type = ObjectType.rgb

    @property
    def reverse_preprocess_flow(self):
        return_list = []
        for i in range(len(self.preprocess_flow)):
            fn = self.preprocess_flow[-1 - i]
            if fn.__qualname__ == 'normalize.<locals>.img_op':
                return_list.append(unnormalize(fn.mean, fn.std))
        return return_list

    def infer_single_image(self, img):
        if self._model.built:
            self._model.eval()
            if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
                self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]

            for func in self.preprocess_flow:
                if (inspect.isfunction(func) or isinstance(func, Transform)) and func is not image_backend_adaption:
                    img = func(img, spec=self._model.input_spec)
            img = image_backend_adaption(img)
            inp = to_tensor(np.expand_dims(img, 0)).to(self._model.device).to(self._model.weights[0].data.dtype)
            result = self._model(inp)
            result = to_numpy(result)[0]

            for func in self.reverse_preprocess_flow:
                if inspect.isfunction(func):
                    result = func(result)
            result = array2image(result)
            return result


class FaceLandmarkModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(FaceLandmarkModel, self).__init__(inputs, input_shape, output)

        if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
            self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.inputs.value_list) > 0 and \
                self._model.signature.inputs.value_list[0].object_type is None:
            self._model.signature.inputs.value_list[0].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.outputs.value_list) > 0 and \
                self._model.signature.outputs.value_list[0].object_type is None:
            self._model.signature.outputs.value_list[0].object_type = ObjectType.landmarks

    def infer_single_image(self, img):
        if self._model.built:
            self._model.eval()
            if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
                self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
            img = image2array(img)
            img_shp = img.shape

            if img.shape[-1] == 4:
                img = img[:, :, :3]
            rescale_scale = 1.0
            for func in self.preprocess_flow:
                if (inspect.isfunction(func) or isinstance(func, Transform)) and not isinstance(func,
                                                                                                image_backend_adaption):
                    img = func(img, spec=self._model.input_spec)
                    if (inspect.isfunction(func) and func.__qualname__ == 'resize.<locals>.img_op') or (
                            isinstance(func, Transform) and func.name == 'resize'):
                        rescale_scale = func.scale

            img = image_backend_adaption(img)
            if isinstance(self._model, Layer):
                inp = to_tensor(np.expand_dims(img, 0)).to(self._model.device).to(self._model.weights[0].data.dtype)
                result = self._model(inp)
                result = to_numpy(result) / rescale_scale
                result[:, :, 0::2] = clip(result[:, :, 0::2], 0, img_shp[1])
                result[:, :, 1::2] = clip(result[:, :, 1::2], 0, img_shp[0])
                return result.astype(np.int32)
            else:

                raise ValueError('the model is not layer.')

        else:
            raise ValueError('the model is not built yet.')


class FaceRecognitionModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(FaceRecognitionModel, self).__init__(inputs, input_shape, output)

        self.preprocess_flow = []
        if self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type is None:
            self._model._signature.inputs[self._model._signature.inputs.key_list[0]].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.inputs.value_list) > 0 and \
                self._model.signature.inputs.value_list[0].object_type is None:
            self._model.signature.inputs.value_list[0].object_type = ObjectType.rgb
        if self._model.signature is not None and len(self._model.signature.outputs.value_list) > 0 and \
                self._model.signature.outputs.value_list[0].object_type is None:
            self._model.signature.outputs.value_list[0].object_type = ObjectType.embedding

    def infer_single_image(self, img):

        if isinstance(self._model, Layer) and self._model.built:
            self._model.eval()

            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            rescale_scale = 1.0
            for func in self.preprocess_flow:
                if (inspect.isfunction(func) or isinstance(func, Transform)) and func is not image_backend_adaption:
                    img = func(img, spec=self._model.input_spec)
                    if (inspect.isfunction(func) and func.__qualname__ == 'resize.<locals>.img_op') or (
                            isinstance(func, Transform) and func.name == 'resize'):
                        rescale_scale = func.scale
            img = image_backend_adaption(img)
            inp = to_tensor(np.expand_dims(img, 0)).to(self._model.device).to(self._model.weights[0].data.dtype)
            result = self._model(inp)[0]
            embedding = to_numpy(result)
            return embedding

        else:
            raise ValueError('the model is not built yet.')


class LanguageModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(LanguageModel, self).__init__(inputs, input_shape, output)
        self.vocabs = None
        self.preprocess_flow = []

    def save_onnx(self, save_path=None, **kwargs):
        pass

    def save_weights(self, save_path=None, **kwargs):
        for callback in self.training_context['callbacks']:
            callback.on_model_saving_start(self.training_context)

        if save_path is not None:
            self._model.save_weights(save_path)
        elif 'save_path' in self.training_context and self.training_context['save_path'] is not None:
            self._model.save_weights(self.training_context['save_path'])

        else:
            if 'Models' is not None and len('Models') > 1 and not os.path.exists('Models'):
                try:
                    os.makedirs('Models')
                except Exception as e:
                    pass
            save_full_path = os.path.join('Models/', 'model_{0}_epoch{1}.h5'.format(self._model.__name__,
                                                                                    self.training_context[
                                                                                        'current_epoch']))
            self._model.save_weights(self.training_context['save_full_path'])

        self._model.train()

    def load_model(self, file_path, **kwargs):
        ctx.print('Loading pretrained model from {}'.format(file_path))
        folder, filename, ext = split_path(file_path)
        if filename == '':
            filename = self.name
        state_dict = None
        pretrained_dict = None
        if ext == '.pth.tar':
            state_dict = load_pthtar(file_path)
        elif ext == '.pth':
            load_path = file_path
            if not os.path.exists(file_path):
                if os.path.exists(file_path.replace(ext, '.pth.tar')):
                    load_path = file_path.replace(ext, '.pth.tar')
                elif os.path.exists(os.path.join(working_directory, filename + ext)):
                    load_path = os.path.join(working_directory, filename + ext)
            recovery_pth = load_pthtar(load_path)

            if isinstance(recovery_pth, dict):
                state_dict = recovery_pth

            elif isinstance(recovery_pth, Layer):
                state_dict = recovery_pth.state_dict()

        if 'backend' in state_dict and state_dict['backend'] != 'tensorflow':
            raise RuntimeError(
                'The model archive {0} is a {1}-based model, but current backend is Tensorflow, so cannot load model properly.'.format(
                    file_path, state_dict['backend']))

        if "state_dict" in state_dict.keys():
            pretrained_dict = state_dict['state_dict']
        else:
            pretrained_dict = state_dict

        if check_keys(self._model, pretrained_dict):
            has_abnormal = False
            for key in pretrained_dict.keys():
                value = pretrained_dict[key]
                if is_tensor(value) and any_abnormal_number(value):
                    has_abnormal = True
                    pretrained_dict[key] = where(is_nan(value),
                                                 random_normal_like(value, mean=0, std=0.02).to(get_device()).cast(
                                                     value.dtype), value)
                if is_tensor(value) and ndim(value) == 0:
                    pretrained_dict[key] = to_tensor(value.item())

            if has_abnormal:
                sys.stderr.write(self._model._name + '  has_abnormal detected and  fixed!!\n')
            self._model.load_state_dict(pretrained_dict, strict=False)
            ctx.print('Model loaded!')
            # must switch to evluate first beforeinference or training
            # Dropout and Batch normalization will behavior change!!!

            self._model.eval()
        if "signature" in state_dict.keys() and (
                self._model.signature is None or state_dict['signature'] != self._model.signature):
            self._model.signature = state_dict['signature']
        self._model.to(get_device())
