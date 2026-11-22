"""Modelbase"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
# import pysnooper
# from torch import autograd
import copy
import inspect
import itertools
from contextlib import nullcontext
import json
import os
import shutil
import sys
import time
import uuid
from typing import Callable

import numpy as np

from trident import context
from trident.context import split_path, make_dir_if_need, sanitize_path

from trident.backend import dtype
from trident.backend.decorators import *
from trident.backend.common import to_list, format_time, get_terminal_size, get_backend, \
    PrintException, OrderedDict, adaptive_format, cyan_color, get_class, camel2snake, is_instance,unpack_singleton
from trident.backend.opencv_backend import array2image
from trident.backend.tensorspec import *

from trident.data.vision_transforms import Unnormalize
from trident.loggers.history import HistoryBase

ctx = context._context()
working_directory = ctx.working_directory

if get_backend() == 'pytorch':
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *

elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *

elif get_backend() == 'jax':
    from trident.backend.jax_backend import *
    from trident.backend.jax_ops import *

__all__ = ['progress_bar', 'ModelBase']

_, term_width = get_terminal_size()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


class ModelBase(object):
    def __init__(self, inputs=None, input_shape=None, output=None, name='', **kwargs):
        if isinstance(inputs, tuple) and isinstance(inputs[0], int):
            input_shape, inputs = inputs, input_shape
        self.batch_index = 0
        self.filter_index = 1 if get_backend() == 'pytorch' else -1

        self._model = None
        self.output_fn: Callable = None
        self.accumulation_steps = 1

        self.lr_scheduler = None
        self._losses = OrderedDict()
        self._metrics = OrderedDict()
        self.optimizer = None
        self.loss_weights = OrderedDict()

        self._regs = OrderedDict()
        self._constraints = OrderedDict()

        self._preprocess_flow = []

        self.current_save_path = None

        self.weights_history = []
        self.gradients_history = []
        self.input_history = []
        self.target_history = []
        # self.callbacks = []
        self.is_autocast_enabled = False
        self.gradscaler = None
        self.grad_clipping_by_norm = False
        self.grad_clipping_threshold = None
        self.use_output_as_loss = False
        self.training_context = {
            'losses': HistoryBase('losses'),
            'metrics': HistoryBase('metrics'),
            'epoch_losses': HistoryBase('epoch_losses'),
            'epoch_metrics': HistoryBase('epoch_metrics'),
            'grads_state': OrderedDict(),
            'tmp_losses': HistoryBase('tmp_losses'),
            'tmp_metrics': HistoryBase('tmp_metrics'),
            'out_sample_metrics': HistoryBase('out_sample_metrics'),
            'print_progress_frequency': 10,
            'print_batch_progress_frequency': 0,
            'collect_data_inteval': 1,
            'print_progress_unit': 'batch',
            'optimizer': None,
            'warmup': 0,
            'grads': None,
            'stop_training': False,  # stop training
            'total_epoch': -1,  # current_epoch
            'total_batch': -1,  # current_batch
            'current_epoch': -1,  # current_epoch
            'current_batch': -1,  # current_batch
            'data_feed': None,
            'current_model': None,  # current model
            'train_data': OrderedDict(),
            'test_data': OrderedDict(),
            'current_input': None,  # current input
            'current_target': None,  # current target
            'steps': 0,
            'current_output': None,  # current output
            'current_loss': to_tensor(0.0, requires_grad=True),  # current loss
            'best_metric': None,  # current loss
            'best_model': None,  # current model
            'loss_history': None, 'metric_history': None, 'base_lr': None,  # current loss
            'current_lr': None,  # current loss
            'save_path': os.path.join(working_directory, 'Models'),
            'is_collect_data': True,
            'callbacks': [],
            'stop_update': 0,
            'retain_graph': False,
            'skip_generate_output': False,
            'skip_reset_total_loss': False}
        if name is not None:
            self.name = name

        if output is not None and (inputs is not None or input_shape is not None):
            self._initial_graph(inputs, input_shape, output)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if isinstance(value, Layer):
            if not value.built:
                value.build(self.model.input_shape)
            self._model = value
        elif isinstance(value, np.ndarray) or 'tensor' in value.__name__.lower():
            self._initial_graph(input_shape=int_shape(value), output=to_tensor(value))
        else:
            raise ValueError('Only Layer, Module, Image and Tensor can be valid model')

    @property
    def inputs(self):
        if self._model is not None and callable(self._model):
            if hasattr(self._model, '_signature'):
                self._model._signature = get_signature(self._model)
            return self._model._signature.inputs
        else:
            return None

    @inputs.setter
    def inputs(self, value):
        if self._model is not None and callable(self._model):
            if hasattr(self._model, '_signature'):
                self._model._signature = get_signature(self._model)
            self._model._signature.inputs = value

    @property
    def outputs(self):
        if self._model is not None and callable(self._model):
            if hasattr(self._model, '_signature'):
                self._model._signature = get_signature(self._model)
            return self._model._signature.outputs
        else:
            return None

    @outputs.setter
    def outputs(self, value):
        if self._model is not None and callable(self._model):
            if hasattr(self._model, '_signature'):
                self._model._signature = get_signature(self._model)
            self._model._signature.outputs = value

    @property
    def targets(self):
        return NotImplemented

    @targets.setter
    def targets(self, value):
        pass

    @property
    def warmup(self):
        return self.training_context['warmup']

    @warmup.setter
    def warmup(self, value):
        self.training_context['warmup'] = value

    @property
    def batch_metric_history(self):
        return self.training_context['metrics']

    @batch_metric_history.setter
    def batch_metric_history(self, value):
        self.training_context['metrics'] = value

    @property
    def batch_loss_history(self):
        return self.training_context['losses']

    @batch_loss_history.setter
    def batch_loss_history(self, value):
        self.training_context['losses'] = value

    @property
    def epoch_metric_history(self):
        return self.training_context['epoch_metrics']

    @epoch_metric_history.setter
    def epoch_metric_history(self, value):
        self.training_context['epoch_metrics'] = value

    @property
    def epoch_loss_history(self):
        return self.training_context['epoch_losses']

    @epoch_loss_history.setter
    def epoch_loss_history(self, value):
        self.training_context['epoch_losses'] = value

    @property
    def losses(self):
        return self._losses

    @property
    def metrics(self):
        return self._metrics

    def update_signature(self, arg_names):
        if self.model is not None and hasattr(self.model, 'signature') and self._signature != self.model.signature:
            self._signature = self.model.signature

        if self._signature is not None and len(self._signature.inputs.key_list) == len(arg_names):
            old_inputs = copy.deepcopy(self._signature.inputs)
            self._signature.inputs = OrderedDict()
            self.inputs = OrderedDict()
            for i in range(len(old_inputs)):
                spec = old_inputs[i]
                spec.name = arg_names[i]
                self._signature.inputs[arg_names[i]] = spec
                self.inputs[arg_names[i]] = spec


        elif self._signature is not None and len(self._signature.inputs.key_list) + len(
                self._signature.outputs.key_list) == len(arg_names):

            old_inputs = copy.deepcopy(self._signature.inputs)
            old_outputs = copy.deepcopy(self._signature.outputs)
            self._signature.inputs = OrderedDict()
            self._signature.outputs = OrderedDict()
            self.inputs = OrderedDict()
            self._outputs = OrderedDict()
            self._targets = OrderedDict()
            for i in range(len(old_inputs)):
                spec = old_inputs[i]
                spec.name = arg_names[i]
                self._signature.inputs[arg_names[i]] = spec
                self.inputs[arg_names[i]] = spec

            for i in range(len(old_inputs)):
                target_arg = arg_names[i].replace('output', 'target')
                if 'target' not in target_arg:
                    target_arg = 'target_' + target_arg

                spec = old_outputs[i]
                spec.name = arg_names[i + len(old_inputs)]
                self._signature.outputs[arg_names[i + len(old_inputs)]] = spec
                self._outputs[arg_names[i + len(old_inputs)]] = spec
                self._targets[target_arg] = spec

            ctx.print(self._signature)
        elif not isinstance(arg_names, (list, tuple)):
            raise ValueError('arg_names should be list or tuple')
        elif len(self._signature) != len(arg_names):
            raise ValueError('data feed and arg_names should be the same length')

    @property
    def preprocess_flow(self):
        return self._preprocess_flow

    @preprocess_flow.setter
    def preprocess_flow(self, value):
        self._preprocess_flow = value

    @property
    def reverse_preprocess_flow(self):
        return_list = [reverse_image_backend_adaption]
        for i in range(len(self._preprocess_flow)):
            fn = self._preprocess_flow[-1 - i]
            if (inspect.isfunction(fn) and fn.__qualname__ == 'normalize.<locals>.img_op') or (
                    fn.__class__.__name__ == 'Normalize'):
                return_list.append(Unnormalize(fn.mean, fn.std))
        return_list.append(array2image)
        return return_list

    def data_preprocess(self, img_data):

        if img_data.ndim == 4:
            return to_numpy([self.data_preprocess(im) for im in img_data])
        if len(self.preprocess_flow) == 0:
            return np.expand_dims(image_backend_adaption(img_data), 0)
        if isinstance(img_data, np.ndarray):
            for fc in self.preprocess_flow:
                if self._model is not None and self.input_spec is not None:
                    img_data = fc(img_data, spec=self.input_spec)

                else:
                    img_data = fc(img_data)
            img_data = np.expand_dims(image_backend_adaption(img_data), 0)
            if self.input_spec is None:
                self._signature = get_signature(self._model)
                self._signature.inputs[self._signature.inputs.key_list[0]].shape = tensor_to_shape(to_tensor(img_data),
                                                                                                   need_exclude_batch_axis=False)
                self._signature.inputs[self._signature.inputs.key_list[0]].object_type = ObjectType.rgb
                self._signature.inputs[self._signature.inputs.key_list[0]].dtype = ctx.float_dtype

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

    def _initial_graph(self, inputs=None, input_shape=None, output=None):
        pass

    def complie(self, optimizer="AdaBelief",
                loss=None,
                metrics=None,
                loss_weights=None,
                **kwargs
                ):
        raise NotImplementedError

    def fit(self,
            x=None,
            y=None,
            batch_size=8,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=8,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            ):
        from trident.data.data_provider import DataProvider
        from trident.data.dataset import Iterator, LabelDataset, NumpyDataset
        from trident.optims.trainers import TrainingPlan

        if x is None or y is None or len(x) == 0 or len(x) != len(y):
            raise ValueError("fit requires aligned, nonempty x and y")
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in [0, 1)")
        if validation_freq < 1 or (validation_steps is not None and validation_steps < 1):
            raise ValueError("validation frequency and steps must be positive")
        if validation_data is None and validation_split:
            count = int(len(x) * validation_split)
            if count == 0:
                raise ValueError("validation_split would produce an empty validation set")
            boundary = len(x) - count
            validation_data = x[boundary:], y[boundary:]
            x, y = x[:boundary], y[:boundary]

        def make_iterator(features, labels, size, randomized):
            features = to_numpy(features) if is_tensor(features) else features
            labels = to_numpy(labels) if is_tensor(labels) else labels
            array = np.asarray(labels)
            categorical = array.ndim == 1 and array.dtype.kind in "iubUS"
            target = (LabelDataset(labels=labels, symbol="target") if categorical
                      else NumpyDataset(data=labels, symbol="target"))
            return Iterator(data=NumpyDataset(data=features, symbol="input"), label=target,
                            batch_size=size, is_shuffle=randomized,
                            buffer_size=max_queue_size, workers=workers)

        train = make_iterator(x, y, batch_size, shuffle)
        held_out = None
        if validation_data is not None:
            val_x, val_y = validation_data
            if not len(val_x) or len(val_x) != len(val_y):
                raise ValueError("validation_data requires aligned, nonempty x and y")
            held_out = make_iterator(val_x, val_y, validation_batch_size or batch_size, False)
            if isinstance(train.label, LabelDataset) and train.label._lab2idx and isinstance(held_out.label, LabelDataset):
                if not set(held_out.label._lab2idx).issubset(train.label._lab2idx):
                    raise ValueError("validation_data contains labels absent from training data")
                held_out.label.binding_class_names(train.label.class_names[train.label.__default_language__])
        provider = DataProvider(traindata=train, testdata=held_out, batch_size=batch_size)
        plan = (TrainingPlan().add_training_item(self).with_data_loader(provider)
                .with_batch_size(batch_size).repeat_epochs(epochs)
                .print_progress_scheduling(1, unit='epoch')
                .out_sample_evaluation_scheduling(validation_freq, unit='epoch', on_epoch_end=False)
                .save_model_scheduling(1, unit='epoch'))
        if callbacks:
            plan.with_callbacks(*callbacks)
        if held_out is not None:
            # The legacy provider normally keeps train and test batch sizes in sync.
            plan.fit_validation = (held_out, validation_batch_size or batch_size, validation_steps)
        plan.start_now()
        return self

    def _evaluate_fit_batches(self, batches, limit):
        """Evaluate held-out samples without updating weights or training metrics."""
        previous_test = self.training_context['test_data']
        was_training = getattr(self._model, 'training', True)
        totals = {name: 0.0 for name in self._metrics}
        samples = 0
        context = nullcontext()
        if get_backend() == 'pytorch':
            import torch
            context = torch.no_grad()
        try:
            self._model.eval()
            with context:
                for batch in itertools.islice(batches, limit):
                    data = OrderedDict((spec.name, to_tensor(value)) for spec, value in batch.items())
                    size = len(next(iter(data.values())))
                    self.training_context['test_data'] = data
                    if self.output_fn is None:
                        self.do_calculate_forward(is_training=False)
                    else:
                        self.output_fn(model=self, training_context=self.training_context,
                                       is_training=False, is_autocast_enabled=self.is_autocast_enabled)
                    for name, metric in self._metrics.items():
                        value = try_map_args_and_call(metric, data, self.training_context['data_feed'],
                                                      self.is_autocast_enabled)
                        totals[name] += float(np.mean(to_numpy(value))) * size
                    samples += size
            if samples:
                for name, total in totals.items():
                    self.training_context['out_sample_metrics'].collect(name, self.steps, total / samples)
        finally:
            self.training_context['test_data'] = previous_test
            if was_training:
                self._model.train()

    def fit_generator(self,
                      generator,
                      batch_size=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_split=0.0,
                      validation_data=None,
                      shuffle=True,
                      class_weight=None,
                      sample_weight=None,
                      initial_epoch=0,
                      steps_per_epoch=None,
                      validation_steps=None,
                      validation_batch_size=None,
                      validation_freq=1,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      ):
        raise NotImplementedError

    def __getattr__(self, name):
        if name in ['training_context', '_input_shape', '_output_shape', '_class_names', 'class_names', 'output_fn',
                    'optimizer']:
            return self.__dict__[name]
        elif name in ['reverse_preprocess_flow']:
            return self.__getattribute__(name)
        elif name in ['name']:
            if '_model' in self.__dict__:
                _model = self.__dict__['_model']
                if isinstance(_model, Layer) or hasattr(_model, 'name'):
                    return _model._name if hasattr(_model, '_name') else _model.name
                elif is_tensor(_model):
                    object.__setattr__(self, name, 'model_' + str(uuid.uuid4().node))
                    return self.__dict__[name]
        elif name == 'signature' or name == '_signature':
            _model = self.__dict__['_model']
            if _model is not None and hasattr(_model, '_signature'):
                return _model._signature
            elif _model is not None and hasattr(_model, 'signature'):
                return _model.signature
            else:
                return None
        if 'training_context' in self.__dict__:
            if name in self.__dict__['training_context']:
                return self.__dict__['training_context'][name]
        if '_model' in self.__dict__:
            _model = self.__dict__['_model']
            if isinstance(_model, Layer):
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
        if name in ['_input_shape', '_output_shape', '_class_names', 'class_names', 'output_fn', 'optimizer']:
            object.__setattr__(self, name, value)
        elif name in ['trainable']:
            _model = self.__dict__['_model']
            if hasattr(_model, 'trainable'):
                _model.trainable = value

        elif name in ['_model']:
            object.__setattr__(self, '_model', value)
        elif name in ['name']:
            if '_model' in self.__dict__:
                _model = self.__dict__['_model']
                if isinstance(_model, Layer):
                    _model._name = value
                    if hasattr(_model, '_name'):
                        _model.name = value

                elif is_tensor(_model):
                    object.__setattr__(self, name, value)

        else:

            if name == 'signature' or name == '_signature':
                _model = self.__dict__['_model']
                if _model is not None:
                    object.__setattr__(self.__dict__['_model'], "_" + name, value)

            if 'training_context' in self.__dict__ and name in self.__dict__['training_context']:
                self.__dict__['training_context'][name] = value
            elif '_model' in self.__dict__ and self.__dict__['_model'] is not None:
                _model = self.__dict__['_model']
                if isinstance(_model, Layer):
                    if _model is not None and name in _model.__dict__['_parameters']:
                        _model.__dict__['_parameters'][name] = value
                    elif _model is not None and name in _model.__dict__['_modules']:
                        _model.__dict__['_modules'][name] = value
                    elif _model is not None and name in _model.__dict__['_buffers']:
                        _model.__dict__['_buffers'][name] = value
                    elif _model is not None and name in _model.__dict__:
                        object.__setattr__(self.__dict__['_model'], name, value)
                    elif _model is not None and "_" + name in _model.__dict__:
                        object.__setattr__(self.__dict__['_model'], "_" + name, value)
                    else:
                        object.__setattr__(self, name, value)
                else:
                    object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, value)

    def __call__(self, *x, **kwargs):
        if is_tensor(self._model):
            return self._model
        else:
            return self._model(*x, **kwargs)

    def with_optimizer(self, optimizer, **kwargs):
        return self

    def with_loss(self, loss, loss_weight=1, start_epoch=0, as_metric=False, name='', **kwargs):
        return self

    def with_metric(self, metric, name='', print_only=False, **kwargs):
        return self

    def with_regularizer(self, reg, reg_weight=None, **kwargs):
        return self

    def with_constraint(self, constraint, **kwargs):
        return self

    def with_initializer(self, initializer, **kwargs):
        return self

    def with_model_save_path(self, save_path, **kwargs):
        return self

    def with_callbacks(self, *callbacks):
        if len(self.callbacks) == 0:
            self.callbacks = to_list(callbacks)
        else:
            self.callbacks.extend(callbacks)
        self.training_context['callbacks'] = self.callbacks
        return self

    def with_learning_rate_scheduler(self, lr_schedule=None, warmup=0, **kwargs):
        """Configure a main LR callback and optional warmup policy (or integer epochs)."""
        from trident.callbacks.lr_schedulers import AdjustLRCallbackBase, get_lr_scheduler
        from trident.callbacks.warmup import WarmupScheduler, resolve_warmup

        warmup = resolve_warmup(warmup, self.optimizer)
        if lr_schedule is None and not warmup:
            return self
        if isinstance(lr_schedule, str):
            name = lr_schedule
            lr_schedule = get_lr_scheduler(name)
            if lr_schedule is None:
                raise ValueError("unknown learning rate scheduler: " + name)
        if callable(lr_schedule):
            lr_schedule = lr_schedule(**kwargs)
        elif kwargs:
            raise TypeError("scheduler options require a scheduler factory")
        if lr_schedule is not None and not isinstance(lr_schedule, AdjustLRCallbackBase):
            raise TypeError("expected a Trident learning rate scheduler")
        if warmup:
            lr_schedule = WarmupScheduler(warmup, self.optimizer, lr_schedule)
        if self.lr_scheduler is not None and (warmup or isinstance(self.lr_scheduler, WarmupScheduler)):
            self.callbacks[:] = [cb for cb in self.callbacks if cb is not self.lr_scheduler]
        self.lr_scheduler = lr_schedule
        self.warmup = warmup
        if warmup:
            self.callbacks.insert(0, lr_schedule)
        else:
            self.callbacks.append(lr_schedule)
        return self

    def with_automatic_mixed_precision_training(self, **kwargs):
        """Enable automatic mixed precision training
            only enable when using pytorch 1.6 (or higher) as backend and cuda is available.

        Args:
            **kwargs ():

        Returns:
            the model self

        """
        return self

    def with_grad_clipping(self, **kwargs):
        """Enable grad clipping


               Args:
                   **kwargs ():

               Returns:
                   the model self

               """
        return self

    def with_accumulate_grads(self, accumulation_steps=2):
        if not isinstance(accumulation_steps, int) or accumulation_steps < 1:
            raise ValueError("accumulation_steps must be a positive integer")
        self.accumulation_steps = accumulation_steps
        return self

    def adjust_learning_rate_scheduling(self, index: int, unit='batch', new_value: float = None):
        return self

    def reset_training_context(self):
        self.training_context = {
            'losses': HistoryBase('losses'),
            'metrics': HistoryBase('metrics'),
            'epoch_losses': HistoryBase('epoch_losses'),
            'epoch_metrics': HistoryBase('epoch_metrics'),
            'grads_state': OrderedDict(),
            'tmp_losses': HistoryBase('tmp_losses'),
            'tmp_metrics': HistoryBase('tmp_metrics'),
            'out_sample_metrics': HistoryBase('out_sample_metrics'),
            'print_progress_frequency': 10,
            'print_batch_progress_frequency': 0,
            'collect_data_inteval': 1,
            'print_progress_unit': 'batch',
            'optimizer': None,
            'warmup': 0,
            'grads': None,
            'stop_training': False,  # stop training
            'total_epoch': -1,  # current_epoch
            'total_batch': -1,  # current_batch
            'current_epoch': -1,  # current_epoch
            'current_batch': -1,  # current_batch
            'data_feed': None,
            'current_model': None,  # current model
            'train_data': OrderedDict(),
            'test_data': OrderedDict(),
            'current_input': None,  # current input
            'current_target': None,  # current target
            'steps': 0,
            'current_output': None,  # current output
            'current_loss': to_tensor(0.0, requires_grad=True),  # current loss
            'best_metric': None,  # current loss
            'best_model': None,  # current model
            'loss_history': None, 'metric_history': None, 'base_lr': None,  # current loss
            'current_lr': None,  # current loss
            'save_path': os.path.join(working_directory, 'Models'), 'is_collect_data': True, 'callbacks': [],
            'stop_update': 0, 'retain_graph': False,
            'skip_generate_output': False,
            'skip_reset_total_loss': False}

    def adjust_learning_rate(self, lr):
        raise NotImplementedError

    def rebinding_input_output(self, input_shape):
        pass

    @measure_perf
    def do_on_epoch_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_epoch_start(self.training_context)

    @measure_perf
    def do_on_epoch_end(self):
        for callback in self.training_context['callbacks']:
            callback.on_epoch_end(self.training_context)

        current_epoch = self.training_context['current_epoch']
        batch_steps, batch_values = self.training_context['losses'].get_series('total_losses')
        steps_values, epoch_values = self.training_context['losses'].get_series('epoch')

        if len(batch_steps) > 0:
            epoch_steps = [s for s in steps_values if epoch_values[steps_values.index(s)] == current_epoch]
            filtered_batch_values = [v for step, v in zip(batch_steps, batch_values) if step in epoch_steps]
            if len(filtered_batch_values) > 0:
                filtered_batch_values = np.array(filtered_batch_values[-10:]).mean() if len(
                    filtered_batch_values) > 10 else np.array(filtered_batch_values).mean()
                self.epoch_loss_history.collect('total_losses', self.training_context['current_epoch'],
                                                filtered_batch_values)

        steps_values, epoch_values = self.training_context['metrics'].get_series('epoch')

        for k, v in self.training_context['metrics'].items():
            if k != 'epoch':
                metric_steps, metric_values = self.training_context['metrics'].get_series(k)
                epoch_steps = [s for s in steps_values if epoch_values[steps_values.index(s)] == current_epoch]
                filtered_batch_metric_values = [v for step, v in zip(metric_steps, metric_values) if
                                                step in epoch_steps]
                if len(filtered_batch_metric_values) > 0:
                    filtered_batch_metric_values = np.array(filtered_batch_metric_values[-10:]).mean() if len(
                        filtered_batch_metric_values) > 10 else np.array(
                        filtered_batch_metric_values).mean()
                    self.epoch_metric_history.collect(k, self.training_context['current_epoch'],
                                                      filtered_batch_metric_values)

            # self.training_context['print_epoch_progress_frequency'] = 1

    @measure_perf
    def do_on_batch_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_batch_start(self.training_context)

    @measure_perf
    def do_on_batch_end(self):
        for callback in self.training_context['callbacks']:
            callback.on_batch_end(self.training_context)

    @measure_perf
    def do_on_data_received(self, train_data, test_data):
        for callback in self.training_context['callbacks']:
            callback.on_data_received(self.training_context)
        return train_data, test_data

    @measure_perf
    def do_calculate_forward(self, is_training=True):
        data = self.training_context['train_data'] if is_training or self.training_context['test_data'] is None or len(
            self.training_context['test_data']) == 0 else \
            self.training_context['test_data']
        data_feed = self.training_context['data_feed']
        model = self.training_context['current_model']
        if is_tensor(model):
            data['output'] = self._model

        else:
            if not is_training:
                model.eval()
            else:
                model.train()

            signature = model.signature
            # if any_abnormal_number(data['image']):
            #     print('{0} before calculate forward'.format(''))
            output = try_map_args_and_call(model, data, data_feed, self.is_autocast_enabled)
            # if any_abnormal_number(output):
            #     print('{0} after calculate forward'.format(''))
            output=unpack_singleton(output)
            if isinstance(output, (list, tuple)) and len(output)==len(signature.outputs):
                for i in range(len(output)):
                    data[signature.outputs.key_list[i]] = output[i]
            elif is_instance(output, 'OrderedDict'):
                for k, v in output.items():
                    data[k] = v
            else:
                if self._signature.outputs is None or len(self._signature.outputs) == 0:
                    self._signature.outputs['output'] = TensorSpec.tensor_to_spec(output)
                    data[self._signature.outputs.key_list[0]] = output
                else:
                    data[self._signature.outputs.key_list[0]] = output
        if is_training and self.use_output_as_loss and is_tensor(data['output']):
            this_loss = data['output'].sum()
            self._add_loss_term(this_loss, 'sum')
            self.training_context['losses'].collect('output', self.training_context['steps'],
                                                    to_scalar(this_loss.copy()))

    @measure_perf
    def do_on_loss_calculation_start(self):
        for callback in self.callbacks:
            callback.on_loss_calculation_start(self.training_context)

    @measure_perf
    def do_on_loss_calculation_end(self):

        for callback in self.callbacks:
            callback.on_loss_calculation_end(self.training_context)

    def _should_update_gradients(self):
        if self.training_context.get('waiting_to_start', False):
            return False
        stop = self.training_context['stop_update']
        if stop > 1:
            self.training_context['stop_update'] -= 1
        if 0 < stop < 1:
            import random
            return random.random() <= stop
        return stop == 0

    def _gradient_batch_weight(self):
        """Callbacks may set accumulation_weight for a custom weighted mean."""
        context = self.training_context
        if context.get('accumulation_weight') is not None:
            weight = float(context['accumulation_weight'])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("accumulation_weight must be finite and nonnegative")
            return weight
        data = context['train_data']
        feed = context.get('data_feed') or {}
        signature = getattr(self, 'signature', None)
        names = signature.inputs if signature is not None else ()
        inputs = (data.get(feed.get(name, name)) for name in names)
        for value in itertools.chain(inputs, data.values()):
            if getattr(value, 'ndim', 0) > 0 or isinstance(value, (list, tuple)):
                return len(value)
        return 1

    def _add_loss_term(self, value, reduction, name=None, denominator=None):
        if denominator is not None and float(denominator) == 0:
            # An entirely ignored term must not create a zero gradient that advances momentum.
            value = value.detach()
        self.training_context['current_loss'] = self.training_context['current_loss'] + value
        if self.accumulation_steps > 1:
            if reduction in ('sum', 'batch_sum'):
                prior = self.training_context.get('sum_reduced_loss')
                self.training_context['sum_reduced_loss'] = value if prior is None else prior + value
            elif denominator is not None:
                self.training_context.setdefault('normalized_losses', {})[name] = (value, denominator)

    def _reset_loss_terms(self):
        self.training_context['sum_reduced_loss'] = None
        self.training_context['normalized_losses'] = {}
        self.training_context['accumulation_weight'] = None

    @measure_perf
    def do_calculate_losses(self):
        self.do_on_loss_calculation_start()

        # losss
        for k, v in self._losses.items():
            if not hasattr(v, 'start_epoch') or (
                    hasattr(v, 'start_epoch') and v.start_epoch <= self.training_context['current_epoch']):

                loss_weight = to_tensor(1.0)
                if k in self.loss_weights:
                    loss_weight = self.loss_weights[k]
                loss_weight = to_tensor(loss_weight, ctx.float_dtype)
                # print(-1, 'train_data', self.train_data.value_list[-1].shape, 'abnormal:', any_abnormal_number( self.train_data.value_list[-1]))
                # print(self.train_data.value_list[-1])
                this_loss = try_map_args_and_call(v, self.train_data, self.training_context['data_feed'],
                                                  self.is_autocast_enabled)

                if isinstance(this_loss, tuple):
                    overall_loss = to_tensor(0.0, requires_grad=True)
                    for i in range(len(this_loss)):
                        if any_abnormal_number(this_loss[i]):
                            raise FloatingPointError("Nonfinite loss: {0}".format(k))
                        else:
                            # a leaf Variable that requires grad connotused in an in-place operation.
                            overall_loss = overall_loss + loss_weight * this_loss[i]
                    self._add_loss_term(overall_loss, getattr(v, 'reduction', 'mean'), k,
                                        getattr(v, 'gradient_denominator', None))

                    if hasattr(v, 'as_metric') and v.as_metric == True:
                        self.training_context['tmp_metrics'].collect(camel2snake(k), self.training_context['steps'],
                                                                     to_numpy(overall_loss / loss_weight))

                    self.training_context['tmp_losses'].collect(k, self.training_context['steps'], overall_loss)

                else:
                    this_loss = loss_weight * this_loss
                    if any_abnormal_number(this_loss):
                        raise FloatingPointError("Nonfinite loss: {0}".format(k))
                    else:
                        # a leaf Variable that requires grad connotused in an in-place operation.
                        self._add_loss_term(this_loss, getattr(v, 'reduction', 'mean'), k,
                                            getattr(v, 'gradient_denominator', None))

                    if hasattr(v, 'as_metric') and v.as_metric == True:
                        self.training_context['tmp_metrics'].collect(camel2snake(k), self.training_context['steps'],
                                                                     to_numpy(this_loss / loss_weight))

                    self.training_context['tmp_losses'].collect(k, self.training_context['steps'], this_loss)





        self.do_on_loss_calculation_end()

    @measure_perf
    def do_calculate_regularizations(self):
        # regularizer
        for k, v in self._regs.items():
            this_loss = to_tensor(0.0, requires_grad=True)
            if 'model' in v.signature.inputs:
                reg_weight = v.signature.inputs['reg_weight'].default if 'reg_weight' in v.signature.inputs and hasattr(
                    v.signature.inputs['reg_weight'], 'default') else 1e-6

                this_loss = v(self._model, reg_weight=reg_weight) if self.training_context[
                                                                         'stop_update'] < 1 else to_tensor(0.0,
                                                                                                           requires_grad=True)
            elif 'output' in v.signature.inputs:
                this_loss = try_map_args_and_call(v, self.train_data, self.training_context['data_feed'],
                                                  self.is_autocast_enabled) if self.training_context[
                                                                                   'stop_update'] < 1 else to_tensor(
                    0.0)
            if any_abnormal_number(this_loss):
                raise FloatingPointError('Nonfinite regularizer: {}'.format(k))
            self._add_loss_term(this_loss, 'mean')

            self.training_context['tmp_losses'].collect(k + '_Loss', self.training_context['steps'], this_loss)

    @measure_perf
    def do_calculate_constraints(self):
        for k, v in self._constraints.items():
            if self.training_context['stop_update'] == 0:
                v(self._model)

    @measure_perf
    def on_optimization_step_start(self):
        self.training_context['gradient_accumulation_steps'] = self.accumulation_steps
        for callback in self.training_context['callbacks']:
            callback.on_optimization_step_start(self.training_context)

    @measure_perf
    def do_gradient_update(self, log_gradients=False):
        pass

    @measure_perf
    def do_on_optimization_step_end(self):
        for callback in self.training_context['callbacks']:
            callback.on_optimization_step_end(self.training_context)

    @measure_perf
    def do_on_metrics_evaluation_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_metrics_evaluation_start(self.training_context)

    @measure_perf
    def do_calculate_metrics(self):
        # ON_EVALUATION_START
        self.do_on_metrics_evaluation_start()
        for k, v in self._metrics.items():
            print_only = (getattr(v, 'print_only') if hasattr(v, 'print_only') else False)
            if not print_only:
                self.training_context['metrics'].regist(k)
            self.training_context['tmp_metrics'].regist(k)

            this_metric = try_map_args_and_call(v, self.train_data, self.training_context['data_feed'],
                                                self.is_autocast_enabled)
            self.training_context['tmp_metrics'].collect(k, self.training_context['steps'], this_metric)

            if self.training_context['is_out_sample_evaluation'] == True and self.test_data is not None and len(
                    self.test_data) > 0:
                this_out_metric = try_map_args_and_call(v, self.test_data, self.training_context['data_feed'],
                                                        self.is_autocast_enabled)
                self.training_context['out_sample_metrics'].collect(k, self.training_context['steps'], this_out_metric)

        # ON_EVALUATION_END
        self.do_on_metrics_evaluation_end()

    @measure_perf
    def do_on_metrics_evaluation_end(self):
        for callback in self.training_context['callbacks']:
            callback.on_metrics_evaluation_end(self.training_context)

    @measure_perf
    def do_on_progress_start(self):
        for callback in self.training_context['callbacks']:
            callback.on_progress_start(self.training_context)

    @measure_perf
    def do_on_progress_end(self):
        for callback in self.training_context['callbacks']:
            callback.on_progress_end(self.training_context)

    @measure_perf
    def do_on_excution_exception(self):
        for callback in self.training_context['callbacks']:
            callback.on_excution_exception(self.training_context)

    @measure_perf
    def log_gradient(self, grads=None):
        raise NotImplementedError

    @measure_perf
    def log_weight(self, weghts=None):
        raise NotImplementedError

    @measure_perf
    def save_model(self, save_path=None):
        return NotImplemented

    @measure_perf
    def save_onnx(self, save_path=None, dynamic_axes=None, **kwargs):
        return NotImplemented

    @measure_perf
    def save_history(self, save_path=None, **kwargs):

        default_file_name = '{0}_history_{1}.json_'.format(self._model.name, self.training_context['execution_id'])
        save_path = self.training_context['save_path']
        folder, filename, ext = split_path(save_path)
        if filename == '':
            save_path = sanitize_path(os.path.join(folder, default_file_name))
        out = OrderedDict()
        out['batch_loss_history'] = self.batch_loss_history
        out['batch_metric_history'] = self.batch_metric_history
        out['epoch_loss_history'] = self.epoch_loss_history
        out['epoch_metric_history'] = self.epoch_metric_history
        with open(save_path, 'w') as f:
            jstring = json.dumps(out, indent=4)
            f.write(jstring)
            shutil.copy(save_path, save_path.replace('.json_', '.json'))

    @measure_perf
    def save_weights(self, save_path=None, **kwargs):
        if save_path is not None:
            pass
        else:
            save_path = self.training_context['save_path']

    @measure_perf
    def load_model(self, file_path, **kwargs):
        return NotImplemented

    @measure_perf
    def print_batch_progress(self, print_batch_progress_frequency):
        self.do_on_progress_start()

        if 'max_name_length' not in self.training_context:
            self.training_context['max_name_length'] = len(self.name) + 1
        metric_strings = []

        for k in self.batch_metric_history.key_list:
            if k != 'epoch':
                metric_value = None
                batch_steps, batch_values = self.batch_metric_history.get_series(k)
                if len(batch_values) == 0:
                    batch_steps, batch_values = self.tmp_metrics.get_series(k)
                    metric_value = np.array(batch_values).mean()
                else:
                    metric_value = batch_values[-1]
                metric_strings.append(
                    '{0}: {1} '.format(k, adaptive_format(metric_value, batch_values, value_type='metric', name=k)))

        if len(self.training_context['losses']) == 0:
            loss_value = None
        else:
            if len(self.training_context['losses'].get_series('total_losses')[0]) > 0:
                loss_value = self.training_context['losses'].get_last('total_losses')[-1]
            else:
                steps, values = self.tmp_losses.get_series('total_losses')
                loss_value = np.array(values).mean()

        step_time = self.training_context['time_batch_progress'] / builtins.min(self.steps + 1,
                                                                                print_batch_progress_frequency)
        progress_bar(step_time, self.training_context['current_batch'] + 1,
                     self.training_context['total_batch'] if self.training_context['total_batch'] is not None else '*',
                     'Loss: {0} | {1} | lr: {2:<10.3e} | epoch: {3}'.format(
                         adaptive_format(loss_value, value_type='loss'), ', '.join(metric_strings),
                         self.training_context['current_lr'],
                         self.training_context['current_epoch']),
                     name=self.training_context['model_name'].ljust(self.training_context['max_name_length'] + 1, ' '))
        self.training_context['time_batch_progress'] = 0

        self.do_on_progress_end()

    @measure_perf
    def print_epoch_progress(self, *args, **kwargs):
        self.do_on_progress_start()

        if 'max_name_length' not in self.training_context:
            self.training_context['max_name_length'] = len(self.name) + 1
        metric_strings = []

        for k in self.epoch_metric_history.key_list:
            if k != 'epoch':
                metric_strings.append('{0}: {1}'.format(k, adaptive_format(self.epoch_metric_history.get_last(k)[-1],
                                                                           self.epoch_metric_history.get_series(k)[-1],
                                                                           'metric', k)))
        step_time = self.training_context['time_epoch_progress']

        progress_bar(step_time, self.training_context['current_epoch'] + 1, self.training_context['total_epoch'],
                     'Loss: {0}| {1} | lr: {2:<10.3e}'.format(
                         adaptive_format(self.epoch_loss_history.get_last('total_losses')[-1],
                                         self.epoch_loss_history.get_series('total_losses')[-1], value_type='loss',
                                         name='total_losses'), ', '.join(metric_strings),
                         self.training_context['current_lr']),
                     name=self.training_context['model_name'].ljust(self.training_context['max_name_length'] + 1, ' '))
        self.training_context['time_epoch_progress'] = 0

        self.do_on_progress_end()

    # @pysnooper.snoop()
    def train_model(self, train_data, test_data, current_epoch, current_batch, total_epoch, total_batch=None,
                    done=False,
                    is_collect_data=True, is_print_batch_progress=True, is_print_epoch_progress=True,
                    is_print_batch_gradients=True, log_gradients=False, log_weights=False, accumulate_grads=False,
                    is_out_sample_evaluation=False, **kwargs):
        # with autograd.detect_anomaly():

        self.training_context['current_epoch'] = current_epoch
        self.training_context['current_batch'] = current_batch
        self.training_context['total_epoch'] = total_epoch
        self.training_context['total_batch'] = total_batch
        self.training_context['done'] = done
        self.training_context['is_collect_data'] = is_collect_data
        self.training_context['log_gradients'] = log_gradients
        self.training_context['log_weights'] = log_weights
        self.training_context['current_model'] = self._model

        if self.training_context['current_batch'] == 0:
            if self.training_context['current_epoch'] == 0:
                # epoch is not the logical inteval for us to control the flow
                self.training_context['steps'] = 0
                # on_epoch_start
                for callback in self.callbacks:
                    callback.on_epoch_start(self.training_context)

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
        self.training_context['current_lr'] = (self.optimizer.lr if hasattr(self.optimizer, 'lr')
                                               else self.optimizer.param_groups[0]['lr'])
        self.batch_loss_history.collect('epoch', self.training_context['steps'], current_epoch)
        self.batch_metric_history.collect('epoch', self.training_context['steps'], current_epoch)
        is_epoch_end = (self.training_context['current_batch'] == self.training_context['total_batch'] - 1) if \
            self.training_context['total_batch'] is not None else done
        # print('step:{0}'.format(self.training_context['steps']))
        self.do_on_batch_start()
        for callback in self.callbacks:
            callback.on_batch_start(self.training_context)

        train_data, test_data = self.do_on_data_received(train_data, test_data)
        # if any_abnormal_number(train_data.value_list[0]):
        #     print('{0} after data_received'.format(''))
        self._reset_loss_terms()
        for callback in self.callbacks:
            callback.on_data_received(self.training_context)

        if (not self.training_context[
            'skip_reset_total_loss']):  # and (not (self.training_context['steps'] + 1) % self.accumulation_steps != 0):
            self.training_context['current_loss'] = to_tensor(0.0)

        if 'skip_generate_output' not in self.training_context or self.training_context[
            'skip_generate_output'] == False:
            if self.output_fn is not None and callable(self.output_fn):
                self.output_fn(model=self, training_context=self.training_context, is_training=True,
                               is_autocast_enabled=self.is_autocast_enabled)
            else:
                self.do_calculate_forward(is_training=True)


        # write output in to data
        # print(-1, 'output',self.training_context['train_data'].value_list[-1].shape, 'abnormal:', any_abnormal_number(self.training_context['train_data'].value_list[-1]))

        # if any_abnormal_number(train_data['output']):
        #     print('{0} before calculate loss'.format(''))
        self.do_calculate_losses()
        self.do_calculate_regularizations()

        self.do_gradient_update(log_gradients and is_collect_data)

        if (hasattr(self._model, 'parameters') or is_tensor(self._model)) and any_abnormal_number(self._model):
            raise FloatingPointError("Nonfinite model parameters after optimizer update")
        if log_weights and is_collect_data:
            if isinstance(self._model, Layer):
                self.log_weight(weghts=self._model.weights)
            elif is_tensor(self._model):
                self.log_weight(weghts=self._model)

        if is_out_sample_evaluation == True and self.test_data is not None and len(self.test_data) > 0 and \
                self.training_context['stop_update'] < 1:
            if self.output_fn is not None and callable(self.output_fn):
                self.output_fn(model=self, training_context=self.training_context, is_training=False,
                               is_autocast_enabled=self.is_autocast_enabled)
            else:
                self.do_calculate_forward(is_training=False)

        self.do_calculate_metrics()
        # aggregate the step loss and metrics

        if is_print_batch_progress or is_print_epoch_progress or is_epoch_end:
            # aggregate tmp data and move to metrics history
            for k, v in self.training_context['tmp_metrics'].items():
                steps, values = self.training_context['tmp_metrics'].get_series(k)

                # check
                if len(steps) > 0:
                    self.training_context['metrics'].collect(k, self.training_context['steps'],
                                                             to_scalar(to_numpy(values).mean()))
            self.training_context['tmp_metrics'].reset()

        # ON_BATCH_END

        self.do_on_batch_end()

        # print batch progresss
        if is_print_batch_progress:

            self.print_batch_progress(self.training_context['print_batch_progress_frequency'])

            self.training_context['print_batch_progress_frequency'] = 1

        else:
            self.training_context['print_batch_progress_frequency'] += 1

        if is_out_sample_evaluation and self.test_data:
            self.model.eval()
            verbose = []
            for k in self.training_context['out_sample_metrics'].get_keys():
                test_steps, test_values = self.training_context['out_sample_metrics'].get_series(k)
                metric_value = test_values[-1]
                history_metric_value = to_scalar(test_values[-1])
                verbose.append('{0}: {1}'.format(k, adaptive_format(history_metric_value, prev_value=test_values,
                                                                    value_type='metric', name=k)))
            out_sample_evaluation_str = cyan_color(
                self.training_context['model_name'] + ' ' + 'out-of-sample evaluation: ' + ','.join(verbose))
            ctx.print(out_sample_evaluation_str)
            self.model.train()
            # self.training_context['steps'] += 1


    def summary(self, inputs=None):
        raise NotImplementedError

    def predict(self, input):
        raise NotImplementedError

    def test(self, input, target):
        raise NotImplementedError

    def trigger_when(self, when='on_batch_end', frequency=1, unit='batch', action=None):
        if 'epoch' in when:
            unit = 'epoch'

        cb = get_class('LambdaCallback', ['trident.callbacks', 'trident.callbacks.callback_base'])
        new_callbacks = cb(when=when, frequency=frequency, unit=unit, action=action)
        self.with_callbacks(new_callbacks)
        return self

    def unfreeze_model_scheduling(self, frequency: int, unit='epoch', slice_from=None, slice_to=None, module_name=None):
        cb = get_class('UnfreezeModelCallback', ['trident.callbacks'])
        self.callbacks.append(cb(frequency, unit, slice_from, slice_to, module_name=module_name))
        return self

    def cpu(self):
        if self._model is not None and isinstance(self._model, Layer):
            set_device('cpu')
        elif self._model is not None and isinstance(self._model, Tensor):
            self._model.cpu()

    def cuda(self):
        if self._model is not None and isinstance(self._model, Layer):
            set_device('cuda')
        elif self._model is not None and isinstance(self._model, Tensor):
            self._model.cuda()

    def gpu(self):
        self.cuda()

    def xpu(self):
        self._model.xpu()

    def train(self, mode=True):
        if isinstance(self._model, Layer):
            self._model.train(mode)

    def eval(self):
        if isinstance(self._model, Layer):
            self._model.eval()
            #
    # def fit(self, x = None, y = None, batch_size = 8, epochs = 10,
    #   verbose = getOption("keras.fit_verbose", default = 1),
    #   callbacks = None, view_metrics = getOption("keras.view_metrics",
    #   default = "auto"), validation_split = 0, validation_data = NULL,
    #   shuffle = TRUE, class_weight = None, sample_weight = None,
    #   initial_epoch = 0, steps_per_epoch = NULL, validation_steps = NULL,
    #   ...):


def progress_bar(step_time, current, total, msg=None, name=''):
    # cur_len = builtins.max(int(TOTAL_BAR_LENGTH * float(current) / total), 1)
    # rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1 + cur_len

    L = ['{0}'.format(name), ' Step: {0:<8s}'.format(format_time(step_time))]
    # L.append(' | Tot: {0:<12s}'.format(format_time(tot_time)))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    # for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
    #     sys.stdout.write(' ')
    sys.stdout.write(' ')
    if isinstance(total, str):
        sys.stdout.write(' ( {0:d}/{1} )'.format(current, total))
    else:
        sys.stdout.write(' ( {0:d}/{1:d} )'.format(current, total))
    sys.stdout.write('\n')
    sys.stdout.flush()  # # Go back to the center of the bar.  # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):  #     sys.stdout.write('\b')  # sys.stdout.write(' %d/%d
    # ' % (current+1, total))  # if current < total-1:  #     sys.stdout.write('\r')  # else:  #     sys.stdout.write('\n')  # sys.stdout.flush()
