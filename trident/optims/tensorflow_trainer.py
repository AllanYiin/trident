from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import copy
import inspect
import os
import random
import shutil
import sys
import time
import uuid
from functools import partial
import numbers
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.eager import context, tape, function
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.ops.losses import util as tf_losses_utils


from trident.backend.opencv_backend import array2image, image2array

from trident import __version__
from trident.backend.common import *
from trident.backend.model import ModelBase, HistoryBase, progress_bar
from trident.backend.tensorflow_backend import Sequential, Layer, Combine, try_map_args_and_call, summary, get_device, fix_layer, set_device
from trident.backend.tensorflow_ops import *
from trident.backend.tensorflow_ops import is_tensor
from trident.backend.tensorflow_serialization import save, load, load_pthtar
from trident.callbacks.lr_schedulers import get_lr_scheduler, AdjustLRCallbackBase, AdjustLRCallback
from trident.data.image_common import *
from trident.data.vision_transforms import *
from trident.backend.tensorspec import *
from trident.layers.tensorflow_layers import SoftMax

from trident.optims.tensorflow_constraints import get_constraint
from trident.optims.tensorflow_losses import get_loss, _ClassificationLoss
from trident.optims.tensorflow_metrics import get_metric
from trident.optims.tensorflow_optimizers import get_optimizer
from trident.optims.tensorflow_regularizers import *

# from tensorflow.python.framework.ops import EagerTensor

__all__ = ['Model', 'ImageClassificationModel','ImageRegressionModel', 'ImageDetectionModel', 'ImageSegmentationModel','FaceLandmarkModel','FaceRecognitionModel', 'LanguageModel']
_session = get_session()
_backend = get_backend()
working_directory = _session.working_directory
_device = get_device()


def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,





class Model(ModelBase):
    def __init__(self, inputs=None, input_shape=None, output=None, name=None):
        super().__init__(inputs, input_shape, output, name)
        self.batch_index = 0
        self.filter_index = -1
        self._enable_tensorboard = False
        self.summary_writer = None

    def _initial_graph(self, inputs=None, input_shape=None, output=None, initializer=None):
        if output is None:
            raise ValueError('There is at least one output')

        # if isinstance(output,(np.ndarray,Tensor)) and input_shape is None:
        #     input_shape=squeeze(output.shape)

        if inputs is None:
            if input_shape is None:
                raise ValueError('You should assign inputs or input shape')
            elif isinstance(input_shape, TensorSpec):
                self.inputs[input_shape.name] = input_shape
            elif isinstance(input_shape, dict):
                for k, v in input_shape.items():
                    if is_tensor(v):
                        self.inputs[k] = TensorSpec(shape=TensorShape(v), name=k)
                    elif isinstance(v, TensorSpec):
                        self.inputs[v.name] = v
            elif isinstance(input_shape, (tuple, list)) and all([isinstance(item, int) for item in input_shape]):
                input_name = 'input'
                input_shape = TensorShape((None,) + tuple(input_shape))
                self.inputs[input_name] = TensorSpec(shape=input_shape, name=input_name)
            else:
                input_shape = TensorShape(unpack_singleton(input_shape))
                if isinstance(input_shape, TensorShape):
                    input_name = 'input'
                    self.inputs[input_name] = TensorSpec(shape=input_shape, name=input_name)
                else:
                    for m in range(len(input_shape)):
                        self.inputs['input_{0}'.format(m)] = TensorSpec(shape=(input_shape[m]), name='input_{0}'.format(m))
        elif isinstance(inputs, (tuple, list)):
            if len(inputs) == 1 and is_tensor(inputs[0]):
                input_name = 'input'
                self.inputs[input_name] = TensorSpec(shape=tensor_to_shape(inputs[0]), name=input_name)
            else:
                for m in range(len(inputs)):
                    inp = inputs[m]
                    if is_tensor(inp) or isinstance(inp, np.ndarray):
                        input_name = 'input_{0}'.format(m)
                        self.inputs[input_name] = TensorSpec(shape=tensor_to_shape(inputs[m]), name=input_name)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, TensorSpec):
                    self.inputs[k] = v
                elif is_tensor(v) or isinstance(v, np.ndarray):
                    if  isinstance(v, np.ndarray):
                        v = to_tensor(v)
                    self.inputs[k] = TensorSpec(shape=tensor_to_shape(v),dtype=v.dtype, name=k)
        elif is_tensor(inputs):
            self.inputs['input'] = TensorSpec(shape=tensor_to_shape(inputs),dtype=inputs.dtype, name='input')
        elif isinstance(inputs, np.ndarray):
            inputs = to_tensor(inputs)
            self.inputs['input'] =   TensorSpec(shape=tensor_to_shape(inputs),dtype=inputs.dtype,name='input')


        # single model
        if isinstance(output, (Layer, tf.Module)):
            # update notes
            output.nodes = OrderedDict([(mod.uuid, mod) for mod in list(output.modules()) if isinstance(mod, Layer)])
            for mod in output.modules():
                if isinstance(mod, Layer):
                    mod.nodes = output.nodes

            # output.cpu()
            if output.built and hasattr(output, '_output_shape') and output._output_shape is not None:
                self._model = output
                self._model.input_spec = self.inputs.value_list[0]
                self.signature = None
                if self.signature is not None and hasattr(self.signature, "outputs"):
                    self._outputs = OrderedDict()
                    self._targets = OrderedDict()
                    for name, spec in self.signature.outputs.item_list:
                        self._outputs[name] = spec
                        self._targets[name.replace("output", "target")] = spec

            else:
                out = None
                if inputs is not None:
                    args = None
                    if isinstance(inputs, dict):
                        out = output(*list(inputs.values()))
                    elif isinstance(inputs, (list, tuple)):
                        out = output(*inputs)
                    else:
                        out = output(inputs)

                else:

                    dummay_input = to_tensor(input_shape.get_dummy_tensor()).to(get_device())
                    # prevent pytorch 'ValueError: Expected more than 1 value per channel when training, got input size ....
                    output.to(get_device())
                    output.eval()
                    out = output(dummay_input)

                self._model = output
                self._model.input_spec = self.inputs.value_list[0]
                if isinstance(out, Tensor):
                    self._outputs['output'] = TensorSpec(shape=tensor_to_shape(out), name='output')
                    self._targets['target'] = TensorSpec(shape=tensor_to_shape(out), name='target')
                elif isinstance(out, OrderedDict):
                    for k, v in out.items():
                        self._outputs[k] = TensorSpec(shape=tensor_to_shape(v), name=k)
                        self._targets[k.replace('output', 'target').replace('student', 'teatcher')] = TensorSpec(shape=tensor_to_shape(v),
                                                                                                                 name=k.replace('output', 'target').replace('student', 'teatcher'))

                else:
                    for i in range(len(out)):
                        self._outputs['output_{0}'.format(i)] = TensorSpec(shape=tensor_to_shape(out[i]), name='output_{0}'.format(i))
                        self._targets['target_{0}'.format(i)] = TensorSpec(shape=tensor_to_shape(out[i]), name='target_{0}'.format(i))

            if self._model.signature.maybe_not_complete():
                self._model.signature = None


        elif isinstance(output, (list, tuple)) and all([isinstance(m, (tf.Module)) for m in output]):
            output_list = []
            model_list = []
            dummay_input = to_tensor(input_shape.get_dummy_tensor()).to(get_device()) if not is_tensor(inputs) else inputs.to(get_device())

            for op in output:
                if isinstance(op, (Layer, tf.Module)):
                    op.to(get_device())
                    # prevent pytorch 'ValueError: Expected more than 1 value per channel when training, got input size ....
                    op.eval()
                    out = op(dummay_input)
                    model_list.append(op)
                    output_list.extend(*out)
            model = Combine(model_list)
            self._model = model

            for i in range(len(output_list)):
                self._outputs['output_{0}'.format(i)] = TensorSpec(shape=tensor_to_shape(output_list[i]), name='output_{0}'.format(i))
                self._targets['target_{0}'.format(i)] = TensorSpec(shape=tensor_to_shape(output_list[i]), name='target_{0}'.format(i))

            self.signature = None


        elif isinstance(output, (np.ndarray, Tensor)):
            # style transfer , or adversarial attack
            self._model = to_tensor(output, requires_grad=True)
            out = self._model
            self._outputs['output'] = TensorSpec(shape=tensor_to_shape(out), name='output')
            self._targets['target'] = TensorSpec(shape=tensor_to_shape(out), name='target')

        else:
            raise ValueError('Invalid output')

        self.training_context['current_model'] = self._model
        if hasattr(self._model, 'name') and 'name' in self.__dict__:
            delattr(self, 'name')
        if self.save_path is None:
            save_path = os.path.join('Models', '{0}.pth.tar_'.format(self._model._name))
            self.save_path = sanitize_path(make_dir_if_need(save_path))
        else:
            self.save_path = sanitize_path(make_dir_if_need(self.save_path))
        self.training_context['save_path'] = self.save_path

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

    @property
    def outputs(self):
        if self._model is not None and is_tensor(self._model):
            return self._outputs
        elif self._model is None or not self._model.built:
            return None
        elif len(self._outputs) >= 1:
            return self._outputs
        else:
            return self._outputs

    @property
    def device(self):
        return get_device()

    def train(self):
        if self._model is not None and is_tensor(self._model):
            pass
        elif self._model is not None and isinstance(self._model, Layer):
            self._model.train()
        else:
            raise ValueError('There is no built model ,nothing to learn')

    def eval(self):
        if self._model is not None and is_tensor(self._model):
            pass
        elif self._model is not None and isinstance(self._model, Layer):
            self._model.eval()
        else:
            raise ValueError('There is no built model ,nothing to evaluate')

    @property
    def layers(self):
        if self._model is not None and isinstance(self._model, Layer):
            return self._model.nodes
        else:
            return []

    #
    # def complie(self, optimizer, losses=None, metrics=None, loss_weights=None, sample_weight_mode=None,
    #             weighted_metrics=None, target_tensors=None):
    #     self.with_optimizer(optimizer)
    #     if losses is not None and isinstance(losses, (list, tuple)):
    #         for loss in losses:
    #             self.with_loss(loss)
    #     if metrics is not None and isinstance(metrics, (list, tuple)):
    #         for metric in metrics:
    #             self.with_metric(metric)
    #
    #     return self

    def with_optimizer(self, optimizer, **kwargs):
        if 'learning_rate' in kwargs:
            learning_rate = kwargs['learning_rate']
            kwargs['lr'] = learning_rate
            kwargs.pop('learning_rate')

        if isinstance(optimizer, str):
            optimizer_class = get_optimizer(optimizer)
            self.optimizer = optimizer_class(
                self._model.parameters() if isinstance(self._model, Layer) else [self._model], **kwargs)

        else:
            self.optimizer = optimizer(self._model.parameters() if isinstance(self._model, Layer) else [self._model], **kwargs)
        self.base_lr = kwargs.get('lr', 1e-3)
        self.training_context['base_lr'] = self.base_lr
        self.training_context['current_lr'] = self.base_lr

        return self

    def with_loss(self, loss, loss_weight=1, output_idx=0, start_epoch=0, name='', **kwargs):
        alias = name
        argnames = Signature()
        if (alias is None or len(alias) == 0) and hasattr(loss, '__name__'):
            alias = loss.__name__

        if isinstance(loss, str):
            if loss == 'output':
                self.use_output_as_loss = True
                return self
            else:
                loss_class = get_loss(loss)
                alias = loss if loss_class is not None else alias
                if alias in self._losses:
                    dup_keys = [key for key in self._losses.key_list if alias + '_' in key]
                    alias = alias + '_' + str(len(dup_keys) + 1)
                self._losses[alias] = loss_class(**kwargs) if len(kwargs) > 0 else loss_class()
                if hasattr(loss, 'forward'):
                    argnames = get_signature(self._losses[alias].forward, alias)
                else:
                    argnames = get_signature(self._losses[alias].__call__, alias)
        elif inspect.isclass(loss) and  loss.__class__.__name__=="type":
            alias = loss.__class__.__name__ if alias is None or len(alias) == 0 else alias
            if alias in self._losses:
                dup_keys = [key for key in self._losses.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)

            self._losses[alias] = loss(**kwargs) if len(kwargs) > 0 else loss()
            if hasattr(self._losses[alias], 'forward'):
                argnames = get_signature(self._losses[alias].forward, alias)
            else:
                argnames = get_signature(self._losses[alias].__call__, alias)
        elif not inspect.isfunction(loss) and callable(loss):
            alias = loss.__class__.__name__ if alias is None or len(alias) == 0 else alias
            if alias in self._losses:
                dup_keys = [key for key in self._losses.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            self._losses[alias] = loss
            if hasattr(loss, 'forward'):
                argnames = get_signature(self._losses[alias].forward, alias)
            else:
                argnames = get_signature(self._losses[alias], alias)
            self._losses[alias].signature = argnames
        elif inspect.isfunction(loss):
            if alias in self._losses:
                dup_keys = [key for key in self._losses.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            spec = inspect.getfullargspec(loss)
            if len(spec.args) >= 2 and len(spec.args) - 0 if spec.defaults is None else len(spec.defaults) == 2:
                self._losses[alias] = loss
            else:
                self._losses[alias] = partial(loss, **kwargs)
            argnames = get_signature(loss, alias)
            self._losses[alias].signature = argnames

        # create signature
        if hasattr(self._losses[alias], 'signature') and self._losses[alias].signature is not None:
            pass
        else:
            try:
                self._losses[alias].signature = argnames
                self._losses[alias].signature.name = alias
                if (len(self._losses[alias].signature.outputs) == 1 and self._losses[alias].signature.outputs.value_list[0] is None) or len(
                        self._losses[alias].signature.outputs) == 0:
                    self._losses[alias].signature.outputs = OrderedDict()
                    self._losses[alias].signature.outputs[alias] = None
                if hasattr(self._losses[alias], 'is_logsoftmax'):
                    if isinstance(self._model, Layer):
                        last_module = list(self._model.modules())[-1]
                        if isinstance(last_module, SoftMax):
                            self._losses[alias].is_logsoftmax = True
                print(self._losses[alias].signature)
            except:
                print(argnames)

        self.loss_weights[alias] = float(loss_weight)
        self._losses[alias].__name__ = alias
        # self._losses[alias].signature = argnames
        self._losses[alias].start_epoch = start_epoch

        return self

    def with_metric(self, metric, output_idx=0, collect_history=None, name='', **kwargs):
        if collect_history is None:
            collect_history = True
        alias = name
        argnames = Signature()
        if (alias is None or len(alias) == 0) and hasattr(metric, '__name__'):
            alias = metric.__name__

        if isinstance(metric, str):
            alias = metric if len(alias) == 0 else alias
            metric_fn = get_metric(metric)
            if alias in self._metrics:
                dup_keys = [key for key in self._metrics.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            spec = inspect.getfullargspec(metric_fn)
            if len(spec.args) >= 2 and len(spec.args) - 0 if spec.defaults is None else len(spec.defaults) == 2:
                self._metrics[alias] = metric_fn
            else:
                self._metrics[alias] = partial(metric_fn, **kwargs)
            argnames = get_signature(metric_fn, alias)
        elif inspect.isclass(metric) and metric.__class__.__name__=="type":
            alias = metric.__class__.__name__ if len(alias) == 0 else alias
            if alias in self._metrics:
                dup_keys = [key for key in self._metrics.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            self._metrics[alias] = metric(**kwargs) if len(kwargs) > 0 else metric()
            if hasattr(self._metrics[alias], 'forward'):
                argnames = get_signature(self._metrics[alias].forward)
            else:
                argnames = get_signature(self._metrics[alias].__call__)
        elif not inspect.isfunction(metric) and callable(metric):
            alias = metric.__class__.__name__ if len(alias) == 0 else alias
            if alias in self._metrics:
                dup_keys = [key for key in self._metrics.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            self._metrics[alias] = metric
            if hasattr(metric, 'forward'):
                argnames = get_signature(self._metrics[alias].forward, alias)
            else:
                argnames = get_signature(self._metrics[alias].__call__, alias)
        elif inspect.isfunction(metric):
            if alias in self._metrics:
                dup_keys = [key for key in self._metrics.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            spec = inspect.getfullargspec(metric)
            if len(spec.args) >= 2 and len(spec.args) - 0 if spec.defaults is None else len(spec.defaults) == 2:
                self._metrics[alias] = metric
            else:
                self._metrics[alias] = partial(metric, **kwargs)
            argnames = get_signature(metric, alias)
        # create signature
        if hasattr(self._metrics[alias], 'signature') and self._metrics[alias].signature is not None:
            pass
        else:
            self._metrics[alias].signature = argnames
        self._metrics[alias].signature.name = alias
        if (len(self._metrics[alias].signature.outputs) == 1 and self._metrics[alias].signature.outputs.value_list[
            0] is None) or len(self._metrics[alias].signature.outputs) == 0:
            self._metrics[alias].signature.outputs = OrderedDict()
            self._metrics[alias].signature.outputs[alias] = None
        print(self._metrics[alias].signature)
        # outputs = self.outputs
        # targets = self.targets
        # if all([k in targets.key_list or k in outputs.key_list for k in argnames.key_list]):
        #     pass
        # elif outputs is not None and len(outputs) == 1 and len(argnames) == 2 and argnames.key_list[0] in ['input',
        #                                                                                                  'output',
        #                                                                                                  'y_pred']
        #                                                                                                  and \
        #         argnames.key_list[1] in ['target', 'label', 'y_true']:
        #     argnames = OrderedDict()
        #     argnames[outputs.key_list[0]] = outputs[outputs.key_list[0]]
        #     argnames[targets.key_list[0]] = targets[targets.key_list[0]]
        # elif outputs is not None and len(outputs) == 1 and len(argnames) == 2:
        #     argnames[argnames.key_list[0]] = outputs[outputs.key_list[0]]
        #     argnames[argnames.key_list[1]] = targets[targets.key_list[0]]
        # elif outputs is not None and len(outputs) > 1:
        #     output_idx = list(output_idx) if isinstance(output_idx, (list, tuple)) else [output_idx]
        #     if len(output_idx) == 1 and len(argnames) == 2:
        #         argnames = OrderedDict()
        #         out = outputs.key_list[output_idx[0]]
        #         target = targets.key_list[output_idx[0]]
        #         argnames[argnames.key_list[0]] = outputs[out]
        #         argnames[argnames.key_list[1]] = targets[target]
        #     elif len(output_idx) > 1 and len(argnames) == 2 * len(output_idx):
        #         for idx in output_idx:
        #             out = outputs.key_list[idx]
        #             target = targets.key_list[idx]
        #             if out in argnames:
        #                 argnames[out] = outputs[out]
        #             if target in argnames:
        #                 argnames[target] = targets[target]
        self._metrics[alias].__name__ = alias
        self._metrics[alias].collect_history = collect_history
        return self

    def with_regularizer(self, reg, **kwargs):
        if reg is None:
            return self
        reg_fn = None
        if isinstance(reg, str):
            reg_fn = get_reg(reg)
        elif reg is callable:
            reg_fn = reg
        args = get_signature(reg_fn)
        if 'reg_weight' in args.inputs.key_list:
            args.pop('reg_weight')

        self._regs[reg_fn.__name__] = partial(reg_fn, **kwargs)
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

    def with_learning_rate_scheduler(self, lr_schedule, warmup=0, **kwargs):
        if lr_schedule is None:
            return self
        if isinstance(lr_schedule, str):
            lr_schedule = get_lr_scheduler(lr_schedule)
        if callable(lr_schedule):
            lr_scheduler = lr_schedule(**kwargs)
            self.callbacks.append(lr_scheduler)
        elif isinstance(lr_schedule, AdjustLRCallbackBase):
            self.callbacks.append(lr_schedule)
        self.warmup = warmup
        if self.warmup > 0:
            self.optimizer.adjust_learning_rate(1e-6, False)
            self.training_context['current_lr'] = 1e-6
        return self

    def with_automatic_mixed_precision_training(self, **kwargs):
        """Enable automatic mixed precision training
            only enable when using pytorch 1.6 (or higher) as backend and cuda is available.

        Args:
            **kwargs ():

        Returns:
            the model self

        """
        sys.stderr.write('automatic mixed precision training only enable when using pytorch 1.6 (or higher) as backend and cuda is available.')

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

    def do_on_training_start(self):
        self.train()

    def do_on_training_end(self):
        self.eval()

    def do_on_epoch_start(self):
        self.training_context['time_epoch_start'] = time.time()
        if self.training_context['steps'] == 0:
            self.training_context['time_epoch_progress'] = self.training_context['time_epoch_start']

        if self.warmup > 0 and self.training_context['current_epoch'] < self.warmup:
            lr = 1e-6 * (self.training_context['current_epoch'] + 1)
            self.optimizer.adjust_learning_rate(self.base_lr, verbose=True)
            self.training_context['current_lr'] = lr
        elif self.warmup > 0 and self.training_context['current_epoch'] == self.warmup:
            self.optimizer.adjust_learning_rate(self.base_lr, verbose=True)
            self.training_context['current_lr'] = self.base_lr


    def do_on_epoch_end(self):
        self.training_context['time_epoch_end'] = time.time()

    def do_on_batch_start(self):
        self.training_context['time_batch_start'] = time.time()
        if self.training_context['steps'] == 0:
            self.training_context['time_batch_progress'] = self.training_context['time_batch_start']

    def do_on_batch_end(self):
        self.training_context['time_batch_end'] = time.time()
        if self.training_context['steps'] % 100 == 0:
            gc.collect()
        if self.training_context['steps'] % 200 == 0:
            if 'gpu' in get_device() or 'cuda' in get_device():
                self._model.cpu()
                self._model.cuda()

        if self.training_context['steps'] % _session.epoch_equivalent == 0:
            if self.warmup > 0 and self.warmup == self.training_context['steps'] // _session.epoch_equivalent:
                self.adjust_learning_rate(self.training_context['base_lr'])
                self.warmup = 0

        if self.training_context['current_batch'] == 0 and self.training_context['is_print_batch_progress'] == True:
            temp = OrderedDict()
            for k in self.training_context['losses'].key_list:
                if len(self.training_context['losses'][k]) > 0:
                    temp[k] = self.training_context['losses'][k][-1][-1]
            print('{ '+', '.join(['{0}: {1}'.format(k,adaptive_format(v)) for k,v in temp.items()])+' }')

    def do_on_data_received(self, train_data, test_data):
        if train_data is None and test_data is None:
            return self.training_context['train_data'], self.training_context['test_data']
        if 'data_feed' not in self.training_context or len(self.training_context['data_feed']) == 0 or self.training_context['current_batch'] + self.training_context[
            'current_epoch'] == 0:
            try:

                data_feed = OrderedDict() if 'data_feed' not in self.training_context else self.training_context['data_feed']
                inshapes = self.inputs.value_list
                outshapes = self.targets.value_list
                available_fields = copy.deepcopy(train_data.key_list)
                if train_data is not None:
                    # check input
                    for arg in self._model.signature.inputs.key_list:
                        if arg in data_feed and data_feed[arg] in available_fields:
                            available_fields.remove(data_feed[arg])
                        else:
                            data_feed[arg] = ''
                            if len(train_data) == 1 and len(self._model.signature.inputs.key_list) == 1:
                                data_feed[arg] = train_data.key_list[0]
                                available_fields.remove(train_data.key_list[0])
                            elif arg in available_fields:
                                data_feed[arg] = arg
                                available_fields.remove(arg)
                            elif arg in ['x', 'input'] and 'data' in available_fields:
                                data_feed[arg] = 'data'
                                available_fields.remove('data')
                            elif arg in ['x', 'input'] and 'image' in available_fields:
                                data_feed[arg] = 'image'
                                available_fields.remove('image')
                            elif arg == 'x' and 'input' in available_fields:
                                data_feed[arg] = 'input'
                                available_fields.remove('input')
                            elif len(self._model.signature.inputs.key_list) == 1:
                                for item in available_fields:
                                    data_shape = tensor_to_shape(train_data[item])
                                    if 'target' not in item and 'output' != item and data_shape == inshapes[0].shape:
                                        data_feed[arg] = item
                                        available_fields.remove(item)
                                        break
                            else:
                                Warning(
                                    'input argment {0} cannot mapping to any data, please check it and update the datafeed'.format(
                                        arg))

                    # check for target
                    if len(available_fields) > 0:
                        if len(available_fields) == 1:
                            data_feed['target'] = available_fields[0]
                        else:
                            for i in range(len(self.targets)):
                                arg = self.targets.key_list[i]
                                data_feed[arg] = ''
                                if len(train_data) == 1:
                                    data_feed[self.targets.key_list[0]] = train_data.key_list[0]
                                elif arg in available_fields:
                                    data_feed[arg] = arg
                                    available_fields.remove(arg)
                                elif arg == 'target' and 'label' in available_fields:
                                    data_feed[arg] = 'label'
                                    available_fields.remove('label')
                                elif arg == 'target' and len(available_fields) == 1:
                                    data_feed[arg] = available_fields[0]
                                    available_fields.remove(available_fields[0])
                                elif len(available_fields) > 0:
                                    target_shape = outshapes
                                    for item in available_fields:
                                        data_shape = tensor_to_shape(train_data[item])
                                        if target_shape == data_shape:
                                            data_feed[arg] = item
                                            available_fields.remove(item)
                                        elif ('int64' in str(train_data[item].dtype) or 'int32' in str(
                                                train_data[item].dtype)) and target_shape== data_shape:
                                            data_feed[arg] = item
                                            available_fields.remove(item)
                                        else:
                                            Warning(
                                                'target argment {0} cannot mapping to any data, please check it and update the datafeed'.format(
                                                    arg))
                            # if len(self.targets) == 1 and data_feed[self.targets.key_list[0]] != None:
                            #     self.training_context['current_target'] = train_data[data_feed[self.targets.key_list[0]]]

                    # if len(self._signature.inputs.key_list) == 1 and data_feed[self._signature.inputs.key_list[0]] != None:
                    #     self.training_context['data_feed'] = data_feed
                    # elif '' not in data_feed.value_list:
                    self.training_context['data_feed'] = data_feed

                    print('data_feed', data_feed)
            except:
                PrintException()

        try:
            data_feed = self.training_context['data_feed']
            input_list = [data_feed[arg] for arg in self.signature.inputs.key_list]
            for item in train_data.key_list:
                if item in input_list:
                    # only model 's input argments
                    train_data[item] = to_tensor(train_data[item].copy(), requires_grad=True)
                elif item in self.targets.key_list or data_feed:
                    train_data[item] = to_tensor(train_data[item].copy(), requires_grad=False)
                else:
                    train_data[item] = to_tensor(train_data[item].copy())

                if test_data is not None and item in test_data:
                    test_data[item] = to_tensor(test_data[item].copy())

                    # check target

            self.training_context['train_data'] = train_data
            self.training_context['test_data'] = test_data

        except:
            PrintException()

        return self.training_context['train_data'], self.training_context['test_data']

    def do_preparation_for_loss(self):
        pass

    def get_current_loss(self):
        return self.training_context['current_loss']

    def do_gradient_update(self, log_gradients=False):
        if isinstance(self._model, (Layer, tf.Module)):
            # double check!!!
            self._model.train()
        if log_gradients:
            self.log_gradient(self.optimizer.grads_and_vars)
        # vars=self.training_context['vars']
        # cal_grads=self.training_context['grads']
        #
        # if isinstance(self._model,Layer) and self.grad_clipping_by_norm:
        #     cal_grads = [(tf.clip_by_norm(grad, -1.0*self.grad_clipping_threshold, 1.0*self.grad_clipping_threshold)) for grad in cal_grads]
        #

        if self.training_context['stop_update'] < 1:
            for callback in self.training_context['callbacks']:
                callback.on_optimization_step_start(self.training_context)

            if self.training_context['stop_update'] == 0:
                self.optimizer.step(self.training_context['grads_and_vars'])

            elif 0 < self.training_context['stop_update'] < 1:
                if random.random() <= self.training_context['stop_update']:
                    self.optimizer.step(self.training_context['grads_and_vars'])
            else:
                self.training_context['stop_update'] = self.training_context['stop_update'] - 1

            for callback in self.training_context['callbacks']:
                callback.on_optimization_step_end(self.training_context)

    def do_post_gradient_update(self):

        self.training_context['tmp_losses'].collect('total_losses', self.training_context['steps'], to_numpy(self.training_context['current_loss']).mean())
        if self.training_context['is_collect_data'] == True:
            steps, values = self.training_context['tmp_losses'].get_series('total_losses')
            self.training_context['losses'].collect('total_losses', self.training_context['steps'], float(to_numpy(values).mean()))
            self.training_context['tmp_losses'].reset()

    def do_on_progress_end(self):
        if self.training_context['current_epoch'] > self.warmup:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(np.array(self.training_context['metrics'][list(self._metrics.keys())[0]]).mean(), )
                self.training_context['current_lr'] = self.optimizer.lr

    def do_on_excution_exception(self):
        pass

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

    def save_model(self, save_path=None):
        for callback in self.training_context['callbacks']:
            callback.on_model_saving_start(self.training_context)

        if isinstance(self._model, Layer) and any_abnormal_number(self._model):
            for para in self._model.parameters():
                if any_abnormal_number(para.value()):
                    para.assign(where(is_nan(para), random_normal_like(para.value(), mean=0, std=0.02).to(para.device), para))

        if save_path is not None:
            pass
        else:
            save_path = self.training_context['save_path']
        folder, filename, ext = split_path(save_path)
        if filename == '':
            filename = self.name
        if not filename.endswith('_tf'):
            filename += '_tf'
        save_path = os.path.join(folder, filename + ext)
        self.training_context['save_path'] = save_path

        if isinstance(self._model, Layer):
            folder, filename, ext = split_path(save_path)
            ext = '.pth.tar_'
            save_path = os.path.join(folder, filename + ext)
            make_dir_if_need(sanitize_path(save_path))
            save_path = sanitize_path(save_path)
            device = get_device()
            self._model.eval()

            with tf.device('/cpu:0'):
                save({
                    'state_dict': self._model.state_dict(),
                    'backend': 'tensorflow',
                    'trident_version': __version__,
                    'tensorflow_version': tf.version.VERSION,
                    'signature': self.signature
                }, save_path, is_compressed=True)

                shutil.copy2(save_path, save_path.replace('.pth.tar_', '.pth.tar'))
                os.remove(save_path)
                save_path = save_path.replace('pth.tar_', 'pth_')
                save(self._model, save_path)
                shutil.copy2(save_path, save_path.replace('.pth_', '.pth'))
                os.remove(save_path)
            gc.collect()
            with tf.device(device):
                self._model.train()





        elif is_tensor(self._model):
            save_path = self.get_save_path(save_path, default_folder='Models',
                                           default_file_name='{0}_epoch{1}.npy_'.format(self._model.name,
                                                                                        self.training_context[
                                                                                            'current_epoch']))
            numpy_model = to_numpy(self._model)
            save_path = sanitize_path(save_path)
            self.current_save_path = save_path
            np.save(save_path, numpy_model)
            shutil.copy2(save_path, save_path.replace('.npy_', '.npy'))
            os.remove(save_path)
            sys.stdout.write('Yor model is a Tensor not a nn.Module, it has saved as numpy array(*.npy) successfully. ')
        else:
            raise ValueError(
                'only Layer or nn.Module as model can export to onnx, yours model is {0}'.format(type(self._model)))
        for callback in self.training_context['callbacks']:
            callback.on_model_saving_end(self.training_context)

    def save_onnx(self, file_path):
        pass

    def save_weights(self, file_path):
        for callback in self.training_context['callbacks']:
            callback.on_model_saving_start(self.training_context)

        if file_path is not None:
            self._model.save_weights(file_path)
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

    def load_model(self, file_path):
        print('Loading pretrained model from {}'.format(file_path))
        folder, filename, ext = split_path(file_path)
        if filename == '':
            filename = self.name

        ext = '.pth.tar'
        save_path = os.path.join(folder, filename + ext)
        if not os.path.exists(save_path):
            save_path = os.path.join(working_directory, filename + ext)
        pretrained_dict = load_pthtar(file_path)
        state_dict = None
        if "state_dict" in pretrained_dict.keys():
            state_dict = pretrained_dict['state_dict']

            # pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        if check_keys(self._model, state_dict):
            self._model.load_state_dict(state_dict, strict=False)
            print('Model loaded!')

        self.signature = None
        if 'signature' in pretrained_dict:
            self.signature = pretrained_dict['signature']

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

    def train_model(self, train_data, test_data, current_epoch, current_batch, total_epoch, total_batch,
                    is_collect_data=True, is_print_batch_progress=True, is_print_epoch_progress=True,
                    is_print_batch_gradients=True, log_gradients=False, log_weights=False, accumulate_grads=False, is_out_sample_evaluation=False, **kwargs):
        with tf.device(get_device()):
            try:
                self.training_context['current_epoch'] = current_epoch
                self.training_context['current_batch'] = current_batch
                self.training_context['total_epoch'] = total_epoch
                self.training_context['total_batch'] = total_batch
                self.training_context['is_collect_data'] = is_collect_data
                self.training_context['log_gradients'] = log_gradients
                self.training_context['log_weights'] = log_weights
                self.training_context['current_model'] = self._model
                self.training_context['current_lr'] = self.optimizer.lr
                self.training_context['train_data'] = train_data
                self.training_context['test_data'] = test_data

                if self.training_context['current_batch'] == 0:
                    if self.training_context['current_epoch'] == 0:
                        self.do_on_training_start()
                        # epoch is not the logical inteval for us to control the flow
                        self.training_context['steps'] = 0
                        self.training_context['grads_state'] = OrderedDict()
                        self.training_context['grads_state']['first_layer'] = []
                        self.training_context['grads_state']['last_layer'] = []
                    self.training_context['is_print_batch_progress'] = is_print_batch_progress
                    self.training_context['is_print_epoch_progress'] = is_print_epoch_progress
                    self.training_context['print_batch_progress_frequency'] = 1
                    self.training_context['print_epoch_progress_frequency'] = 1

                    self.do_on_epoch_start()
                    for callback in self.callbacks:
                        callback.on_epoch_start(self.training_context)

                self.do_on_batch_start()
                for callback in self.callbacks:
                    callback.on_batch_start(self.training_context)

                train_data, test_data = self.do_on_data_received(train_data, test_data)

                for callback in self.callbacks:
                    callback.on_data_received(self.training_context)

                if accumulate_grads == False:
                    self.training_context['current_loss'] = to_tensor(0.0, requires_grad=True)
                    self.do_preparation_for_loss()
                    self.training_context['optimizer'] = self.optimizer

                with tf.GradientTape() as grad_tape:
                    if 'skip_generate_output' not in self.training_context or self.training_context['skip_generate_output'] == False:
                        try:
                            if self.output_fn is not None:
                                self.output_fn()
                            else:
                                output = try_map_args_and_call(self._model, self.train_data, self.training_context['data_feed'])
                                if isinstance(output, (list, tuple)):
                                    for i in range(len(output)):
                                        self.train_data[self.outputs.key_list[i]] = output[i]
                                elif isinstance(output, (OrderedDict)):
                                    for k, v in output.items():
                                        self.train_data[k] = v
                                elif 'tensor' in output.__class__.__name__.lower():
                                    self.train_data[self.outputs.key_list[0]] = output
                                    if self.use_output_as_loss == True:
                                        this_loss = output.sum()
                                        self.training_context['losses'].collect(self.outputs.key_list[0], self.training_context['steps'], this_loss)
                                        self.training_context['current_loss'] = self.training_context['current_loss'] + this_loss
                                else:
                                    self.train_data[self.outputs.key_list[0]] = output
                                    if self.use_output_as_loss == True:
                                        this_loss = output.sum()
                                        self.training_context['losses'].collect(self.outputs.key_list[0], self.training_context['steps'], this_loss)
                                        self.training_context['current_loss'] = self.training_context['current_loss'] + this_loss
                        except Exception as e:
                            print(e)
                            PrintException()
                            if isinstance(self._model, Layer) and any_abnormal_number(self._model):
                                for para in self._model.parameters():
                                    if para is not None and any_abnormal_number(para):
                                        para.data.copy_(where(is_nan(para), random_normal_like(para, mean=0, std=0.02).to(get_device()), para))

                    # write output in to data

                    # confirm singleton
                    # output=unpack_singleton(output)

                    # losss
                    for k, v in self._losses.items():
                        if not hasattr(v, 'start_epoch') or (hasattr(v, 'start_epoch') and v.start_epoch <= self.training_context['current_epoch']):
                            try:
                                loss_weight = 1.0
                                if k in self.loss_weights:
                                    loss_weight = self.loss_weights[k]
                                loss_weight = to_tensor(loss_weight, 'float32')
                                this_loss = loss_weight * try_map_args_and_call(v, self.train_data,
                                                                                self.training_context['data_feed'])  # v.forward(output, target) if hasattr(v, 'forward') else v(
                                if self.training_context['stop_update'] >= 1:
                                    pass  # this_loss= to_tensor(0.0,requires_grad=True)
                                # output, target)

                                if isinstance(this_loss, tuple):
                                    overall_loss = to_tensor(0.0)
                                    for i in range(len(this_loss)):
                                        if any_abnormal_number(this_loss[i]):
                                            sys.stderr.write(
                                                'Loss {0} have abnormal number (nan, inf,-inf), trident will skip it automaticly, please check anything wrong!!!/n'.format(k))
                                        else:
                                            # a leaf Variable that requires grad connotused in an in-place operation.
                                            overall_loss = overall_loss + this_loss[i]
                                    self.training_context['current_loss'] = self.training_context['current_loss'] + overall_loss
                                    if is_collect_data:
                                        self.training_context['losses'].collect(k, self.training_context['steps'], overall_loss)
                                else:
                                    if any_abnormal_number(this_loss):
                                        sys.stderr.write(
                                            'Loss {0} have abnormal number (nan, inf,-inf), trident will skip it automaticly, ' 'please check anything wrong!!!/n'.format(k))
                                    else:
                                        # a leaf Variable that requires grad connotused in an in-place operation.
                                        self.training_context['current_loss'] = self.training_context['current_loss'] + this_loss
                                    if is_collect_data:
                                        self.training_context['losses'].collect(k, self.training_context['steps'], this_loss)
                            except Exception as e:
                                print(e)
                                PrintException()

                    self.do_post_loss_calculation()
                    for callback in self.callbacks:
                        callback.on_loss_calculation_end(self.training_context)

                    if accumulate_grads == False:
                        # regularizer
                        for k, v in self._regs.items():
                            this_loss = to_tensor(0.0)
                            if 'model' in v.signature.inputs:
                                this_loss = v(self._model) if self.training_context['stop_update'] < 1 else to_tensor(0.0, requires_grad=True)
                            elif 'output' in v.signature.inputs:

                                this_loss = try_map_args_and_call(v, self.train_data, self.training_context['data_feed']) if self.training_context[
                                                                                                                                 'stop_update'] < 1 else to_tensor(0.0)
                            if not any_abnormal_number(this_loss):
                                # a leaf Variable that requires grad connotused in an in-place operation.
                                self.training_context['current_loss'] = self.training_context['current_loss'] + this_loss  # self.training_context[

                            if is_collect_data:
                                self.training_context['losses'].collect(k + '_Loss', self.training_context['steps'], this_loss)

                vars = grad_tape.watched_variables()
                grads = grad_tape.gradient(self.training_context['current_loss'], vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                # grads = tuple([where(is_nan(grad), zeros_like(grad), grad) for grad in grads])

                self.training_context['grads_and_vars']= zip(grads, vars);
                self.optimizer.grads_and_vars = zip(grads, vars)
                # self.training_context['grads'] = grads
                # self.training_context['vars'] = vars

                self.do_pre_optimization_step()
                # self.optimizer.step(zip(grads,vars))
                self.do_gradient_update(log_gradients and is_collect_data)
                self.training_context['current_lr'] = self.optimizer.lr

                # ON_POSTBACKWARD_CALCULATION
                self.do_post_gradient_update()

                # if isinstance(self._model, Layer) and any_abnormal_number(self._model):
                #     for para in self._model.parameters():
                #         if any_abnormal_number(para):
                #             para.assign(where(is_nan(para), random_normal_like(para, mean=0, std=0.02), para.value()))

                # model comfirm
                for k, v in self._constraints.items():
                    if self.training_context['stop_update'] == 0:
                        v(self._model)

                if log_weights and is_collect_data:
                    if isinstance(self._model, Layer):
                        self.log_weight(weghts=self._model.weights)
                    elif is_tensor(self._model):
                        self.log_weight(weghts=self._model)

                if is_out_sample_evaluation == True and self.test_data is not None and len(self.test_data) > 0 and self.training_context['stop_update'] < 1:
                    tmp_output = try_map_args_and_call(self._model, self.test_data, self.training_context['data_feed'])
                    if isinstance(tmp_output, (list, tuple)):
                        for i in range(len(tmp_output)):
                            self.test_data[self.outputs.key_list[i]] = tmp_output[i]
                    elif 'tensor' in tmp_output.__class__.__name__.lower():
                        self.test_data[self.outputs.key_list[0]] = tmp_output
                    else:
                        self.test_data[self.outputs.key_list[0]] = tmp_output

                # ON_EVALUATION_START
                self.do_on_metrics_evaluation_start()
                for callback in self.training_context['callbacks']:
                    callback.on_metrics_evaluation_start(self.training_context)

                for k, v in self._metrics.items():
                    collect_history = getattr(v, 'collect_history') if hasattr(v, 'collect_history') else True
                    if not collect_history == False:
                        self.training_context['metrics'].regist(k)
                        self.training_context['tmp_metrics'].regist(k)

                    this_metric = try_map_args_and_call(v, self.train_data, self.training_context['data_feed']) if self.training_context['stop_update'] < 1 else to_tensor(0)
                    self.training_context['tmp_metrics'].collect(k, self.training_context['steps'], float(to_numpy(this_metric)))

                    if is_out_sample_evaluation == True and self.test_data is not None and len(self.test_data) > 0 and collect_history != False:
                        this_out_metric = try_map_args_and_call(v, self.test_data, self.training_context['data_feed'])
                        self.training_context['out_sample_metrics'].collect(k, self.training_context['steps'], float(to_numpy(this_out_metric)))

                # ON_EVALUATION_END
                self.do_on_metrics_evaluation_end()
                for callback in self.training_context['callbacks']:
                    callback.on_metrics_evaluation_end(self.training_context)

                # callback's metric can keep in epoch_metric_history

                if is_collect_data:
                    # aggregate tmp data and move to metrics history
                    for k, v in self.training_context['tmp_metrics'].items():
                        steps, values = self.training_context['tmp_metrics'].get_series(k)
                        self.training_context['metrics'].collect(k, self.training_context['steps'], float(to_numpy(values).mean()))
                    self.training_context['tmp_metrics'].reset()

                # ON_BATCH_END
                self.do_on_batch_end()
                for callback in self.training_context['callbacks']:
                    callback.on_batch_end(self.training_context)

                # print batch progresss
                if is_print_batch_progress:
                    self.do_on_progress_start()
                    for callback in self.training_context['callbacks']:
                        callback.on_progress_start(self.training_context)

                    self.print_batch_progress(self.training_context['print_batch_progress_frequency'])

                    self.training_context['print_batch_progress_frequency'] = 1
                    self.do_on_progress_end()
                    for callback in self.training_context['callbacks']:
                        callback.on_progress_end(self.training_context)
                else:
                    self.training_context['print_batch_progress_frequency'] += 1

                if is_out_sample_evaluation == True and self.test_data is not None and len(self.test_data) > 0:
                    verbose = []
                    for k in self.training_context['out_sample_metrics'].get_keys():
                        test_steps, test_values = self.training_context['out_sample_metrics'].get_series(k)
                        metric_value = test_values[-1]
                        history_metric_value = np.array(test_values).mean()

                        format_string = '.3%'
                        if history_metric_value > 3:
                            format_string = '.3f'
                        elif history_metric_value < 1e-3:
                            format_string = '.3e'
                        verbose.append('{0}: {1:<8{2}}'.format(k, metric_value, format_string))
                    print(self.training_context['model_name'] + ': out-of-sample evaluation: ', ','.join(verbose))

                if self.training_context['current_batch'] == self.training_context['total_batch'] - 1:
                    self.do_on_epoch_end()
                    batch_steps, batch_values = self.training_context['losses'].get_series('total_losses')
                    if not hasattr(self.training_context['losses'], 'last_aggregate_idx'):
                        self.epoch_loss_history.collect('total_losses', self.training_context['current_epoch'], np.array(batch_values).mean())
                        self.training_context['losses'].last_aggregate_idx = len(batch_values)
                    else:
                        self.epoch_loss_history.collect('total_losses', self.training_context['current_epoch'],
                                                        np.array(batch_values[self.training_context['losses'].last_aggregate_idx:]).mean())
                        self.training_context['losses'].last_aggregate_idx = len(batch_values)

                    for k, v in self.training_context['metrics'].items():
                        metric_steps, metric_values = self.training_context['metrics'].get_series(k)
                        if not hasattr(self.training_context['metrics'], 'last_aggregate_idx'):
                            self.epoch_metric_history.collect(k, self.training_context['current_epoch'], np.array(metric_values).mean())
                            self.training_context['metrics'].last_aggregate_idx = len(metric_values)
                        else:
                            if self.training_context['metrics'].last_aggregate_idx < len(metric_values):
                                self.epoch_metric_history.collect(k, self.training_context['current_epoch'],
                                                                  np.array(metric_values[self.training_context['metrics'].last_aggregate_idx:]).mean())
                                self.training_context['metrics'].last_aggregate_idx = len(metric_values)

                    if is_print_epoch_progress:
                        self.do_on_progress_start()
                        for callback in self.training_context['callbacks']:
                            callback.on_progress_start(self.training_context)
                        self.print_epoch_progress(self.training_context['print_epoch_progress_frequency'])
                        self.training_context['print_epoch_progress_frequency'] = 1
                        self.do_on_progress_end()
                        for callback in self.training_context['callbacks']:
                            callback.on_progress_end(self.training_context)
                    else:
                        self.training_context['print_epoch_progress_frequency'] += 1

                    for callback in self.training_context['callbacks']:
                        callback.on_epoch_end(self.training_context)

                    if self.training_context['current_epoch'] == self.training_context['total_epoch'] - 1:
                        self.do_on_training_end()
                        for callback in self.training_context['callbacks']:
                            callback.on_training_end(self.training_context)
            except Exception:
                self.do_on_excution_exception()
                PrintException()

    def summary(self):
        if self._model.built:
            summary(self._model, [TensorShape(item.shape) for item in self.inputs.value_list])
            return self
        else:
            raise ValueError('This model has not yet been built. ')

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
                if self._model is not None and self.signature is not None and len(self.signature) > 1 and self.input_spec is not None:
                    img_data = fc(img_data, spec=self.input_spec)
                else:
                    img_data = fc(img_data)
            img_data = np.expand_dims(image_backend_adaption(img_data))
            if self.input_spec is None:
                self.input_spec = TensorSpec(shape=tensor_to_shape(to_tensor(img_data), need_exclude_batch_axis=False), object_type=ObjectType.rgb, name='input')

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
        # We treat the extra repr like the sub-module, one item per line
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
        attrs = list(self.__dict__.keys()) if self.__dict__ is not None else {}
        losses = list(self._losses.keys()) if self._losses is not None else {}
        metrics = list(self._metrics.keys()) if self._metrics is not None else {}
        regs = list(self._regs.keys()) if self._regs is not None else {}

        constraints = list(self._constraints.keys()) if self._constraints is not None else {}
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
                    self.summary_writer = SummaryWriter(os.path.join(working_directory, 'Logs'))

                except Exception as e:
                    print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                    print(e)
                    PrintException()
            elif get_backend() == 'tensorflow':
                try:
                    from trident.loggers.tensorflow_tensorboard import SummaryWriter
                    self.summary_writer = SummaryWriter(os.path.join(working_directory, 'Logs'))
                except Exception as e:
                    print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                    print(e)
                    PrintException()


class ImageClassificationModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(ImageClassificationModel, self).__init__(inputs, input_shape, output)

        self._class_names = []
        self.preprocess_flow = []
        self._idx2lab = {}
        self._lab2idx = {}

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
                if self._model.input_spec.object_type is None:
                    self._model.input_spec.object_type = ObjectType.rgb
            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            for func in self.preprocess_flow:
                if inspect.isfunction(func) and func is not image_backend_adaption:
                    img = func(img)
            img = image_backend_adaption(img)
            inp = to_tensor(np.expand_dims(img, 0))
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


    def infer_single_image(self, img):
        if self._model.built:
            self._model.eval()
            if self._model.input_spec.object_type is None:
                self._model.input_spec.object_type = ObjectType.rgb
            img = image2array(img)
            img_shp = img.shape
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            rescale_scale = 1.0
            for func in self.preprocess_flow:
                if inspect.isfunction(func) and func is not image_backend_adaption:
                    img = func(img, spec=self._model.input_spec)
                    if func.__qualname__ == 'resize.<locals>.img_op':
                        rescale_scale = func.scale
            img = image_backend_adaption(img)
            if isinstance( self._model,Layer):
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
        self.preprocess_flow = []
        self.detection_threshould = 0.5

    def infer_single_image(self, img, scale=1):
        if self._model.built:
            self._model.to(self.device)
            self._model.eval()
            if self._model.input_spec.object_type is None:
                self._model.input_spec.object_type = ObjectType.rgb
            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]

            for func in self.preprocess_flow:
                if inspect.isfunction(func):
                    img = func(img)

            result = self._model(to_tensor(np.expand_dims(img, 0)))
            self._model._set_inputs(to_tensor(np.expand_dims(img, 0)))
            bboxes = self.generate_bboxes(*result, threshould=self.detection_threshould, scale=scale)
            bboxes = self.nms(bboxes)
            # idx=int(np.argmax(result,-1)[0])
            return bboxes
        else:
            raise ValueError('the model is not built yet.')

    def generate_bboxes(self, *outputs, threshould=0.5, scale=1):
        raise NotImplementedError

    def nms(self, bboxes):
        raise NotImplementedError


class ImageSegmentationModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(ImageSegmentationModel, self).__init__(inputs, input_shape, output)
        self.preprocess_flow = []

class ImageGenerationModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(ImageGenerationModel, self).__init__(inputs, input_shape, output)
        self.preprocess_flow = []

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
            if self._model.input_spec.object_type is None:
                self._model.input_spec.object_type = ObjectType.rgb
            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]

            for func in self.preprocess_flow:
                if inspect.isfunction(func) and func is not image_backend_adaption:
                    img = func(img)
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


    def infer_single_image(self, img):
        if self._model.built:
            self._model.eval()
            if self._model.input_spec.object_type is None:
                self._model.input_spec.object_type=ObjectType.rgb
            img = image2array(img)
            img_shp=img.shape

            if img.shape[-1] == 4:
                img = img[:, :, :3]
            rescale_scale=1.0
            for func in self.preprocess_flow:
                if inspect.isfunction(func) and func is not image_backend_adaption:
                    img = func(img,spec=self._model.input_spec)
                    if func.__qualname__ == 'resize.<locals>.img_op':
                        rescale_scale = func.scale
            img = image_backend_adaption(img)
            if isinstance(self._model, Layer):
                inp = to_tensor(np.expand_dims(img, 0)).to(self._model.device).to(self._model.weights[0].data.dtype)
                result = self._model(inp)
                result = to_numpy(result)/ rescale_scale
                result[:,:, 0::2] =clip(result[:,:, 0::2] ,0,img_shp[1])
                result[:,:, 1::2] =clip(result[:,:, 1::2],0,img_shp[0])
                return result.astype(np.int32)
            else:

                raise ValueError('the model is not layer.')

        else:
            raise ValueError('the model is not built yet.')


class FaceRecognitionModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(FaceRecognitionModel, self).__init__(inputs, input_shape, output)

        self._class_names = []
        self.preprocess_flow = []

        self._idx2lab = {}
        self._lab2idx = {}

    @property
    def reverse_preprocess_flow(self):
        return_list = []
        return_list.append(reverse_image_backend_adaption)
        for i in range(len(self.preprocess_flow)):
            fn = self.preprocess_flow[-1 - i]
            if fn.__qualname__ == 'normalize.<locals>.img_op':
                return_list.append(unnormalize(fn.mean, fn.std))
        return_list.append(array2image)
        return return_list

    def get_embedded(self, img_path):
        def norm(x):
            b = np.sqrt(np.sum(np.square(x)))
            return x / (b if b != 0 else 1)

        img = image2array(img_path)
        img = Resize((224, 224))(img)
        img = Normalize([131.0912, 103.8827, 91.4953], [1, 1, 1])(img)
        img = to_tensor(np.expand_dims(img.transpose([2, 0, 1]), 0))
        embedding = self.model(img)[0]
        return norm(embedding)

    def infer_single_image(self, img):
        def norm(x):
            b = np.sqrt(np.sum(np.square(x)))
            return x / (b if b != 0 else 1)

        if isinstance(self._model, Layer) and self._model.built:
            self._model.eval()
            if self._model.input_spec.object_type is None:
                self._model.input_spec.object_type = ObjectType.rgb
            img = image2array(img)
            if img.shape[-1] == 4:
                img = img[:, :, :3]

            for func in self.preprocess_flow:
                if inspect.isfunction(func) and func is not image_backend_adaption:
                    img = func(img)
            img = image_backend_adaption(img)
            inp = to_tensor(np.expand_dims(img, 0)).to(self._model.device).to(self._model.weights[0].data.dtype)
            result = self._model(inp)[0]
            embedding = to_numpy(result)
            return norm(embedding)

        else:
            raise ValueError('the model is not built yet.')


class LanguageModel(Model):
    def __init__(self, inputs=None, input_shape=None, output=None):
        super(LanguageModel, self).__init__(inputs, input_shape, output)
        self.preprocess_flow = []
