from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import inspect
import os
import random
import shutil
import sys
import time
import uuid
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.eager import context, tape, function
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops.losses import util as tf_losses_utils

from trident import __version__
from trident.backend.common import *
from trident.backend.model import ModelBase, progress_bar
from trident.backend.tensorflow_backend import Sequential, Layer, try_map_args_and_call, summary
from trident.backend.tensorflow_ops import *
from trident.backend.tensorflow_serialization import save, load
from trident.callbacks.lr_schedulers import get_lr_scheduler
from trident.data.image_common import *
from trident.layers.tensorflow_layers import SoftMax

from trident.optims.tensorflow_constraints import get_constraint
from trident.optims.tensorflow_losses import get_loss, _ClassificationLoss
from trident.optims.tensorflow_metrics import get_metric
from trident.optims.tensorflow_optimizers import get_optimizer
from trident.optims.tensorflow_regularizers import *

# from tensorflow.python.framework.ops import EagerTensor

__all__ = ['Model', 'ImageClassificationModel', 'ImageDetectionModel', 'ImageSegmentationModel', 'LanguageModel']

_device = 'CPU'
for device in device_lib.list_local_devices():
    if tf.DeviceSpec.from_string(device.name).device_type == 'GPU':
        _device = 'GPU'
        break


def _to_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


class Model(ModelBase):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(Model, self).__init__(inputs, output, input_shape)

    def _initial_graph(self, inputs=None, output=None, input_shape=None):
        input_var = inputs
        out_var = output
        if output is None:
            raise ValueError('There is at least one output')

        if inputs is None:
            if input_shape is None:
                pass
            else:
                input_shape = tf.TensorShape(to_list(input_shape))
                input_name = 'input'
                # input_var =tf.placeholder(dtype=tf.float32, shape=[None])
                # input_var = Input(input_shape, name=input_name)
                self.inputs[input_name] = input_shape

        elif isinstance(inputs, (tuple, list)):
            for inp in inputs:
                if isinstance(inp, tf.keras.Input):
                    input_name = inp.name if inp.name != '' else 'input_{0}'.format(len(self.inputs) + 1)
                    self.inputs[input_name] = inp.get_shape()[1:]
        elif isinstance(inputs, dict):
            input_var = list(inputs.values())
            for k, v in inputs.items():
                if isinstance(v, tf.keras.Input):
                    self.inputs[k] = v.get_shape()[1:]

        if isinstance(output, (Layer, tf.Module)):
            # update notes
            output.nodes = OrderedDict([(mod.uuid, mod) for mod in list(output.modules()) if isinstance(mod, Layer)])
            for mod in output.modules():
                if isinstance(mod, Layer):
                    mod.nodes = output.nodes

            # output.cpu()
            if output.built and hasattr(output, '_output_shape') and is_tensor(output._output_shape):
                self._model = output
                self._outputs['output'] = to_list(output._output_shape)
                self._targets['target'] = to_list(output._output_shape)
            else:
                dummay_input = to_tensor(np.random.standard_normal((1, *input_shape)).astype(np.float32))
                out = out_var(dummay_input)
                # out_var=out_var,input_signature=tf.TensorSpec(shape, dtype=tf.dtypes.float32))
                self._model = out_var

                if is_tensor(out):
                    self._outputs['output'] = out.get_shape()[1:]
                    self._targets['target'] = out.get_shape()[1:]
                else:
                    for i in range(len(out)):
                        self._outputs['output_{0}'.format(i)] = out[i].get_shape()[1:]
                        self._targets['target_{0}'.format(i)] = out[i].get_shape()[1:]
                self._model.signature = get_signature(self._model.forward, 'model')
                self._model.signature.inputs = copy.deepcopy(self.inputs)
                self._model.signature.outputs = copy.deepcopy(self._outputs)
                self._signature = self._model.signature


        elif is_tensor(out_var):
            self._model = out_var
            self._outputs['output'] = out_var.get_shape()[1:]
            self._targets['target'] = out_var.get_shape()[1:]

            self._model.signature = Signature('model')
            self._model.signature.inputs['x'] = list(int_shape(self._model))[1:]
            self._model.signature.outputs = copy.deepcopy(self._outputs)
            self._signature = self._model.signature

        else:
            raise ValueError('')

        self.training_context['current_model'] = self._model
        save_path = os.path.join('Models', '{0}.pth.tar_'.format(self._model.name))
        self.save_path = sanitize_path(make_dir_if_need(save_path))
        self.training_context['save_path'] = self.save_path

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
            return self._model.layers
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
            self.optimizer = optimizer(self._model.parameters() if isinstance(self._model, Layer) else [self._model],**kwargs)
        self.base_lr = kwargs.get('lr', 1e-3)
        self.training_context['optimizer'] = self.optimizer
        self.training_context['base_lr'] = self.base_lr
        self.training_context['current_lr'] = self.base_lr

        return self

    def with_loss(self, loss, loss_weight=1, output_idx=0, start_epoch=0, name='', **kwargs):
        alias = name
        argnames = Signature()
        if (alias is None or len(alias) == 0) and hasattr(loss, '__name__'):
            alias = loss.__name__

        if isinstance(loss, str):

            loss_class = get_loss(loss)
            alias = loss if loss_class is not None else alias
            if alias in self._losses:
                dup_keys = [key for key in self._losses.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            self._losses[alias] = loss_class(**kwargs) if len(kwargs) > 0 else loss()
            if hasattr(loss, 'forward'):
                argnames = get_signature(self._losses[alias].forward, alias)
            else:
                argnames = get_signature(self._losses[alias].__call__, alias)
        elif inspect.isclass(loss) and inspect._is_type(loss):
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
            alias = loss.__class__.__name__ if len(alias) == 0 else alias
            if alias in self._losses:
                dup_keys = [key for key in self._losses.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            self._losses[alias] = loss
            if hasattr(loss, 'forward'):
                argnames = get_signature(self._losses[alias].forward, alias)
            else:
                argnames = get_signature(self._losses[alias].__call__, alias)
        elif inspect.isfunction(loss):
            if alias in self._losses:
                dup_keys = [key for key in self._losses.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            spec = inspect.getfullargspec(loss)
            if len(spec.args) >= 2 and len(spec.args) - 0 if spec.defaults is None else len(spec.defaults) == 2:
                self._losses[alias] = loss
            else:
                self._losses[alias] = partial(loss, **kwargs) if len(kwargs) > 0 else loss()
            argnames = get_signature(loss, alias)
        # create signature
        if hasattr(self._losses[alias], 'signature') and self._losses[alias].signature is not None:
            pass
        else:
            self._losses[alias].signature = argnames
        self._losses[alias].signature.name = alias
        if (len(self._losses[alias].signature.outputs) == 1 and self._losses[alias].outputs.value_list[
            0] is None) or len(self._losses[alias].signature.outputs) == 0:
            self._losses[alias].signature.outputs = OrderedDict()
            self._losses[alias].signature.outputs[alias] = None
        print(self._losses[alias].signature)
        if hasattr(self._losses[alias], 'is_logsoftmax'):
            if isinstance(self._model, Layer):
                last_module = list(self._model.modules())[-1]
                if isinstance(last_module, SoftMax):
                    self._losses[alias].is_logsoftmax = True
        self.loss_weights[alias] = loss_weight

        self._losses[alias].__name__ = alias
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
            metric_class = get_metric(metric)
            if alias in self._metrics:
                dup_keys = [key for key in self._metrics.key_list if alias + '_' in key]
                alias = alias + '_' + str(len(dup_keys) + 1)
            self._metrics[alias] = metric_class(**kwargs) if len(kwargs) > 0 else metric_class()
            if hasattr(metric, 'forward'):
                argnames = get_signature(self._metrics[alias].forward, alias)
            else:
                argnames = get_signature(self._metrics[alias].__call__, alias)
        elif inspect.isclass(metric) and inspect._is_type(metric):
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
        if (len(self._metrics[alias].signature.outputs) == 1 and self._metrics[alias].outputs.value_list[
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
        if 'reg_weight' in args:
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
        self.warmup = warmup
        if self.warmup > 0:
            self.optimizer.adjust_learning_rate(1e-6, False)
            self.training_context['current_lr'] = 1e-6
        return self

    def adjust_learning_rate(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
        self.training_context['current_lr'] = lr

    def do_on_training_start(self):
        self.train()

    def do_on_training_end(self):
        self.eval()

    def do_on_epoch_start(self):
        pass

    def do_on_epoch_end(self):
        pass

    def do_on_batch_end(self):
        if self.training_context['current_batch'] == 0:
            temp = OrderedDict()
            for k in self.training_context['losses'].key_list:
                temp[k] = self.training_context['losses'][k][-1]
            print(temp)

    def do_on_data_received(self, train_data, test_data):
        if 'data_feed' not in self.training_context or len(self.training_context['data_feed']) == 0:
            try:
                data_feed = OrderedDict()
                inshapes = self.inputs.value_list
                outshapes = self.targets.value_list
                available_fields = copy.deepcopy(train_data.key_list)
                if train_data is not None:
                    # check input
                    for arg in self._model.signature.inputs.key_list:
                        data_feed[arg] = ''
                    for arg in self._model.signature.inputs.key_list:
                        if len(train_data) == 1:
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
                                data_shape = list(train_data[item].shape[1:]) if len(train_data[item].shape) > 1 else []
                                if 'target' not in item and 'output' != item and data_shape == inshapes[0]:
                                    data_feed[arg] = item
                                    available_fields.remove(item)
                                    break
                        else:
                            Warning(
                                'input argment {0} cannot mapping to any data, please check it and update the datafeed'.format(
                                    arg))

                    if len(self._signature.inputs.key_list) == 1 and data_feed[self._signature.inputs.key_list[0]] != None:
                        self.training_context['data_feed'] = data_feed

                    # check for target
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
                            target_shape = outshapes[i]
                            for item in available_fields:
                                data_shape = list(train_data[item].shape[1:]) if len(train_data[item].shape) > 1 else []
                                if target_shape == data_shape:
                                    data_feed[arg] = item
                                    available_fields.remove(item)
                                elif ('int64' in str(train_data[item].dtype) or 'int32' in str(
                                        train_data[item].dtype)) and target_shape[:-1] == data_shape:
                                    data_feed[arg] = item
                                    available_fields.remove(item)
                                else:
                                    Warning(
                                        'target argment {0} cannot mapping to any data, please check it and update the datafeed'.format(
                                            arg))
                    if len(self.targets) == 1 and data_feed[self.targets.key_list[0]] != None:
                        self.training_context['current_target'] = train_data[data_feed[self.targets.key_list[0]]]

                    self.training_context['data_feed'] = data_feed
                    print('data_feed', data_feed)
            except:
                PrintException()
        # convert to tensor
        try:
            data_feed = self.training_context['data_feed']
            input_list = [data_feed[arg] for arg in self._signature.inputs.key_list]
            for item in train_data.key_list:
                if item in input_list:
                    # only model 's input argments
                    train_data[item] = to_tensor(train_data[item].copy())
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
        if self.optimizer.grad_tape is None:
            self.optimizer.grad_tape = tf.GradientTape(watch_accessed_variables=False)
        self.optimizer.grad_tape.__enter__()
        if self.training_context['current_epoch'] + self.training_context['current_batch'] > 0:
            self.optimizer.grad_tape.reset()
        self.optimizer.grad_tape.watch(self._model.trainable_variables)
        if len(self.optimizer.grad_tape.watched_variables()) == 0:
            sys.stderr.write('There is no trainable parameters in GradientTape.')

        return train_data, test_data

    def do_preparation_for_loss(self):
        pass

    def get_current_loss(self):
        return self.training_context['current_loss']

    def do_gradient_update(self, log_gradients=False):
        grads = self.optimizer.grad_tape
        if grads._recording:
            grads._pop_tape()
        vars = grads.watched_variables()
        cal_grads = grads.gradient(self.training_context['current_loss'], vars)

        grads_and_vars = zip(cal_grads, vars)
        self.optimizer.grads_and_vars = self.optimizer._filter_grads(grads_and_vars,
                                                                     self.optimizer.gradient_centralization)

        if self.training_context['stop_update'] < 1:
            for callback in self.training_context['callbacks']:
                callback.on_optimization_step_start(self.training_context)

            # grads = tape.gradient(this_loss, self._model.trainable_variables)

            # Gradients does not exist for variables during training using gradienttape
            # for handling this issue, need filter
            # new_vars = []
            # new_grads = []

            if self.training_context['stop_update'] == 0:
                self.optimizer.step(grads_and_vars)

            elif 0 < self.training_context['stop_update'] < 1:
                if random.random() <= self.training_context['stop_update']:
                    self.optimizer.step(grads_and_vars)
            else:
                self.training_context['stop_update'] = self.training_context['stop_update'] - 1

            for callback in self.training_context['callbacks']:
                callback.on_optimization_step_end(self.training_context)

            if log_gradients:
                self.log_gradient(self.optimizer.grads_and_vars)

    def do_on_progress_end(self):
        if self.training_context['current_epoch'] > self.warmup:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(np.array(self.training_context['metrics'][list(self._metrics.keys())[0]]).mean())
                self.training_context['current_lr'] = self.optimizer.lr

    def do_on_excution_exception(self):
        if self.optimizer.grad_tape._recording:
            self.optimizer.grad_tape._pop_tape()

    def log_gradient(self, grads=None):
        grads=tuple(grads)
        grad_dict = OrderedDict()
        for i,(g,v) in enumerate(grads):
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

        if any_abnormal_number(self._model):
            raise ValueError(self._get_name() + '  nan detected!!')

        if isinstance(self._model, Layer):
            save_path = self.get_save_path(save_path, default_folder='Models',
                                           default_file_name='{0}_epoch{1}.pth.tar_'.format(self._model.name,
                                                                                            self.training_context[
                                                                                                'current_epoch']))
            folder, _, _ = split_path(save_path)
            save_path = sanitize_path(save_path)
            self.current_save_path = save_path
            self._model.eval()
            save({'state_dict': self._model.state_dict(), 'backend': 'tensorflow', 'trident_version': __version__,
                  'tensorflow_version': tf.version.VERSION, 'signature': self.signature}, save_path)
            shutil.copy(save_path, save_path.replace('.pth.tar_', '.pth.tar'))
            os.remove(save_path)

            # tf.saved_model.save(self._model, "new_models")
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
            shutil.copy(save_path, save_path.replace('.npy_', '.npy'))
            os.remove(save_path)
            sys.stdout.write('Yor model is a Tensor not a nn.Module, it has saved as numpy array(*.npy) successfully. ')
        else:
            raise ValueError(
                'only Layer or nn.Module as model can export to onnx, yours model is {0}'.format(type(self._model)))

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
        pretrained_dict = load(file_path)

        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']

            # pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        if check_keys(self._model, pretrained_dict):
            self._model.load_state_dict(pretrained_dict, strict=False)
            print('Model loaded!')

        if self.signature is None:
            self.signature = get_signature(self._model.forward)

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

    # def train_model(self, train_data, test_data, current_epoch, current_batch, total_epoch, total_batch,
    #                 is_collect_data=True, is_print_batch_progress=True, is_print_epoch_progress=True,
    #                 is_print_batch_gradients=True, log_gradients=False, log_weights=False, accumulate_grads=False):
    #     try:
    #         self.training_context['current_epoch'] = current_epoch
    #         self.training_context['current_batch'] = current_batch
    #         self.training_context['total_epoch'] = total_epoch
    #         self.training_context['total_batch'] = total_batch
    #         self.training_context['is_collect_data'] = is_collect_data
    #         self.training_context['log_gradients'] = log_gradients
    #         self.training_context['log_weights'] = log_weights
    #         self.training_context['current_model'] = self._model
    #         self.training_context['current_lr'] = self.optimizer.lr
    #         self.training_context['train_data'] = train_data
    #         self.training_context['test_data'] = test_data
    #
    #         self.sample_collect_history.append(1 if is_collect_data else 0)
    #
    #         if self.training_context['current_batch'] == 0:
    #             if self.training_context['current_epoch'] == 0:
    #                 self.do_on_training_start()
    #                 # epoch is not the logical inteval for us to control the flow
    #                 self.training_context['tmp_losses'] = []
    #                 self.training_context['tmp_metrics'] = OrderedDict()
    #                 self.training_context['losses'] = OrderedDict()
    #                 self.training_context['losses']['total_losses'] = []
    #                 self.training_context['metrics'] = OrderedDict()
    #                 self.training_context['grads_state'] = OrderedDict()
    #                 self.training_context['grads_state']['first_layer'] = []
    #                 self.training_context['grads_state']['last_layer'] = []
    #
    #             self.training_context['print_batch_progress_frequency'] = 1
    #             self.training_context['print_epoch_progress_frequency'] = 1
    #
    #             self.do_on_epoch_start()
    #             for callback in self.callbacks:
    #                 callback.on_epoch_start(self.training_context)
    #         self.do_on_batch_start()
    #         for callback in self.callbacks:
    #             callback.on_batch_start(self.training_context)
    #
    #         self.do_on_batch_start()
    #         train_data, test_data = self.do_on_data_received(train_data, test_data)
    #
    #         for callback in self.callbacks:
    #             callback.on_data_received(self.training_context)
    #
    #         if accumulate_grads == False:
    #             self.training_context['current_loss'] = to_tensor(0.0)
    #
    #             self.do_preparation_for_loss()
    #             self.training_context['optimizer'] = self.optimizer
    #
    #         with self.optimizer.grad_tape as g:
    #             # if g.watched_variables() is None or len(g.watched_variables())!=len(
    #             self._model.trainable_variables):
    #             g.reset()
    #             g.watch(self._model.trainable_variables)
    #             output = try_map_args_and_call(self._model, train_data, self.training_context['data_feed'])
    #
    #             if isinstance(output, (list, tuple)):
    #                 for i in range(len(output)):
    #                     train_data[self.outputs.key_list[i]] = output[i]
    #             elif is_tensor(output):
    #                 train_data[self.outputs.key_list[0]] = output
    #             else:
    #                 train_data[self.outputs.key_list[0]] = output
    #
    #             for k, v in self._losses.items():
    #                 if not hasattr(v, 'start_epoch') or (
    #                         hasattr(v, 'start_epoch') and v.start_epoch <= self.training_context['current_epoch']):
    #                     if k not in self.training_context['losses']:
    #                         self.training_context['losses'][k] = []
    #                     try:
    #                         loss_weight = 1
    #                         if k in self.loss_weights:
    #                             loss_weight = self.loss_weights[k]
    #
    #                         this_loss = try_map_args_and_call(v, train_data,
    #                                                           self.training_context['data_feed']) * loss_weight * (
    #                                         1 if self.training_context[
    #                                                  'stop_update'] < 1 else 0)  # v.forward(output, target) if
    #                                                  hasattr(v, 'forward') else v(
    #
    #                         # output, target)
    #
    #                         if isinstance(this_loss, tuple):
    #                             overall_loss = 0
    #                             for i in range(len(this_loss)):
    #                                 if is_abnormal_number(this_loss[i]):
    #                                     sys.stderr.write(
    #                                         'Loss {0} have abnormal number (nan, inf,-inf), trident will skip it '
    #                                         'automaticly, please check anything wrong!!!/n'.format(k))
    #                                 else:
    #                                     overall_loss += this_loss[i]
    #
    #                             self.training_context['current_loss'] += overall_loss
    #
    #                             if is_collect_data:
    #                                 self.training_context['losses'][k].append(float(to_numpy(overall_loss)))
    #                         else:
    #                             if is_abnormal_number(this_loss):
    #                                 sys.stderr.write(
    #                                     'Loss {0} have abnormal number (nan, inf,-inf), trident will skip it '
    #                                     'automaticly, ' 'please check anything wrong!!!/n'.format(k))
    #                             else:
    #                                 self.training_context['current_loss'] += this_loss
    #                             if is_collect_data:
    #                                 self.training_context['losses'][k].append(float(to_numpy(this_loss)))
    #                     except Exception as e:
    #                         print(e)
    #                         PrintException()
    #
    #             self.do_post_loss_calculation()
    #             for callback in self.callbacks:
    #                 callback.on_loss_calculation_end(self.training_context)
    #
    #             if accumulate_grads == False:
    #                 # regularizer
    #                 for k, v in self._regs.items():
    #                     if k + '_Loss' not in self.training_context['losses']:
    #                         self.training_context['losses'][k + '_Loss'] = []
    #                     if 'model' in v.signature.inputs:
    #                         this_loss = v(self._model) if self.training_context['stop_update'] < 1 else to_tensor(0)
    #                     elif 'output' in v.signature.inputs:
    #
    #                         this_loss = try_map_args_and_call(v, train_data, self.training_context['data_feed']) * (
    #                             1 if self.training_context['stop_update'] < 1 else 0)
    #
    #                     self.training_context['current_loss'] += this_loss  # self.training_context[
    #
    #                     # 'current_loss'] + this_loss
    #                     if is_collect_data:
    #                         self.training_context['losses'][k + '_Loss'].append(float(to_numpy(this_loss)))
    #
    #                 self.training_context['optimizer'] = self.optimizer
    #
    #             self.do_pre_optimization_step()
    #
    #         self.do_gradient_update(log_gradients and is_collect_data, grads=g)
    #         self.training_context['optimizer'] = self.optimizer
    #         self.training_context['current_lr'] = self.optimizer.lr
    #
    #         # ON_POSTBACKWARD_CALCULATION
    #         self.do_post_gradient_update()
    #         self.training_context['grads'] = 0
    #
    #         # model comfirm
    #         for k, v in self._constraints.items():
    #             v(self._model)
    #
    #         if log_weights and is_collect_data:
    #             if isinstance(self._model, Layer):
    #                 self.log_weight(weghts=self._model.weights)
    #             elif is_tensor(self._model):
    #                 self.log_weight(weghts=self._model)
    #
    #         if test_data is not None and len(test_data) > 0 and self.training_context['stop_update'] < 1:
    #             output = try_map_args_and_call(self._model, test_data, self.training_context['data_feed'])
    #             if isinstance(output, (list, tuple)):
    #                 for i in range(len(output)):
    #                     test_data[self.outputs.key_list[i]] = output[i]
    #             elif 'tensor' in output.__class__.__name__.lower():
    #                 test_data[self.outputs.key_list[0]] = output
    #             else:
    #                 test_data[self.outputs.key_list[0]] = output
    #
    #         # ON_EVALUATION_START
    #         self.do_on_metrics_evaluation_start()
    #         for callback in self.training_context['callbacks']:
    #             callback.on_metrics_evaluation_start(self.training_context)
    #
    #         for k, v in self._metrics.items():
    #             collect_history = getattr(v, 'collect_history')
    #             if k not in self.training_context['metrics']:
    #                 self.training_context['tmp_metrics'][k] = []
    #                 self.training_context['metrics'][k] = []
    #                 if not collect_history == False:
    #                     self.training_context['metrics'][k] = []
    #
    #             this_metric = try_map_args_and_call(v, train_data, self.training_context['data_feed']) * (
    #                 1 if self.training_context['stop_update'] < 1 else 0)
    #             self.training_context['tmp_metrics'][k].append(to_numpy(this_metric).mean())
    #
    #             if test_data is not None and len(test_data) > 0 and collect_history != False:
    #                 if k not in self.training_context['out_sample_metrics']:
    #                     self.training_context['out_sample_metrics'][k] = []
    #
    #                 this_out_metric = try_map_args_and_call(v, test_data, self.training_context['data_feed'])
    #                 self.training_context['out_sample_metrics'][k].append(to_numpy(this_out_metric).mean())
    #
    #         # ON_EVALUATION_END
    #         self.do_on_metrics_evaluation_end()
    #         for callback in self.training_context['callbacks']:
    #             callback.on_metrics_evaluation_end(self.training_context)
    #
    #         # callback's metric can keep in epoch_metric_history
    #         for k, v in self.training_context['tmp_metrics'].items():
    #             if not getattr(self._metrics[k], 'collect_history') == False:
    #                 if k not in self.epoch_metric_history:
    #                     self.epoch_metric_history[k] = []
    #
    #         if is_collect_data:
    #             for k, v in self.training_context['tmp_metrics'].items():
    #                 if not getattr(self._metrics[k], 'collect_history') == False and v is not None:
    #                     self.training_context['metrics'][k].append(float(to_numpy(v).mean()))
    #                     self.training_context['tmp_metrics'][k] = []
    #
    #         # print batch progresss
    #         if is_print_batch_progress:
    #             self.do_on_progress_start()
    #             for callback in self.training_context['callbacks']:
    #                 callback.on_progress_start(self.training_context)
    #
    #             self.print_batch_progress(self.training_context['print_batch_progress_frequency'])
    #
    #             self.training_context['print_batch_progress_frequency'] = 1
    #             self.do_on_progress_end()
    #             for callback in self.training_context['callbacks']:
    #                 callback.on_progress_end(self.training_context)
    #         else:
    #             self.training_context['print_batch_progress_frequency'] += 1
    #
    #         if test_data is not None and len(test_data) > 0:
    #             print(self.training_context['model_name'] + ': out-of-sample evaluation: ', ','.join(
    #                 ['{0}: {1:<8.3%}'.format(k, v[-1]) for k, v in
    #                  self.training_context['out_sample_metrics'].items()]))
    #
    #         # ON_BATCH_END
    #         self.do_on_batch_end()
    #         for callback in self.training_context['callbacks']:
    #             callback.on_batch_end(self.training_context)
    #
    #         if self.training_context['current_batch'] == self.training_context['total_batch'] - 1:
    #             self.do_on_epoch_end()
    #
    #             slice_cnt = sum(self.sample_collect_history[-1 * total_batch:])
    #             self.epoch_loss_history['total_losses'].append(
    #                 np.array(self.training_context['losses']['total_losses'][-1 * slice_cnt:]).mean())
    #             for k, v in self.training_context['metrics'].items():
    #                 if len(v) >= slice_cnt:
    #                     self.epoch_metric_history[k].append(np.array(v[-1 * slice_cnt:]).mean())
    #
    #             if is_print_epoch_progress:
    #                 self.do_on_progress_start()
    #                 for callback in self.training_context['callbacks']:
    #                     callback.on_progress_start(self.training_context)
    #                 self.print_epoch_progress(self.training_context['print_epoch_progress_frequency'])
    #                 self.training_context['print_epoch_progress_frequency'] = 1
    #                 self.do_on_progress_end()
    #                 for callback in self.training_context['callbacks']:
    #                     callback.on_progress_end(self.training_context)
    #             else:
    #                 self.training_context['print_epoch_progress_frequency'] += 1
    #
    #             for callback in self.training_context['callbacks']:
    #                 callback.on_epoch_end(self.training_context)
    #
    #             if self.training_context['current_epoch'] == self.training_context['total_epoch'] - 1:
    #                 self.do_on_training_end()
    #                 for callback in self.training_context['callbacks']:
    #                     callback.on_training_end(self.training_context)
    #     except Exception:
    #         PrintException()

    def summary(self):
        if self._model.built:
            return summary(self._model, self.inputs.value_list)
        else:
            raise ValueError('This model has not yet been built. ')

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
                mod_str = repr(value)
                mod_str = addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
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


class ImageClassificationModel(Model):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(ImageClassificationModel, self).__init__(inputs, output, input_shape)

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


class ImageDetectionModel(Model):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(ImageDetectionModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = []
        self.detection_threshould = 0.5

    def infer_single_image(self, img, scale=1):
        if self._model.built:
            self._model.to(self.device)
            self._model.eval()
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
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(ImageSegmentationModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = []


class LanguageModel(Model):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(LanguageModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = []
