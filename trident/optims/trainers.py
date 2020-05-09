import copy
import inspect
import json
import os
import shutil
import sys
import time
import uuid
from functools import partial

import numpy as np

from ..backend.common import to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path
from ..callbacks import *
from ..callbacks.visualization_callbacks import *
from ..data.data_provider import *
from ..misc.ipython_utils import *
from ..misc.visualization_utils import tile_rgb_images, loss_metric_curve

__all__ = ['progress_bar',  'ModelBase', 'TrainingPlan']

_session = get_session()
_backend = _session.backend
if _backend == 'pytorch':
    import torch
    import torch.nn as nn
    from ..backend.pytorch_backend import *
    from ..backend.pytorch_ops import *
    from ..optims.pytorch_optimizers import *
elif _backend == 'tensorflow':
    import tensorflow as tf
    from ..backend.tensorflow_backend import *
    from ..backend.tensorflow_ops import *
    from ..optims.tensorflow_optimizers import *


_, term_width = get_terminal_size()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time




class ModelBase(object):
    def __init__(self, inputs=None, output=None, input_shape=None, name='', **kwargs):
        self.inputs = OrderedDict()
        self._outputs = OrderedDict()
        self._targets = OrderedDict()
        self._model = None
        self.name = name
        self.optimizer = None
        self.lr_scheduler = None
        self._losses = OrderedDict()
        self._metrics = OrderedDict()
        self.loss_weights = OrderedDict()
        self._signature = None
        self._regs = OrderedDict()
        self._constraints = OrderedDict()
        self.base_lr = None
        self.warmup = 0
        self.sample_collect_history = []
        self.preprocess_flow = []

        self.epoch_loss_history = OrderedDict()
        self.epoch_loss_history['total_losses'] = []
        self.epoch_metric_history = OrderedDict()
        self.weights_history = []
        self.gradients_history = []
        self.input_history = []
        self.target_history = []
        self.callbacks = []
        self.training_context = {'losses': OrderedDict(),  # loss_wrapper
                                 'metrics': OrderedDict(),  # loss_wrapper
                                 'grads_state': OrderedDict(),  # loss_wrapper
                                 'tmp_losses': [],  # loss_wrapper
                                 'tmp_metrics': OrderedDict(),  # loss_wrapper
                                 'out_sample_metrics': OrderedDict(),
                                 'grads': None,
                                 'optimizer': None,  # optimizer
                                 'stop_training': False,  # stop training
                                 'total_epoch': -1,  # current_epoch
                                 'total_batch': -1,  # current_batch
                                 'current_epoch': -1,  # current_epoch
                                 'current_batch': -1,  # current_batch
                                 'current_model': None,  # current model
                                 'current_input': None,  # current input
                                 'current_target': None,  # current target
                                 'current_output': None,  # current output
                                 'current_loss': None,  # current loss
                                 'best_metric': None,  # current loss
                                 'best_model': None,  # current model
                                 'loss_history': None, 'metric_history': None, 'base_lr': self.base_lr,  # current loss
                                 'current_lr': None,  # current loss
                                 'save_path': None, 'is_collect_data': True, 'callbacks': self.callbacks,
                                 'stop_update': 0, 'retain_graph': False}
        self.training_context['losses']['total_losses'] = []

        self._initial_graph(inputs, output, input_shape)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if isinstance(value, Layer):
            if value.built:
                inp_shape = copy.deepcopy(self.inputs.value_list[0])
                self.inputs = OrderedDict()
                self.targets = OrderedDict()
                self._initial_graph(output=value, input_shape=inp_shape)
                self._signature =get_signature(value.forward)
            else:
                self.inputs = OrderedDict()
                self.targets = OrderedDict()
                if self._model is not None:
                    self._model.cpu()
                self._initial_graph(output=value, input_shape=value.input_shape)
                self._signature = get_signature(value.forward)
        elif isinstance(value, nn.Module):
            self._model = value
            self._signature = get_signature(value.forward)
        elif isinstance(value, np.ndarray) or 'tensor' in value.__name__.lower():
            self._model = to_tensor(value)
        else:
            raise ValueError('Only Layer, Module, Image and Tensor can be valid model')


    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs=value


    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets=value


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
    def losses(self):
        return self._losses

    @property
    def metrics(self):
        return self._metrics

    @property
    def signature(self):
        if self._signature is not None:
            return self._signature
        elif self.model is not None and hasattr(self.model, 'signature'):
            return self.model.signature
        else:
            self.model.signature = get_signature(self.model.forward)
            self._signatures = self.model.signature
            return self.model.signature

    @signature.setter
    def signature(self, value):
        if self.model is not None:
            self.model.signature = value

    def update_signature(self, arg_names):
        if self._signature is None:
            new_inputs = OrderedDict()
            for i in range(len(arg_names[:len(self.inputs)])):
                arg = arg_names[:len(self.inputs)][i]
                new_inputs[arg] = self.inputs.value_list[0]
            self.inputs = new_inputs
            self._signature = new_inputs
            self._model.signature = new_inputs
            new_outputs = OrderedDict()
            new_target = OrderedDict()
            outputs_args = arg_names[len(self.signature.key_list):]
            outputs=self._outputs
            targets=self._targets
            for i in range(len(outputs_args)):
                arg = outputs_args[i]
                new_outputs[arg] = outputs.value_list[0]

                target_arg = arg.replace('output', 'target')
                if 'target' not in target_arg:
                    target_arg = 'target_' + target_arg
                new_target[target_arg] = targets.value_list[0]
            self._outputs = new_outputs
            self._targets = new_target
            print(self._signature.key_list)
            print(self._outputs.key_list)
            print(self._targets.key_list)
        elif not isinstance(arg_names, (list, tuple)):
            raise ValueError('arg_names should be list or tuple')
        elif len(self.signature.key_list) < len(arg_names) and len(self.signature.key_list) + len(self.outputs) == len(
                arg_names):
            self.signature = OrderedDict()
            for arg in arg_names[:len(self.signature.key_list)]:
                self.signature[arg] = None
            self.inputs = self.signature

            new_outputs = OrderedDict()
            new_target = OrderedDict()
            outputs_args = arg_names[len(self.signature.key_list):]
            for i in range(len(outputs_args)):
                arg = outputs_args[i]
                new_outputs[arg] = self.outputs.value_list[0]

                target_arg = arg.replace('output', 'target')
                if 'target' not in target_arg:
                    target_arg = 'target_' + target_arg
                new_target[target_arg] = self.targets.value_list[0]
            self.outputs = new_outputs
            self.targets = new_target
            print(self._signature.key_list)
            print(self.outputs.key_list)
            print(self.targets.key_list)
        elif len(self._signature.key_list) != len(arg_names):
            raise ValueError('data deed and arg_names should be the same length')
        else:
            #self.signature = namedtuple('signature', arg_names)
            print(self.signature.key_list)

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

    def data_preprocess(self, img_data):
        if img_data.ndim==4:
            return to_tensor(to_numpy([self.data_preprocess(im) for im in img_data]))
        if len(self.preprocess_flow) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self.preprocess_flow:
                if not fc.__qualname__.startswith('random_') or  'crop' in fc.__qualname__  or  'rescale' in fc.__qualname__  or  (fc.__qualname__.startswith('random_') and random.randint(0,10)%2==0):
                    img_data = fc(img_data)
            img_data = to_tensor(image_backend_adaption(img_data)).unsqueeze(0)
            return img_data
        else:
            return img_data

    def reverse_data_preprocess(self, img_data: np.ndarray):
        if img_data.ndim==4:
            return to_numpy([self.reverse_data_preprocess(im) for im in img_data])
        if len(self.reverse_preprocess_flow) == 0:
            return reverse_image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_preprocess_flow:
                img_data = fc(img_data)
            img_data = reverse_image_backend_adaption(img_data)
        return img_data

    def _initial_graph(self, inputs=None, output=None, input_shape=None):
        pass

    def complie(self, optimizer, losses=None, metrics=None, loss_weights=None, sample_weight_mode=None,
                weighted_metrics=None, target_tensors=None):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self._model(*input, **kwargs)

    def with_optimizer(self, optimizer, **kwargs):
        return self

    def with_loss(self, loss, loss_weight=1, output_idx=0, name='', **kwargs):
        return self

    def with_metric(self, metric, output_idx=0, name='', **kwargs):
        return self

    def with_regularizer(self, reg, **kwargs):
        return self

    def with_constraint(self, constraint, **kwargs):
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

    def with_learning_rate_scheduler(self, lr_schedule, warmup=0, **kwargs):
        return self

    def reset_training_context(self):
        self.training_context = {'losses': OrderedDict(),
                                 'metrics': OrderedDict(),
                                 'grads_state': OrderedDict(),
                                 'tmp_losses': [],
                                 'tmp_metrics': OrderedDict(),
                                 'out_sample_metrics': OrderedDict(),
                                 'grads': None, 'optimizer': None,  # optimizer
                                 'stop_training': False,  # stop training
                                 'total_epoch': -1,  # current_epoch
                                 'total_batch': -1,  # current_batch
                                 'current_epoch': -1,  # current_epoch
                                 'current_batch': -1,  # current_batch
                                 'current_model': self._model,  # current model
                                 'current_input': None,  # current input
                                 'current_target': None,  # current target
                                 'current_output': None,  # current output
                                 'current_loss': None,  # current loss
                                 'best_metric': None,  # current loss
                                 'best_model': None,  # current model
                                 'loss_history': None, 'metric_history': None, 'base_lr': self.base_lr,  # current loss
                                 'current_lr': None,  # current loss
                                 'save_path': None, 'callbacks': self.callbacks, 'stop_update': 0,
                                 'retain_graph': False}

    def adjust_learning_rate(self, lr):
        raise NotImplementedError

    def rebinding_input_output(self, input_shape):
        pass

    def do_on_training_start(self):
        # set model as training state
        # zero grad
        pass

    def do_on_training_end(self):
        # set model as training state
        # zero grad
        pass

    def do_on_epoch_start(self):
        # set model as training state
        # zero grad
        pass

    def do_on_epoch_end(self):
        # set model as training state
        # zero grad
        pass

    def do_on_batch_start(self):
        # set model as training state
        # zero grad
        pass

    def do_on_batch_end(self):
        # set model as training state
        # zero grad
        pass

    def do_on_data_received(self, train_data, test_data):

        return train_data, test_data

    def do_preparation_for_loss(self):
        pass

    def do_post_loss_calculation(self):
        pass

    def do_pre_optimization_step(self):
        # set model as training state
        # zero grad
        pass

    def do_gradient_update(self, log_gradients=False):
        pass

    def do_post_gradient_update(self):
        self.training_context['tmp_losses'].append(to_numpy(self.training_context['current_loss']).mean())
        if self.training_context['is_collect_data'] == True:
            self.training_context['losses']['total_losses'].append(
                float(to_numpy(self.training_context['tmp_losses']).mean()))
            self.training_context['tmp_losses'] = []

    def do_on_metrics_evaluation_start(self):
        pass

    def do_on_metrics_evaluation_end(self):
        pass

    def do_on_progress_start(self):
        # set model as training state
        # zero grad
        pass

    def do_on_progress_end(self):
        # set model as training state
        # zero grad
        pass

    def log_gradient(self, grads=None):
        raise NotImplementedError

    def log_weight(self, weghts=None):
        raise NotImplementedError

    def merge_grads(self, old_grads, new_grades):
        raise NotImplementedError

    def get_save_path(self,save_path='',default_folder='log',default_file_name=''):
        _,default_filename,default_ext=split_path(default_file_name)
        if save_path is None or len(save_path) == 0:
            save_path = self.training_context.get('save_path', getattr(self, 'save_path',''))

        folder,filename,ext=split_path(save_path)
        if folder=='':
            folder=default_folder
        if filename=='':
            filename=default_filename
        save_path = os.path.join(folder, filename+default_ext)
        make_dir_if_need(sanitize_path(save_path))
        return sanitize_path(save_path)



    def save_model(self, file_path, ):
        raise NotImplementedError

    def save_onnx(self, file_path):
        raise NotImplementedError

    def save_history(self, file_path=None):
        default_file_name = '{0}_history_{1}.json_'.format(self._model.name, self.training_context['execution_id'])
        save_path = self.get_save_path(file_path, default_folder='Log',default_file_name=default_file_name)
        folder,filename,ext=split_path(save_path)
        save_path=os.path.join(folder,default_file_name)
        out=OrderedDict()
        out['batch_loss_history']=self.batch_loss_history
        out['batch_metric_history'] = self.batch_metric_history
        out['epoch_loss_history'] = self.epoch_loss_history
        out['epoch_metric_history'] = self.epoch_metric_history
        with open(save_path, 'w') as f:
            jstring=json.dumps(out, indent=4)
            f.write(jstring)
            shutil.copy(save_path, save_path.replace('.json_', '.json'))


    def save_weights(self, file_path):
        raise NotImplementedError

    def load_model(self, file_path, ):
        raise NotImplementedError

    def print_batch_progress(self, print_batch_progress_frequency):
        slice_cnt = sum(self.sample_collect_history[-1 * print_batch_progress_frequency:])
        metric_strings=[]
        for k, v in self._metrics.items():
            format_string='.3%'
            collect_history = self._metrics[k].collect_history
            if collect_history!=False and k in self.batch_metric_history and len(self.batch_metric_history[k])>=slice_cnt:
                metric_value=np.array(self.batch_metric_history[k][-1 * slice_cnt:]).mean()
                if metric_value.item()>5:
                    format_string = '.3f'
                elif metric_value.item()<1e-4:
                    format_string = '.3e'
                metric_strings.append('{0}: {1:<8{2}}'.format(k,metric_value,format_string))
            elif collect_history==False and k in self.training_context['tmp_metrics']:
                metric_value = np.array(self.training_context['tmp_metrics'][k]).mean()
                if metric_value.item() > 5:
                    format_string = '.3%'
                metric_strings.append('{0}: {1:<8{2}}'.format(k,metric_value,format_string))
                self.training_context['tmp_metrics'][k]=[]


        progress_bar(self.training_context['current_batch'], self.training_context['total_batch'],
                 'Loss: {0:<8.3f}| {1} | learning rate: {2:<10.3e}| epoch: {3}'.format(
                     np.array(self.batch_loss_history['total_losses'][-1 * slice_cnt:]).mean(), ','.join(metric_strings), self.training_context['current_lr'],
                     self.training_context['current_epoch']), name=self.name)

    def print_epoch_progress(self, print_epoch_progress_frequency):
        progress_bar(self.training_context['current_epoch'], self.training_context['total_epoch'],
                     'Loss: {0:<8.3f}| {1} | learning rate: {2:<10.3e}'.format(
                         np.array(self.epoch_loss_history['total_losses']).mean(), ','.join(
                             ['{0}: {1:<8.3%}'.format(k, np.array(v).mean()) for k, v in
                              self.epoch_metric_history.items()]), self.training_context['current_lr']), name=self.name)

    def train_model(self, train_data, test_data, current_epoch, current_batch, total_epoch, total_batch,
                    is_collect_data=True, is_print_batch_progress=True, is_print_epoch_progress=True,
                    is_print_batch_gradients=True, log_gradients=False, log_weights=False, accumulate_grads=False):
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


            self.sample_collect_history.append(1 if is_collect_data else 0)

            if self.training_context['current_batch'] == 0:
                if self.training_context['current_epoch'] == 0:
                    self.do_on_training_start()
                    # epoch is not the logical inteval for us to control the flow
                    self.training_context['tmp_losses'] = []
                    self.training_context['tmp_metrics'] = OrderedDict()
                    self.training_context['losses'] = OrderedDict()
                    self.training_context['losses']['total_losses'] = []
                    self.training_context['metrics'] = OrderedDict()
                    self.training_context['grads_state'] = OrderedDict()
                    self.training_context['grads_state']['first_layer'] = []
                    self.training_context['grads_state']['last_layer'] = []

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
                self.training_context['current_loss'] = 0
                self.do_preparation_for_loss()
                self.training_context['optimizer'] = self.optimizer

            output = try_map_args_and_call(self._model, train_data, self.training_context['data_feed'])
            # write output in to data

            if isinstance(output, (list, tuple)):
                for i in range(len(output)):
                    train_data[self.outputs.key_list[i]] = output[i]
            elif 'tensor' in output.__class__.__name__.lower():
                train_data[self.outputs.key_list[0]] =output
            else:
                train_data[self.outputs.key_list[0]] = output


            # confirm singleton
            # output=unpack_singleton(output)

            # losss
            for k, v in self._losses.items():
                if not hasattr(v,'start_epoch') or (hasattr(v,'start_epoch') and v.start_epoch<=self.training_context['current_epoch']):
                    if k not in self.training_context['losses']:
                        self.training_context['losses'][k] = []
                    try:
                        loss_weight = 1
                        if k in self.loss_weights:
                            loss_weight = self.loss_weights[k]

                        this_loss = to_tensor(loss_weight)*try_map_args_and_call(v, train_data, self.training_context['data_feed']) if  self.training_context['stop_update']<1 else torch.tensor(0)# v.forward(output, target) if hasattr(v, 'forward') else v(
                        # output, target)

                        if isinstance(this_loss, tuple):
                            overall_loss =0
                            for i in range(len(this_loss)):
                                if is_abnormal_number(this_loss[i]):
                                    sys.stderr.write('Loss {0} have abnormal number (nan, inf,-inf), trident will skip it automaticly, please check anything wrong!!!/n'.format(k))
                                else:
                                    overall_loss += this_loss[i]
                            self.training_context['current_loss'] += overall_loss
                            if is_collect_data:
                                self.training_context['losses'][k].append(float(to_numpy(overall_loss)))
                        else:
                            if is_abnormal_number(this_loss):
                                sys.stderr.write( 'Loss {0} have abnormal number (nan, inf,-inf), trident will skip it automaticly, ' 'please check anything wrong!!!/n'.format(k))
                            else:
                                self.training_context['current_loss'] += this_loss
                            if is_collect_data:
                                self.training_context['losses'][k].append(float(to_numpy(this_loss)))
                    except Exception as e:
                        print(e)
                        PrintException()

            self.do_post_loss_calculation()
            for callback in self.callbacks:
                callback.on_loss_calculation_end(self.training_context)

            if accumulate_grads == False:
                # regularizer
                for k, v in self._regs.items():
                    if k + '_Loss' not in self.training_context['losses']:
                        self.training_context['losses'][k + '_Loss'] = []
                    this_loss=0
                    if 'model' in v.signature:
                        this_loss = v(self._model) if self.training_context['stop_update'] < 1 else torch.tensor(0)
                    elif 'output' in v.signature:

                        this_loss = try_map_args_and_call(v, train_data, self.training_context['data_feed']) if self.training_context['stop_update'] < 1 else to_tensor(0)


                    self.training_context['current_loss'] += this_loss  # self.training_context[
                    # 'current_loss'] + this_loss
                    if is_collect_data:
                        self.training_context['losses'][k + '_Loss'].append(float(to_numpy(this_loss)))


                self.training_context['optimizer'] = self.optimizer
                # self.do_post_loss_calculation()
                #
                # for callback in self.callbacks:
                #     callback.on_loss_calculation_end(self.training_context)

                self.do_pre_optimization_step()
                self.do_gradient_update(log_gradients and is_collect_data)
                self.training_context['optimizer'] = self.optimizer
                self.training_context['current_lr'] = self.optimizer.lr

                # ON_POSTBACKWARD_CALCULATION
                self.do_post_gradient_update()

                # model comfirm
                for k, v in self._constraints.items():
                    if self.training_context['stop_update'] == 0 :
                        v(self._model)

                if log_weights and is_collect_data:
                    if isinstance(self._model,Layer):
                        self.log_weight(weghts=self._model.weights)
                    elif is_tensor(self._model):
                        self.log_weight(weghts=self._model)

                if test_data is not None and len(test_data) > 0 and  self.training_context['stop_update']<1 :
                    output = try_map_args_and_call(self._model, test_data, self.training_context['data_feed'])
                    if isinstance(output, (list, tuple)):
                        for i in range(len(output)):
                            test_data[self.outputs.key_list[i]] = output[i]
                    elif 'tensor' in output.__class__.__name__.lower():
                        test_data[self.outputs.key_list[0]] = output
                    else:
                        test_data[self.outputs.key_list[0]] = output



                # ON_EVALUATION_START
                self.do_on_metrics_evaluation_start()
                for callback in self.training_context['callbacks']:
                    callback.on_metrics_evaluation_start(self.training_context)


                for k, v in self._metrics.items():
                    collect_history = getattr(v,'collect_history')
                    if k not in self.training_context['metrics'] :
                        self.training_context['tmp_metrics'][k] = []
                        self.training_context['metrics'][k] = []
                        if not collect_history==False:
                            self.training_context['metrics'][k] = []

                    this_metric = try_map_args_and_call(v, train_data, self.training_context['data_feed']) if  self.training_context['stop_update']<1 else torch.tensor(0)
                    self.training_context['tmp_metrics'][k].append(to_numpy(this_metric).mean())

                    if test_data is not None and len(test_data) > 0 and collect_history!=False :
                        if k not in self.training_context['out_sample_metrics']:
                            self.training_context['out_sample_metrics'][k] = []

                        this_out_metric = try_map_args_and_call(v, test_data , self.training_context['data_feed'])
                        self.training_context['out_sample_metrics'][k].append(to_numpy(this_out_metric).mean())



                # ON_EVALUATION_END
                self.do_on_metrics_evaluation_end()
                for callback in self.training_context['callbacks']:
                    callback.on_metrics_evaluation_end(self.training_context)

                #callback's metric can keep in epoch_metric_history
                for k, v in self.training_context['tmp_metrics'].items():
                    if not getattr(self._metrics[k], 'collect_history') ==False:
                        if k not in self.epoch_metric_history:
                            self.epoch_metric_history[k] = []

                if is_collect_data:
                    for k, v in self.training_context['tmp_metrics'].items():
                        if not getattr(self._metrics[k], 'collect_history')== False:
                            self.training_context['metrics'][k].append(float(to_numpy(v).mean()))
                            self.training_context['tmp_metrics'][k] = []

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

                if test_data is not None and len(test_data) > 0:
                    print(self.training_context['model_name']+': out-of-sample evaluation: ',','.join(['{0}: {1:<8.3%}'.format(k, v[-1]) for k, v in self.training_context['out_sample_metrics'].items()]))


                # ON_BATCH_END
                self.do_on_batch_end()
                for callback in self.training_context['callbacks']:
                    callback.on_batch_end(self.training_context)

            if self.training_context['current_batch'] == self.training_context['total_batch'] - 1:
                self.do_on_epoch_end()

                slice_cnt = sum(self.sample_collect_history[-1 * total_batch:])
                self.epoch_loss_history['total_losses'].append(
                    np.array(self.training_context['losses']['total_losses'][-1 * slice_cnt:]).mean())
                for k, v in self.training_context['metrics'].items():
                    if len(v)>=slice_cnt:
                        self.epoch_metric_history[k].append(np.array(v[-1 * slice_cnt:]).mean())

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
            PrintException()

    def summary(self):
        raise NotImplementedError

    def predict(self,input):
        if isinstance(input,(torch.Tensor,np.ndarray)):
            if isinstance(input,np.ndarray):
                input=to_tensor(input)
            if len(input.shape)==len(self.inputs.value_list[0])+1:
                return self._model(input.unsqueeze(0))
            elif len(input.shape)==len(self.inputs.value_list[0]):
                return self._model(input)
            else:
                return None
        else:
            raise NotImplementedError


    def test(self, input,target):
        raise NotImplementedError


class TrainingPlan(object):
    def __init__(self):
        self.training_items = OrderedDict()
        self.training_names = OrderedDict()
        self._dataloaders = OrderedDict()
        self.num_epochs = 1
        self._minibatch_size = 1
        self.warmup = 0
        self.default_collect_data_inteval = 1
        self.print_progress_frequency = 10
        self.print_progress_unit = 'batch'
        self.print_progress_on_epoch_end = False
        self.out_sample_evaluation_frequency =1
        self.out_sample_evaluation_unit = 'epoch'
        self.out_sample_evaluation_on_epoch_end = True
        self.save_model_frequency = -1
        self.save_model_unit = 'batch'
        self.execution_id = None

        self._is_optimizer_warmup = False

        self.callbacks = []  # if self.callbacks is None:  #     self.callbacks = [  # NumberOfEpochsStoppingCriterionCallback(1)]  # elif not any([issubclass(type(cb),  # StoppingCriterionCallback) for cb in self.callbacks]):  #  #     self.callbacks.append(  # NumberOfEpochsStoppingCriterionCallback(1))

    @property
    def minibatch_size(self):
        return self._minibatch_size

    @minibatch_size.setter
    def minibatch_size(self, value):
        self._minibatch_size = value
        for i, (k, v) in enumerate(self._dataloaders.items()):
            v.minibatch_size = value
            self._dataloaders[k] = v

    def with_callbacks(self, *callbacks):
        if len(self.callbacks) == 0:
            self.callbacks = to_list(callbacks)
        else:
            self.callbacks.extend(callbacks)
        return self

    def __getattr__(self, name):
        if name == 'self':
            return self
        if '_training_items' in self.__dict__:
            _training_items = self.__dict__['_training_items']
            if name in _training_items:
                return _training_items[name]

        if '_dataloaders' in self.__dict__:
            _dataloaders = self.__dict__['_dataloaders']
            if name in _dataloaders:
                return _dataloaders[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
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
        module_attrs = dir(self.__class__)

        # training_items = list(self.training_items.keys())
        keys = module_attrs

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    @classmethod
    def create(cls):
        plan = cls()
        return plan

    def add_training_item(self, training_item,name='', start_epoch=0):
        n = len(self.training_items)

        alias=name if len(name)>0 else  training_item.name
        alias = alias if len(alias) > 0 else training_item.model.name
        alias = alias if len(alias) > 0 else  'model {0}'.format(n)


        if len(training_item.name) > 0:
            self.training_names[n] = training_item.name
        else:
            if len(name) > 0:
                training_item.name = name
                if isinstance(training_item.model,Layer):
                    training_item.model._name=name
                self.training_names[n] = name
            else:
                training_item.name = 'model {0}'.format(n)
                if isinstance(training_item.model, Layer):
                    training_item.model._name ='model {0}'.format(n)
                self.training_names[n] = 'model {0}'.format(n)
        self.training_items[n] = training_item
        self.training_items[n].start_epoch = start_epoch
        return self

    def with_data_loader(self, data_loader, **kwargs):
        self._dataloaders[data_loader.__class__.__name__] = data_loader
        return self

    def repeat_epochs(self, num_epochs: int):
        self.num_epochs = num_epochs
        return self

    def within_minibatch_size(self, minibatch_size: int):
        self.minibatch_size = minibatch_size
        return self

    def out_sample_evaluation_scheduling(self, frequency: int, unit='batch', on_epoch_end=True):
        self.out_sample_evaluation_on_epoch_end = on_epoch_end
        self.out_sample_evaluation_frequency = frequency
        if unit not in ['batch', 'epoch']:
            raise ValueError('unit should be batch or epoch')
        else:
            self.out_sample_evaluation_unit = unit

        return self
    def print_progress_scheduling(self, frequency: int, unit='batch', on_epoch_end=True, show_loss_metric_curve=True):
        self.print_progress_on_epoch_end = on_epoch_end
        self.print_progress_frequency = frequency
        self.default_collect_data_inteval = frequency
        if unit not in ['batch', 'epoch']:
            raise ValueError('unit should be batch or epoch')
        else:
            self.print_progress_unit = unit

        return self

    def print_gradients_scheduling(self, frequency: int, unit='batch'):

        pg = PrintGradientsCallback(batch_inteval=frequency)
        pg.is_shared = False
        self.callbacks.append(pg)
        return self

    def save_model_scheduling(self, frequency: int, unit='batch'):
        self.save_model_frequency = frequency
        if unit not in ['batch', 'epoch']:
            raise ValueError('unit should be batch or epoch')
        else:
            self.save_model_unit = unit
        return self

    def display_tile_image_scheduling(self, frequency: int, unit='batch', save_path: str = None,
                                      name_prefix: str = 'tile_image_{0}.png', include_input=True, include_output=True,
                                      include_target=True, include_mask=None, imshow=None):
        if (is_in_ipython() or is_in_colab()) and imshow is None:
            imshow=True
        elif not is_in_ipython() and not  is_in_colab()and imshow is None:
            imshow=False
        if unit not in ['batch', 'epoch']:
            raise ValueError('unit should be batch or epoch')

        tile = TileImageCallback(frequency if unit == 'epoch' else -1, frequency if unit == 'batch' else -1,
                                 save_path=save_path, name_prefix=name_prefix, include_input=include_input,
                                 include_output=include_output, include_target=include_target,
                                 include_mask=include_mask, imshow=imshow)
        self.callbacks.append(tile)

        return self

    def display_loss_metric_curve_scheduling(self, frequency: int, unit='batch', save_path: str = None,
                                             name_prefix: str = 'loss_metric_curve_{0}.png',
                                             clean_ipython_output_frequency=5, imshow=None):
        if (is_in_ipython() or is_in_colab()) and imshow is None:
            imshow=True
        elif not is_in_ipython() and not  is_in_colab()and imshow is None:
            imshow=False

        if save_path is not None:
            folder, _,_ =split_path(save_path)
            if not os.path.exists(folder):
                try:
                    os.makedirs(folder)
                except Exception as e:
                    PrintException()
                    raise ValueError('save_path:{0} is not valid path'.format(folder))
        plot = PlotLossMetricsCallback(frequency if unit == 'epoch' else -1, frequency if unit == 'batch' else -1,
                                       save_path=save_path, name_prefix=name_prefix,
                                       clean_ipython_output_frequency=clean_ipython_output_frequency, imshow=imshow)
        self.callbacks.append(plot)
        return self

    def generate_datafeed(self,data_loader):
        if data_loader.signature is None:
            _ = data_loader.next()
        for trainingitem in self.training_items.value_list:
            data_feed =None
            if 'data_feed' in trainingitem.training_context and  all([  True for key in data_loader.signature.key_list if key in trainingitem.training_context['data_feed'].value_list]):
                data_feed = trainingitem.training_context['data_feed']
            else:
                if 'data_feed' in trainingitem.training_context:
                    data_feed = trainingitem.training_context['data_feed']
                else:
                    data_feed=OrderedDict()
                outputs=trainingitem.outputs
                targets=trainingitem.targets
                if len(self._dataloaders)==1:
                    if len(trainingitem.model.signature)+len(targets)==len(data_loader.signature) or len(data_loader.signature)==1:
                        if len(outputs)==1 and len(trainingitem.model.signature)==1  :
                            if  1<=len(data_loader.signature)<=2:
                                    data_feed[trainingitem.model.signature.key_list[0]]=data_loader.signature.key_list[0]
                                    for loss in trainingitem._losses.value_list:
                                        args=get_signature(loss)
                                        if hasattr(loss,'signature'):
                                            args=loss.signature
                                        if len(args)==2:
                                            if args.key_list[0] not in data_feed or (args.key_list[0] in data_feed and data_feed[args.key_list[0]] is None):
                                                data_feed[args.key_list[0]]=outputs.key_list[0]
                                            if args.key_list[1] not in data_feed or (args.key_list[1] in data_feed and data_feed[args.key_list[1]] is None):
                                                if args.key_list[1] in data_loader.signature.key_list:
                                                    data_feed[args.key_list[1]] = args.key_list[1]
                                                else:
                                                    data_feed[args.key_list[1]] =data_loader.signature.key_list[-1]   #-1 is for handel autoencoder scenario
                                        else:
                                            raise ValueError('loss shoud only 2 argments when one-input-one-output model with 2 dataset items in data loaders')

                                    trainingitem.training_context['data_feed']=data_feed
                                    print('data_feed for {0} :{1}'.format(trainingitem.name,data_feed))
                        #elif len(outputs)==1

                    else:
                        raise RuntimeError('the number of models input plus the numbers of  targets should equal to the numbers of dataset items')
                else:
                    raise  RuntimeError('Multiple data loader data_feed auto-generation is not support Now.')







    def start_now(self, collect_data_inteval=1, is_resume=False,only_steps=False,max_batches=np.inf,keep_weights_history=False, keep_gradient_history=False):
        try:
            self.execution_id = get_time_suffix()

            # update callback
            if not is_resume or only_steps==True:
                for item in self.training_items.values():
                    item.training_context['execution_id']=self.execution_id
                    for callback in self.callbacks:
                        if callback not in item.callbacks:
                            #private callback
                            if callback.is_shared == False:
                                item.with_callbacks(copy.deepcopy(callback))
                            else:
                                #shared callback
                                item.with_callbacks(callback)
                #shared callbacks will access training plan dict instead of training_context
                for callback in self.callbacks:
                    if callback.is_shared == True:
                        callback.on_training_start(self.__dict__)

            data_loader = self._dataloaders.value_list[0]
            data_loader.minibatch_size = self.minibatch_size
            #generate data feed

            if not is_resume or only_steps==True:
                self.generate_datafeed(data_loader)
                if collect_data_inteval == 1 and len(data_loader.batch_sampler) * self.num_epochs > 1000:
                    collect_data_inteval = self.default_collect_data_inteval
            if only_steps==True:
                self.num_epochs=(max_batches//len(data_loader.batch_sampler))+2

            for epoch in range(self.num_epochs):
                try:
                    for mbs, return_data in enumerate(data_loader):
                        num_batches=len(data_loader.batch_sampler)*epoch+mbs
                        iter_data = OrderedDict()
                        for i in range(len(data_loader.signature.key_list)):
                            name = data_loader.signature.key_list[i]
                            iter_data[name] = return_data[i]

                        #check weather need out-of-sample evaluation
                        need_out_sample_evaluation=False
                        if  only_steps==False and self.out_sample_evaluation_on_epoch_end==True and mbs==len(data_loader.batch_sampler)-1:
                            need_out_sample_evaluation=True
                        elif only_steps==True and self.out_sample_evaluation_on_epoch_end==True and num_batches==max_batches-1:
                            need_out_sample_evaluation=True
                        elif only_steps==False and self.out_sample_evaluation_unit=='batch' and mbs>0 and mbs%self.out_sample_evaluation_frequency==0:
                            need_out_sample_evaluation = True
                        elif only_steps==True and self.out_sample_evaluation_unit=='batch' and num_batches>0 and num_batches%self.out_sample_evaluation_frequency==0:
                            need_out_sample_evaluation = True
                        elif only_steps==False and self.out_sample_evaluation_unit=='epoch' and mbs==len(data_loader.batch_sampler)-1 and epoch%self.out_sample_evaluation_frequency==0:
                            need_out_sample_evaluation = True
                        elif only_steps==True and self.out_sample_evaluation_unit=='epoch' and num_batches==max_batches-1 :
                            need_out_sample_evaluation = True


                        iter_testdata = None
                        if isinstance(data_loader, DataProviderV2) and data_loader.testdata is not None and need_out_sample_evaluation:
                            return_test = data_loader.next_test()
                            if return_test is not None:
                                iter_testdata = OrderedDict()
                                for i in range(len(data_loader.signature.key_list)):
                                    name = data_loader.signature.key_list[i]
                                    iter_testdata[name] = return_test[i]

                        # input, target = Variable(input).to(self.device), Variable(target).to(self.device)

                        for trainitem_name, trainitem in zip(self.training_names.value_list,self.training_items.value_list):
                            train_data=OrderedDict()
                            test_data = None if iter_testdata is None else OrderedDict()
                            for k,v in iter_data.items():
                                train_data[k]=v.copy()

                            if iter_testdata is not None:
                                for k,v in iter_testdata.items():
                                    test_data[k]=v.copy()

                            trainitem.training_context['model_name']=trainitem_name
                            if epoch<int(trainitem.start_epoch):
                                trainitem.training_context['stop_update']=1
                            trainitem.train_model(train_data,test_data, epoch if only_steps==False else 0,mbs if only_steps==False else  num_batches, self.num_epochs if only_steps==False else  1,
                                                  len(data_loader.batch_sampler) if only_steps==False else max_batches,
                                                  is_collect_data=mbs % collect_data_inteval == 0,
                                                  is_print_batch_progress=self.print_progress_unit == 'batch' and mbs % self.print_progress_frequency == 0,
                                                  is_print_epoch_progress=self.print_progress_unit == 'epoch' and (
                                                              epoch + 1) % self.print_progress_frequency == 0,
                                                  log_gradients=keep_gradient_history, log_weights=keep_weights_history, accumulate_grads=False)

                        for k, trainitem in self.training_items.items():
                            for callback in trainitem.training_context['callbacks']:
                                if callback.is_shared == False:
                                    callback.on_overall_batch_end(trainitem.training_context)
                        for callback in self.callbacks:
                            if callback.is_shared == True:
                                callback.on_overall_batch_end(self.__dict__)

                        if self.save_model_frequency > 0 and self.save_model_unit == 'batch' and mbs % self.save_model_frequency == 0:
                            for k, trainitem in self.training_items.items():
                                trainitem.save_model()


                        if only_steps==True and num_batches >= max_batches - 1:
                            for k, trainitem in self.training_items.items():
                                try:
                                     trainitem.save_model()
                                except Exception as e:
                                    print(e)
                            return True

                        if only_steps==False and (mbs + 1) % len(data_loader.batch_sampler) == 0:
                            break

                except StopIteration:
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_epoch_end()
                    pass
                except ValueError as ve:
                    print(ve)
                    PrintException()
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_training_end()
                except Exception as e:
                    print(e)
                    PrintException()
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_training_end()
                if self.save_model_frequency > 0 and self.save_model_unit == 'epoch' and (
                        epoch + 1) % self.save_model_frequency == 0:
                    for k, trainitem in self.training_items.items():
                        trainitem.save_model()


        except KeyboardInterrupt:
            for k, trainitem in self.training_items.items():
                trainitem.save_model()
        except Exception as e:
            print(e)
            PrintException()
            for k, trainitem in self.training_items.items():
                trainitem.save_model()

    def resume(self):
        self.start_now(is_resume=True)

    def only_steps(self, num_steps, collect_data_inteval=1, keep_weights_history=False, keep_gradient_history=False):
        return self.start_now(collect_data_inteval=collect_data_inteval, is_resume=False,only_steps=True,max_batches=num_steps,keep_weights_history=keep_weights_history, keep_gradient_history=keep_gradient_history)


last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None, name=''):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = max(int(TOTAL_BAR_LENGTH * float(current) / total), 1)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1 + cur_len
    # sys.stdout.write(' [')
    # for i in range(cur_len):
    #     sys.stdout.write('=')
    # sys.stdout.write('>')
    # for i in range(rest_len):
    #     sys.stdout.write('.')
    # sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('{0:<12s}'.format(name))
    L.append(' Step: {0:<8s}'.format(format_time(step_time)))
    # L.append(' | Tot: {0:<12s}'.format(format_time(tot_time)))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    # for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
    #     sys.stdout.write(' ')
    sys.stdout.write(' ')
    sys.stdout.write(' ( %d/%d )' % (current, total))
    sys.stdout.write('\n')
    sys.stdout.flush()  # # Go back to the center of the bar.  # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):  #     sys.stdout.write('\b')  # sys.stdout.write(' %d/%d ' % (current+1, total))  # if current < total-1:  #     sys.stdout.write('\r')  # else:  #     sys.stdout.write('\n')  # sys.stdout.flush()

