"""Modelbase"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import inspect
import os
import builtins
import random
import shutil
import string
import sys
import time
import uuid
import json
from typing import List

import numpy as np
from trident.backend.common import to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session,get_backend, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path,make_dir_if_need,Signature
from trident.backend.tensorspec import *
from trident.data.image_common import *
from trident.callbacks import LambdaCallback, UnfreezeModelCallback

_session = get_session()
_backend = get_backend()
if _backend == 'pytorch':
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *




__all__ = ['progress_bar','ModelBase','HistoryBase']



_, term_width = get_terminal_size()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time





class ModelBase(object):
    def __init__(self, inputs=None,  input_shape=None,output=None, name='', **kwargs):
        if isinstance(inputs,tuple) and isinstance(inputs[0],int):
            input_shape,inputs=inputs,input_shape
        self.batch_index = 0
        self.filter_index = 1
        self.inputs = OrderedDict()
        self._outputs = OrderedDict()
        self._targets = OrderedDict()
        self._model = None

        self.lr_scheduler = None
        self._losses = OrderedDict()
        self._metrics = OrderedDict()
        self.loss_weights = OrderedDict()

        self._regs = OrderedDict()
        self._constraints = OrderedDict()


        self.preprocess_flow = []

        self.current_save_path=None

        self.weights_history = []
        self.gradients_history = []
        self.input_history = []
        self.target_history = []
        #self.callbacks = []
        self.gradscaler = None
        self.grad_clipping_by_norm = False
        self.grad_clipping_threshold = None
        self.use_output_as_loss = False
        self.training_context = {
                                 'losses': HistoryBase('losses'),  # loss_wrapper
                                 'metrics': HistoryBase('metrics'),  # loss_wrapper
                                 'epoch_losses': HistoryBase('epoch_losses'),  # loss_wrapper
                                 'epoch_metrics': HistoryBase('epoch_metrics'),  # loss_wrapper
                                 'grads_state': OrderedDict(),  # loss_wrapper
                                 'tmp_losses': HistoryBase('tmp_losses'), # loss_wrapper
                                 'tmp_metrics': HistoryBase('tmp_metrics'),  # loss_wrapper
                                 'out_sample_metrics': HistoryBase('out_sample_metrics'),
                                'print_progress_frequency': 10,
                                'print_progress_unit': 'batch',
                                'optimizer':None,
                                 'warmup':0,
                                 'grads': None,
                                 'stop_training': False,  # stop training
                                 'total_epoch': -1,  # current_epoch
                                 'total_batch': -1,  # current_batch
                                 'current_epoch': -1,  # current_epoch
                                 'current_batch': -1,  # current_batch
                                 'current_model': None,  # current model
                                 'current_input': None,  # current input
                                 'current_target': None,  # current target
                                 'steps': 0,
                                 'current_output': None,  # current output
                                 'current_loss': None,  # current loss
                                 'best_metric': None,  # current loss
                                 'best_model': None,  # current model
                                'loss_history': None, 'metric_history': None, 'base_lr': None,  # current loss
                                'current_lr': None,  # current loss
                                'save_path': None, 'is_collect_data': True, 'callbacks': [],
                                 'stop_update': 0, 'retain_graph': False,
                                 'skip_generate_output':False}
        if name is not None:
            self.name=name



        self._initial_graph(inputs, input_shape,output)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):

        self._outputs = OrderedDict()
        self._targets = OrderedDict()

        if isinstance(value, Layer):
            value.signature = None
            inp_shape=None
            if hasattr(value,'signature') and value.signature is not None and len(value.signature.inputs)>0:
                inp_shape=value.input_shape
            else:
                inp_shape = copy.deepcopy(value.input_shape)
            self.inputs = OrderedDict()
            self._initial_graph(input_shape=inp_shape,output=value)
        elif isinstance(value, np.ndarray) or 'tensor' in value.__name__.lower():
            self.inputs = OrderedDict()
            self._initial_graph(input_shape=value.input_shape,output= to_tensor(value))
        else:
            raise ValueError('Only Layer, Module, Image and Tensor can be valid model')


    @property
    def outputs(self):
        if self._model is not None and isinstance(self._model ,Layer):
            if len(self._outputs)==1 and is_tensor(self._model.output_shape)  and not assert_input_compatibility(self._outputs.value_list[0],self._model.output_shape):
                self._outputs[self._outputs.key_list[0]]=TensorSpec(shape=self._model.output_shape,name=self._outputs.key_list[0])

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
    def warmup(self):
        return self.training_context['warmup']

    @warmup.setter
    def warmup(self, value):
        self.training_context['warmup']=value

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
        if self.model is not None and hasattr(self.model, 'signature') and self.signature != self.model.signature:
            self.signature = self.model.signature
        if self.signature is None or len(self.signature.inputs.key_list)+len(self.signature.outputs.key_list) == len(arg_names):

            new_inputs =  OrderedDict()
            for i in range(len(arg_names[:len(self.inputs)])):
                arg = arg_names[:len(self.inputs)][i]
                new_inputs[arg] = self.inputs.value_list[0]
            self.inputs = new_inputs

            new_outputs = OrderedDict()
            new_target = OrderedDict()
            outputs_args = arg_names[len(self.signature.inputs.key_list):]
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
            # if self.model is not None:
            #     self.signature = get_signature(self._model.forward, 'model')
            #     self.signature.inputs = copy.deepcopy(self.inputs)
            #     self.signature.outputs = copy.deepcopy(self._outputs)

            print(self.signature)
        elif not isinstance(arg_names, (list, tuple)):
            raise ValueError('arg_names should be list or tuple')
        elif len(self.signature) != len(arg_names):
            raise ValueError('data deed and arg_names should be the same length')


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

    def _initial_graph(self, inputs=None, input_shape=None, output=None):
        pass

    def complie(self,optimizer="Adam",
                    loss=None,
                    metrics=None,
                    loss_weights=None,
                    **kwargs
                 ):
        raise NotImplementedError

    def fit(self,
                x=None,
                y=None,
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
        if name == 'signature' or name == '_signature':
            _model = self.__dict__['_model']
            if _model is not None and isinstance(_model, Layer):
                return _model.signature
        if 'training_context' in self.__dict__:
            if name in  self.__dict__['training_context']:
                return  self.__dict__['training_context'][name]
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
            elif _model is not None and "_"+name in _model.__dict__:
                return _model.__dict__["_"+name]

        if name in self.__dict__:
            return self.__dict__[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name, value):
        if name=='signature' or name=='_signature':
            _model = self.__dict__['_model']
            if _model is not None and isinstance(_model ,Layer):
                object.__setattr__(_model, "_" + 'signature', value)
        if 'training_context' in self.__dict__ and name in self.__dict__['training_context']:
            self.__dict__['training_context'][name]=value
        elif '_model' in self.__dict__ and self.__dict__['_model']  :
            _model = self.__dict__['_model']
            if _model is not None and name in _model.__dict__['_parameters']:
                _model.__dict__['_parameters'][name]=value
            elif _model is not None and name in _model.__dict__['_buffers']:
                 _model.__dict__['_buffers'][name]=value
            elif _model is not None and name in _model.__dict__['_modules']:
                _model.__dict__['_modules'][name]=value

            elif _model is not None and name in _model.__dict__:
                object.__setattr__(self.__dict__['_model'], name, value)
            elif _model is not None and "_"+name in _model.__dict__:
                object.__setattr__(self.__dict__['_model'], "_"+name, value)
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

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

    def reset_training_context(self):
        self.training_context = {
            'losses': HistoryBase('losses'),  # loss_wrapper
            'metrics': HistoryBase('metrics'),  # loss_wrapper
            'epoch_losses': HistoryBase('epoch_losses'),  # loss_wrapper
            'epoch_metrics': HistoryBase('epoch_metrics'),  # loss_wrapper
            'grads_state': OrderedDict(),  # loss_wrapper
            'tmp_losses': HistoryBase('tmp_losses'),  # loss_wrapper
            'tmp_metrics': HistoryBase('tmp_metrics'),  # loss_wrapper
            'out_sample_metrics': HistoryBase('out_sample_metrics'),
            'print_progress_frequency': 10,
            'print_progress_unit': 'batch',
            'optimizer': None,
            'warmup': 0,
            'grads': None,
            'stop_training': False,  # stop training
            'total_epoch': -1,  # current_epoch
            'total_batch': -1,  # current_batch
            'current_epoch': -1,  # current_epoch
            'current_batch': -1,  # current_batch
            'current_model': None,  # current model
            'current_input': None,  # current input
            'current_target': None,  # current target
            'steps': 0,
            'current_output': None,  # current output
            'current_loss': None,  # current loss
            'best_metric': None,  # current loss
            'best_model': None,  # current model
            'loss_history': None, 'metric_history': None, 'base_lr': None,  # current loss
            'current_lr': None,  # current loss
            'save_path': None, 'is_collect_data': True, 'callbacks': [],
            'stop_update': 0, 'retain_graph': False,
            'skip_generate_output': False}

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
        self.training_context['time_epoch_start'] = time.time()
        if self.training_context['steps'] == 0:
            self.training_context['time_epoch_progress'] = self.training_context['time_epoch_start']

    def do_on_epoch_end(self):
        # set model as training state
        # zero grad
        pass

    def do_on_batch_start(self):
        self.training_context['time_batch_start'] = time.time()
        if self.training_context['steps'] == 0:
            self.training_context['time_batch_progress'] = self.training_context['time_batch_start']


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

        self.training_context['tmp_losses'].collect('total_losses',self.training_context['steps'],to_numpy(self.training_context['current_loss']).mean())
        if self.training_context['is_collect_data'] == True:
            steps,values=self.training_context['tmp_losses'].get_series('total_losses')
            self.training_context['losses'].collect('total_losses',self.training_context['steps'],float(to_numpy(values).mean()))
            self.training_context['tmp_losses'].reset()

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

    def do_on_excution_exception(self):
        pass

    def log_gradient(self, grads=None):
        raise NotImplementedError

    def log_weight(self, weghts=None):
        raise NotImplementedError

    def merge_grads(self, old_grads, new_grades):
        raise NotImplementedError

    # def get_save_path(self,save_path='',default_folder='log',default_file_name=''):
    #     _,default_filename,default_ext=split_path(default_file_name)
    #     if save_path is None or len(save_path) == 0:
    #         save_path = self.training_context.get('save_path', getattr(self, 'save_path',''))
    #
    #     folder,filename,ext=split_path(save_path)
    #     if folder=='':
    #         folder=default_folder
    #     if filename=='':
    #         filename=default_filename
    #     save_path = os.path.join(folder, filename+default_ext)
    #     make_dir_if_need(sanitize_path(save_path))
    #     return sanitize_path(save_path)



    def save_model(self, save_path ):
        raise NotImplementedError

    def save_onnx(self, save_path):
        raise NotImplementedError

    def save_history(self, save_path=None):
        default_file_name = '{0}_history_{1}.json_'.format(self._model.name, self.training_context['execution_id'])
        save_path = self.get_save_path(save_path, default_folder='Log',default_file_name=default_file_name)
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


    def save_weights(self, save_path):
        raise NotImplementedError

    def load_model(self, file_path, ):
        raise NotImplementedError

    def print_batch_progress(self, print_batch_progress_frequency):
        metric_strings=[]
        for k, v in self._metrics.items():
            collect_history = self._metrics[k].collect_history
            metric_value=None
            batch_steps, batch_values = self.batch_metric_history.get_series(k)
            if collect_history != False:
                if k in self.batch_metric_history and len(batch_values)>=print_batch_progress_frequency:
                    metric_value=np.array(batch_values[-1*print_batch_progress_frequency:]).mean()
                elif k in self.training_context['tmp_metrics']:
                    tmp_steps,tmp_values=self.training_context['tmp_metrics'].get_series(k)
                    metric_value = np.array(tmp_values).mean()
                    self.training_context['tmp_metrics'][k]=[]
                format_string='.3%'
                if metric_value > 3:
                    format_string = '.3f'
                elif metric_value< 1e-3:
                    format_string = '.3e'
                metric_strings.append('{0}: {1:<8{2}}'.format(k, metric_value, format_string))

        loss_steps,loss_values=self.batch_loss_history.get_series('total_losses')
        loss_value=float(np.array(loss_values[-1*print_batch_progress_frequency:]).mean())
        progress_start = self.training_context['time_batch_progress']
        progress_end = time.time()
        step_time = progress_end - progress_start
        self.training_context['time_batch_progress'] = progress_end
        progress_bar(step_time,self.training_context['current_batch'], self.training_context['total_batch'],
                 'Loss: {0:<8.5f}| {1} | learning rate: {2:<10.3e}| epoch: {3}'.format(loss_value, ','.join(metric_strings), self.training_context['current_lr'],
                     self.training_context['current_epoch']), name=self.name)

    def print_epoch_progress(self, print_epoch_progress_frequency):

        metric_strings=[]
        loss_steps, loss_values = self.epoch_loss_history.get_series('total_losses')
        if print_epoch_progress_frequency>len(loss_values):
            print_epoch_progress_frequency=len(loss_values)

        loss_value = float(np.array(loss_values[-1 * print_epoch_progress_frequency:]).mean())
        for k, v in self.epoch_metric_history.items():
            format_string='.3%'
            steps,values=self.epoch_metric_history.get_series(k)
            metric_value=to_numpy(values)
            if np.max(metric_value)>5:
                format_string = '.3f'
            elif np.max(metric_value)<1e-3:
                format_string = '.3e'

            metric_strings.append('{0}: {1:<8{2}}'.format(k,float(metric_value[-1*int(print_epoch_progress_frequency):].mean()),format_string))
        progress_start = self.training_context['time_epoch_progress']
        progress_end = time.time()
        step_time = progress_end-progress_start
        self.training_context['time_epoch_progress']=progress_end
        progress_bar(step_time,self.training_context['current_epoch'], self.training_context['total_epoch'],
                     'Loss: {0:<8.3f}| {1} | learning rate: {2:<10.3e}'.format(loss_value, ','.join(metric_strings), self.training_context['current_lr']), name=self.name)

    def train_model(self, train_data, test_data, current_epoch, current_batch, total_epoch, total_batch,
                    is_collect_data=True, is_print_batch_progress=True, is_print_epoch_progress=True,
                    is_print_batch_gradients=True, log_gradients=False, log_weights=False, accumulate_grads=False,is_out_sample_evaluation=False,**kwargs):
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
                    self.training_context['steps']=0
                    self.training_context['tmp_losses'] = HistoryBase(name='tmp_losses')
                    self.training_context['tmp_metrics'] =HistoryBase(name='tmp_metrics')
                    self.training_context['out_sample_metrics'] = HistoryBase(name='out_sample_metrics')
                    self.training_context['losses'] = HistoryBase(name='losses')
                    self.training_context['losses'].regist('total_losses')
                    self.training_context['metrics'] = HistoryBase(name='metrics')
                    self.training_context['grads_state'] = OrderedDict()
                    self.training_context['grads_state']['first_layer'] = []
                    self.training_context['grads_state']['last_layer'] = []
                self.training_context['is_print_batch_progress']=is_print_batch_progress
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
                self.training_context['current_loss'] = to_tensor(0.0,requires_grad=True)
                self.do_preparation_for_loss()
                self.training_context['optimizer'] = self.optimizer


            if  'skip_generate_output' not in self.training_context or self.training_context['skip_generate_output']==False:
                try:
                    output = try_map_args_and_call(self._model, train_data, self.training_context['data_feed'])
                    if isinstance(output, (list, tuple)):
                        for i in range(len(output)):
                            train_data[self.outputs.key_list[i]] = output[i]

                    elif 'tensor' in output.__class__.__name__.lower():
                        train_data[self.outputs.key_list[0]] = output
                        if self.use_output_as_loss==True:

                            this_loss=output.sum()
                            self.training_context['losses'].collect(self.outputs.key_list[0],self.training_context['steps'],this_loss)
                            self.training_context['current_loss'] = self.training_context['current_loss'] + this_loss
                    else:
                        train_data[self.outputs.key_list[0]] = output
                        if self.use_output_as_loss==True:

                            this_loss=output.sum()
                            self.training_context['losses'].collect(self.outputs.key_list[0], self.training_context['steps'], this_loss)
                            self.training_context['current_loss'] = self.training_context['current_loss'] + this_loss
                except Exception as e:
                    print(e)
                    PrintException()
                    if isinstance(self._model, Layer) and any_abnormal_number(self._model):
                        for para in self._model.parameters():
                            if any_abnormal_number(para):
                                para.data.copy_(where(is_nan(para), random_normal_like(para, mean=0, std=0.02).to(get_device()), para))

            # write output in to data




            # confirm singleton
            # output=unpack_singleton(output)

            # losss
            for k, v in self._losses.items():
                if not hasattr(v,'start_epoch') or (hasattr(v,'start_epoch') and v.start_epoch<=self.training_context['current_epoch']):

                    try:
                        loss_weight = to_tensor(1.0)
                        if k in self.loss_weights:
                            loss_weight = self.loss_weights[k]
                        loss_weight=to_tensor(loss_weight,'float32')
                        this_loss = loss_weight*try_map_args_and_call(v, train_data, self.training_context['data_feed']) # v.forward(output, target) if hasattr(v, 'forward') else v(

                        if isinstance(this_loss, tuple):
                            overall_loss =to_tensor(0.0,requires_grad=True)
                            for i in range(len(this_loss)):
                                if any_abnormal_number(this_loss[i]):
                                    sys.stderr.write('Loss {0} have abnormal number (nan, inf,-inf), trident will skip it automaticly, please check anything wrong!!!/n'.format(k))
                                else:
                                    # a leaf Variable that requires grad connotused in an in-place operation.
                                    overall_loss =overall_loss+ this_loss[i]
                            self.training_context['current_loss'] =self.training_context['current_loss']+ overall_loss

                            if is_collect_data:
                                self.training_context['losses'].collect(k,self.training_context['steps'],float(to_numpy(overall_loss)))

                        else:
                            if any_abnormal_number(this_loss):
                                sys.stderr.write( 'Loss {0} have abnormal number (nan, inf,-inf), trident will skip it automaticly, ' 'please check anything wrong!!!/n'.format(k))
                            else:
                                #a leaf Variable that requires grad connotused in an in-place operation.
                                self.training_context['current_loss'] =self.training_context['current_loss'] + this_loss
                            if is_collect_data:
                                self.training_context['losses'].collect(k, self.training_context['steps'], float(to_numpy(this_loss)))
                    except Exception as e:
                        print(e)
                        PrintException()

            self.do_post_loss_calculation()
            for callback in self.callbacks:
                callback.on_loss_calculation_end(self.training_context)

            if accumulate_grads == False:
                # regularizer
                for k, v in self._regs.items():
                    this_loss=to_tensor(0.0,requires_grad=True)
                    if 'model' in v.signature.inputs:
                        this_loss = v(self._model) if self.training_context['stop_update'] < 1 else to_tensor(0.0,requires_grad=True)
                    elif 'output' in v.signature.inputs:

                        this_loss = try_map_args_and_call(v, train_data, self.training_context['data_feed']) if self.training_context['stop_update'] < 1 else to_tensor(0.0)
                    if not any_abnormal_number(this_loss):
                        # a leaf Variable that requires grad connotused in an in-place operation.
                        self.training_context['current_loss'] =self.training_context['current_loss'] + this_loss  # self.training_context[
                    # 'current_loss'] + this_loss
                    if is_collect_data:
                        self.training_context['losses'].collect(k + '_Loss', self.training_context['steps'], float(to_numpy(this_loss)))



                # self.do_post_loss_calculation()
                #
                # for callback in self.callbacks:
                #     callback.on_loss_calculation_end(self.training_context)

                self.do_pre_optimization_step()
                self.do_gradient_update(log_gradients and is_collect_data)

                self.training_context['current_lr'] = self.optimizer.lr

                # ON_POSTBACKWARD_CALCULATION
                self.do_post_gradient_update()

                if isinstance(self._model, Layer) and any_abnormal_number(self._model):
                    for para in self._model.parameters():
                        if any_abnormal_number(para):
                            para.data.copy_(where(is_nan(para), random_normal_like(para, mean=0, std=0.02).to(get_device()), para))

                # model comfirm
                for k, v in self._constraints.items():
                    if self.training_context['stop_update'] == 0 :
                        v(self._model)

                if log_weights and is_collect_data:
                    if isinstance(self._model,Layer):
                        self.log_weight(weghts=self._model.weights)
                    elif is_tensor(self._model):
                        self.log_weight(weghts=self._model)



                if is_out_sample_evaluation==True and test_data is not None and len(test_data) > 0 and  self.training_context['stop_update']<1 :
                    tmp_output = try_map_args_and_call(self._model, test_data, self.training_context['data_feed'])
                    if isinstance(tmp_output, (list, tuple)):
                        for i in range(len(tmp_output)):
                            test_data[self.outputs.key_list[i]] = tmp_output[i]
                    elif 'tensor' in tmp_output.__class__.__name__.lower():
                        test_data[self.outputs.key_list[0]] = tmp_output
                    else:
                        test_data[self.outputs.key_list[0]] = tmp_output




                # ON_EVALUATION_START
                self.do_on_metrics_evaluation_start()
                for callback in self.training_context['callbacks']:
                    callback.on_metrics_evaluation_start(self.training_context)


                for k, v in self._metrics.items():
                    collect_history =getattr(v,'collect_history') if  hasattr(v,'collect_history') else True
                    if not collect_history == False:
                        self.training_context['metrics'].regist(k)
                        self.training_context['tmp_metrics'].regist(k)

                    this_metric = try_map_args_and_call(v, train_data, self.training_context['data_feed']) if  self.training_context['stop_update']<1 else to_tensor(0)
                    self.training_context['tmp_metrics'].collect(k, self.training_context['steps'], float(to_numpy(this_metric)))


                    if is_out_sample_evaluation==True and test_data is not None and len(test_data) > 0 and collect_history!=False :
                        this_out_metric = try_map_args_and_call(v, test_data , self.training_context['data_feed'])
                        self.training_context['out_sample_metrics'].collect(k, self.training_context['steps'], float(to_numpy(this_out_metric)))

                # ON_EVALUATION_END
                self.do_on_metrics_evaluation_end()
                for callback in self.training_context['callbacks']:
                    callback.on_metrics_evaluation_end(self.training_context)

                #callback's metric can keep in epoch_metric_history


                if is_collect_data:
                    #aggregate tmp data and move to metrics history
                    for k, v in self.training_context['tmp_metrics'].items():
                        steps,values=self.training_context['tmp_metrics'].get_series(k)
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

                if is_out_sample_evaluation==True and test_data is not None and len(test_data) > 0:
                    verbose=[]
                    for k in self.training_context['out_sample_metrics'].get_keys():
                        test_steps, test_values = self.training_context['out_sample_metrics'].get_series(k)
                        metric_value= test_values[-1]
                        history_metric_value=np.array(test_values).mean()

                        format_string = '.3%'
                        if history_metric_value > 3:
                            format_string = '.3f'
                        elif history_metric_value < 1e-3:
                            format_string = '.3e'
                        verbose.append('{0}: {1:<8{2}}'.format(k, metric_value, format_string))
                    print(self.training_context['model_name'] + ': out-of-sample evaluation: ',','.join(verbose))

            if self.training_context['current_batch'] == self.training_context['total_batch'] - 1:
                self.do_on_epoch_end()
                batch_steps,batch_values=self.training_context['losses'].get_series('total_losses')
                if not hasattr(self.training_context['losses'],'last_aggregate_idx'):
                    self.epoch_loss_history.collect('total_losses',self.training_context['current_epoch'],np.array(batch_values).mean())
                    self.training_context['losses'].last_aggregate_idx=len(batch_values)
                else:
                    self.epoch_loss_history.collect('total_losses', self.training_context['current_epoch'], np.array(batch_values[self.training_context['losses'].last_aggregate_idx:]).mean())
                    self.training_context['losses'].last_aggregate_idx = len(batch_values)



                for k, v in self.training_context['metrics'].items():
                    metric_steps, metric_values = self.training_context['metrics'].get_series(k)
                    if not hasattr(self.training_context['metrics'], 'last_aggregate_idx'):
                        self.epoch_metric_history.collect(k, self.training_context['current_epoch'], np.array(metric_values).mean())
                        self.training_context['metrics'].last_aggregate_idx = len(metric_values)
                    else:
                        self.epoch_metric_history.collect(k, self.training_context['current_epoch'], np.array(metric_values[self.training_context['metrics'].last_aggregate_idx:]).mean())
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
        raise NotImplementedError

    def predict(self,input):
        if isinstance(input,(Tensor,np.ndarray)):
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

    def trigger_when(self, when='on_batch_end',epoch=None,batch=None,action=None):
        new_callbacks=LambdaCallback(when,epoch,batch,action)
        self.with_callbacks(new_callbacks)
        return self

    def unfreeze_model_scheduling(self, frequency: int, unit='epoch', slice_from=0, slice_to=None):
        self.callbacks.append(UnfreezeModelCallback(frequency, unit, slice_from, slice_to))
        return self

    def cpu(self):
       if self._model is not None and isinstance(self._model,Layer):
           set_device('cpu')
       elif self._model is not None and isinstance(self._model,Tensor):
           self._model.cpu()

    def cuda(self):
        if self._model is not None and isinstance(self._model, Layer):
            set_device('cuda')
        elif self._model is not None and isinstance(self._model, Tensor):
            self._model.cuda()
    def gpu(self):
        self.cuda()

    #
    # def fit(self, x = None, y = None, batch_size = 8, epochs = 10,
    #   verbose = getOption("keras.fit_verbose", default = 1),
    #   callbacks = None, view_metrics = getOption("keras.view_metrics",
    #   default = "auto"), validation_split = 0, validation_data = NULL,
    #   shuffle = TRUE, class_weight = None, sample_weight = None,
    #   initial_epoch = 0, steps_per_epoch = NULL, validation_steps = NULL,
    #   ...):



def progress_bar(step_time,current, total, msg=None, name=''):
    cur_len = builtins.max(int(TOTAL_BAR_LENGTH * float(current) / total), 1)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1 + cur_len
    # sys.stdout.write(' [')
    # for i in range(cur_len):
    #     sys.stdout.write('=')
    # sys.stdout.write('>')
    # for i in range(rest_len):
    #     sys.stdout.write('.')
    # sys.stdout.write(']')

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



class HistoryBase(OrderedDict):
    def __init__(self, name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name=name

    def regist(self,data_name:str):
        if data_name not in self:
            self[data_name]=[]

    def collect(self,data_name:str,step:int,value:(float,Tensor)):
        if data_name not in self:
            self.regist(data_name)
        if is_tensor(value):
            self[data_name].append((step, to_numpy(value)))
        else:
            self[data_name].append((step, value))


    def reset(self):
        for i in range(len(self)):
            self.value_list[i]=[]
    def get_keys(self):
        return self.key_list

    def get_series(self,data_name):
        if data_name in self:
            steps,values=zip(*self[data_name].copy())
            return steps,values
        else:
            raise ValueError('{0} is not in this History.'.format(data_name))

    def get_last(self,data_name):
        if data_name in self:
            return self[data_name][-1]
        else:
            raise ValueError('{0} is not in this History.'.format(data_name))

    def get_best(self,data_name,is_larger_better=True):
            if data_name in self:
                steps,values=zip(*self[data_name].copy())
                if is_larger_better:
                    return builtins.max(values)
                else:
                    return builtins.min(values)
            else:
                raise ValueError('{0} is not in this History.'.format(data_name))