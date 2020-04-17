import inspect
import os
import random
import string
import sys
import time
import uuid
from collections import OrderedDict
from functools import partial

import cntk as C
import cntk.learners
import numpy as np

from .cntk_constraints import get_constraint
from .cntk_losses import get_loss
from .cntk_metrics import get_metric
from .cntk_optimizers import get_optimizer
from .cntk_regularizers import get_reg
from .trainers import ModelBase, OptimizerBase, progress_bar
from ..backend.cntk_backend import *
from ..backend.common import *
from ..callbacks.lr_schedulers import get_lr_scheduler
from ..data.image_common import *
from ..misc.visualization_utils import tile_rgb_images, loss_metric_curve

__all__ = ['TrainingItem', 'Model','ImageClassificationModel','ImageDetectionModel']

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

#
# def make_deterministic(seed: int = 19260817, cudnn_deterministic: bool = False):
#     r"""Make experiment deterministic by using specific random seeds across
#     all frameworks and (optionally) use deterministic algorithms.
#     Args:
#         seed (int): The random seed to set.
#         cudnn_deterministic (bool): If `True`, set CuDNN to use
#             deterministic algorithms. Setting this to `True` can negatively
#             impact performance, and might not be necessary for most cases.
#             Defaults to `False`.
#     """
#
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#     if cudnn_deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#


class dummy_optimizer(C.UserLearner):
    def __init__(self,parameters, lr=0.001, as_numpy=True):
        super(dummy_optimizer, self).__init__(parameters, lr, as_numpy=True)
        self.gradient_values=None
    def update(self, gradient_values, training_sample_count, sweep_end):
        self.gradient_values=gradient_values
        return True



class Model(ModelBase):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(Model, self).__init__(inputs, output, input_shape)
        self._dummy_optimizer=None
        self._internal_loss=0

    def _initial_graph(self, inputs=None, output=None, input_shape=None):
        out_args=[]
        if output is None:
            raise ValueError('There is at least one output')
        if isinstance(output,C.Function) and len(output.arguments)>0 and output.arguments[0].shape[0]!=-2:
            out_args=list(output.arguments)


        if inputs is None:
            if input_shape is None and len(out_args)==0:
                raise ValueError('You should assign inputs or input shape')
            elif len(out_args)>0:
                for arg in output.arguments:
                    self.inputs[arg.name] = arg
            else:
                input_shape = _to_tuple(input_shape)
                input_name = 'input_{0}'.format(len(self.inputs))
                input_var = Input(input_shape, name=input_name)
                self.inputs[input_name] = input_var
        elif inputs is C.input_variable:
            input_name = inputs.name if inputs.name!='' else 'input_{0}'.format(len(self.inputs))
            input_shape=inputs.input_shape
            self.inputs[input_name] = inputs
        elif isinstance(inputs, (tuple, list)):
            for inp in inputs:
                if inp is C.input_variable:
                    input_name = inp.name if inp.name != '' else 'input_{0}'.format(len(self.inputs))
                    self.inputs[input_name] = inp
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, Input):
                    self.inputs[k] = v

        if isinstance(output, C.Function) and len(output.arguments) > 0 and output.arguments[0].shape[0] != -2:
            self.model = output
        elif isinstance(output, (Layer, C.Function)):
            out=output(self.inputs.value_list[0])
            self.model = out
        elif inspect.isfunction(output):
            output=Sequential(output)
            out = output(self.inputs.value_list[0])
            self.model = out
        elif isinstance(output, (list,tuple)):
            output_list=[]
            for op in output:
                if isinstance(op, (Layer, C.Function)):
                    output_list.append(op)
                elif inspect.isfunction(op):
                    output_list.append(Sequential(op))
            output=C.combine(output_list)
            out = output(self.inputs.value_list[0])
            self.model =out
        else:
            raise ValueError('Invalid output')

        self.training_context['current_model'] = self.model
        for out in self.model.outputs:
            self.targets[out.name]=C.input_variable(out.shape,dtype=np.float32)



    @property
    def layers(self):
        layers1 = list(C.logging.depth_first_search(self.model.root_function, lambda x: isinstance(x, C.Function), depth=-1))
        layers1.reverse()
        return layers1

    def complie(self, optimizer, losses=None, metrics=None, loss_weights=None, sample_weight_mode=None,
                weighted_metrics=None, target_tensors=None):
        self.with_optimizer(optimizer)
        if losses is not None and isinstance(losses, (list, tuple)):
            for loss in losses:
                self.with_loss(loss)
        if metrics is not None and isinstance(metrics, (list, tuple)):
            for metric in metrics:
                self.with_metric(metric)

        return self

    def with_optimizer(self, optimizer, **kwargs):
        if isinstance(optimizer, str):
            optimizer_class = get_optimizer(optimizer)
            self.optimizer = optimizer_class(self.model.parameters, **kwargs)

        else:
            self.optimizer = optimizer(self.model.parameters(), **kwargs)
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, mode='min',
        #                                                                factor=0.5, patience=5, threshold=1e-4,
        #                                                                cooldown=0, min_lr=1e-10, eps=1e-8)
        self.base_lr = kwargs.get('lr', kwargs.get('learning_rate', 1e-3))
        self.training_context['optimizer'] = self.optimizer
        self.training_context['base_lr'] = self.base_lr
        self.training_context['current_lr'] = self.base_lr
        self._dummy_optimizer = dummy_optimizer(self.model.parameters, lr=C.learning_rate_schedule([1e-3], C.UnitType.sample))
        return self

    def _update_dummy_traniner(self):
        self._internal_loss=0
        for loss in self._losses.value_list:
            self._internal_loss=C.plus(self._internal_loss,loss)
        self._dummy_trainer = C.Trainer(self.model, (self._internal_loss, None), self._dummy_optimizer)

    def with_loss(self, loss, loss_weight=1,output_idx=0,name='',**kwargs):
        if isinstance(loss, str):
            loss = get_loss(loss)
        alias=name
        if inspect.isclass(loss):
            alias=loss.__name__ if len(alias)==0 else alias
        if len(alias)==0 and hasattr(loss,'__name__') :
             alias=  loss.__name__
        self.loss_weights[alias]=loss_weight

        if callable(loss):
            loss_fn=loss_weight*C.reduce_mean(loss(**kwargs)(self.model.outputs[output_idx],self.targets.value_list[output_idx]))
            self._losses[alias] = loss_fn
        self._update_dummy_traniner()
        return self

    def with_metric(self, metric, output_idx=0,name='', **kwargs):
        if isinstance(metric, str):
            metric = get_metric(metric)
        alias = name
        if inspect.isfunction(metric):
            alias = metric.__name__ if len(alias) == 0 else alias
        if len(alias) == 0 and hasattr(metric, 'name'):
            alias = metric.name
        if  callable(metric):
            self._metrics[alias] =C.reduce_mean(metric(self.model.outputs[output_idx],self.targets.value_list[output_idx],**kwargs))
        return self

    def with_regularizer(self, reg, **kwargs):
        if reg is None:
            return self
        reg_fn = None
        if isinstance(reg, str):
            reg_fn = get_reg(reg)
        elif reg is callable:
            reg_fn = reg
        args = reg_fn.__code__.co_varnames
        if 'reg_weight' in args:
            if 'model' in args:
                self._model_regs[reg_fn.__name__] = partial(reg_fn, **kwargs)
            elif 'output' in args:
                self._output_regs[reg_fn.__name__] = partial(reg_fn, **kwargs)
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
        self.training_context['save_path'] =make_dir_if_need(save_path)
        return self

    def with_learning_rate_scheduler(self, lr_schedule, warmup=0, **kwargs):
        if lr_schedule is None:
            return self
        if isinstance(lr_schedule,str):
            lr_schedule=get_lr_scheduler(lr_schedule)
        if callable(lr_schedule) :
           lr_scheduler= lr_schedule(**kwargs)
           self.callbacks.append(lr_scheduler)
        self.warmup = warmup
        if self.warmup > 0:
            self.optimizer.adjust_learning_rate(1e-5,False)
            self.training_context['current_lr'] =1e-5
        return self



    def adjust_learning_rate(self,lr):
        self.optimizer.param_groups[0]['lr']=lr
        self.training_context['current_lr']=lr

    def do_on_training_start(self):
        pass
    def do_on_training_end(self):
        pass

    def do_on_epoch_start(self):
        if self.training_context['current_epoch'] < self.warmup:
            lr = 1e-5 * (self.training_context['current_epoch'] + 1)
            self.optimizer.param_groups[0]['lr'] = lr
            self.training_context['current_lr'] = lr
        elif self.training_context['current_epoch'] == self.warmup:
            self.optimizer.param_groups[0]['lr'] = self.base_lr
            self.training_context['current_lr'] =self.base_lr


    def do_on_epoch_end(self):
        if self.training_context['current_epoch'] > self.warmup:

            if self.optimizer.lr< 1e-8:
                self.optimizer.adjust_learning_rate( 0.05 * self.base_lr, True)
                self.training_context['current_lr'] =  0.05 * self.base_lr
        elif self.training_context['current_epoch'] == self.warmup:
            self.optimizer.adjust_learning_rate(self.base_lr, True)
            self.training_context['current_lr'] =self.base_lr
        elif self.training_context['current_epoch'] < self.warmup:
            self.optimizer.adjust_learning_rate(1e-5*(self.training_context['current_epoch']+1), True)
            self.training_context['current_lr'] = 1e-5*(self.training_context['current_epoch']+1)

    def do_on_batch_start(self):
        #set model as training state
        #zero grad
        pass

    def do_on_batch_end(self):
        #set model as training state
        #zero grad
        pass

    def do_on_data_received(self, input=None, target=None):
        return input,target


    def do_preparation_for_loss(self):
        pass



    def do_post_loss_calculation(self):
        pass

    def get_current_loss(self):
        return self.training_context['current_loss']

    def do_gradient_update(self,log_gradients=False):
        arg_map={}
        input_data =self.training_context['current_input']
        target_data=self.training_context['current_target']

        model=self.training_context['current_model']

        maybegrads=model.parameters.grad(arg_map)

        for i in range(len(self.inputs)):
            inp=self.inputs.value_list[i]
            if isinstance(input_data,np.ndarray) and i==0:
                arg_map[inp]=input_data
            elif isinstance(input_data,(list,tuple))  and len(input_data)>i:
                arg_map[inp] = input_data[i]

        for i in range(len(self.targets)):
            tar=self.targets.value_list[i]
            if isinstance(target_data,np.ndarray) and i==0:
                arg_map[tar]=target_data
            elif isinstance(target_data,(list,tuple))  and len(target_data)>i:
                arg_map[tar] = target_data[i]

        self._dummy_trainer.train_minibatch(arg_map)
        grads=self._dummy_optimizer.gradient_values
        if log_gradients :
            self.log_gradient(grads=grads)
        self.optimizer.step(grads)

    def do_post_gradient_update(self):
        pass

    def do_on_progress_end(self):
        pass
    def log_gradient(self,grads=None):
        grads_dict=OrderedDict()
        for p, g in grads.items():
            grads_dict[p.uid]=g.asarray()
        self.gradients_history.append(grads_dict)


    def log_weight(self,weghts=None):
        weight_dict = {}
        for para in self.model.parameters:
            weight_dict[para.uid] =para.value
        self.weights_history.append(weight_dict)


    def save_model(self,save_path=None):
        for callback in self.training_context['callbacks']:
            callback.on_model_saving_start(self.training_context)

        save_path=save_path if save_path is not None else self.training_context.get('save_path',save_path)
        if save_path is not None:
            #torch.save(self.model, file_path)
            self.model.save(save_path)
        else:
            save_full_path = 'Models/model_{0}_epoch{1}.model'.format(self.model.name,self.training_context['current_epoch'])
            #torch.save(self.model, save_full_path)
            self.model.save(save_full_path)

    def save_onnx(self, save_path):
        for callback in self.training_context['callbacks']:
            callback.on_model_saving_start(self.training_context)
        save_path = save_path if save_path is not None else self.training_context.get('save_path', save_path)
        if save_path is not None:
            # torch.save(self.model, file_path)
            self.model.save(save_path,C.ModelFormat.ONNX)
        else:
            save_full_path = 'Models/model_{0}_epoch{1}.onnx'.format(self.model.name,
                                                                    self.training_context['current_epoch'])
            # torch.save(self.model, save_full_path)
            self.model.save(save_full_path,C.ModelFormat.ONNX)


    def train_model(self, input, target, current_epoch, current_batch, total_epoch, total_batch, is_collect_data=True, is_print_batch_progress=True, is_print_epoch_progress=True, log_gradients=False, log_weights=False, accumulate_grads=False):
        try:
            self.training_context['current_epoch'] =current_epoch
            self.training_context['current_batch'] = current_batch
            self.training_context['total_epoch'] = total_epoch
            self.training_context['total_batch'] = total_batch
            self.training_context['is_collect_data'] = is_collect_data
            self.training_context['log_gradients'] = log_gradients
            self.training_context['log_weights'] = log_weights
            self.training_context['current_model'] = self.model
            self.training_context['current_lr'] = self.optimizer.lr

            self.sample_collect_history.append(1 if  is_collect_data else 0)

            if self.training_context['current_batch']==0 :
                if self.training_context['current_epoch']==0:
                    self.do_on_training_start()
                self.training_context['print_batch_progress_frequency'] = 1
                self.training_context['print_epoch_progress_frequency'] = 1
                self.training_context['losses'] =OrderedDict()
                self.training_context['losses']['total_losses'] = []
                self.training_context['metrics'] =OrderedDict()
                self.training_context['tmp_losses'] = OrderedDict()
                self.training_context['tmp_losses']= []
                self.training_context['tmp_metrics'] = OrderedDict()
                self.do_on_epoch_start()
                for callback in self.callbacks:
                    callback.on_epoch_start(self.training_context)
            self.do_on_batch_start()

            input,target=self.do_on_data_received(input, target)
            self.training_context['current_input'] = input
            self.training_context['current_target'] =target
            self.training_context['current_model'] = self.model

            if accumulate_grads == False:
                self.training_context['current_loss'] = 0
                self.do_preparation_for_loss()
                self.training_context['current_model'] = self.model
                self.training_context['optimizer'] = self.optimizer



            output =self.model(input)
            #confirm singleton
            #output=unpack_singleton(output)
            self.training_context['current_model'] = self.model
            self.training_context['current_output'] = output



            #losss
            for k, v in self._losses.items():
                if k not in self.training_context['losses']:
                    self.training_context['losses'][k] = []
                loss_weight=1
                if k in self.loss_weights:
                    loss_weight=self.loss_weights[k]
                this_loss = v.eval({v.arguments[0]:input, v.arguments[1]:target}).mean()
                self.training_context['current_loss'] = self.training_context['current_loss'] + this_loss*loss_weight

                if is_collect_data:
                    self.training_context['losses'][k].append(float(to_numpy(this_loss)*loss_weight))

            self.do_post_loss_calculation()
            for callback in self.callbacks:
                callback.on_loss_calculation_end(self.training_context)

            if accumulate_grads==False:
                #regularizer
                # for k, v in self._output_regs.items():
                #     if k + '_Loss' not in self.training_context['losses']:
                #         self.training_context['losses'][k + '_Loss'] = []
                #     this_loss = v(output)
                #     self.training_context['current_loss'] = self.training_context['current_loss']+ this_loss#self.training_context['current_loss'] + this_loss
                #     if is_collect_data:
                #         self.training_context['losses'][k + '_Loss'].append(float(to_numpy(this_loss)))
                #
                # for k, v in self._model_regs.items():
                #     if k + '_Loss' not in self.training_context['losses']:
                #         self.training_context['losses'][k + '_Loss'] = []
                #     this_loss=v(self.model)
                #     self.training_context['current_loss'] =self.training_context['current_loss']+this_loss
                #     if is_collect_data:
                #         self.training_context['losses'][k + '_Loss'].append(float(to_numpy( this_loss)))

                self.training_context['optimizer'] = self.optimizer
                self.do_pre_optimization_step()
                # ON_PRE_OPTIMIZATION_STEP
                for callback in self.training_context['callbacks']:
                    callback.on_optimization_step_start(self.training_context)


                self.do_gradient_update(log_gradients  and is_collect_data)
                self.training_context['optimizer'] = self.optimizer
                self.training_context['current_lr'] = self.optimizer.lr


                # ON_POSTBACKWARD_CALCULATION
                self.do_post_gradient_update()
                for callback in self.training_context['callbacks']:
                    callback.on_optimization_step_end(self.training_context)


                self.training_context['tmp_losses'].append(float(to_numpy(self.training_context['current_loss'])))
                if is_collect_data:
                    self.training_context['losses']['total_losses'].append(float(to_numpy(self.training_context['tmp_losses']).mean()))
                    self.training_context['tmp_losses']=[]


                #model comfirm
                for k, v in self._constraints.items():
                    v(self.model)

                if log_weights and is_collect_data:
                    self.log_weight()

                output = self.model(input)
                self.training_context['current_model'] = self.model
                self.training_context['current_output'] = output

                # ON_EVALUATION_START
                self.do_on_metrics_evaluation_start()
                for callback in self.training_context['callbacks']:
                    callback.on_metrics_evaluation_start(self.training_context)

                for k, v in self._metrics.items():
                    if k not in self.training_context['metrics']:
                        self.training_context['tmp_metrics'][k] = []
                        self.training_context['metrics'][k] = []

                    self.training_context['tmp_metrics'][k].append(float(to_numpy(v.forward(output, target))) if hasattr(v, 'forward') else float( to_numpy(v(output, target))))

                if is_collect_data:
                    for k ,v in self.training_context['tmp_metrics'].items():
                        self.training_context['metrics'][k].append(float(to_numpy(v).mean()))
                        self.training_context['tmp_metrics'][k]=[]

                #ON_EVALUATION_END
                self.do_on_metrics_evaluation_end()
                for callback in self.training_context['callbacks']:
                    callback.on_metrics_evaluation_end(self.training_context)

                if is_print_batch_progress:
                    self.do_on_progress_start()
                    for callback in self.training_context['callbacks']:
                        callback.on_progress_start(self.training_context)

                    self.print_batch_progress(self.training_context['print_batch_progress_frequency'])
                    self.training_context['print_batch_progress_frequency']=1
                    self.do_on_progress_end()
                    for callback in self.training_context['callbacks']:
                        callback.on_progress_end(self.training_context)
                else:
                    self.training_context['print_batch_progress_frequency']+=1

                # ON_BATCH_END
                self.do_on_batch_end()
                for callback in self.training_context['callbacks']:
                    callback.on_batch_end(self.training_context)

            if self.training_context['current_batch']==self.training_context['total_batch']-1:
                #epoch end
                if self.training_context['current_epoch'] == 0:
                    self.batch_loss_history = self.training_context['losses']
                    self.batch_metric_history = self.training_context['metrics']
                    for k, v in self.training_context['losses'].items():
                        self.epoch_loss_history[k] = []
                        self.epoch_loss_history[k].append(np.array(v).mean())
                    for k, v in  self.training_context['metrics'] .items():
                        self.epoch_metric_history[k] = []
                        self.epoch_metric_history[k].append(np.array(v).mean())


                else:
                    [self.batch_loss_history[k].extend(v) for k, v in self.training_context['losses'].items()]
                    [self.batch_metric_history[k].extend(v) for k, v in  self.training_context['metrics'] .items()]
                    for k, v in self.training_context['losses'].items():
                        self.epoch_loss_history[k].append(np.array(v).mean())
                    for k, v in  self.training_context['metrics'] .items():
                        self.epoch_metric_history[k].append(np.array(v).mean())

                if is_print_epoch_progress:
                    self.do_on_progress_start()
                    for callback in self.training_context['callbacks']:
                        callback.on_progress_start(self.training_context)
                    self.print_epoch_progress(self.training_context['print_epoch_progress_frequency'])
                    self.training_context['print_epoch_progress_frequency']=1
                    self.do_on_progress_end()
                    for callback in self.training_context['callbacks']:
                        callback.on_progress_end(self.training_context)
                else:
                    self.training_context['print_epoch_progress_frequency']+=1

                self.training_context['loss_history']=self.epoch_loss_history
                self.training_context['metric_history']=self.epoch_metric_history
                self.do_on_epoch_end()
                for callback in self.training_context['callbacks']:
                    callback.on_epoch_end(self.training_context)

                self.training_context['current_lr'] = self.optimizer.lr
                if self.training_context['current_epoch']==self.training_context['total_epoch']-1:
                    self.do_on_training_end()
                    for callback in self.training_context['callbacks']:
                        callback.on_training_end(self.training_context)
        except Exception:
            PrintException()

    def summary(self):
        pass
            #summary(self.model,tuple(self.model.input_shape))


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
        module_attrs = dir(self.model.__class__)
        optimizer_attrs = dir(self.optimizer.__class__)
        attrs = list(self.__dict__.keys())
        losses = list(self._losses.keys())
        metrics = list(self._metrics.keys())
        output_regs = list(self._output_regs.keys())
        model_regs = list(self._model_regs.keys())
        constraints = list(self._constraints.keys())
        keys = module_attrs +optimizer_attrs+ attrs + losses + metrics+output_regs+model_regs+constraints
        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)



class ImageClassificationModel(Model):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(ImageClassificationModel, self).__init__(inputs, output, input_shape)

        self._class_names=[]
        self.preprocess_flow=[]

        self._idx2lab={}
        self._lab2idx={}


    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self,value):
        if self._class_names!=value:
            self._class_names=value
            self._lab2idx = {v: k for k, v in enumerate(self._class_names)}
            self._idx2lab = {k: v for k, v in enumerate(self._class_names)}

    @property
    def reverse_preprocess_flow(self):
        return_list=[]
        return_list.append(reverse_image_backend_adaptive)
        for i in range(len(self.preprocess_flow)):
            fn=self.preprocess_flow[-1-i]
            if fn.__qualname__=='normalize.<locals>.img_op':
                return_list.append(unnormalize(fn.mean,fn.std))
        return_list.append(array2image)
        return return_list


    def index2label(self, idx:int):
        if self._idx2lab  is None or len(self._idx2lab .items())==0:
            raise ValueError('You dont have proper mapping class names')
        elif  idx not in self._idx2lab :
            raise ValueError('Index :{0} is not exist in class names'.format(idx))
        else:
            return self._idx2lab[idx]

    def label2index(self ,label):
        if self._lab2idx  is None or len(self._lab2idx .items())==0:
            raise ValueError('You dont have proper mapping class names')
        elif  label not in self._lab2idx :
            raise ValueError('label :{0} is not exist in class names'.format(label))
        else:
            return self._lab2idx[label]

    def infer_single_image(self,img,topk=1):
        if self.model.built:
            self.model.to(self.device)
            self.model.eval()
            img=image2array(img)
            if img.shape[-1]==4:
                img=img[:,:,:3]

            for func in self.preprocess_flow:
                if inspect.isfunction(func) and func is not image_backend_adaptive:
                    img=func(img)
            img=image_backend_adaptive(img)
            result=self.model(to_tensor(np.expand_dims(img,0)))
            result = C.softmax(result,axis=0)
            result=to_numpy(result)[0]

            argresult = np.argsort(result)
            argresult1 =argresult[::-1]
            answer=OrderedDict()
            idxs = list(np.argsort(result)[::-1][:topk])
            for idx in idxs:
                prob=result[idx]
                answer[self.index2label(idx)]=(idx,prob)
            #idx=int(np.argmax(result,-1)[0])


            return answer
        else:
            raise  ValueError('the model is not built yet.')




class ImageDetectionModel(Model):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(ImageDetectionModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = []
        self.detection_threshould=0.5

    @property
    def reverse_preprocess_flow(self):
        return_list = []
        for i in range(len(self.preprocess_flow)):
            fn = self.preprocess_flow[-1 - i]
            if fn.__qualname__ == 'normalize.<locals>.img_op':
                return_list.append(unnormalize(fn.mean, fn.std))
        return return_list

    def infer_single_image(self,img,scale=1):
        if self.model.built:
            img=image2array(img)
            if img.shape[-1]==4:
                img=img[:,:,:3]

            for func in self.preprocess_flow:
                if inspect.isfunction(func):
                    img=func(img)

            result=self.model(to_tensor(np.expand_dims(img,0)))
            bboxes = self.generate_bboxes(*result,threshould=self.detection_threshould,scale=scale)
            bboxes = self.nms(bboxes)
            #idx=int(np.argmax(result,-1)[0])
            return bboxes
        else:
            raise  ValueError('the model is not built yet.')
    def generate_bboxes(self,*outputs,threshould=0.5,scale=1):
        raise NotImplementedError
    def nms(self,bboxes):
        raise NotImplementedError

class ImageSegmentationModel(Model):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(ImageSegmentationModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = []



class LanguageModel(Model):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(LanguageModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = []




TrainingItem=Model






