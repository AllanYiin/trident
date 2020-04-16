import os
import sys
import matplotlib
matplotlib.use('TKAgg')
from IPython import display

import time
import uuid
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
import numpy as np
from ..backend.common import addindent, get_time_suffix, format_time, get_terminal_size, \
    snake2camel
from ..backend.pytorch_backend import *
from ..optimizers.pytorch_optimizers import *
from ..optimizers.pytorch_regularizers import *
from ..layers.pytorch_constraints import *
from misc.visualization_utils import tile_rgb_images,loss_metric_curve
from ..misc.callbacks import *

__all__ = ['progress_bar', 'TrainingItem', 'TrainingPlan']

_, term_width = get_terminal_size()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
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
    L.append('  Step: {0:<8s}'.format(format_time(step_time)))
    L.append(' | Tot: {0:<8s}'.format(format_time(tot_time)))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')
    sys.stdout.write(' ( %d/%d )' % (current, total))
    sys.stdout.write('\n')
    sys.stdout.flush()  # # Go back to the center of the bar.  # for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):  #     sys.stdout.write('\b')  # sys.stdout.write(' %d/%d ' % (current+1, total))  # if current < total-1:  #     sys.stdout.write('\r')  # else:  #     sys.stdout.write('\n')  # sys.stdout.flush()


class TrainingItem(object):
    def __init__(self, model: nn.Module, optimizer, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.optimizer_settings = kwargs
        self.reg = None
        self.constraint = None
        self._losses = OrderedDict()
        self._metrics = OrderedDict()
        self.base_lr = None
        self._is_optimizer_initialized = False
        self._is_optimizer_warmup = False
        self.batch_loss_history = {}
        self.batch_metric_history = {}
        self.epoch_loss_history = {}
        self.epoch_metric_history = {}
        self.weights_history = OrderedDict()
        self.gradients_history = OrderedDict()
        self.input_history = []
        self.target_history =[]

        if isinstance(self.optimizer, str):
            optimizer_class = get_optimizer(self.optimizer)
            self.optimizer = optimizer_class(self.model.parameters(), **self.optimizer_settings)
            self._is_optimizer_initialized = True
        else:
            self.optimizer = self.optimizer(self.model.parameters(), **self.optimizer_settings)
            self._is_optimizer_initialized = True

        self.base_lr = kwargs.get('lr', 1e-3)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())

        losses = list(self._losses.keys())
        metrics = list(self._metrics.keys())
        keys = module_attrs + attrs + losses + metrics

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def is_optimizer_initialized(self):
        return self._is_optimizer_initialized

    def update_optimizer(self):
        if isinstance(self.optimizer, str):
            optimizer_class = get_optimizer(self.optimizer)
            self.optimizer = optimizer_class(self.model.parameters(), **self.optimizer_settings)
            self._is_optimizer_initialized = True
        else:
            self.optimizer = self.optimizer(self.model.parameters(), **self.optimizer_settings)
            self._is_optimizer_initialized = True
        if self._is_optimizer_warmup == True:
            self.optimizer.param_groups[0]['lr'] = 1e-5

    def with_loss(self, loss, **kwargs):
        if hasattr(loss, 'forward'):
            self._losses[loss.__name__] = loss(**kwargs)
            if hasattr(self._losses[loss.__name__], 'reduction'):
                setattr(self._losses[loss.__name__], 'reduction', 'mean')
        elif callable(loss):
            self._losses[loss.__name__] = partial(loss, **kwargs)
        return self

    def with_metrics(self, metrics, **kwargs):
        if hasattr(metrics, 'forward'):
            self._metrics[metrics.__name__] = metrics(**kwargs)
            if hasattr(self._metrics[metrics.__name__], 'reduction'):
                setattr(self._metrics[metrics.__name__], 'reduction', 'mean')
        elif callable(metrics):
            self._metrics[metrics.__name__] = partial(metrics, **kwargs)
        return self

    def with_regularizers(self, reg, **kwargs):
        if reg is None:
            self.reg = None
        elif isinstance(reg, str):
            reg_fn = get_reg(reg)
            self.reg = partial(reg_fn, **kwargs)
        else:
            reg_fn = reg
            self.reg = partial(reg_fn, **kwargs)
        return self

    def with_constraints(self, constraint, **kwargs):
        if constraint is None:
            self.constraint = partial(min_max_norm, max_value=3, min_value=-3)
        elif isinstance(constraint, str):
            constraint_fn = get_constraints(constraint)
            self.constraint = partial(constraint_fn, **kwargs)
        else:
            constraint_fn = constraint
            self.constraint = partial(constraint_fn, **kwargs)
        return self


class TrainingPlan(object):
    def __init__(self, callbacks=None,gradient_accumulation_steps=1,):
        self.training_context = {
            '_results_history': [],
            # loss_wrapper
            'losses': None,
            # optimizer
            'optimizer': None,
            # stop training
            'stop_training': False,
            # current_epoch
            '_current_epoch': -1,
            # current_batch
            'current_batch': None,
            # current output
            'current_output': None,
            # current loss
            'current_loss': None

        }
        self.training_items = OrderedDict()

        self._dataloaders = OrderedDict()
        self.lr_scheduler = None
        self.num_epochs = 1
        self._minibatch_size = 1


        self.warmup = 0
        self.print_progress_frequency = 10
        self.print_progress_unit = 'batch'
        self.save_model_frequency = 50
        self.save_model_unit = 'batch'
        self._is_optimizer_warmup = False

        self.need_tile_image = False
        self.tile_image_save_path = None
        self.tile_image_name_prefix = None
        self.tile_image_save_model_frequency = None
        self.tile_image_save_model_unit = 'batch'
        self.tile_image_include_input = False
        self.tile_image_include_output = False
        self.tile_image_include_target = False
        self.tile_image_include_mask = False
        self.tile_image_imshow = False
        self.callbacks = callbacks
        if self.callbacks is None:
            self.callbacks = [NumberOfEpochsStoppingCriterionCallback(1)]

        elif not any([issubclass(type(cb), StoppingCriterionCallback) for cb in self.callbacks]):

            self.callbacks.append(NumberOfEpochsStoppingCriterionCallback(1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def minibatch_size(self):
        return self._minibatch_size

    @minibatch_size.setter
    def minibatch_size(self, value):
        self._minibatch_size = value
        for i, (k, v) in enumerate(self._dataloaders.items()):
            v.minibatch_size = value
            self._dataloaders[k] = v



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
        attrs = list(self.__dict__.keys())
        training_items = list(self.training_items.keys())
        keys = module_attrs + attrs + training_items

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    @classmethod
    def create(cls):
        plan = cls()
        return plan

    def add_training_item(self, training_item: TrainingItem):
        training_item.model.to(self.device)
        self.training_items[len(self.training_items)] = training_item
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(training_item.optimizer, mode='min', factor=0.75,
                                                                       patience=5, verbose=False, threshold=0.001,
                                                                       threshold_mode='rel', cooldown=2, min_lr=1e-9,
                                                                       eps=1e-8)
        return self

    def with_data_loader(self, data_loader, **kwargs):
        self._dataloaders[data_loader.__class__.__name__] = data_loader
        return self

    def with_learning_rate_schedule(self, lr_schedule, warmup=0, **kwargs):
        # self.lr_schedule=partial(lr_schedule,**kwargs)
        self.warmup = warmup
        self._is_optimizer_warmup = False if self.warmup == 0 else True
        if self.warmup > 0:
            self._is_optimizer_warmup = True
            for key, item in self.training_items.items():
                item.optimizer.param_groups[0]['lr'] = 1e-5

        return self

    def repeat_epochs(self, num_epochs: int):
        self.num_epochs = num_epochs
        return self

    def within_minibatch_size(self, minibatch_size: int):
        self.minibatch_size = minibatch_size
        return self

    def print_progress_scheduling(self, frequency: int, unit='batch',on_epoch_end=True):
        self.print_progress_frequency = frequency
        if unit not in ['batch', 'epoch']:
            raise ValueError('unit should be batch or epoch')
        else:
            self.print_progress_unit = unit
        return self


    def save_model_scheduling(self, frequency: int, save_path: str, unit='batch'):
        if not os.path.exists(save_path):
            try:
                os.mkdir(save_path)
            except Exception as e:
                sys.stderr.write(e.__str__())
        self.save_path = save_path


        self.save_model_frequency = frequency
        if unit not in ['batch', 'epoch']:
            raise ValueError('unit should be batch or epoch')
        else:
            self.save_model_unit = unit
        return self

    def display_tile_image_scheduling(self, frequency: int, unit='batch', save_path: str = None,
                              name_prefix: str = 'tile_image_{0}.png', include_input=True, include_output=True,
                              include_target=True, include_mask=None, imshow=False):
        if not os.path.exists(save_path):
            try:
                os.mkdir(save_path)
            except Exception as e:
                sys.stderr.write(e.__str__())
        self.need_tile_image = True
        self.tile_image_save_path = save_path
        self.tile_image_name_prefix = name_prefix
        self.tile_image_save_model_frequency = frequency
        self.tile_image_include_input = include_input
        self.tile_image_include_output = include_output
        self.tile_image_include_target = include_target
        self.tile_image_include_mask = include_mask
        self.tile_image_imshow = imshow

        if unit not in ['batch', 'epoch']:
            raise ValueError('unit should be batch or epoch')
        else:
            self.tile_image_save_model_unit = unit
        return self


    def plot_loss_metric_curve(self,unit='batch'):
        if unit=='batch':
            loss_metric_curve(self.training_items[0].batch_loss_history, self.training_items[0].batch_metric_history,
                              max_iteration=None, calculate_base=unit,
                              save_path=None,
                              imshow=True)
        elif unit=='epoch':
            loss_metric_curve(self.training_items[0].epoch_loss_history, self.training_items[0].epoch_metric_history,
                              max_iteration=self.num_epochs,calculate_base=unit,
                              save_path=None,
                              imshow=True)
        else:
            raise  ValueError('unit should be batch or epoch.')

    def start_now(self, ):
        self.execution_id = str(uuid.uuid4())[:8].__str__().replace('-', '')
        trainingitem =self.training_items[0]

        trainingitem.batch_loss_history = {}
        trainingitem.batch_metric_history = {}
        trainingitem.epoch_loss_history = {}
        trainingitem.epoch_metric_history = {}

        data_loader = list(self._dataloaders.items())[0][1]
        data_loader.minibatch_size = self.minibatch_size

        loss_fn = trainingitem._losses

        metrics_fn = trainingitem._metrics
        model_in_train = trainingitem.model

        optimizer = trainingitem.optimizer
        regularizer = trainingitem.reg
        constraint = trainingitem.constraint

        losses = {}
        losses['total_losses']=[]
        metrics = {}
        tile_images_list = []
        print(self.__repr__)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False, mode='min',  factor=0.75, patience=10, threshold=1e-4,   cooldown=5,min_lr=1e-10,eps=1e-8)

        for callback in self.callbacks:
            callback.on_training_start(self.training_context)

        for epoch in range(self.num_epochs):
            if self.device=='cuda':
                torch.cuda.empty_cache()
            for callback in self.callbacks:
                callback.on_epoch_start(self.training_context)


            try:
                # if self.lr_schedule is not None and trainingitem.is_optimizer_initialized()==True:
                # don't print learning rate if print_progress_every unit is epoch
                for mbs, iter_data in enumerate(data_loader):
                    if len(iter_data) == 1:
                        input, target = torch.from_numpy(iter_data[0]), torch.from_numpy(iter_data[0])
                    if len(iter_data) == 2:
                        input, target = torch.from_numpy(iter_data[0]), torch.from_numpy(iter_data[1])
                    # input, target = Variable(input).to(self.device), Variable(target).to(self.device)
                    input, target = input.to(self.device), target.to(self.device)
                    model_in_train.train()
                    model_in_train.zero_grad()
                    optimizer.zero_grad()

                    output = model_in_train(input)
                    # if epoch==0 and mbs==0:
                    #     trainingitem.update_optimizer()
                    #     optimizer = trainingitem.optimizer
                    #     if self._is_optimizer_warmup==True:
                    #         self.lr_schedule(optimizer=optimizer, current_epoch=epoch, num_epochs=self.num_epochs, verbose=False if self.print_progress_unit == 'epoch' else False)

                    current_loss = 0
                    for k, v in loss_fn.items():
                        if k not in losses:
                            losses[k] = []
                        this_loss=v.forward(output, target) if hasattr(v, 'forward') else v(output, target)
                        current_loss+=this_loss
                        losses[k].append(float(to_numpy(this_loss)))

                    ##regularizer(l1,l2.....)
                    if regularizer is not None:
                        regularizer(model_in_train, current_loss)
                    losses['total_losses'].append(float(to_numpy(current_loss)))

                    for callback in self.callbacks:
                        callback.post_loss_calculation(self.training_context)

                    for k, v in metrics_fn.items():
                        if k not in metrics:
                            metrics[k] = []
                        metrics[k].append(float(to_numpy(v.forward(output, target))) if hasattr(v, 'forward') else  float(to_numpy(v(output, target))))



                    current_loss.backward()
                    for callback in self.callbacks:
                        callback.post_backward_calculation(self.training_context)
                    optimizer.step()
                    for callback in self.callbacks:
                        callback.pre_optimization_step(self.training_context)

                    # cconstraint
                    if constraint is not None:
                        constraint(model_in_train)


                    if self.need_tile_image == True and self.tile_image_save_model_unit == 'batch' and (mbs + 1) % self.tile_image_save_model_frequency == 0:
                        display.clear_output(wait=True)

                    if self.need_tile_image == True and self.tile_image_save_model_unit == 'batch' and (mbs + 1) % self.tile_image_save_model_frequency == 0:
                        #display.clear_output(wait=True)
                        if self.tile_image_include_input:
                            tile_images_list.append(to_numpy(input).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                        if self.tile_image_include_target:
                            tile_images_list.append(to_numpy(target).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                        if self.tile_image_include_output:
                            tile_images_list.append(to_numpy(output).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                        # if self.tile_image_include_mask:
                        #     tile_images_list.append(input*127.5+127.5)
                        tile_rgb_images(*tile_images_list, save_path=os.path.join(self.tile_image_save_path,  self.self.tile_image_name_prefix),  imshow=self.tile_image_imshow)
                        tile_images_list = []

                    if self.print_progress_unit == 'batch' and (mbs + 1) % self.print_progress_frequency == 0:
                        metric_key = list(metrics.keys())[0]
                        if len(metrics[metric_key]) > self.print_progress_frequency  and (mbs + 1) % self.print_progress_frequency  == 0:
                            if epoch < self.warmup:
                                lr = 1e-5 * (epoch + 1)
                                optimizer.param_groups[0]['lr'] = lr
                            else:
                                if epoch == self.warmup:
                                    optimizer.param_groups[0]['lr'] = trainingitem.base_lr
                                self.lr_scheduler.step(np.array(metrics[metric_key][-1*self.print_progress_frequency :]).mean())

                        progress_bar(mbs+1, len(data_loader.batch_sampler), 'Loss: {0:<8.3f}| {1} | learning rate: {2:<10.4e}| epoch: {3}'.format(np.array(losses['total_losses'][-1*self.print_progress_frequency:]).mean(), ','.join( ['{0}: {1:<8.3%}'.format(snake2camel(k), np.array(v[-1*self.print_progress_frequency:]).mean()) for k, v in  metrics.items()]), optimizer.param_groups[0]['lr'],epoch+1))

                    if self.save_model_unit == 'batch' and (mbs + 1) % self.save_model_frequency == 0:
                        save_full_path = os.path.join(self.self.save_path,
                                                      'model_{0}_{1}_{2}.pth'.format(model_in_train.__class__.__name__,
                                                                                     self.execution_id,
                                                                                     get_time_suffix()))
                        torch.save(model_in_train, save_full_path)

                    for callback in self.callbacks:
                        callback.on_batch_end(self.training_context)
                    if (mbs + 1) % len(data_loader.batch_sampler) == 0:
                        break


                if self.need_tile_image == True and self.tile_image_save_model_unit == 'epoch' and ( epoch + 1) % self.tile_image_save_model_frequency == 0:
                    display.clear_output(wait=True)


                if epoch==0:
                    trainingitem.batch_loss_history = losses
                    trainingitem.batch_metric_history = metrics
                    for k, v in losses.items():
                        trainingitem.epoch_loss_history[k]=[]
                        trainingitem.epoch_loss_history[k].append(np.array(v).mean())
                    for k, v in metrics.items():
                        trainingitem.epoch_metric_history[k] = []
                        trainingitem.epoch_metric_history[k].append(np.array(v).mean())
                else:
                    [trainingitem.batch_loss_history[k].extend(v)    for k, v in losses.items()]
                    [trainingitem.batch_metric_history[k].extend(v) for k, v in metrics.items()]
                    for k, v in losses.items():
                        trainingitem.epoch_loss_history[k].append(np.array(v).mean())
                    for k, v in metrics.items():
                        trainingitem.epoch_metric_history[k].append(np.array(v).mean())


                if self.need_tile_image == True and self.tile_image_save_model_unit == 'epoch' and (epoch + 1) % self.tile_image_save_model_frequency == 0:
                    display.clear_output(wait=True)
                    if self.tile_image_include_input:
                        tile_images_list.append(to_numpy(input).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                    if self.tile_image_include_target:
                        tile_images_list.append(to_numpy(target).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                    if self.tile_image_include_output:
                        tile_images_list.append(to_numpy(output).transpose([0, 2, 3, 1]) * 127.5 + 127.5)
                    # if self.tile_image_include_mask:
                    #     tile_images_list.append(input*127.5+127.5)
                    tile_rgb_images(*tile_images_list,
                                    save_path=os.path.join(self.tile_image_save_path, self.self.tile_image_name_prefix),
                                    imshow=self.tile_image_imshow)
                    loss_metric_curve(trainingitem.epoch_loss_history,trainingitem.epoch_metric_history,max_iteration=self.num_epochs,save_path=os.path.join(self.tile_image_save_path, 'loss_metric_curve.png'),
                                    imshow=self.tile_image_imshow)
                    tile_images_list = []


                if self.print_progress_unit == 'epoch' and (epoch + 1) % self.print_progress_frequency == 0:
                    progress_bar(epoch+1, self.num_epochs, 'Loss: {0:<8.3f}| {1} | learning rate: {2:<10.4e}'.format( np.array(losses['total_losses']).mean(), ','.join(['{0}: {1:<8.3%}'.format(snake2camel(k), np.array(v).mean()) for k, v in metrics.items()]),  optimizer.param_groups[0]['lr']))


                if self.save_model_unit == 'epoch' and (epoch + 1) % self.save_model_frequency == 0:
                    save_full_path = os.path.join(self.self.save_path,  'model_{0}_ep_{1}.pth'.format(model_in_train.__class__.__name__, epoch))
                    torch.save(model_in_train, save_full_path)  # copyfile(src, dst)

                metric_key = list(metrics.keys())[0]

                if epoch < self.warmup:
                    lr = 1e-5 * (epoch + 1)
                    optimizer.param_groups[0]['lr'] = lr
                else:
                    if epoch == self.warmup:
                        optimizer.param_groups[0]['lr'] = trainingitem.base_lr
                    self.lr_scheduler.step(np.array(metrics[metric_key]).mean())
                    if optimizer.param_groups[0]['lr']<1e-8:
                        optimizer.param_groups[0]['lr'] =0.05* trainingitem.base_lr


                losses ={}
                losses['total_losses'] = []
                metrics = {}
            except StopIteration:
                model_in_train.eval()
                pass
            except ValueError as ve:
                model_in_train.eval()
                print(ve)
            except Exception as e:
                model_in_train.eval()
                print(e)

            for callback in self.callbacks:
                callback.on_epoch_end(self.training_context)



        for callback in self.callbacks:
            callback.on_training_end(self.training_context)
        model_in_train.eval()



    def only_steps(self,num_steps,keep_weights_history=False,keep_gradient_history=False,keep_input_history=False,keep_target_history=False ):
        self.execution_id = str(uuid.uuid4())[:8].__str__().replace('-', '')
        trainingitem = list(self.training_items.items())[0][1]

        trainingitem.batch_loss_history = {}
        trainingitem.batch_metric_history = {}
        trainingitem.epoch_loss_history = {}
        trainingitem.epoch_metric_history = {}

        data_loader = list(self._dataloaders.items())[0][1]

        data_loader.minibatch_size=self.minibatch_size
        loss_fn = trainingitem._losses

        metrics_fn = trainingitem._metrics
        model_in_train = trainingitem.model

        optimizer = trainingitem.optimizer
        regularizer = trainingitem.reg
        constraint = trainingitem.constraint

        losses = {}
        losses['total_losses']=[]
        metrics = {}
        tile_images_list = []
        try:
            # if self.lr_schedule is not None and trainingitem.is_optimizer_initialized()==True:
            # don't print learning rate if print_progress_every unit is epoch
            for mbs, iter_data in enumerate(data_loader):
                if mbs<num_steps:
                    if len(iter_data) == 1:
                        input, target = torch.from_numpy(iter_data[0]), torch.from_numpy(iter_data[0])

                    if len(iter_data) == 2:
                        input, target = torch.from_numpy(iter_data[0]), torch.from_numpy(iter_data[1])
                    # input, target = Variable(input).to(self.device), Variable(target).to(self.device)
                    if keep_input_history == True:
                        trainingitem.input_history.append(input)
                    if keep_target_history == True:
                        trainingitem.target_history.append(target)
                    input, target = input.to(self.device), target.to(self.device)


                    output = model_in_train(input)

                    current_loss = 0
                    for k, v in loss_fn.items():
                        if k not in losses:
                            losses[k] = []
                        this_loss=v.forward(output, target) if hasattr(v, 'forward') else v(output, target)
                        current_loss+=this_loss
                        losses[k].append(float(to_numpy(this_loss)))

                    ##regularizer(l1,l2.....)
                    if regularizer is not None:
                        regularizer(model_in_train, current_loss)
                    losses['total_losses'].append(float(to_numpy(current_loss)))

                    for k, v in metrics_fn.items():
                        if k not in metrics:
                            metrics[k] = []
                        metrics[k].append(float(to_numpy(v.forward(output, target))) if hasattr(v, 'forward') else  float(to_numpy(v(output, target))))


                    optimizer.zero_grad()
                    current_loss.backward()

                    grad_dict={}
                    if keep_gradient_history==True:
                        for k,v in model_in_train.named_parameters():
                            grad_dict[k]=to_numpy(v.grad)
                    trainingitem.gradients_history[mbs]=grad_dict
                    optimizer.step()

                    weight_dict = {}
                    if keep_weights_history == True:
                        for k, v in model_in_train.named_parameters():
                            weight_dict[k] =to_numpy(v.data)
                    trainingitem.weights_history[mbs] = weight_dict

                    # cconstraint
                    if constraint is not None:
                        constraint(model_in_train)

                    progress_bar(mbs+1, len(data_loader.batch_sampler), 'Loss: {0:<8.3f}| {1} | learning rate: {2:<10.4e}| epoch: {3}'.format(losses['total_losses'][-1], ','.join( ['{0}: {1:<8.3%}'.format(snake2camel(k), v[-1]) for k, v in  metrics.items()]), optimizer.param_groups[0]['lr'],1))


                    if (mbs + 1) % len(data_loader.batch_sampler) == 0:
                        break
                else:
                    break
            trainingitem.batch_loss_history = losses
            trainingitem.batch_metric_history = metrics

        except StopIteration:
            pass
        except ValueError as ve:
            print(ve)
        except Exception as e:
            print(e)
