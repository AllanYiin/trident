from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

from trident.backend.common import to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path
from trident.backend.model import ModelBase, progress_bar
from trident.callbacks.visualization_callbacks import *
from trident.data.data_provider import *
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import tile_rgb_images, loss_metric_curve

__all__ = [  'ModelBase', 'TrainingPlan']

_session = get_session()
_backend = _session.backend
if _backend == 'pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *
    from trident.optims.pytorch_optimizers import *
elif _backend == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *
    from trident.optims.tensorflow_optimizers import *


_, term_width = get_terminal_size()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time




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

