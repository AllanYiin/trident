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

from trident.backend.decorators import deprecated
from trident.callbacks.lr_schedulers import AdjustLRCallback

from trident.backend import iteration_tools
from trident.data.dataset import ZipDataset
from trident.backend.common import get_backend,to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path,make_dir_if_need
from trident.backend.model import ModelBase, progress_bar
from trident.callbacks.visualization_callbacks import *
from trident.data.data_provider import *
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import tile_rgb_images, loss_metric_curve
from trident.backend.tensorspec import TensorSpec, assert_spec_compatibility
from trident.loggers.history import HistoryBase
__all__ = ['TrainingPlan']

_session = get_session()
_backend =get_backend()
working_directory=_session.working_directory


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


class TrainingPlan(object):
    def __init__(self):
        self.training_items = OrderedDict()
        self.training_names = OrderedDict()
        self._dataloaders = OrderedDict()
        self.num_epochs = 1
        self._minibatch_size = 1
        self.steps=0
        self.warmup = 0
        self.default_collect_data_inteval = 1
        self.print_progress_frequency = 10
        self.print_progress_unit = 'batch'
        self.print_progress_on_epoch_end = False
        self.out_sample_evaluation_frequency = 1
        self.out_sample_evaluation_unit = 'epoch'
        self.out_sample_evaluation_on_epoch_end = True
        self.save_model_frequency = -1
        self.save_model_unit = 'batch'
        self.execution_id = None
        self.enable_tensorboard=False
        self._is_optimizer_warmup = False
        self.summary_writer = None

        self.callbacks = []  # if self.callbacks is None:  #     self.callbacks = [  #
        # NumberOfEpochsStoppingCriterionCallback(1)]  # elif not any([issubclass(type(cb),
        # StoppingCriterionCallback) for cb in self.callbacks]):  #  #     self.callbacks.append(  #
        # NumberOfEpochsStoppingCriterionCallback(1))
        self.is_terminate = False

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

    def add_training_item(self, training_item, name=None, start_epoch=0):
        n = len(self.training_items)
        if name is not None and len(name) > 0:
            self.training_names[n] = name
            training_item.name = name
        elif training_item.name is not None and len(training_item.name) > 0:
            self.training_names[n] = training_item.name
        else:
            training_item.name = 'model {0}'.format(n)
            self.training_names[n] = 'model {0}'.format(n)
        self.training_items[n] = training_item
        training_item.training_context['training_name'] = self.training_names[n]
        self.training_items[n].start_epoch = start_epoch
        # backward compatibility
        for k, v in training_item.inputs.items():
            if isinstance(v, tuple) and all([isinstance(item, numbers.Integral) for item in v]):
                training_item.inputs[k] = TensorSpec(shape=to_tensor(v), name=training_item.name)
                training_item.signature.inputs[k] = TensorSpec(shape=to_tensor(v), name=training_item.name)
            elif isinstance(v, TensorSpec):
                training_item.signature.inputs[k] = v

        for k, v in training_item.outputs.items():
            if isinstance(v, tuple) and all([isinstance(item, numbers.Integral) for item in v]):
                training_item.outputs[k] = TensorSpec(shape=to_tensor(v), name=training_item.name)
                training_item.signature.outputs[k] = TensorSpec(shape=to_tensor(v), name=training_item.name)
            elif isinstance(v, TensorSpec):
                training_item.signature.outputs[k] = v
        if isinstance(training_item.model, Layer) and training_item.signature != training_item.model.signature:
            training_item.model.signature = None
            training_item.signature = training_item.model.signature
        return self

    def with_data_loader(self, data_loader, **kwargs):
        self._dataloaders[data_loader.__class__.__name__] = data_loader
        return self

    def repeat_epochs(self, num_epochs: int):
        self.num_epochs = num_epochs
        return self

    @deprecated('0.7.0','with_batch_size')
    def within_minibatch_size(self, minibatch_size: int):
        self.minibatch_size = minibatch_size
        return self

    def with_batch_size(self, minibatch_size: int):
        self.minibatch_size = minibatch_size
        return self

    @deprecated('0.7.0', 'with_tensorboard')
    def within_tensorboard(self):
        self.enable_tensorboard=True
        #check weather have tensorboard
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
        return self

    def with_tensorboard(self):
        self.enable_tensorboard = True
        # check weather have tensorboard
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
        for i in range(len(self.training_items)):
            self.training_items[i].training_context['print_progress_frequency'] = frequency
            self.training_items[i].training_context['print_progress_unit'] = self.print_progress_unit
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
            imshow = True
        elif not is_in_ipython() and not is_in_colab() and imshow is None:
            imshow = False
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
            imshow = True
        elif not is_in_ipython() and not is_in_colab() and imshow is None:
            imshow = False

        if save_path is not None:
            folder, _, _ = split_path(save_path)
            if not os.path.exists(folder):
                try:
                    os.makedirs(folder)
                except Exception as e:
                    PrintException()
                    raise ValueError('save_path:{0} is not valid path'.format(folder))
        plot = PlotLossMetricsCallback(frequency if unit == 'epoch' else -1, frequency if unit == 'batch' else -1,
                                       save_path=save_path, name_prefix=name_prefix.format(get_time_suffix()),
                                       clean_ipython_output_frequency=clean_ipython_output_frequency, imshow=imshow)
        self.callbacks.append(plot)
        return self

    def generate_datafeed(self, data_provider):
        if data_provider.signature is None:
            _ = data_provider.next()
        # data_input=data_provider.traindata.data.symbol
        # if len(data_provider.traindata.unpair)>0:
        #     data_unpair = data_provider.traindata.unpair
        data_symbols = iteration_tools.flatten([data_provider.traindata.data.symbol], iterable_types=(list, tuple))
        label_symbols = iteration_tools.flatten([data_provider.traindata.label.symbol], iterable_types=(list, tuple))
        unpair_symbols = iteration_tools.flatten([data_provider.traindata.unpair.symbol], iterable_types=(list, tuple))
        if "" in label_symbols:
            label_symbols.remove("")
        if "" in unpair_symbols:
            unpair_symbols.remove("")

        for trainingitem in self.training_items.value_list:
            existing_data_feed = None
            if 'data_feed' in trainingitem.training_context:
                existing_data_feed = trainingitem.training_context['data_feed']

            data_feed = OrderedDict()
            #datasets = data_provider.traindata.get_datasets()

            available_items =list(set(data_symbols+label_symbols+unpair_symbols+trainingitem.signature.outputs.key_list))
            if "" in available_items:
                available_items.remove("")


            for inp in trainingitem.signature.inputs.key_list:
                data_feed[inp] = None
            for k, v in trainingitem._losses.items():
                for inp in v.signature.inputs.key_list:
                    data_feed[inp] = None
            for k, v in trainingitem._metrics.items():
                for inp in v.signature.inputs.key_list:
                    data_feed[inp] = None

            if ( 'x' in data_feed  or  'input' in data_feed ) and 'input' in available_items:
                if 'x' in data_feed :
                    data_feed['x'] = 'input'
                    available_items.remove('input')
                elif  'input' in data_feed :
                    data_feed['input'] = 'input'
                    available_items.remove('input')



            if len(trainingitem.signature.inputs) == len(data_symbols) == 1:
                #if trainingitem.signature.inputs.value_list[0].shape.is_compatible_with(data_provider.traindata.data.element_spec.shape):
                data_feed[trainingitem.signature.inputs.key_list[0]] = data_provider.traindata.data.symbol
                available_items.remove(data_provider.traindata.data.symbol)

            if len(trainingitem.signature.outputs) ==1 and  len(label_symbols) == 0:
                data_feed[trainingitem.signature.outputs.key_list[0].replace("output", "target").replace("student", "teacher")] = data_provider.traindata.data.symbol
                if data_provider.traindata.label.symbol in available_items:
                    available_items.remove(data_provider.traindata.label.symbol)
            elif len(trainingitem.signature.outputs) == len(label_symbols) == 1:
                data_feed[trainingitem.signature.outputs.key_list[0].replace("output", "target").replace("student", "teacher")] = data_provider.traindata.label.symbol
                available_items.remove(data_provider.traindata.label.symbol)

            for out in trainingitem.signature.outputs.key_list:  # fill the data_feed by key
                if out in available_items:  # output=putput
                    data_feed[out] = out
                    available_items.remove(out)

            for key in data_feed.keys():
                if data_feed[key] == None and key in available_items:
                    data_feed[key] = key
                    available_items.remove(key)
                elif data_feed[key] == None:
                    data_feed[key] = key

                # elif out.replace("output","target").replace("student","teacher") in available_items:
                #     data_feed[out] =out.replace("output","target").replace("student","teacher")
                #     available_items.remove(out.replace("output","target").replace("student","teacher"))

            trainingitem.training_context['data_feed'] = data_feed
            print('data_feed for {0} :{1}'.format(trainingitem.name, data_feed))

    def start_now(self, collect_data_inteval=1, is_resume=False, only_steps=False, max_batches=np.inf,
                  keep_weights_history=False, keep_gradient_history=False):
        data_provider = self._dataloaders.value_list[0]
        data_provider.minibatch_size = self.minibatch_size
        data_provider.mode = 'dict'
        try:
            self.execution_id = get_time_suffix()
            exception_cnt = 0
            abnormal_num_count = 0
            # update callback
            if self.enable_tensorboard:
                for idx,(item, item_name) in enumerate(zip(self.training_items.value_list,self.training_names.value_list)):
                    if hasattr(item,'training_context'):
                        for context_item in list(item.training_context.values()):
                            if isinstance(context_item,HistoryBase):
                                context_item.enable_tensorboard=True
                                context_item.training_name=item_name
                                context_item.summary_writer=self.summary_writer
                        item.training_context['training_name'] = item_name
                        item.training_context['summary_writer']=self.summary_writer

                make_dir_if_need(os.path.join(working_directory,'Logs'))

                sys.stdout.writelines(['Please execute the command to initial tensorboard:  tensorboard --logdir={0}  --port 6006 \n\r'.format(os.path.join(working_directory,'Logs'))])
                sys.stdout.writelines(['Tensorboard is initialized. You can access tensorboard at http://localhost:6006/   \n\r'])

            if not is_resume or only_steps == True:
                max_name_length = builtins.max([len(name) for name in self.training_names.value_list])
                for item in self.training_items.values():
                    #sysnc device
                    item.model.to(get_device())
                    item.training_context['execution_id'] = self.execution_id
                    item.training_context['max_name_length'] = max_name_length
                    for callback in self.callbacks:
                        if callback not in item.callbacks:
                            # private callback
                            if not callback.is_shared:
                                item.with_callbacks(copy.deepcopy(callback))
                            else:
                                # shared callback
                                item.with_callbacks(callback)
                # shared callbacks will access training plan dict instead of training_context
                for callback in self.callbacks:
                    if callback.is_shared:
                        callback.on_training_start(self.__dict__)



            # generate data feed
            if not is_resume or only_steps == True:
                self.generate_datafeed(data_provider)
                if collect_data_inteval == 1 and len(data_provider.batch_sampler) * self.num_epochs > 1000:
                    collect_data_inteval = self.default_collect_data_inteval
            if only_steps:
                self.num_epochs = (max_batches // len(data_provider.batch_sampler)) + 2



            for epoch in range(self.num_epochs):
                try:
                    for mbs, return_data in enumerate(data_provider):

                        if self.is_terminate:
                            for callback in self.callbacks:
                                if callback.is_shared == True:
                                    callback.on_training_terminated(self.__dict__)

                            for k, trainitem in self.training_items.items():

                                for callback in trainitem.training_context['callbacks']:
                                    if callback.is_shared == False:
                                        callback.on_training_terminated(trainitem.training_context)
                            data_provider.mode = 'tuple'
                        else:

                            num_batches = len(data_provider.batch_sampler) * epoch + mbs
                            iter_data = OrderedDict()

                            if isinstance(return_data, OrderedDict):
                                for spec, data in return_data.item_list:
                                    iter_data[spec.name] = data
                            elif isinstance(return_data, tuple):
                                for i in range(len(return_data)):
                                    iter_data[data_provider.traindata.data_template.key_list[i].name] = return_data[i]

                            # check weather need out-of-sample evaluation
                            need_out_sample_evaluation = False
                            if self.out_sample_evaluation_on_epoch_end == True and mbs > 0 and self.out_sample_evaluation_unit == 'batch' and  mbs %  self.out_sample_evaluation_frequency == 0:
                                need_out_sample_evaluation=True
                            elif self.out_sample_evaluation_on_epoch_end == True and only_steps == False and self.out_sample_evaluation_unit == 'epoch' and mbs == len(
                                    data_provider.batch_sampler) - 1 and epoch % self.out_sample_evaluation_frequency == 0:
                                need_out_sample_evaluation = True


                            iter_testdata = None
                            if isinstance(data_provider, DataProvider) and data_provider.testdata is not None and need_out_sample_evaluation:
                                return_test = data_provider.next_test()
                                if return_test is not None:
                                    iter_testdata = OrderedDict()
                                    if isinstance(return_test, OrderedDict):
                                        for spec, data in return_test.item_list:
                                            iter_testdata[spec.name] = data
                                    elif isinstance(return_test, tuple):
                                        for i in range(len(return_test)):
                                            iter_testdata[data_provider.traindata.data_template.key_list[i].name] = return_test[i]

                            # input, target = Variable(input).to(self.device), Variable(target).to(self.device)

                            for trainitem_name, trainitem in zip(self.training_names.value_list, self.training_items.value_list):
                                train_data = copy.deepcopy(iter_data)
                                test_data = copy.deepcopy(iter_testdata)
                                trainitem.training_context['data_template'] = data_provider.traindata.data_template

                                trainitem.training_context['model_name'] = trainitem_name
                                if epoch < int(trainitem.start_epoch):
                                    trainitem.training_context['stop_update'] = 1

                                trainitem.train_model(train_data, test_data,
                                                      epoch if only_steps == False else 0,
                                                      mbs if only_steps == False else num_batches,
                                                      self.num_epochs if only_steps == False else 1,
                                                      len(data_provider.batch_sampler) if only_steps == False else max_batches,
                                                      is_collect_data=mbs % collect_data_inteval == 0,
                                                      is_print_batch_progress=self.print_progress_unit == 'batch' and mbs % self.print_progress_frequency == 0,
                                                      is_print_epoch_progress=self.print_progress_unit == 'epoch' and (epoch + 1) % self.print_progress_frequency == 0,
                                                      log_gradients=keep_gradient_history, log_weights=keep_weights_history,
                                                      accumulate_grads=False, is_out_sample_evaluation=need_out_sample_evaluation)
                            self.steps +=1


                            if self.enable_tensorboard and len(self.training_items)>1 and mbs % collect_data_inteval == 0:
                                compare_dict = OrderedDict()
                                step=None
                                for trainitem_name, trainitem in zip(self.training_names.value_list, self.training_items.value_list):
                                    for k,v in trainitem.training_context["losses"].items():
                                        if k not in compare_dict:
                                            compare_dict[k]=OrderedDict()
                                        compare_dict[k][k+"/"+trainitem_name]=v[-1][1]
                                        step=v[-1][0]
                                    for k,v in trainitem.training_context["metrics"].items():
                                        if k not in compare_dict:
                                            compare_dict[k]=OrderedDict()
                                        compare_dict[k][k+"/"+trainitem_name]=v[-1][1]
                                for k,v in compare_dict.items():
                                    self.summary_writer.add_scalars(k,v,step)







                            if (self.print_progress_unit == 'batch' and mbs % self.print_progress_frequency == 0) or \
                                    (self.print_progress_unit == 'epoch' and (epoch + 1) % self.print_progress_frequency == 0):
                                if len(self.training_items)>1:
                                    print(' \n', flush=True)

                            for k, trainitem in self.training_items.items():
                                for callback in trainitem.training_context['callbacks']:
                                    if callback.is_shared == False:
                                        callback.on_overall_batch_end(trainitem.training_context)
                            for callback in self.callbacks:
                                if callback.is_shared == True:
                                    callback.on_overall_batch_end(self.__dict__)

                            if self.save_model_frequency > 0 and self.save_model_unit == 'batch' and (num_batches + 1) % \
                                    self.save_model_frequency == 0:
                                for k, trainitem in self.training_items.items():
                                    trainitem.save_model(trainitem.training_context['save_path'])
                                    # if self.enable_tensorboard:
                                    #     trainitem.save_onnx(trainitem.training_context['save_path'].replace('.pth','.onnx'))
                                    #     self.summary_writer.add_onnx_graph(trainitem.training_context['save_path'].replace('.pth','.onnx'));

                            if only_steps == True and num_batches >= max_batches - 1:
                                for k, trainitem in self.training_items.items():
                                    try:
                                        trainitem.save_model()
                                    except Exception as e:
                                        print(e)
                                data_provider.mode = 'tuple'
                                return True

                            if only_steps == False and (mbs + 1) % len(data_provider.batch_sampler) == 0:
                                break
                        trainitem.do_on_epoch_end()
                except StopIteration:
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_epoch_end()
                    trainitem.do_on_epoch_end()
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
            data_provider.mode = 'tuple'


        except KeyboardInterrupt:
            for k, trainitem in self.training_items.items():
                trainitem.save_model()
            data_provider.mode = 'tuple'
        except Exception as e:
            print(e)
            PrintException()
            for k, trainitem in self.training_items.items():
                trainitem.save_model()
            data_provider.mode = 'tuple'

    def resume(self):
        self.start_now(is_resume=True)

    def only_steps(self, num_steps, collect_data_inteval=1, keep_weights_history=False, keep_gradient_history=False):
        return self.start_now(collect_data_inteval=collect_data_inteval, is_resume=False, only_steps=True,
                              max_batches=num_steps, keep_weights_history=keep_weights_history,
                              keep_gradient_history=keep_gradient_history)

