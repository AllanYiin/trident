from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import gc
import inspect
import itertools
import json
import math
import numbers
import os
import shutil
import sys
import time
import uuid
from functools import partial
import builtins
import threading
import numpy as np

from trident import context

from trident.backend.decorators import deprecated
from trident.callbacks.lr_schedulers import AdjustLRCallback, StepLR

from trident.backend import iteration_tools
from trident.data.dataset import ZipDataset, RandomNoiseDataset, ImageDataset
from trident.backend.common import get_backend, to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path, make_dir_if_need,open_browser,launchTensorBoard,launchMLFlow
from trident.backend.model import ModelBase, progress_bar
from trident.callbacks.visualization_callbacks import *
from trident.data.data_provider import *
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import tile_rgb_images, loss_metric_curve
from trident.backend.tensorspec import TensorSpec, assert_spec_compatibility, get_signature
from trident.loggers.history import HistoryBase

__all__ = ['TrainingPlan', 'GanTrainingPlan']

ctx = context._context()
_backend = get_backend()
working_directory = ctx.working_directory

if _backend == 'pytorch':

    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *
    from trident.layers.pytorch_activations import Sigmoid, Tanh
    from trident.layers.pytorch_layers import Flatten, Dense
    from trident.layers.pytorch_pooling import GlobalAvgPool2d
    from trident.optims.pytorch_losses import L1Loss, L2Loss, BCELoss, MSELoss
    from trident.optims.pytorch_optimizers import *

elif _backend == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *
    from trident.layers.tensorflow_activations import Sigmoid, Tanh
    from trident.layers.tensorflow_layers import Flatten, Dense
    from trident.layers.tensorflow_pooling import GlobalAvgPool2d
    from trident.optims.tensorflow_losses import L1Loss, L2Loss, BCELoss, MSELoss
    from trident.optims.tensorflow_optimizers import *


class TrainingPlan(object):
    def __init__(self,name=None):
        self.name = name
        self.training_items = OrderedDict()
        self.training_names = OrderedDict()
        self._dataloaders = OrderedDict()
        self.num_epochs = 1
        self._batch_size = 1
        self.steps = 0
        self.epochs = 0
        self.warmup = 0
        self.default_collect_data_inteval = 10
        self.print_progress_frequency = 10
        self.print_progress_unit = 'batch'
        self.print_progress_on_epoch_end = False
        self.out_sample_evaluation_frequency = 1
        self.out_sample_evaluation_unit = 'epoch'
        self.out_sample_evaluation_on_epoch_end = True
        self.save_model_frequency = -1
        self.save_model_unit = 'batch'
        self.execution_id = None
        self.enable_tensorboard = ctx.enable_tensorboard
        self._is_optimizer_warmup = False

        self.callbacks = []  # if self.callbacks is None:  #     self.callbacks = [  #
        # NumberOfEpochsStoppingCriterionCallback(1)]  # elif not any([issubclass(type(cb),
        # StoppingCriterionCallback) for cb in self.callbacks]):  #  #     self.callbacks.append(  #
        # NumberOfEpochsStoppingCriterionCallback(1))
        self.is_terminate = False


    @property
    def minibatch_size(self):
        return self._batch_size

    @minibatch_size.setter
    def minibatch_size(self, value):
        self._batch_size = value
        for k, v in self._dataloaders.item_list:
            v.batch_size = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        for k, v in self._dataloaders.item_list:
            v.batch_size = value

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

        if self.name is None and len(self.training_items)==1:
            self.name='TrainingPlan_{0}'.format(self.training_names[0])
        elif self.name=='TrainingPlan_{0}'.format(self.training_names[0]) and len(self.training_items)==1:
            self.name ='TrainingPlan_{0}'.format(uuid.uuid4().node)


        # backward compatibility
        # for k, v in training_item.signature.inputs.items():
        #     if isinstance(v, tuple) and all([isinstance(item, numbers.Integral) for item in v]):
        #         training_item.inputs[k] = TensorSpec(shape=to_tensor(v), name=training_item.name)
        #         training_item.signature.inputs[k] = TensorSpec(shape=to_tensor(v), name=training_item.name)
        #     elif isinstance(v, TensorSpec):
        #         training_item.signature.inputs[k] = v
        #
        # for k, v in training_item.signature.outputs.items():
        #     if isinstance(v, tuple) and all([isinstance(item, numbers.Integral) for item in v]):
        #         training_item.outputs[k] = TensorSpec(shape=to_tensor(v), name=training_item.name)
        #         training_item.signature.outputs[k] = TensorSpec(shape=to_tensor(v), name=training_item.name)
        #     elif isinstance(v, TensorSpec):
        #         training_item.signature.outputs[k] = v
        # if isinstance(training_item.model, Layer) and training_item.signature != training_item.model.signature:
        #     training_item.model.signature = None
        #     training_item.signature = training_item.model.signature
        return self

    def with_data_loader(self, data_loader, **kwargs):
        self._dataloaders[data_loader.__class__.__name__] = data_loader
        return self

    def repeat_epochs(self, num_epochs: int):
        self.num_epochs = num_epochs
        return self


    def with_batch_size(self, batch_size: int):
        self._batch_size = batch_size
        return self

    def with_tensorboard(self):
        make_dir_if_need(os.path.join(working_directory, 'Logs'))
        # check weather have tensorboard
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
        return self

    def with_mlflow(self):
        from trident.loggers.mlflow_logger import MLFlowLogger
        ctx.try_enable_mlflow(MLFlowLogger(experiment_name=self.name))

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

        self.default_collect_data_inteval = frequency if unit in ['batch','step'] else 10

        if unit not in ['batch', 'epoch','step']:
            raise ValueError('unit should be batch or epoch')
        else:
            self.print_progress_unit = unit
        for i in range(len(self.training_items)):
            self.training_items[i].training_context['print_progress_frequency'] = frequency
            self.training_items[i].training_context['print_progress_unit'] = self.print_progress_unit
        return self

    def print_gradients_scheduling(self, frequency: int, unit='batch'):

        pg = PrintGradientsCallback(frequency=frequency, unit=unit, )
        pg.is_shared = True
        self.callbacks.append(pg)
        return self

    def print_gpu_utilization(self, frequency: int, unit='batch'):
        pg = PrintGpuUtilizationCallback(frequency=frequency, unit=unit, )
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

        tile = TileImageCallback(frequency=frequency, unit=unit,
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
        plot = PlotLossMetricsCallback(frequency=frequency, unit=unit,
                                       save_path=save_path, name_prefix=name_prefix.format(get_time_suffix()),
                                       clean_ipython_output_frequency=clean_ipython_output_frequency, imshow=imshow)
        plot.is_shared=True
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

        for trainingitem,trainingitem_name in zip(self.training_items.value_list,self.training_names.value_list):
            existing_data_feed = None
            if 'data_feed' in trainingitem.training_context and trainingitem.training_context['data_feed'] is not None:
                existing_data_feed = trainingitem.training_context['data_feed']
            if not is_tensor(trainingitem._model) and not hasattr(trainingitem._model, '_signature') or trainingitem._model._signature is None:
                trainingitem._model._signature = get_signature(trainingitem._model)

            data_feed = OrderedDict()
            # datasets = data_provider.traindata.get_datasets()

            available_items = list(set(data_symbols + label_symbols + unpair_symbols + trainingitem.signature.outputs.key_list))
            if "" in available_items:
                available_items.remove("")

            for inp in trainingitem.signature.inputs.key_list:
                if existing_data_feed is not None and isinstance(existing_data_feed,OrderedDict) and inp in existing_data_feed:
                    data_feed[inp] = existing_data_feed[inp]
                elif inp in data_symbols:
                    data_feed[inp]=inp
                    available_items.remove(inp)
                elif inp in unpair_symbols:
                    data_feed[inp]=inp
                    available_items.remove(inp)
                else:
                    data_feed[inp] = None
            for k, v in trainingitem._losses.items():
                for inp in v.signature.inputs.key_list:
                    if existing_data_feed is not None and inp in existing_data_feed :
                        data_feed[inp] = existing_data_feed[inp]
                    elif inp in data_feed and  data_feed[inp] is not None:
                        pass
                    elif inp in label_symbols:
                        data_feed[inp] = inp
                        available_items.remove(inp)
                    elif inp in unpair_symbols:
                        data_feed[inp] = inp
                        available_items.remove(inp)
                    else:
                        data_feed[inp] = None
            for k, v in trainingitem._metrics.items():
                for inp in v.signature.inputs.key_list:
                    if existing_data_feed is not None and inp in existing_data_feed:
                        data_feed[inp] = existing_data_feed[inp]
                    elif inp in data_feed and data_feed[inp] is not None:
                        pass
                    elif inp in label_symbols:
                        data_feed[inp] = inp
                        available_items.remove(inp)
                    elif inp in unpair_symbols:
                        data_feed[inp] = inp
                        available_items.remove(inp)
                    else:
                        data_feed[inp] = None
            if 'output' in data_feed and data_feed['output'] is None:
                data_feed['output']='output'
            if 'x' in data_feed and data_feed['x'] is None:
                if len(data_symbols) == 1 and data_symbols[0] in available_items:
                    data_feed['x'] = data_symbols[0]
                    available_items.remove(data_symbols[0])

                if 'x' in available_items:
                    data_feed['x'] = 'x'
                    available_items.remove('x')
                elif 'input' in available_items:
                    data_feed['x'] = 'input'
                    available_items.remove('input')
            if len([item for item  in data_feed.value_list if item is None])==0:
                pass
            else:

                if len(trainingitem.signature.inputs) == len(data_symbols) == 1:
                    # if trainingitem.signature.inputs.value_list[0].shape.is_compatible_with(data_provider.traindata.data.element_spec.shape):
                    data_feed[trainingitem.signature.inputs.key_list[0]] = data_provider.traindata.data.symbol
                    if data_provider.traindata.data.symbol in available_items:
                        available_items.remove(data_provider.traindata.data.symbol)

                if len(trainingitem.signature.outputs) == 1 and len(label_symbols) == 0:
                    data_feed[trainingitem.signature.outputs.key_list[0].replace("output", "target").replace("student", "teacher")] = data_provider.traindata.data.symbol
                    if data_provider.traindata.label.symbol in available_items:
                        available_items.remove(data_provider.traindata.label.symbol)
                elif len(trainingitem.signature.outputs) == len(label_symbols) == 1:
                    data_feed[trainingitem.signature.outputs.key_list[0].replace("output", "target").replace("student", "teacher")] = data_provider.traindata.label.symbol
                    # if data_provider.traindata.label.symbol in available_items:
                    #     available_items.remove(data_provider.traindata.label.symbol)

                for out in trainingitem.signature.outputs.key_list:  # fill the data_feed by key
                    if out in available_items:  # output=putput
                        data_feed[out] = out
                        available_items.remove(out)

                if 'target' in data_feed:
                    if len(label_symbols) == 1 and label_symbols[0] in available_items:
                        data_feed['target'] = label_symbols[0]
                        available_items.remove(label_symbols[0])

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
            ctx.print('data_feed for {0} :'.format(trainingitem_name))
            ctx.print(json.dumps(data_feed, indent=4, sort_keys=True))

    def do_on_training_start(self):
        for callback in self.callbacks:
            callback.on_training_start(self.__dict__)
        for item in self.training_items.value_list:
            item.train()

    def do_on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end(self.__dict__)
        for item in self.training_items.value_list:
            item.save_model()
            item.eval()

    def do_on_overall_epoch_end(self):
        for callback in self.callbacks:
            callback.on_overall_epoch_end(self.__dict__)

    def do_on_overall_batch_end(self):
        for callback in self.callbacks:
            callback.on_overall_batch_end(self.__dict__)

    def start_now(self, collect_data_inteval=None, is_resume=False, only_steps=False, max_batches=np.inf,
            keep_weights_history=False, keep_gradient_history=False):
        data_provider = self._dataloaders.value_list[-1]
        data_provider.batch_size = self.batch_size
        data_provider.mode = 'dict'

        if collect_data_inteval is None:
            collect_data_inteval = self.default_collect_data_inteval if self.print_progress_unit == 'epoch' else self.print_progress_frequency

        try:
            self.execution_id = get_time_suffix()
            exception_cnt = 0
            abnormal_num_count = 0
            # update callback

            #enable tensorboard & mlflow
            if ctx.enable_tensorboard:
                for idx, (item, item_name) in enumerate(zip(self.training_items.value_list, self.training_names.value_list)):
                    if hasattr(item, 'training_context'):
                        for context_item in list(item.training_context.values()):
                            if isinstance(context_item, HistoryBase):
                                context_item.training_name = item_name

                        item.training_context['training_name'] = item_name
                        item.training_context['summary_writer'] = ctx.summary_writer

                t1 = threading.Thread(target=launchTensorBoard, args=([]))
                t1.setDaemon(True)
                t1.start()
                open_browser('http://{0}:{1}/'.format(ctx.tensorboard_server, ctx.tensorboard_port), 5)
            if ctx.enable_mlflow:
                t2 = threading.Thread(target=launchMLFlow, args=([]))
                t2.setDaemon(True)
                t2.start()
                ctx.mlflow_logger.start_run()
                open_browser('http://{0}:{1}/'.format(ctx.mlflow_server, ctx.mlflow_port), 5)



            if not is_resume or only_steps == True:
                max_name_length = builtins.max([len(name) for name in self.training_names.value_list])
                for item in self.training_items.values():
                    # sysnc device
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

            if len(data_provider._batch_transform_funcs) > 0:
                data_provider.traindata.batch_sampler._batch_transform_funcs = data_provider._batch_transform_funcs
            self.do_on_training_start()
            for epoch in range(self.num_epochs):
                self.epochs=epoch
                for mbs, return_data in enumerate(data_provider):
                    try:

                        if self.is_terminate:
                            for callback in self.callbacks:
                                if callback.is_shared:
                                    callback.on_training_terminated(self.__dict__)

                            for k, trainitem in self.training_items.items():
                                for callback in trainitem.training_context['callbacks']:
                                    if not callback.is_shared:
                                        callback.on_training_terminated(trainitem.training_context)
                            data_provider.mode = 'tuple'
                            return True

                        else:

                            iter_data = OrderedDict()

                            if isinstance(return_data, OrderedDict):
                                for spec, data in return_data.item_list:
                                    iter_data[spec.name] = data
                            elif isinstance(return_data, tuple):
                                for i in range(len(return_data)):
                                    iter_data[data_provider.traindata.data_template.key_list[i].name] = return_data[i]

                            # check weather need out-of-sample evaluation
                            need_out_sample_evaluation = False
                            if self.out_sample_evaluation_on_epoch_end == True and mbs > 0 and self.out_sample_evaluation_unit == 'batch' and mbs % \
                                    self.out_sample_evaluation_frequency == 0:
                                need_out_sample_evaluation = True
                            elif self.out_sample_evaluation_on_epoch_end == True and only_steps == False and self.out_sample_evaluation_unit == 'epoch' and mbs == len(
                                    data_provider.batch_sampler) - 1 and epoch % self.out_sample_evaluation_frequency == 0:
                                need_out_sample_evaluation = True

                            iter_testdata = None
                            if isinstance(data_provider, (DataProvider, TextSequenceDataProvider)) and data_provider.testdata is not None and need_out_sample_evaluation:
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
                            num_epoch = self.num_epochs if only_steps == False else 1
                            num_batches = len(data_provider.batch_sampler) if only_steps == False else max_batches
                            current_batch = mbs if only_steps == False else self.steps
                            current_epoch = epoch if only_steps == False else 0
                            is_epoch_end = current_batch== num_batches - 1
                            for trainitem_name, trainitem in zip(self.training_names.value_list, self.training_items.value_list):
                                train_data = copy.deepcopy(iter_data)
                                test_data = copy.deepcopy(iter_testdata)
                                trainitem.training_context['data_template'] = data_provider.traindata.data_template
                                trainitem.training_context['collect_data_inteval'] = collect_data_inteval
                                trainitem.training_context['model_name'] = trainitem_name
                                start_epoch = 0 if not hasattr(trainitem, 'start_epoch') else trainitem.start_epoch
                                if epoch < start_epoch:
                                    trainitem.training_context['stop_update'] = 1

                                if current_batch == 0:
                                    trainitem.do_on_epoch_start()
                                trainitem.steps = self.steps
                                trainitem.current_epoch = current_epoch
                                trainitem.current_batch = current_batch

                                trainitem.train_model(train_data, test_data,
                                                      current_epoch,
                                                      current_batch,
                                                      num_epoch,
                                                      num_batches,
                                                      done=None,
                                                      is_collect_data=current_batch == 0 or (self.steps+1) % collect_data_inteval == 0,
                                                      is_print_batch_progress=self.print_progress_unit == 'batch' and self.steps > 0 and (current_batch+1) % self.print_progress_frequency == 0,
                                                      is_print_epoch_progress=self.print_progress_unit == 'epoch' and epoch > 0 and epoch % self.print_progress_frequency == 0,
                                                      log_gradients=keep_gradient_history, log_weights=keep_weights_history,
                                                      accumulate_grads=(self.steps +1)  % trainitem.accumulation_steps != 0,
                                                      is_out_sample_evaluation=need_out_sample_evaluation)

                                if is_epoch_end:
                                    trainitem.do_on_epoch_end()

                            self.steps += 1

                            if ctx.enable_tensorboard and len(self.training_items) > 1 and mbs % collect_data_inteval == 0:
                                compare_dict = OrderedDict()
                                step = None
                                for trainitem_name, trainitem in zip(self.training_names.value_list, self.training_items.value_list):
                                    for k, v in trainitem.training_context["losses"].items():
                                        if k not in compare_dict:
                                            compare_dict[k] = OrderedDict()
                                        compare_dict[k][k + "/" + trainitem_name] = v[-1][1]
                                        step = v[-1][0]
                                    for k, v in trainitem.training_context["metrics"].items():
                                        if k not in compare_dict:
                                            compare_dict[k] = OrderedDict()
                                        compare_dict[k][k + "/" + trainitem_name] = v[-1][1]
                                for k, v in compare_dict.items():
                                    ctx.summary_writer.add_scalars(k, v, step)

                            if (self.print_progress_unit == 'batch' and (current_batch+1)% self.print_progress_frequency == 0) or \
                                    (self.print_progress_unit == 'epoch' and (epoch + 1) % self.print_progress_frequency == 0):
                                if len(self.training_items) > 1:
                                    ctx.print(' \n', flush=True)

                            self.do_on_overall_batch_end()
                            if is_epoch_end:
                                self.do_on_overall_epoch_end()

                            if self.save_model_frequency > 0 and self.save_model_unit == 'batch' and (current_batch+ 1) % \
                                    self.save_model_frequency == 0:
                                for k, trainitem in self.training_items.items():
                                    trainitem.save_model(trainitem.training_context['save_path'], )
                                    # if ctx.enable_tensorboard and ('upload_onnx' not in trainitem.training_context or trainitem.training_context['upload_onnx'] == False):
                                    #     trainitem.save_onnx(trainitem.training_context['save_path'].replace('.pth', '.onnx'))
                                    #     ctx.summary_writer.add_onnx_graph(trainitem.training_context['save_path'].replace('.pth', '.onnx'));
                                    #     trainitem.training_context['upload_onnx'] = True
                            if only_steps == True and self.steps >= max_batches - 1:
                                for k, trainitem in self.training_items.items():
                                    try:
                                        trainitem.save_model(trainitem.training_context['save_path'], )
                                    except Exception as e:
                                        ctx.print(e)
                                data_provider.mode = 'tuple'
                                return True

                            if only_steps == False and (self.steps + 1) % len(data_provider.batch_sampler) == 0:
                                break

                    except RuntimeError as runtime_e:
                        if 'CUDA out of memory' in str(runtime_e):
                            gc.collect()

                    except StopIteration:
                        for k, trainitem in self.training_items.items():
                            trainitem.do_on_epoch_end()
                            trainitem.save_model(trainitem.training_context['save_path'], )

                    except ValueError as ve:
                        ctx.print(ve)
                        PrintException()
                        for k, trainitem in self.training_items.items():
                            trainitem.do_on_excution_exception()

                    except Exception as e:
                        ctx.print(e)
                        PrintException()
                for k, trainitem in self.training_items.items():
                    trainitem.do_on_excution_exception()
                if self.save_model_frequency > 0 and self.save_model_unit == 'epoch' and (
                        epoch + 1) % self.save_model_frequency == 0:
                    for k, trainitem in self.training_items.items():
                        trainitem.save_model(trainitem.training_context['save_path'], )
            self.do_on_training_end()
            data_provider.mode = 'tuple'


        except KeyboardInterrupt:
            for k, trainitem in self.training_items.items():
                trainitem.save_model(trainitem.training_context['save_path'], )
            data_provider.mode = 'tuple'
        except Exception as e:
            ctx.print(e)
            PrintException()
            for k, trainitem in self.training_items.items():
                trainitem.save_model(trainitem.training_context['save_path'], )
            data_provider.mode = 'tuple'

    def resume(self):
        self.start_now(is_resume=True)

    def only_steps(self, num_steps, collect_data_inteval=1, keep_weights_history=False, keep_gradient_history=False):
        return self.start_now(collect_data_inteval=collect_data_inteval, is_resume=False, only_steps=True,
                              max_batches=num_steps, keep_weights_history=keep_weights_history,
                              keep_gradient_history=keep_gradient_history)


class GanTrainingPlan(TrainingPlan):
    def __init__(self):
        super().__init__()
        self.is_generator_first = None
        self.gan_type = None
        self.is_condition_gan = False
        self.discriminator = None
        self.generator = None
        self._use_label_smoothing = False
        self.max_noise_intensity = 0
        self.min_noise_intensity = 0
        self.decay = 10000
        self._use_total_variation_loss = False
        self.total_variation_reg_weight = 0.005
        self.total_variation_start_epoch = 3
        self._use_pull_away_term_loss = False
        self._use_feature_matching = False
        self._use_label_smoothing = None
        self.discriminator_feature_uuid = None

    def with_generator(self, generator, name='modelG'):
        if len(self.training_items) == 0:
            self.is_generator_first = True

        generator.training_context['gan_role'] = 'generator'
        if generator.optimizer is None:
            generator.with_optimizer(Adam, 2e-4, betas=(0.5, 0.999))
        generator.with_callbacks(StepLR(frequency=5, unit='epoch', gamma=0.75))
        if not any([isinstance(cb, TileImageCallback) for cb in generator.callbacks]):
            generator.with_callbacks(GanTileImageCallback(frequency=50,unit='batch'))

        self.generator = generator
        return self.add_training_item(self.generator, name=name, start_epoch=0)

    def with_discriminator(self, discriminator, name='modelD'):
        if len(self.training_items) == 0:
            self.is_generator_first = False

        discriminator.training_context['gan_role'] = 'discriminator'
        if discriminator.optimizer is None:
            discriminator.with_optimizer(Adam, 2e-4, betas=(0.5, 0.999))
        discriminator.with_callbacks(StepLR(frequency=5, unit='epoch', gamma=0.75))
        self.discriminator = discriminator
        # if not self.is_generator_first:
        #     self.discriminator.training_context['retain_graph'] = True
        # else:
        #     self.discriminator.training_context['retain_graph'] = False
        return self.add_training_item(self.discriminator, name=name, start_epoch=0)

    def with_label_smoothing(self, one_side=True):
        self._use_label_smoothing = "one_side" if one_side else "two_side"
        return self

    def with_noised_real_images(self, max_noise_intensity=0.1, min_noise_intensity=0, decay=10000):
        self.max_noise_intensity = max_noise_intensity
        self.min_noise_intensity = min_noise_intensity
        self.decay = decay
        return self

    def with_total_variation_loss(self, reg_weight=0.005, start_epoch=3):
        self._use_total_variation_loss = True
        self.total_variation_start_epoch = start_epoch
        self.total_variation_reg_weight = reg_weight
        return self

    def with_pull_away_term_loss(self, loss_weight=0.1):

        def pullaway_loss():
            embeddings = None
            traindata = self.discriminator.training_context['train_data']
            if 'embeddings' in self.discriminator._model._modules:
                embeddings = self.discriminator._model.embeddings._output_tensor
            elif 'fake_feature' in traindata:
                embeddings = traindata['fake_feature']
            if embeddings is not None:
                embeddings = Flatten()(embeddings)
                norm = sqrt(sum(embeddings ** 2.0, -1, keepdim=True))
                normalized_emb = embeddings / norm
                similarity = matmul(normalized_emb, normalized_emb, transpose_b=True)
                batch_size = int_shape(embeddings)[0]
                loss_pt = (sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
                return loss_pt
            else:
                return to_tensor(0.0)

        self._use_pull_away_term_loss = True
        self.generator.with_loss(pullaway_loss, loss_weight=loss_weight)
        return self

    def with_gradient_penalty(self, lambda_term=10):

        def gradient_penalty(img_real, img_fake):
            shp = int_shape(img_real)

            eta = random_uniform((shp[0], 1, 1, 1), 0.0, 1.0).to(get_device())
            interpolated = eta * img_real + ((1 - eta) * img_fake)
            gradients = None
            if get_backend() == 'pytorch':
                from torch import autograd
                interpolated.requires_grad = True

                # calculate probability of interpolated examples
                prob_interpolated = self.discriminator(interpolated)

                # calculate gradients of probabilities with respect to examples
                gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                          grad_outputs=ones_like(prob_interpolated, requires_grad=False).to(get_device()),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]
                return ((sqrt(reduce_sum(gradients ** 2, axis=[2, 3])) - 1.0) ** 2).mean()
            elif get_backend() == 'tensorflow':
                with tf.GradientTape() as t:
                    t.watch(interpolated)
                    prob_interpolated = self.discriminate(interpolated)
                gradients = t.gradient(prob_interpolated, interpolated)
                return ((sqrt(reduce_sum(gradients ** 2, axis=[1, 2])) - 1.0) ** 2).mean()

        for modual in self.discriminator.model.children():
            if 'BatchNorm' in modual.__class__.__name__:
                sys.stdout.write('Your discriminator already use batch normalization, suggest not use gradient_penalty.' + '\n\r')
                break
        self.discriminator.with_loss(gradient_penalty, loss_weight=lambda_term)
        return self

    def with_feature_matching(self, loss_weight=0.5):
        self._use_feature_matching = True
        moduals = list(self.discriminator.model.children())
        for i in range(len(moduals)):
            m = moduals[i]
            if isinstance(m, Flatten) and i + 1 < len(moduals):
                if 'MinibatchDiscrimination' in moduals[i + 1].__class__.__name__:
                    moduals[i].keep_output = True
                    self.discriminator_feature_uuid = moduals[i].uuid
                    break
                else:
                    moduals[i + 1].keep_output = True
                    self.discriminator_feature_uuid = moduals[i + 1].uuid
                    break
            elif isinstance(m, Dense) or isinstance(m, GlobalAvgPool2d) and i + 1 < len(moduals):
                moduals[i].keep_output = True
                self.discriminator_feature_uuid = moduals[i].uuid
                break

        def feature_matching(real_features, fake_features):
            if fake_features is not None and real_features is not None:
                return L2Loss(reduction='mean')(fake_features, real_features)

            return to_tensor(0.0)

        self.generator.with_loss(feature_matching, loss_weight=loss_weight)
        return self

    def with_ttur(self, multiplier=2.5):
        d_lr = self.discriminator.optimizer.lr
        g_lr = self.generator.optimizer.lr
        self.discriminator.optimizer.lr = g_lr * multiplier
        self.discriminator.optimizer.base_lr = g_lr * multiplier
        return self

    def with_gan_type(self, gan_type=''):
        self.gan_type = gan_type
        real_label, fake_label = 1, 0

        def metric_dfake(d_fake):
            return d_fake.mean()

        def metric_dreal(d_real):
            return d_real.mean()

        def g_loss(d_fake, real_label):
            return BCELoss()(d_fake, real_label)

        def real_loss(d_real, real_label):
            return BCELoss()(d_real, real_label)

        def fake_loss(d_fake, fake_label):
            return BCELoss()(d_fake, fake_label)

        if self.gan_type == 'gan':

            self.generator.with_loss(g_loss)
            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(fake_loss, loss_weight=0.5)

        elif self.gan_type == 'dcgan':
            self.generator.with_loss(g_loss)
            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(fake_loss, loss_weight=0.5)


        elif self.gan_type == 'wgan':
            if isinstance(self.discriminator.model[-1], Sigmoid):
                self.discriminator.model.remove_at(-1)

            def g_loss(d_fake):
                return -d_fake.mean()

            self.generator.with_loss(g_loss)

        elif self.gan_type == 'wgan-gp':
            if isinstance(self.discriminator.model[-1], Sigmoid):
                self.discriminator.model.remove_at(-1)

            def g_loss(d_fake):
                return -d_fake.mean()

            self.generator.with_loss(g_loss)

        elif self.gan_type == 'wgan-div':
            if isinstance(self.discriminator.model[-1], Sigmoid):
                self.discriminator.model.remove_at(-1)

            def g_loss(d_fake):
                return -d_fake.mean()

            self.generator.with_loss(g_loss)
        elif self.gan_type == 'began':  # Boundary Equilibrium GAN
            if self.is_generator_first:
                self.generator.training_context['retain_graph'] = True

            def g_loss(img_fake, d_fake):
                return L1Loss()(img_fake, d_fake)

            self.generator.with_loss(g_loss)

            def real_loss(img_real, d_real):
                return L1Loss()(img_real, d_real)

            def fake_loss(img_fake, d_fake):
                return L1Loss()(img_fake, d_fake)

            def weight_fake_loss(img_fake, d_fake):
                return - self.k_t * L1Loss()(img_fake.detach(), d_fake)

            # if self.is_generator_first:
            # self.generator.training_context['retain_graph']=True
            self.k_t = 0
            self.measure = 1
            self.discriminator.training_context['train_data']['k_t'] = 0
            self.discriminator.training_context['train_data']['measure'] = 1
            self.discriminator.with_loss(real_loss)
            self.discriminator.with_loss(weight_fake_loss, name='weight_fake_loss')
        elif self.gan_type == 'ebgan':  # Energy-based GAN
            if self.is_generator_first:
                self.generator.training_context['retain_graph'] = True

            def g_loss(img_fake, d_fake):
                return L2Loss()(img_fake, d_fake)

            self.generator.with_loss(g_loss)

            def real_loss(img_real, d_real):
                return L2Loss()(img_real, d_real)

            def fake_loss(img_fake, d_fake):
                return L2Loss()(img_fake, d_fake)

            def weight_fake_loss(img_fake, d_fake):
                return clip(self.margin - L2Loss(reduction='mean')(img_fake.detach(), d_fake), min=0)

            # if self.is_generator_first:
            # self.generator.training_context['retain_graph']=True
            self.margin = 0.1
            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(weight_fake_loss, name='weight_fake_loss', loss_weight=0.5)
        elif self.gan_type == 'lsgan':  # least squared
            def g_loss(d_fake, real_label):
                return MSELoss()(d_fake, real_label)

            self.generator.with_loss(g_loss)

            def real_loss(d_real, real_label):
                return MSELoss()(d_real, real_label)

            def fake_loss(d_fake, fake_label):
                return MSELoss()(d_fake, fake_label)

            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(fake_loss, loss_weight=0.5)
        elif self.gan_type == 'lsgan1':  # loss sensitive
            def g_loss(d_fake, real_label):
                return MSELoss(d_fake, real_label)

            self.generator.with_loss(g_loss)
        elif self.gan_type == 'rasgan':
            if isinstance(self.discriminator.model[-1], Sigmoid):
                self.discriminator.model.remove_at(-1)
                # self.discriminator.model.add_module('tanh',Tanh())

            def g_loss(d_fake, d_real, real_label, fake_label):
                d_fake_logit = sigmoid(d_fake - d_real.mean().detach())
                d_real_logit = sigmoid(d_real.detach() - d_fake.mean())
                return - ((d_fake_logit + 1e-8).log()).mean() - ((1 - d_real_logit + 1e-8).log()).mean()

            self.generator.with_loss(g_loss, loss_weight=0.5)

            def real_loss(d_real, d_fake, real_label):
                d_real_logit = sigmoid(d_real - d_fake.mean())
                return - ((d_real_logit + 1e-8).log()).mean()

            def fake_loss(d_fake, d_real, fake_label):
                d_fake_logit = sigmoid(d_fake - d_real.mean())
                return - ((1 - d_fake_logit + 1e-8).log()).mean()

            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(fake_loss, loss_weight=0.5)

        if self.gan_type == 'began':
            def metric_k_t():
                return self.k_t

            def metric_measure():
                return self.measure

            self.generator.with_metric(g_loss, name='g_loss')
            self.discriminator.with_metric(real_loss, name='real_loss')
            self.discriminator.with_metric(fake_loss, name='fake_loss')
            self.discriminator.with_metric(metric_k_t, name='k_t')
            self.discriminator.with_metric(metric_measure, name='measure')

        elif self.gan_type == 'ebgan':
            self.generator.with_metric(g_loss, name='g_loss')
            self.discriminator.with_metric(real_loss, name='real_loss')
            self.discriminator.with_metric(fake_loss, name='fake_loss')
        else:
            self.generator.with_metric(metric_dfake, name='d_fake')
            self.discriminator.with_metric(metric_dfake, name='d_fake')
            self.discriminator.with_metric(metric_dreal, name='d_real')

        return self

    def generate_datafeed(self, data_provider):
        if data_provider.signature is None:
            _ = data_provider.next()
        # data_input=data_provider.traindata.data.symbol
        self.is_condition_gan = False
        if len(data_provider.traindata.unpair) > 0:
            data_unpair = data_provider.traindata.unpair
            data_provider.traindata.unpair.symbol = 'noise'

        elif isinstance(data_provider.traindata.label, ImageDataset) and len(data_provider.traindata.data) == len(data_provider.traindata.label):
            self.is_condition_gan = True

        self.generator.data_feed = OrderedDict()
        self.generator.data_feed['input'] = data_provider.traindata.data.symbol if self.is_condition_gan else data_provider.traindata.unpair
        self.generator.data_feed['img_fake'] = 'output'
        self.generator.data_feed['img_real'] = data_provider.traindata.label.symbol if self.is_condition_gan else data_provider.traindata.data.symbol
        self.generator.data_feed['d_fake'] = 'd_fake'
        self.generator.data_feed['d_real'] = 'd_real'
        self.generator.data_feed['real_label'] = 'real_label'
        self.generator.data_feed['fake_label'] = 'fake_label'
        if self.is_condition_gan:
            self.generator.data_feed['target'] = data_provider.traindata.label.symbol if self.is_condition_gan else data_provider.traindata.data.symbol
        if self._use_feature_matching:
            self.generator.data_feed['real_features'] = 'real_features'
            self.generator.data_feed['fake_features'] = 'fake_features'
        ctx.print('generator data_feed:{0}'.format(self.generator.data_feed))

        self.discriminator.data_feed = OrderedDict()
        self.discriminator.data_feed['input'] = data_provider.traindata.label.symbol if self.is_condition_gan else data_provider.traindata.data.symbol
        self.discriminator.data_feed['img_real'] = data_provider.traindata.label.symbol if self.is_condition_gan else data_provider.traindata.data.symbol
        self.discriminator.data_feed['img_fake'] = 'img_fake'
        self.discriminator.data_feed['d_real'] = "output"
        self.discriminator.data_feed['d_fake'] = "d_fake"

        if self.gan_type == 'began':
            self.discriminator.optimizer.lr = 2e-4
            self.generator.optimizer.lr = 2e-4
            self.discriminator.data_feed['k_t'] = 'k_t'
            self.discriminator.data_feed['measure'] = 'measure'

        self.discriminator.data_feed['real_label'] = 'real_label'
        self.discriminator.data_feed['fake_label'] = 'fake_label'
        ctx.print('discriminator data_feed:{0}'.format(self.discriminator.data_feed))

    def start_now(self, collect_data_inteval=1, is_resume=False, only_steps=False, max_batches=np.inf,
                  keep_weights_history=False, keep_gradient_history=False):

        def g_get_dfake(training_context):
            traindata = training_context['train_data']
            traindata['d_fake'] = self.discriminator(traindata['output'].to(get_device()))
            if self._use_feature_matching and self.discriminator_feature_uuid in self.discriminator.nodes:
                traindata['fake_features'] = self.discriminator.nodes[self.discriminator_feature_uuid].output

            traindata['real_label'] = ones_like(traindata['d_fake']).detach().to(get_device())
            traindata['fake_label'] = zeros_like(traindata['d_fake']).detach().to(get_device())

        def g_get_dreal(training_context):
            traindata = training_context['train_data']

            if (self.is_generator_first and 'output' not in self.discriminator.training_context['train_data']) or self._use_feature_matching:
                traindata['d_real'] = self.discriminator(traindata[training_context['data_feed']['img_real']]).detach()
            else:
                traindata['d_real'] = self.discriminator.training_context['train_data']['output'].detach()

            if self._use_feature_matching and self.discriminator_feature_uuid in self.discriminator.nodes:
                traindata['real_features'] = self.discriminator.nodes[self.discriminator_feature_uuid].output

        def d_get_dfake(training_context):
            traindata = training_context['train_data']
            if self.gan_type == 'began':
                traindata['k_t'] = to_tensor(self.k_t)
                traindata['measure'] = to_tensor(self.measure)

            if not self.is_generator_first and 'output' not in self.generator.training_context['train_data']:
                traindata[training_context['data_feed']['img_fake']] = self.generator(traindata[data_provider.traindata.data.symbol if self.is_condition_gan else 'noise']).detach()
            else:
                traindata[training_context['data_feed']['img_fake']] = self.generator.training_context['train_data']['output'].detach()
            traindata['d_fake'] = self.discriminator(traindata[training_context['data_feed']['img_fake']])

            traindata['real_label'] = ones_like(traindata['d_fake']).detach().to(get_device())
            if self._use_label_smoothing is not None:
                traindata['real_label'] = clip(random_normal_like(traindata['d_fake'], mean=1, std=0.02), 0.8, 1.2).detach().to(get_device())
            traindata['fake_label'] = zeros_like(traindata['d_fake']).detach().to(get_device())
            if self._use_label_smoothing == 'two_side':
                traindata['fake_label'] = clip(abs(random_normal_like(traindata['d_fake'], mean=0, std=0.02)), 0.0, 0.2).detach().to(get_device())

        def d_get_dreal(training_context):
            traindata = training_context['train_data']
            traindata['d_real'] = traindata['output']
            traindata['real_label'] = ones_like(traindata['output']).detach().to(get_device())
            if self._use_label_smoothing is not None:
                traindata['real_label'] = random_uniform_like(traindata['output'], 0.9, 1).detach().to(get_device())
            traindata['fake_label'] = zeros_like(traindata['output']).detach().to(get_device())
            if self._use_label_smoothing == 'two_side':
                traindata['fake_label'] = clip(abs(random_normal_like(traindata['d_fake'], mean=0, std=0.02)), 0.0, 0.2).detach().to(get_device())

        data_provider = self._dataloaders.value_list[0]

        data_provider.batch_size = self.batch_size
        data_provider.mode = 'dict'
        # generate data feed
        self.generate_datafeed(data_provider)
        try:
            self.execution_id = get_time_suffix()
            exception_cnt = 0
            abnormal_num_count = 0
            # update callback
            if ctx.enable_tensorboard:
                for idx, (item, item_name) in enumerate(zip(self.training_items.value_list, self.training_names.value_list)):
                    if hasattr(item, 'training_context'):
                        for context_item in list(item.training_context.values()):
                            if isinstance(context_item, HistoryBase):
                                context_item.training_name = item_name

                        item.training_context['training_name'] = item_name
                        item.training_context['summary_writer'] = ctx.summary_writer

                make_dir_if_need(os.path.join(working_directory, 'Logs'))

                sys.stdout.writelines(
                    ['Please execute the command to initial tensorboard:  tensorboard --logdir={0}  --port 6006 \n\r'.format(os.path.join(working_directory, 'Logs'))])
                sys.stdout.writelines(['Tensorboard is initialized. You can access tensorboard at http://localhost:6006/   \n\r'])

            if not is_resume or only_steps == True:
                max_name_length = builtins.max([len(name) for name in self.training_names.value_list])
                for item in self.training_items.values():
                    # sysnc device
                    item.model.train()
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
                self.generator.trigger_when(when='on_loss_calculation_start', action=g_get_dfake)
                if self.gan_type in ('rasgan') or self._use_feature_matching:
                    self.generator.trigger_when(when='on_loss_calculation_start', action=g_get_dreal)
                if self.gan_type in ('began'):
                    def update_k_t(training_context):
                        traindata = training_context['train_data']
                        if 'k_t' not in traindata:
                            traindata['k_t'] = 0
                        if len(self.discriminator.training_context['losses']) > 0 and len(self.discriminator.training_context['metrics']) > 0:
                            d_real_loss = self.discriminator.training_context['losses']['real_loss'][-1][1]
                            d_fake_loss = self.discriminator.training_context['metrics']['fake_loss'][-1][1]
                            gamma = 0.5
                            lambda_k = 0.001
                            g_d_balance = (gamma * d_real_loss - d_fake_loss)
                            self.k_t += lambda_k * g_d_balance
                            self.k_t = max(min(1, self.k_t), 0)
                            self.measure = d_real_loss + abs(g_d_balance)
                        traindata['k_t'] = to_tensor(self.k_t)

                        traindata['measure'] = to_tensor(self.measure)

                    self.discriminator.trigger_when(when='on_loss_calculation_end', action=update_k_t)

                self.discriminator.trigger_when(when='on_loss_calculation_start', action=d_get_dfake)
                self.discriminator.trigger_when(when='on_loss_calculation_start', action=d_get_dreal)
                # shared callbacks will access training plan dict instead of training_context
                for callback in self.callbacks:
                    if callback.is_shared:
                        callback.on_training_start(self.__dict__)

            if not is_resume or only_steps == True:
                if collect_data_inteval == 1 and len(data_provider.batch_sampler) * self.num_epochs > 1000:
                    collect_data_inteval = self.default_collect_data_inteval
            if only_steps:
                self.num_epochs = (max_batches // len(data_provider.batch_sampler)) + 2

            for epoch in range(self.num_epochs):
                try:
                    for mbs, return_data in enumerate(data_provider):

                        if self.is_terminate:
                            for callback in self.callbacks:
                                if callback.is_shared:
                                    callback.on_training_terminated(self.__dict__)

                            for k, trainitem in self.training_items.items():

                                for callback in trainitem.training_context['callbacks']:
                                    if not callback.is_shared:
                                        callback.on_training_terminated(trainitem.training_context)
                            data_provider.mode = 'tuple'
                        else:

                            iter_data = OrderedDict()

                            if isinstance(return_data, OrderedDict):
                                for spec, data in return_data.item_list:
                                    iter_data[spec.name] = data
                            elif isinstance(return_data, tuple):
                                for i in range(len(return_data)):
                                    iter_data[data_provider.traindata.data_template.key_list[i].name] = return_data[i]

                            # check weather need out-of-sample evaluation
                            need_out_sample_evaluation = False

                            # input, target = Variable(input).to(self.device), Variable(target).to(self.device)

                            for trainitem_name, trainitem in zip(self.training_names.value_list, self.training_items.value_list):
                                train_data = copy.deepcopy(iter_data)
                                trainitem.training_context['data_template'] = data_provider.traindata.data_template
                                trainitem.training_context['collect_data_inteval'] = collect_data_inteval
                                trainitem.training_context['model_name'] = trainitem_name
                                if epoch < int(trainitem.start_epoch):
                                    trainitem.training_context['stop_update'] = 1
                                if self.max_noise_intensity > 0 and trainitem.training_context['gan_role'] == 'discriminator':
                                    current_intensity = self.min_noise_intensity + (self.max_noise_intensity - self.min_noise_intensity) * math.exp(-1.0 * self.steps / self.decay)
                                    if self.steps % 100 == 0:
                                        ctx.print('noise intensity:{0}'.format(current_intensity))
                                    train_data['img_real'] = train_data['img_real'] + random_normal_like(train_data['img_real']) * current_intensity

                                # if self.is_generator_first:
                                #     trainitem.model.zero_grad()
                                should_collect_data = True

                                dis_k = 1
                                if self.gan_type not in ['began', 'ebgan'] and trainitem.training_context['gan_role'] == 'discriminator' and 'd_fake' in trainitem.training_context[
                                    'metrics']:

                                    if trainitem.training_context['gan_role'] == 'discriminator' and dis_k == 1 and trainitem.training_context['metrics']['d_fake'][-1][1] < 0.1:
                                        dis_k = 1
                                    if trainitem.training_context['gan_role'] == 'discriminator' and dis_k == 1 and trainitem.training_context['metrics']['d_fake'][-1][1] > 0.3:
                                        dis_k = 1
                                if (trainitem.training_context['gan_role'] == 'discriminator' and mbs % dis_k == 0) or (
                                        trainitem.training_context['gan_role'] == 'generator' and mbs % 1 == 0):
                                    trainitem.training_context['stop_update'] = 0
                                else:
                                    trainitem.training_context['stop_update'] = 1
                                    should_collect_data = False
                                if self._use_total_variation_loss and trainitem.training_context[
                                    'gan_role'] == 'generator' and epoch == self.total_variation_start_epoch and mbs == 0:
                                    self.generator.with_regularizer('total_variation_norm_reg', reg_weight=self.total_variation_reg_weight)

                                trainitem.train_model(train_data, None,
                                                      epoch if only_steps == False else 0,
                                                      mbs if only_steps == False else self.steps,
                                                      self.num_epochs if only_steps == False else 1,
                                                      len(data_provider.batch_sampler) if only_steps == False else max_batches,
                                                      is_collect_data=mbs == 0 or (should_collect_data and mbs % collect_data_inteval) == 0,
                                                      is_print_batch_progress=self.print_progress_unit == 'batch' and mbs > 0 and mbs % self.print_progress_frequency == 0,
                                                      is_print_epoch_progress=self.print_progress_unit == 'epoch' and epoch > 0 and epoch % self.print_progress_frequency == 0,
                                                      log_gradients=keep_gradient_history, log_weights=keep_weights_history,
                                                      accumulate_grads=False, is_out_sample_evaluation=need_out_sample_evaluation)
                            self.steps += 1

                            if ctx.enable_tensorboard and len(self.training_items) > 1 and mbs % collect_data_inteval == 0:
                                compare_dict = OrderedDict()
                                step = None
                                for trainitem_name, trainitem in zip(self.training_names.value_list, self.training_items.value_list):
                                    for k, v in trainitem.training_context["losses"].items():
                                        if k not in compare_dict:
                                            compare_dict[k] = OrderedDict()
                                        compare_dict[k][k + "/" + trainitem_name] = v[-1][1]
                                        step = v[-1][0]
                                    for k, v in trainitem.training_context["metrics"].items():
                                        if k not in compare_dict:
                                            compare_dict[k] = OrderedDict()
                                        compare_dict[k][k + "/" + trainitem_name] = v[-1][1]
                                for k, v in compare_dict.items():
                                    ctx.summary_writer.add_scalars(k, v, step)

                            if (self.print_progress_unit == 'batch' and mbs % self.print_progress_frequency == 0) or \
                                    (self.print_progress_unit == 'epoch' and (epoch + 1) % self.print_progress_frequency == 0):
                                if len(self.training_items) > 1:
                                    ctx.print(' \n', flush=True)

                            for k, trainitem in self.training_items.items():
                                for callback in trainitem.training_context['callbacks']:
                                    if not callback.is_shared:
                                        callback.on_overall_batch_end(trainitem.training_context)
                            for callback in self.callbacks:
                                if callback.is_shared:
                                    callback.on_overall_batch_end(self.__dict__)

                            if self.save_model_frequency > 0 and self.save_model_unit == 'batch' and (self.steps + 1) % \
                                    self.save_model_frequency == 0:
                                for k, trainitem in self.training_items.items():
                                    trainitem.save_model(trainitem.training_context['save_path'], )
                                    if ctx.enable_tensorboard and ('upload_onnx' not in trainitem.training_context or trainitem.training_context['upload_onnx'] == False):
                                        trainitem.save_onnx(trainitem.training_context['save_path'].replace('.pth', '.onnx'))
                                        ctx.summary_writer.add_onnx_graph(trainitem.training_context['save_path'].replace('.pth', '.onnx'));
                                        trainitem.training_context['upload_onnx'] = True
                            if only_steps == True and self.steps >= max_batches - 1:
                                for k, trainitem in self.training_items.items():
                                    try:
                                        trainitem.save_model(trainitem.training_context['save_path'], )
                                    except Exception as e:
                                        ctx.print(e)
                                data_provider.mode = 'tuple'
                                return True

                            if only_steps == False and (mbs + 1) % len(data_provider.batch_sampler) == 0:
                                break


                except StopIteration:
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_epoch_end()
                        trainitem.save_model(trainitem.training_context['save_path'], )

                except ValueError as ve:
                    ctx.print(ve)
                    PrintException()
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_excution_exception()
                except Exception as e:
                    ctx.print(e)
                    PrintException()
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_excution_exception()

                if self.save_model_frequency > 0 and self.save_model_unit == 'epoch' and (
                        epoch + 1) % self.save_model_frequency == 0:
                    for k, trainitem in self.training_items.items():
                        trainitem.save_model(trainitem.training_context['save_path'], )
            data_provider.mode = 'tuple'


        except KeyboardInterrupt:
            for k, trainitem in self.training_items.items():
                trainitem.save_model(trainitem.training_context['save_path'], )
            data_provider.mode = 'tuple'
        except Exception as e:
            ctx.print(e)
            PrintException()
            for k, trainitem in self.training_items.items():
                trainitem.save_model(trainitem.training_context['save_path'], )
            data_provider.mode = 'tuple'


class CycleGanTrainingPlan(TrainingPlan):
    def __init__(self,):
        super().__init__()
        self.is_generator_first = None
        self.gan_type = None
        self.is_condition_gan = False
        self.discriminator = None
        self.netG_A = None
        self.netG_B = None
        self.netD_A = None
        self.netD_B = None
        self.optimizerG=None
        self.optimizerD = None
        self._use_label_smoothing = False
        self.max_noise_intensity = 0
        self.min_noise_intensity = 0
        self.decay = 10000
        self._use_total_variation_loss = False
        self.total_variation_reg_weight = 0.005
        self.total_variation_start_epoch = 3
        self._use_pull_away_term_loss = False
        self._use_feature_matching = False
        self._use_label_smoothing = None
        self.discriminator_feature_uuid = None

    def with_generator(self, netG_A,netG_B, name='modelG'):
        if len(self.training_items) == 0:
            self.is_generator_first = True
        self.netG_A=netG_A
        self.netG_B=netG_B

        #generator.training_context['gan_role'] = 'generator'
        if self.optimizerG is None:
            self.optimizerG=Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),lr=2e-4, betas=(0.5, 0.999))
        # generator.with_callbacks(StepLR(frequency=5, unit='epoch', gamma=0.75))
        # if not any([isinstance(cb, TileImageCallback) for cb in generator.callbacks]):
        #     generator.with_callbacks(GanTileImageCallback(batch_inteval=50))
        return self.add_training_item(self.netG_A, name='netG_A', start_epoch=0).add_training_item(self.netG_B, name='netG_B', start_epoch=0)

    def with_discriminator(self, netD_A,netD_B,  name='modelD'):
        if len(self.training_items) == 0:
            self.is_generator_first = False

        #discriminator.training_context['gan_role'] = 'discriminator'
        if self.optimizerD is None:
            self.optimizerD=Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),lr=2e-4, betas=(0.5, 0.999))
       # discriminator.with_callbacks(StepLR(frequency=5, unit='epoch', gamma=0.75))
        #self.discriminator = discriminator
        # if not self.is_generator_first:
        #     self.discriminator.training_context['retain_graph'] = True
        # else:
        #     self.discriminator.training_context['retain_graph'] = False
        return self.add_training_item(self.netD_A, name='netD_A', start_epoch=0).add_training_item(self.netD_B, name='netD_B', start_epoch=0)

    def with_label_smoothing(self, one_side=True):
        self._use_label_smoothing = "one_side" if one_side else "two_side"
        return self

    def with_noised_real_images(self, max_noise_intensity=0.1, min_noise_intensity=0, decay=10000):
        self.max_noise_intensity = max_noise_intensity
        self.min_noise_intensity = min_noise_intensity
        self.decay = decay
        return self

    def with_total_variation_loss(self, reg_weight=0.005, start_epoch=3):
        self._use_total_variation_loss = True
        self.total_variation_start_epoch = start_epoch
        self.total_variation_reg_weight = reg_weight
        return self

    def with_pull_away_term_loss(self, loss_weight=0.1):

        def pullaway_loss():
            embeddings = None
            traindata = self.discriminator.training_context['train_data']
            if 'embeddings' in self.discriminator._model._modules:
                embeddings = self.discriminator._model.embeddings._output_tensor
            elif 'fake_feature' in traindata:
                embeddings = traindata['fake_feature']
            if embeddings is not None:
                embeddings = Flatten()(embeddings)
                norm = sqrt(sum(embeddings ** 2.0, -1, keepdim=True))
                normalized_emb = embeddings / norm
                similarity = matmul(normalized_emb, normalized_emb, transpose_b=True)
                batch_size = int_shape(embeddings)[0]
                loss_pt = (sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
                return loss_pt
            else:
                return to_tensor(0.0)

        self._use_pull_away_term_loss = True
        self.generator.with_loss(pullaway_loss, loss_weight=loss_weight)
        return self

    def with_gradient_penalty(self, lambda_term=10):

        def gradient_penalty(img_real, img_fake):
            shp = int_shape(img_real)

            eta = random_uniform((shp[0], 1, 1, 1), 0.0, 1.0).to(get_device())
            interpolated = eta * img_real + ((1 - eta) * img_fake)
            gradients = None
            if get_backend() == 'pytorch':
                from torch import autograd
                interpolated.requires_grad = True

                # calculate probability of interpolated examples
                prob_interpolated = self.discriminator(interpolated)

                # calculate gradients of probabilities with respect to examples
                gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                          grad_outputs=ones_like(prob_interpolated, requires_grad=False).to(get_device()),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]
                return ((sqrt(reduce_sum(gradients ** 2, axis=[2, 3])) - 1.0) ** 2).mean()
            elif get_backend() == 'tensorflow':
                with tf.GradientTape() as t:
                    t.watch(interpolated)
                    prob_interpolated = self.discriminate(interpolated)
                gradients = t.gradient(prob_interpolated, interpolated)
                return ((sqrt(reduce_sum(gradients ** 2, axis=[1, 2])) - 1.0) ** 2).mean()

        for modual in self.discriminator.model.children():
            if 'BatchNorm' in modual.__class__.__name__:
                sys.stdout.write('Your discriminator already use batch normalization, suggest not use gradient_penalty.' + '\n\r')
                break
        self.discriminator.with_loss(gradient_penalty, loss_weight=lambda_term)
        return self

    def with_feature_matching(self, loss_weight=0.5):
        self._use_feature_matching = True
        moduals = list(self.discriminator.model.children())
        for i in range(len(moduals)):
            m = moduals[i]
            if isinstance(m, Flatten) and i + 1 < len(moduals):
                if 'MinibatchDiscrimination' in moduals[i + 1].__class__.__name__:
                    moduals[i].keep_output = True
                    self.discriminator_feature_uuid = moduals[i].uuid
                    break
                else:
                    moduals[i + 1].keep_output = True
                    self.discriminator_feature_uuid = moduals[i + 1].uuid
                    break
            elif isinstance(m, Dense) or isinstance(m, GlobalAvgPool2d) and i + 1 < len(moduals):
                moduals[i].keep_output = True
                self.discriminator_feature_uuid = moduals[i].uuid
                break

        def feature_matching(real_features, fake_features):
            if fake_features is not None and real_features is not None:
                return L2Loss(reduction='mean')(fake_features, real_features)

            return to_tensor(0.0)

        self.generator.with_loss(feature_matching, loss_weight=loss_weight)
        return self

    def with_ttur(self, multiplier=2.5):
        d_lr = self.discriminator.optimizer.lr
        g_lr = self.generator.optimizer.lr
        self.discriminator.optimizer.lr = g_lr * multiplier
        self.discriminator.optimizer.base_lr = g_lr * multiplier
        return self

    def with_gan_type(self, gan_type=''):
        self.gan_type = gan_type
        real_label, fake_label = 1, 0

        def metric_dfake(d_fake):
            return d_fake.mean()

        def metric_dreal(d_real):
            return d_real.mean()

        def g_loss(d_fake, real_label):
            return BCELoss()(d_fake, real_label)

        def real_loss(d_real, real_label):
            return BCELoss()(d_real, real_label)

        def fake_loss(d_fake, fake_label):
            return BCELoss()(d_fake, fake_label)

        if self.gan_type == 'gan':

            self.generator.with_loss(g_loss)
            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(fake_loss, loss_weight=0.5)

        elif self.gan_type == 'dcgan':
            self.generator.with_loss(g_loss)
            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(fake_loss, loss_weight=0.5)


        elif self.gan_type == 'wgan':
            if isinstance(self.discriminator.model[-1], Sigmoid):
                self.discriminator.model.remove_at(-1)

            def g_loss(d_fake):
                return -d_fake.mean()

            self.generator.with_loss(g_loss)

        elif self.gan_type == 'wgan-gp':
            if isinstance(self.discriminator.model[-1], Sigmoid):
                self.discriminator.model.remove_at(-1)

            def g_loss(d_fake):
                return -d_fake.mean()

            self.generator.with_loss(g_loss)

        elif self.gan_type == 'wgan-div':
            if isinstance(self.discriminator.model[-1], Sigmoid):
                self.discriminator.model.remove_at(-1)

            def g_loss(d_fake):
                return -d_fake.mean()

            self.generator.with_loss(g_loss)
        elif self.gan_type == 'began':  # Boundary Equilibrium GAN
            if self.is_generator_first:
                self.generator.training_context['retain_graph'] = True

            def g_loss(img_fake, d_fake):
                return L1Loss()(img_fake, d_fake)

            self.generator.with_loss(g_loss)

            def real_loss(img_real, d_real):
                return L1Loss()(img_real, d_real)

            def fake_loss(img_fake, d_fake):
                return L1Loss()(img_fake, d_fake)

            def weight_fake_loss(img_fake, d_fake):
                return - self.k_t * L1Loss()(img_fake.detach(), d_fake)

            # if self.is_generator_first:
            # self.generator.training_context['retain_graph']=True
            self.k_t = 0
            self.measure = 1
            self.discriminator.training_context['train_data']['k_t'] = 0
            self.discriminator.training_context['train_data']['measure'] = 1
            self.discriminator.with_loss(real_loss)
            self.discriminator.with_loss(weight_fake_loss, name='weight_fake_loss')
        elif self.gan_type == 'ebgan':  # Energy-based GAN
            if self.is_generator_first:
                self.generator.training_context['retain_graph'] = True

            def g_loss(img_fake, d_fake):
                return L2Loss()(img_fake, d_fake)

            self.generator.with_loss(g_loss)

            def real_loss(img_real, d_real):
                return L2Loss()(img_real, d_real)

            def fake_loss(img_fake, d_fake):
                return L2Loss()(img_fake, d_fake)

            def weight_fake_loss(img_fake, d_fake):
                return clip(self.margin - L2Loss(reduction='mean')(img_fake.detach(), d_fake), min=0)

            # if self.is_generator_first:
            # self.generator.training_context['retain_graph']=True
            self.margin = 0.1
            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(weight_fake_loss, name='weight_fake_loss', loss_weight=0.5)
        elif self.gan_type == 'lsgan':  # least squared
            def g_loss(d_fake, real_label):
                return MSELoss()(d_fake, real_label)

            self.generator.with_loss(g_loss)

            def real_loss(d_real, real_label):
                return MSELoss()(d_real, real_label)

            def fake_loss(d_fake, fake_label):
                return MSELoss()(d_fake, fake_label)

            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(fake_loss, loss_weight=0.5)
        elif self.gan_type == 'lsgan1':  # loss sensitive
            def g_loss(d_fake, real_label):
                return MSELoss(d_fake, real_label)

            self.generator.with_loss(g_loss)
        elif self.gan_type == 'rasgan':
            if isinstance(self.discriminator.model[-1], Sigmoid):
                self.discriminator.model.remove_at(-1)
                # self.discriminator.model.add_module('tanh',Tanh())

            def g_loss(d_fake, d_real, real_label, fake_label):
                d_fake_logit = sigmoid(d_fake - d_real.mean().detach())
                d_real_logit = sigmoid(d_real.detach() - d_fake.mean())
                return - ((d_fake_logit + 1e-8).log()).mean() - ((1 - d_real_logit + 1e-8).log()).mean()

            self.generator.with_loss(g_loss, loss_weight=0.5)

            def real_loss(d_real, d_fake, real_label):
                d_real_logit = sigmoid(d_real - d_fake.mean())
                return - ((d_real_logit + 1e-8).log()).mean()

            def fake_loss(d_fake, d_real, fake_label):
                d_fake_logit = sigmoid(d_fake - d_real.mean())
                return - ((1 - d_fake_logit + 1e-8).log()).mean()

            self.discriminator.with_loss(real_loss, loss_weight=0.5)
            self.discriminator.with_loss(fake_loss, loss_weight=0.5)

        if self.gan_type == 'began':
            def metric_k_t():
                return self.k_t

            def metric_measure():
                return self.measure

            self.generator.with_metric(g_loss, name='g_loss')
            self.discriminator.with_metric(real_loss, name='real_loss')
            self.discriminator.with_metric(fake_loss, name='fake_loss')
            self.discriminator.with_metric(metric_k_t, name='k_t')
            self.discriminator.with_metric(metric_measure, name='measure')

        elif self.gan_type == 'ebgan':
            self.generator.with_metric(g_loss, name='g_loss')
            self.discriminator.with_metric(real_loss, name='real_loss')
            self.discriminator.with_metric(fake_loss, name='fake_loss')
        else:
            self.generator.with_metric(metric_dfake, name='d_fake')
            self.discriminator.with_metric(metric_dfake, name='d_fake')
            self.discriminator.with_metric(metric_dreal, name='d_real')

        return self

    def generate_datafeed(self, data_provider):
        if data_provider.signature is None:
            _ = data_provider.next()
        # data_input=data_provider.traindata.data.symbol
        self.is_condition_gan = False
        if len(data_provider.traindata.unpair) > 0:
            data_unpair = data_provider.traindata.unpair
            data_provider.traindata.unpair.symbol = 'noise'

        elif isinstance(data_provider.traindata.label, ImageDataset) and len(data_provider.traindata.data) == len(data_provider.traindata.label):
            self.is_condition_gan = True

        self.generator.data_feed = OrderedDict()
        self.generator.data_feed['input'] = data_provider.traindata.data.symbol if self.is_condition_gan else data_provider.traindata.unpair
        self.generator.data_feed['img_fake'] = 'output'
        self.generator.data_feed['img_real'] = data_provider.traindata.label.symbol if self.is_condition_gan else data_provider.traindata.data.symbol
        self.generator.data_feed['d_fake'] = 'd_fake'
        self.generator.data_feed['d_real'] = 'd_real'
        self.generator.data_feed['real_label'] = 'real_label'
        self.generator.data_feed['fake_label'] = 'fake_label'
        if self.is_condition_gan:
            self.generator.data_feed['target'] = data_provider.traindata.label.symbol if self.is_condition_gan else data_provider.traindata.data.symbol
        if self._use_feature_matching:
            self.generator.data_feed['real_features'] = 'real_features'
            self.generator.data_feed['fake_features'] = 'fake_features'
        ctx.print('generator data_feed:{0}'.format(self.generator.data_feed))

        self.discriminator.data_feed = OrderedDict()
        self.discriminator.data_feed['input'] = data_provider.traindata.label.symbol if self.is_condition_gan else data_provider.traindata.data.symbol
        self.discriminator.data_feed['img_real'] = data_provider.traindata.label.symbol if self.is_condition_gan else data_provider.traindata.data.symbol
        self.discriminator.data_feed['img_fake'] = 'img_fake'
        self.discriminator.data_feed['d_real'] = "output"
        self.discriminator.data_feed['d_fake'] = "d_fake"

        if self.gan_type == 'began':
            self.discriminator.optimizer.lr = 2e-4
            self.generator.optimizer.lr = 2e-4
            self.discriminator.data_feed['k_t'] = 'k_t'
            self.discriminator.data_feed['measure'] = 'measure'

        self.discriminator.data_feed['real_label'] = 'real_label'
        self.discriminator.data_feed['fake_label'] = 'fake_label'
        ctx.print('discriminator data_feed:{0}'.format(self.discriminator.data_feed))

    def start_now(self, collect_data_inteval=1, is_resume=False, only_steps=False, max_batches=np.inf,
                  keep_weights_history=False, keep_gradient_history=False):

        def g_get_dfake(training_context):
            traindata = training_context['train_data']
            traindata['d_fake'] = self.discriminator(traindata['output'].to(get_device()))
            if self._use_feature_matching and self.discriminator_feature_uuid in self.discriminator.nodes:
                traindata['fake_features'] = self.discriminator.nodes[self.discriminator_feature_uuid].output

            traindata['real_label'] = ones_like(traindata['d_fake']).detach().to(get_device())
            traindata['fake_label'] = zeros_like(traindata['d_fake']).detach().to(get_device())

        def g_get_dreal(training_context):
            traindata = training_context['train_data']

            if (self.is_generator_first and 'output' not in self.discriminator.training_context['train_data']) or self._use_feature_matching:
                traindata['d_real'] = self.discriminator(traindata[training_context['data_feed']['img_real']]).detach()
            else:
                traindata['d_real'] = self.discriminator.training_context['train_data']['output'].detach()

            if self._use_feature_matching and self.discriminator_feature_uuid in self.discriminator.nodes:
                traindata['real_features'] = self.discriminator.nodes[self.discriminator_feature_uuid].output

        def d_get_dfake(training_context):
            traindata = training_context['train_data']
            if self.gan_type == 'began':
                traindata['k_t'] = to_tensor(self.k_t)
                traindata['measure'] = to_tensor(self.measure)

            if not self.is_generator_first and 'output' not in self.generator.training_context['train_data']:
                traindata[training_context['data_feed']['img_fake']] = self.generator(traindata[data_provider.traindata.data.symbol if self.is_condition_gan else 'noise']).detach()
            else:
                traindata[training_context['data_feed']['img_fake']] = self.generator.training_context['train_data']['output'].detach()
            traindata['d_fake'] = self.discriminator(traindata[training_context['data_feed']['img_fake']])

            traindata['real_label'] = ones_like(traindata['d_fake']).detach().to(get_device())
            if self._use_label_smoothing is not None:
                traindata['real_label'] = clip(random_normal_like(traindata['d_fake'], mean=1, std=0.02), 0.8, 1.2).detach().to(get_device())
            traindata['fake_label'] = zeros_like(traindata['d_fake']).detach().to(get_device())
            if self._use_label_smoothing == 'two_side':
                traindata['fake_label'] = clip(abs(random_normal_like(traindata['d_fake'], mean=0, std=0.02)), 0.0, 0.2).detach().to(get_device())

        def d_get_dreal(training_context):
            traindata = training_context['train_data']
            traindata['d_real'] = traindata['output']
            traindata['real_label'] = ones_like(traindata['output']).detach().to(get_device())
            if self._use_label_smoothing is not None:
                traindata['real_label'] = random_uniform_like(traindata['output'], 0.9, 1).detach().to(get_device())
            traindata['fake_label'] = zeros_like(traindata['output']).detach().to(get_device())
            if self._use_label_smoothing == 'two_side':
                traindata['fake_label'] = clip(abs(random_normal_like(traindata['d_fake'], mean=0, std=0.02)), 0.0, 0.2).detach().to(get_device())

        data_provider = self._dataloaders.value_list[0]

        data_provider.batch_size = self.batch_size
        data_provider.mode = 'dict'
        # generate data feed
        self.generate_datafeed(data_provider)
        try:
            self.execution_id = get_time_suffix()
            exception_cnt = 0
            abnormal_num_count = 0
            # update callback
            if ctx.enable_tensorboard:
                for idx, (item, item_name) in enumerate(zip(self.training_items.value_list, self.training_names.value_list)):
                    if hasattr(item, 'training_context'):
                        for context_item in list(item.training_context.values()):
                            if isinstance(context_item, HistoryBase):
                                context_item.training_name = item_name

                        item.training_context['training_name'] = item_name
                        item.training_context['summary_writer'] = ctx.summary_writer

                make_dir_if_need(os.path.join(working_directory, 'Logs'))

                sys.stdout.writelines(
                    ['Please execute the command to initial tensorboard:  tensorboard --logdir={0}  --port 6006 \n\r'.format(os.path.join(working_directory, 'Logs'))])
                sys.stdout.writelines(['Tensorboard is initialized. You can access tensorboard at http://localhost:6006/   \n\r'])

            if not is_resume or only_steps == True:
                max_name_length = builtins.max([len(name) for name in self.training_names.value_list])
                for item in self.training_items.values():
                    # sysnc device
                    item.model.train()
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
                self.generator.trigger_when(when='on_loss_calculation_start', action=g_get_dfake)
                if self.gan_type in ('rasgan') or self._use_feature_matching:
                    self.generator.trigger_when(when='on_loss_calculation_start', action=g_get_dreal)
                if self.gan_type in ('began'):
                    def update_k_t(training_context):
                        traindata = training_context['train_data']
                        if 'k_t' not in traindata:
                            traindata['k_t'] = 0
                        if len(self.discriminator.training_context['losses']) > 0 and len(self.discriminator.training_context['metrics']) > 0:
                            d_real_loss = self.discriminator.training_context['losses']['real_loss'][-1][1]
                            d_fake_loss = self.discriminator.training_context['metrics']['fake_loss'][-1][1]
                            gamma = 0.5
                            lambda_k = 0.001
                            g_d_balance = (gamma * d_real_loss - d_fake_loss)
                            self.k_t += lambda_k * g_d_balance
                            self.k_t = max(min(1, self.k_t), 0)
                            self.measure = d_real_loss + abs(g_d_balance)
                        traindata['k_t'] = to_tensor(self.k_t)

                        traindata['measure'] = to_tensor(self.measure)

                    self.discriminator.trigger_when(when='on_loss_calculation_end', action=update_k_t)

                self.discriminator.trigger_when(when='on_loss_calculation_start', action=d_get_dfake)
                self.discriminator.trigger_when(when='on_loss_calculation_start', action=d_get_dreal)
                # shared callbacks will access training plan dict instead of training_context
                for callback in self.callbacks:
                    if callback.is_shared:
                        callback.on_training_start(self.__dict__)

            if not is_resume or only_steps == True:
                if collect_data_inteval == 1 and len(data_provider.batch_sampler) * self.num_epochs > 1000:
                    collect_data_inteval = self.default_collect_data_inteval
            if only_steps:
                self.num_epochs = (max_batches // len(data_provider.batch_sampler)) + 2

            for epoch in range(self.num_epochs):
                try:
                    for mbs, return_data in enumerate(data_provider):

                        if self.is_terminate:
                            for callback in self.callbacks:
                                if callback.is_shared:
                                    callback.on_training_terminated(self.__dict__)

                            for k, trainitem in self.training_items.items():

                                for callback in trainitem.training_context['callbacks']:
                                    if not callback.is_shared:
                                        callback.on_training_terminated(trainitem.training_context)
                            data_provider.mode = 'tuple'
                        else:

                            iter_data = OrderedDict()

                            if isinstance(return_data, OrderedDict):
                                for spec, data in return_data.item_list:
                                    iter_data[spec.name] = data
                            elif isinstance(return_data, tuple):
                                for i in range(len(return_data)):
                                    iter_data[data_provider.traindata.data_template.key_list[i].name] = return_data[i]

                            # check weather need out-of-sample evaluation
                            need_out_sample_evaluation = False

                            # input, target = Variable(input).to(self.device), Variable(target).to(self.device)

                            for trainitem_name, trainitem in zip(self.training_names.value_list, self.training_items.value_list):
                                train_data = copy.deepcopy(iter_data)
                                trainitem.training_context['data_template'] = data_provider.traindata.data_template
                                trainitem.training_context['collect_data_inteval'] = collect_data_inteval
                                trainitem.training_context['model_name'] = trainitem_name
                                if epoch < int(trainitem.start_epoch):
                                    trainitem.training_context['stop_update'] = 1
                                if self.max_noise_intensity > 0 and trainitem.training_context['gan_role'] == 'discriminator':
                                    current_intensity = self.min_noise_intensity + (self.max_noise_intensity - self.min_noise_intensity) * math.exp(-1.0 * self.steps / self.decay)
                                    if self.steps % 100 == 0:
                                        ctx.print('noise intensity:{0}'.format(current_intensity))
                                    train_data['img_real'] = train_data['img_real'] + random_normal_like(train_data['img_real']) * current_intensity

                                # if self.is_generator_first:
                                #     trainitem.model.zero_grad()
                                should_collect_data = True

                                dis_k = 1
                                if self.gan_type not in ['began', 'ebgan'] and trainitem.training_context['gan_role'] == 'discriminator' and 'd_fake' in trainitem.training_context[
                                    'metrics']:

                                    if trainitem.training_context['gan_role'] == 'discriminator' and dis_k == 1 and trainitem.training_context['metrics']['d_fake'][-1][1] < 0.1:
                                        dis_k = 1
                                    if trainitem.training_context['gan_role'] == 'discriminator' and dis_k == 1 and trainitem.training_context['metrics']['d_fake'][-1][1] > 0.3:
                                        dis_k = 1
                                if (trainitem.training_context['gan_role'] == 'discriminator' and mbs % dis_k == 0) or (
                                        trainitem.training_context['gan_role'] == 'generator' and mbs % 1 == 0):
                                    trainitem.training_context['stop_update'] = 0
                                else:
                                    trainitem.training_context['stop_update'] = 1
                                    should_collect_data = False
                                if self._use_total_variation_loss and trainitem.training_context[
                                    'gan_role'] == 'generator' and epoch == self.total_variation_start_epoch and mbs == 0:
                                    self.generator.with_regularizer('total_variation_norm_reg', reg_weight=self.total_variation_reg_weight)

                                trainitem.train_model(train_data, None,
                                                      epoch if only_steps == False else 0,
                                                      mbs if only_steps == False else self.steps,
                                                      self.num_epochs if only_steps == False else 1,
                                                      len(data_provider.batch_sampler) if only_steps == False else max_batches,
                                                      is_collect_data=mbs == 0 or (should_collect_data and mbs % collect_data_inteval) == 0,
                                                      is_print_batch_progress=self.print_progress_unit == 'batch' and mbs > 0 and mbs % self.print_progress_frequency == 0,
                                                      is_print_epoch_progress=self.print_progress_unit == 'epoch' and epoch > 0 and epoch % self.print_progress_frequency == 0,
                                                      log_gradients=keep_gradient_history, log_weights=keep_weights_history,
                                                      accumulate_grads=False, is_out_sample_evaluation=need_out_sample_evaluation)
                            self.steps += 1

                            if ctx.enable_tensorboard and len(self.training_items) > 1 and mbs % collect_data_inteval == 0:
                                compare_dict = OrderedDict()
                                step = None
                                for trainitem_name, trainitem in zip(self.training_names.value_list, self.training_items.value_list):
                                    for k, v in trainitem.training_context["losses"].items():
                                        if k not in compare_dict:
                                            compare_dict[k] = OrderedDict()
                                        compare_dict[k][k + "/" + trainitem_name] = v[-1][1]
                                        step = v[-1][0]
                                    for k, v in trainitem.training_context["metrics"].items():
                                        if k not in compare_dict:
                                            compare_dict[k] = OrderedDict()
                                        compare_dict[k][k + "/" + trainitem_name] = v[-1][1]
                                for k, v in compare_dict.items():
                                    ctx.summary_writer.add_scalars(k, v, step)

                            if (self.print_progress_unit == 'batch' and mbs % self.print_progress_frequency == 0) or \
                                    (self.print_progress_unit == 'epoch' and (epoch + 1) % self.print_progress_frequency == 0):
                                if len(self.training_items) > 1:
                                    ctx.print(' \n', flush=True)

                            for k, trainitem in self.training_items.items():
                                for callback in trainitem.training_context['callbacks']:
                                    if not callback.is_shared:
                                        callback.on_overall_batch_end(trainitem.training_context)
                            for callback in self.callbacks:
                                if callback.is_shared:
                                    callback.on_overall_batch_end(self.__dict__)

                            if self.save_model_frequency > 0 and self.save_model_unit == 'batch' and (self.steps + 1) % \
                                    self.save_model_frequency == 0:
                                for k, trainitem in self.training_items.items():
                                    trainitem.save_model(trainitem.training_context['save_path'], )
                                    if ctx.enable_tensorboard and ('upload_onnx' not in trainitem.training_context or trainitem.training_context['upload_onnx'] == False):
                                        trainitem.save_onnx(trainitem.training_context['save_path'].replace('.pth', '.onnx'))
                                        ctx.summary_writer.add_onnx_graph(trainitem.training_context['save_path'].replace('.pth', '.onnx'));
                                        trainitem.training_context['upload_onnx'] = True
                            if only_steps == True and self.steps >= max_batches - 1:
                                for k, trainitem in self.training_items.items():
                                    try:
                                        trainitem.save_model(trainitem.training_context['save_path'], )
                                    except Exception as e:
                                        ctx.print(e)
                                data_provider.mode = 'tuple'
                                return True

                            if only_steps == False and (mbs + 1) % len(data_provider.batch_sampler) == 0:
                                break


                except StopIteration:
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_epoch_end()
                        trainitem.save_model(trainitem.training_context['save_path'], )

                except ValueError as ve:
                    ctx.print(ve)
                    PrintException()
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_excution_exception()
                except Exception as e:
                    ctx.print(e)
                    PrintException()
                    for k, trainitem in self.training_items.items():
                        trainitem.do_on_excution_exception()

                if self.save_model_frequency > 0 and self.save_model_unit == 'epoch' and (
                        epoch + 1) % self.save_model_frequency == 0:
                    for k, trainitem in self.training_items.items():
                        trainitem.save_model(trainitem.training_context['save_path'], )
            data_provider.mode = 'tuple'


        except KeyboardInterrupt:
            for k, trainitem in self.training_items.items():
                trainitem.save_model(trainitem.training_context['save_path'], )
            data_provider.mode = 'tuple'
        except Exception as e:
            ctx.print(e)
            PrintException()
            for k, trainitem in self.training_items.items():
                trainitem.save_model(trainitem.training_context['save_path'], )
            data_provider.mode = 'tuple'




