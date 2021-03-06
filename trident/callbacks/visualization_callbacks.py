from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import math
import os
import sys
import warnings
import time
import numpy as np
import torch

from trident import context
import numbers
from trident.backend.common import *
from trident.backend.load_backend import *
from trident.backend.pillow_backend import image2array
from trident.callbacks.callback_base import CallbackBase
from trident.data.dataset import MaskDataset, ImageDataset, ZipDataset
from trident.data.image_common import image_backend_adaption
from trident.data.mask_common import label2color
from trident.misc.ipython_utils import is_in_ipython, is_in_colab
from trident.misc.visualization_utils import *
from trident.data.bbox_common import *

if get_backend() == 'pytorch':
    from trident.backend.pytorch_backend import try_map_args_and_call, Layer
    from trident.backend.pytorch_ops import to_numpy, to_tensor, arange, shuffle, cast, clip, sqrt, int_shape, argmax, softmax, any_abnormal_number, reduce_any, ndim, exp, \
        concate, \
        expand_dims

elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_backend import try_map_args_and_call, Layer
    from trident.backend.tensorflow_ops import to_numpy, to_tensor, arange, shuffle, cast, clip, sqrt, int_shape, concate, zeros_like, ones_like, argmax, softmax, concate, \
        any_abnormal_number, ndim, exp, not_equal, reduce_any, expand_dims

if is_in_ipython() or is_in_colab():
    from IPython import display

ctx = context._context()
_backend = get_backend()

__all__ = ['VisualizationCallbackBase', 'TileImageCallback', 'PrintGradientsCallback', 'SegTileImageCallback',
           'PlotLossMetricsCallback', 'DetectionPlotImageCallback', 'GanTileImageCallback']


class VisualizationCallbackBase(CallbackBase):
    def __init__(self, frequency=-1, unit='batch', save_path: str = None, imshow=False):
        super(VisualizationCallbackBase, self).__init__()
        self.is_in_ipython = is_in_ipython()
        self.is_in_colab = is_in_colab()
        self.frequency = frequency
        if unit in ('batch', 'step', 'epoch'):
            self.unit = unit
        else:
            print(red_color('Only [batch, step, epoch] are valid unit.', True))
        if save_path is None:
            save_path = 'results'
        self.save_path = make_dir_if_need(save_path)
        self.imshow = imshow

    pass


class TileImageCallback(VisualizationCallbackBase):
    def __init__(self, frequency=-1, unit='batch', save_path: str = 'results',
                 name_prefix: str = 'tile_image_{0}.png', row=3, include_input=True, include_output=True, include_target=True,
                 include_mask=None, reverse_image_transform=None, imshow=False):
        super(TileImageCallback, self).__init__(frequency, unit, save_path, imshow)
        self.is_in_ipython = is_in_ipython()
        self.is_in_colab = is_in_colab()
        self.tile_image_name_prefix = name_prefix
        self.reverse_image_transform = reverse_image_transform
        self.row = row

        self.include_input = include_input
        self.include_output = include_output
        self.include_target = include_target
        self.include_mask = include_mask

    def plot_tile_image(self, training_context):
        tile_images_list = []

        input = None
        target = None
        output = None
        mask = None
        legend = []

        data_feed = training_context['data_feed']
        data = training_context['train_data']
        model = training_context['current_model']
        dataprovider = enforce_singleton(ctx.get_data_provider())
        if self.include_input:
            input = to_numpy(data[data_feed['input']])
        if 'tensor' in model.__class__.__name__.lower():
            output = to_numpy(model.clone())
        elif isinstance(model, Layer):
            output = to_numpy(data[data_feed['output']].copy())
        if self.include_target:
            target = to_numpy(data[data_feed['target']].copy())
        if self.include_mask:
            if isinstance(dataprovider.traindata.label, MaskDataset):
                mask = to_numpy(data[dataprovider.traindata.label.symbol])
            elif isinstance(dataprovider.traindata.label, ZipDataset):
                for ds in dataprovider.traindata.label._datasets:
                    if isinstance(ds, MaskDataset):
                        mask = to_numpy(data[ds.symbol])

        reverse_image_transform = dataprovider.reverse_image_transform
        reverse_image_transform_funcs = dataprovider.reverse_image_transform_funcs
        if self.include_input and input is not None:
            if len(reverse_image_transform_funcs) > 1:
                input_arr = []
                for i in range(len(input)):
                    input_arr.append(reverse_image_transform(input[i]))
                tile_images_list.append(input_arr)
            else:
                input_arr = to_numpy(input).transpose([0, 2, 3, 1]) if get_backend() != 'tensorflow' else to_numpy(input)
                max_value = input_arr.max()
                min_value = input_arr.min()
                if max_value <= 1.1 and min_value >= 0:
                    tile_images_list.append(input_arr * 255.0)
                elif max_value <= 1.1 and min_value >= -1.1:
                    tile_images_list.append(input_arr * 128 + 127)
                else:
                    print(max_value, min_value)
                    tile_images_list.append(input_arr)
            legend.append('input')
        if self.include_target and target is not None:
            if len(reverse_image_transform_funcs) > 1:
                target_arr = []
                for i in range(len(target)):
                    target_arr.append(reverse_image_transform(target[i]))
                tile_images_list.append(target_arr)
            else:
                target_arr = to_numpy(target).transpose([0, 2, 3, 1]) if get_backend() != 'tensorflow' else to_numpy(target)

                if target_arr.max() <= 1.1 and target_arr.min() >= 0:
                    tile_images_list.append(target_arr * 255)
                elif target_arr.max() <= 1.1 and target_arr.min() >= -1.1:
                    tile_images_list.append(target_arr * 128 + 127)
                else:
                    tile_images_list.append(target_arr)
            legend.append('target')
        if self.include_output and output is not None:
            if len(reverse_image_transform_funcs) > 1:
                output_arr = []
                for i in range(len(output)):
                    out = reverse_image_transform(output[i])
                    output_arr.append(np.clip(out, 0, 255))
                tile_images_list.append(output_arr)
            else:
                output_arr = to_numpy(output).transpose([0, 2, 3, 1]) if get_backend() != 'tensorflow' else to_numpy(output)

                if output_arr.max() <= 1.2 and output_arr.min() >= 0:
                    tile_images_list.append(np.clip(output_arr * 255, 0, 255))
                elif output_arr.max() <= 1.2 and output_arr.min() >= -1.2:

                    tile_images_list.append(np.clip(output_arr * 128 + 127, 0, 255))
                else:
                    tile_images_list.append(np.clip(output_arr, 0, 255))

            legend.append('output')

        # if self.tile_image_include_mask:
        #     tile_images_list.append(input*127.5+127.5)
        fig = tile_rgb_images(*tile_images_list, row=self.row, save_path=os.path.join(self.save_path, self.tile_image_name_prefix), imshow=True, legend=legend)
        if ctx.enable_tensorboard and ctx.summary_writer is not None:
            ctx.summary_writer.add_figure(training_context['training_name'] + '/plot/tile_image', fig, global_step=training_context['steps'], close=True, walltime=time.time())
        plt.close()

    def on_batch_end(self, training_context):
        if self.frequency > 0 and ((self.unit == 'batch' and (training_context['current_batch'] + 1) % self.frequency == 0) or (
                self.unit == 'step' and (training_context['steps'] + 1) % self.frequency == 0)):
            self.plot_tile_image(training_context)

    def on_epoch_end(self, training_context):
        if self.frequency > 0 and (self.unit == 'epoch' and training_context['current_batch'] == 0 and (training_context['current_epoch'] + 1) % self.frequency == 0):
            self.plot_tile_image(training_context)


class GanTileImageCallback(VisualizationCallbackBase):
    def __init__(self, frequency=-1, unit='batch', save_path: str = 'results',
                 name_prefix: str = 'tile_image_{0}.png', row=3,
                 include_mask=None, reverse_image_transform=None, imshow=False):
        super(GanTileImageCallback, self).__init__(frequency, unit, save_path, imshow)
        self.is_in_ipython = is_in_ipython()
        self.is_in_colab = is_in_colab()
        self.tile_image_name_prefix = name_prefix
        self.reverse_image_transform = reverse_image_transform
        self.row = row
        dataprovider = enforce_singleton(ctx.get_data_provider())
        self.accumulate_sample = False
        self.sample_enough = False
        if dataprovider.minibatch_size < row * row:
            self.accumulate_sample = True
        self.tile_images_list = []
        self.output_arr = []

    def plot_tile_image(self, training_context):

        output = None
        legend = []

        data_feed = training_context['data_feed']
        data = training_context['train_data']
        model = training_context['current_model'].eval()
        dataprovider = enforce_singleton(ctx.get_data_provider())
        output = to_numpy(model(data[data_feed['input']])) if data_feed['input'] in data else to_numpy(data['output'])
        model.train()
        reverse_image_transform = dataprovider.reverse_image_transform

        if reverse_image_transform is not None:
            for i in range(len(output)):
                self.output_arr.append(reverse_image_transform(output[i]))
                if len(self.output_arr) == self.row:
                    self.tile_images_list.append(self.output_arr)
                    if len(self.tile_images_list) == self.row:
                        self.sample_enough = True
                        break
                    self.output_arr = []

        legend.append('output')

        if self.sample_enough:
            fig = tile_rgb_images(*self.tile_images_list, row=self.row, save_path=os.path.join(self.save_path, self.tile_image_name_prefix), imshow=True, legend=None)
            if ctx.enable_tensorboard and ctx.summary_writer is not None:
                ctx.summary_writer.add_figure(training_context['training_name'] + '/plot/tile_image', fig, global_step=training_context['steps'], close=True, walltime=time.time())
            plt.close()

    def on_batch_end(self, training_context):
        if self.frequency > 0 and ((self.unit == 'batch' and (training_context['current_batch'] + 1) % self.frequency == 0) or (
                self.unit == 'step' and (training_context['steps'] + 1) % self.frequency == 0) or not self.sample_enough):
            if self.sample_enough:
                self.tile_images_list = []
                self.output_arr = []
                self.sample_enough = False
            self.plot_tile_image(training_context)

    def on_epoch_end(self, training_context):
        if self.frequency > 0 and (
                (self.unit == 'epoch' and training_context['current_batch'] == 0 and (training_context['current_epoch'] + 1) % self.frequency == 0) or not self.sample_enough):
            if self.sample_enough:
                self.tile_images_list = []
                self.output_arr = []
                self.sample_enough = False
            self.plot_tile_image(training_context)


class SegTileImageCallback(VisualizationCallbackBase):
    def __init__(self, frequency=-1, unit='batch', save_path: str = 'results', reverse_image_transform=None,
                 is_label_mask=False, palette=None, background=(120, 120, 120), name_prefix: str = 'segtile_image_{0}.png', imshow=False):
        super(SegTileImageCallback, self).__init__(frequency, unit, save_path, imshow)
        self.is_in_ipython = is_in_ipython()
        self.is_in_colab = is_in_colab()
        self.is_label_mask = is_label_mask
        self.palette = palette
        self.tile_image_name_prefix = name_prefix
        self.reverse_image_transform = reverse_image_transform

        self.background = to_numpy(background)

    def plot_tile_image(self, training_context):
        axis = 1
        if get_backend() == 'tensorflow':
            axis = -1
        new_shape = [1] * 4
        new_shape[axis] = 3
        background = np.reshape(to_numpy(self.background), new_shape)
        tile_images_list = []
        input = None
        target = None
        output = None
        is_label_mask = self.is_label_mask
        data_feed = training_context['data_feed']
        data = training_context['train_data']
        model = training_context['current_model']

        # if len(data) >= 3:
        for data_key in data.key_list:
            if data_key == data_feed[model.signature.inputs.key_list[0]]:
                input = data[data_feed[model.signature.inputs.key_list[0]]]
                model.eval()
                if is_label_mask:
                    output = to_numpy(argmax(model(input), axis=axis))
                else:
                    output = to_numpy(expand_dims(cast(argmax(model(input), axis=axis), input.dtype), axis=axis))

                model.train()

            # elif data_key == data_feed[model.signature.outputs.key_list[0]]:
            #     output = data[data_feed[model.signature.outputs.key_list[0]]]
            #     if output.max() < 0:
            #         output = exp(output)

            elif (
                    'target' in data_key or 'label' in data_key or 'mask' in data_key) and not 'output' in data_key and data_key in data_feed.value_list:
                target = to_numpy(data[data_key])
        output_arr = None
        if 'alpha' not in data:
            output_arr = output.copy()
            if is_label_mask:
                target = label2color(target, self.palette)
                output = label2color(output, self.palette)
        else:
            if get_backend() == 'tensorflow':
                output = output[:, :, :, 1:2] * argmax(output, axis)
            else:
                output = (output[:, 1:2, :, :] * argmax(output, axis)).transpose(0, 2, 3, 1)
            target = to_numpy(data['alpha'])

        input_arr = []
        input = to_numpy(input)
        for i in range(len(input)):
            input_arr.append(image_backend_adaption(self.reverse_image_transform(input[i])))
        input_arr = np.stack(input_arr, axis=0)
        # input_arr=np.asarray(input_arr)
        tile_images_list.append(input_arr)

        if is_label_mask:
            tile_images_list.append(target)
            tile_images_list.append(output)
        else:
            target_arr = target

            if len(target.shape) < len(int_shape(input)):
                if get_backend() == 'tensorflow':
                    target_arr = np.expand_dims(target, -1)
                else:
                    target_arr = np.expand_dims(target, axis=axis)

            if 'alpha' not in data:
                target_arr[target_arr > 0] = 1

            tile_images_list.append(target_arr * input_arr + (1 - target_arr) * background)

            tile_images_list.append(output_arr * input_arr + (1 - output_arr) * background)

        # if self.tile_image_include_mask:
        #     tile_images_list.append(input*127.5+127.5)
        fig = tile_rgb_images(*tile_images_list, save_path=os.path.join(self.save_path, self.tile_image_name_prefix), imshow=True)
        if ctx.enable_tensorboard and ctx.summary_writer is not None:
            ctx.summary_writer.add_figure(training_context['training_name'] + '/plot/segtile_image', fig, global_step=training_context['steps'], close=True,
                                          walltime=time.time())
        plt.close()

    def on_batch_end(self, training_context):
        if self.frequency > 0 and ((self.unit == 'batch' and (training_context['current_batch'] + 1) % self.frequency == 0) or (
                self.unit == 'step' and (training_context['steps'] + 1) % self.frequency == 0)):
            self.plot_tile_image(training_context)

    def on_epoch_end(self, training_context):
        if self.frequency > 0 and (self.unit == 'epoch' and training_context['current_batch'] == 0 and (training_context['current_epoch'] + 1) % self.frequency == 0):
            self.plot_tile_image(training_context)


class DetectionPlotImageCallback(VisualizationCallbackBase):
    def __init__(self, frequency=-1, unit='batch', save_path: str = 'results', reverse_image_transform=None, labels=None,
                 palette=None, background=(120, 120, 120), name_prefix: str = 'detection_plot_image_{0}.png', imshow=False):
        super(DetectionPlotImageCallback, self).__init__(frequency, unit, save_path, imshow)
        self.is_in_ipython = is_in_ipython()
        self.is_in_colab = is_in_colab()
        self.labels = labels
        self.palette = palette
        self.tile_image_name_prefix = name_prefix
        self.reverse_image_transform = reverse_image_transform
        self.background = np.expand_dims(np.expand_dims(to_numpy(background), 0), 0)

    def plot_detection_image(self, training_context):
        tile_images_list = []
        input = None
        target = None
        output = None

        data_feed = training_context['data_feed']
        data = training_context['train_data']
        model = training_context['current_model']
        output = try_map_args_and_call(model, data, data_feed)
        target = data['bbox']
        input = data[data_feed[model.signature.inputs.key_list[0]]]
        input_image = self.reverse_image_transform(to_numpy(input))
        targetmask = (target[:, 4] > 0.9)
        input_image1 = input_image.copy()
        target_boxes = to_numpy(xywh2xyxy(target[targetmask, :]))
        for box in target_boxes:
            plot_one_box(box, input_image1, (255, 128, 128), self.labels[box[5:]])

        # input_arr=np.asarray(input_arr)
        tile_images_list.append(input_image1)

        input_image = self.reverse_image_transform(to_numpy(input))
        mask = (output[:, :, 4] > 0.7)
        if len(output[:, :, 4]) > 0:
            mask2 = (argmax(softmax(output[:, :, 5:], -1), -1) != 0)
            mask = (mask.float() + mask2.float() == 2)
        output = output[mask, :]
        input_image2 = input_image.copy()
        output_boxes = to_numpy(xywh2xyxy(output[mask, :]))
        for box in output_boxes:
            plot_one_box(box, input_image2, (255, 255, 128), self.labels[np.argmax(box[5:])])

        tile_images_list.append(input_image2)
        fig = tile_rgb_images(*tile_images_list, save_path=os.path.join(self.save_path, self.tile_image_name_prefix), imshow=True)
        if ctx.enable_tensorboard and ctx.summary_writer is not None:
            ctx.summary_writer.add_figure(training_context['training_name'] + '/plot/detection_plot', fig, global_step=training_context['steps'], close=True,
                                          walltime=time.time())
        plt.close()

    def on_batch_end(self, training_context):
        if self.frequency > 0 and ((self.unit == 'batch' and (training_context['current_batch'] + 1) % self.frequency == 0) or (
                self.unit == 'step' and (training_context['steps'] + 1) % self.frequency == 0)):
            self.plot_detection_image(training_context)

    def on_epoch_end(self, training_context):
        if self.frequency > 0 and (self.unit == 'epoch' and training_context['current_batch'] == 0 and (training_context['current_epoch'] + 1) % self.frequency == 0):
            self.plot_detection_image(training_context)


class PlotLossMetricsCallback(VisualizationCallbackBase):
    def __init__(self, frequency=-1, unit='batch', save_path: str = 'results', clean_ipython_output_frequency=5,
                 name_prefix: str = 'loss_metric_curve_{0}.png', is_inplace=False, imshow=False):
        super(PlotLossMetricsCallback, self).__init__(frequency, unit, save_path, imshow)
        self.training_items = None
        self.name_prefix = name_prefix
        self.is_inplace = is_inplace

        self.is_shared = True
        self.loss_history_list = []
        self.metric_history_list = []
        self.counter = 0
        self.clean_ipython_output_frequency = clean_ipython_output_frequency

    def on_training_start(self, training_context):
        if not self.is_inplace:
            self.training_items = training_context['training_items']

    def on_overall_batch_end(self, training_context):
        if not self.is_inplace and training_context['steps'] > 10:
            if self.frequency > 0 and ((self.unit == 'batch' and (training_context['steps'] + 1) % self.frequency == 0) or (
                    self.unit == 'step' and (training_context['steps'] + 1) % self.frequency == 0)):
                if is_in_ipython() and self.counter == self.clean_ipython_output_frequency:
                    display.clear_output(wait=True)
                    self.counter = 0
                self.loss_history_list = []
                self.metric_history_list = []
                plotable_metric_names = OrderedDict()
                for i in range(len(self.training_items.value_list)):
                    trainitem = self.training_items.value_list[i]
                    self.loss_history_list.append(trainitem.batch_loss_history)
                    plotable_metric_names[i] = [k for k, v in trainitem._metrics.item_list if v.print_only == False]
                    self.metric_history_list.append(trainitem.batch_metric_history)
                self.counter += 1
                fig = loss_metric_curve(self.loss_history_list, self.metric_history_list, metrics_names=plotable_metric_names,
                                        legend=training_context['training_names'].value_list, calculate_base='batch',
                                        max_iteration=None, save_path=os.path.join(self.save_path, self.name_prefix),
                                        imshow=self.imshow)

                if ctx.enable_tensorboard and ctx.summary_writer is not None:
                    ctx.summary_writer.add_figure('overall/plot/loss_metric_curve', fig, global_step=training_context['steps'], close=True, walltime=time.time())
                plt.close()
                # if self.tile_image_unit == 'epoch' and (epoch + 1) % self.tile_image_frequency == 0:  #     epoch_loss_history = [trainitem.epoch_loss_history for k,
                # trainitem in self.training_items.items()]  #     epoch_metric_history = [trainitem.epoch_metric_history for k, trainitem in self.training_items.items()]  #  #
                # loss_metric_curve(epoch_loss_history, epoch_metric_history, legend=self.training_names.value_list,  #                       calculate_base='epoch',
                # max_iteration=self.num_epochs,  #                       save_path=os.path.join(self.tile_image_save_path, 'loss_metric_curve.png'),  #
                # imshow=True)

    # def on_batch_end(self, training_context):
    #     if self.is_inplace:
    #         if (self.batch_inteval > 0 and (self.training_items.value_list[0].training_context['current_batch'] + 1) % self.batch_inteval == 0) or (self.epoch_inteval > 0 and
    #         self.training_items.value_list[0].training_context['current_batch'] +1==self.training_items.value_list[0].training_context['total_batch']  and (
    #         self.training_items.value_list[0].training_context['current_epoch'] + 1) % self.epoch_inteval == 0):
    #             if is_in_ipython() and self.counter == self.clean_ipython_output_frequency:
    #                 display.clear_output(wait=True)
    #                 self.counter = 0
    #             self.loss_history_list = []
    #             self.metric_history_list = []
    #             self.loss_history_list.append(self.batch_loss_history)
    #             self.metric_history_list.append(self.batch_metric_history)
    #             self.counter += 1
    #             loss_metric_curve(self.loss_history_list, self.metric_history_list,
    #                               legend=training_context['training_names'].value_list, calculate_base='batch',
    #                               max_iteration=None, save_path=os.path.join(self.save_path, self.name_prefix),
    #                               imshow=self.imshow)
    #
    #
    #             # if self.tile_image_unit == 'epoch' and (epoch + 1) % self.tile_image_frequency == 0:  #     epoch_loss_history = [trainitem.epoch_loss_history for k,
    #             trainitem in self.training_items.items()]  #     epoch_metric_history = [trainitem.epoch_metric_history for k, trainitem in self.training_items.items()]  #  #
    #             loss_metric_curve(epoch_loss_history, epoch_metric_history, legend=self.training_names.value_list,  #                       calculate_base='epoch',
    #             max_iteration=self.num_epochs,  #                       save_path=os.path.join(self.tile_image_save_path, 'loss_metric_curve.png'),  #
    #             imshow=True)


class PrintGradientsCallback(VisualizationCallbackBase):
    def __init__(self, frequency=-1, unit='batch'):
        super(PrintGradientsCallback, self).__init__(frequency, unit)

        self.first_layer = OrderedDict()
        self.last_layer = OrderedDict()
        self.is_modulefict = False
        self.lines = []

    def on_optimization_step_start(self, training_context):
        if get_backend() == 'pytorch':
            with torch.no_grad():
                if self.frequency > 0 and ((self.unit == 'batch' and (training_context['current_batch'] + 1) % self.frequency == 0) or (
                        self.unit == 'step' and (training_context['steps'] + 1) % self.frequency == 0) or (
                                                   self.unit == 'epoch' and (training_context['current_epoch'] + 1) % self.frequency == 0)):
                    grad_dict = OrderedDict()
                    if 'grads_state' not in training_context:
                        training_context['grads_state'] = OrderedDict()
                    if training_context['current_batch'] == 0 and training_context['current_epoch'] > 0:
                        # relocate the first/ last layers
                        self.first_layer = OrderedDict()
                        self.last_layer = OrderedDict()

                    if len(self.first_layer) == 0 and len(self.last_layer) == 0:
                        if training_context['current_model'][-1].__class__.__name__ == 'ModuleDict':
                            self.is_modulefict = True
                            for k, v in training_context['current_model'][-1].items():
                                last_layer_name = ''
                                for name, module in v.named_modules():
                                    if len([pk for pk, pv in module._parameters.items() if 'bias' not in pk and pv.requires_grad]) > 0:
                                        last_layer_name = module.relative_name
                                if last_layer_name != '':
                                    self.last_layer[last_layer_name] = k

                        first_layer_name = ''
                        last_layer_name = ''
                        for k, v in training_context['current_model'].named_modules():
                            if len([pk for pk, pv in v._parameters.items() if 'bias' not in pk and pv is not None and pv.requires_grad]) > 0:
                                if first_layer_name == '':
                                    first_layer_name = v.relative_name
                                    self.first_layer[first_layer_name] = 'first_layer'

                                if not self.is_modulefict:
                                    last_layer_name = v.relative_name
                        if last_layer_name != '' and not self.is_modulefict:
                            self.last_layer[last_layer_name] = 'last_layer'

                    for name, module in training_context['current_model'].named_modules():
                        if module.relative_name in self.first_layer or module.relative_name in self.last_layer:
                            grads_data = [np.abs(np.reshape(to_numpy(pv.grad.data), -1)) for pk, pv in module._parameters.items() if
                                          'bias' not in pk and pv is not None and pv.requires_grad and pv.grad is not None]
                            weights_data = [np.abs(np.reshape(to_numpy(pv.data), -1)) for pk, pv in module._parameters.items() if
                                            'bias' not in pk and pv is not None and pv.requires_grad]
                            if ctx.enable_tensorboard and ctx.summary_writer is not None:
                                ctx.summary_writer.add_histogram(
                                    training_context['training_name'] + '/gradients/' + self.first_layer[module.relative_name] if module.relative_name in self.first_layer else
                                    self.last_layer[module.relative_name], np.concatenate(grads_data, axis=0), training_context['steps'])
                                ctx.summary_writer.add_histogram(
                                    training_context['training_name'] + '/weights/' + self.first_layer[module.relative_name] if module.relative_name in self.first_layer else
                                    self.last_layer[module.relative_name], np.concatenate(weights_data, axis=0), training_context['steps'])
                            if len(grads_data) > 0:
                                grads_data = np.concatenate(grads_data, axis=0).mean()
                            else:
                                grads_data = None
                            if module.relative_name in self.first_layer:
                                training_context['grads_state']['first_layer'] = grads_data
                            elif module.relative_name in self.last_layer:
                                training_context['grads_state'][self.last_layer[module.relative_name]] = grads_data

                    if len(training_context['grads_state']) > 0:
                        grads_str = yellow_color('{0:<16s}'.format(training_context['current_model'].name) + '|'.join(
                            ['{0} gradients: {1:<8.3e} '.format(k, v) for k, v in training_context['grads_state'].items() if isinstance(v, numbers.Number)]))
                        self.lines.append(grads_str + '\n')


        elif get_backend() == 'tensorflow':
            if self.frequency > 0 and ((self.unit == 'batch' and (training_context['current_batch'] + 1) % self.frequency == 0) or (
                    self.unit == 'step' and (training_context['steps'] + 1) % self.frequency == 0) or (
                                               self.unit == 'epoch' and training_context['current_batch'] == 0 and (training_context['current_epoch'] + 1) % self.frequency == 0)):
                grad_dict = OrderedDict()
                if 'grads_state' not in training_context:
                    training_context['grads_state'] = OrderedDict()
                if training_context['current_batch'] == 0 and training_context['current_epoch'] > 0:
                    # relocate the first/ last layers
                    self.first_layer = OrderedDict()
                    self.last_layer = OrderedDict()

                if len(self.first_layer) == 0 and len(self.last_layer) == 0:
                    if training_context['current_model'][-1].__class__.__name__ == 'ModuleDict':
                        self.is_modulefict = True
                        for k, v in training_context['current_model'][-1].items():
                            last_layer_name = ''
                            for name, module in v.named_modules():
                                if len([pk for pk, pv in module._parameters.items() if 'bias' not in pk and pv is not None and pv.trainable]) > 0:
                                    last_layer_name = module.relative_name
                            if last_layer_name != '':
                                self.last_layer[last_layer_name] = k
                    first_layer_name = ''
                    last_layer_name = ''
                    for k, v in training_context['current_model'].named_modules():
                        if len([pk for pk, pv in v._parameters.items() if 'bias' not in pk and pv is not None and pv.trainable]) > 0:
                            if first_layer_name == '':
                                first_layer_name = v.relative_name
                                self.first_layer[first_layer_name] = 'first_layer'

                            if not self.is_modulefict:
                                last_layer_name = v.relative_name
                    if last_layer_name != '' and not self.is_modulefict:
                        self.last_layer[last_layer_name] = 'last_layer'

                grads_and_vars = list(training_context['grads_and_vars'])
                grads_dict = OrderedDict()
                for grad, var in grads_and_vars:
                    grads_dict[var.ref()] = to_numpy(grad)

                for name, module in training_context['current_model'].named_modules():
                    if module.relative_name in self.first_layer or module.relative_name in self.last_layer:
                        grads_data = [np.abs(np.reshape(grads_dict[pv.ref()], -1)) for pk, pv in module._parameters.items() if
                                      'bias' not in pk and pv.trainable and grads_dict[pv.ref()] is not None]
                        weights_data = [np.abs(np.reshape(to_numpy(pv.value()), -1)) for pk, pv in module._parameters.items() if 'bias' not in pk and pv.trainable]
                        if ctx.enable_tensorboard and ctx.summary_writer is not None:
                            ctx.summary_writer.add_histogram(
                                training_context['training_name'] + '/gradients/' + self.first_layer[module.relative_name] if module.relative_name in self.first_layer else
                                self.last_layer[module.relative_name], np.concatenate(grads_data, axis=0), training_context['steps'])
                            ctx.summary_writer.add_histogram(
                                training_context['training_name'] + '/weights/' + self.first_layer[module.relative_name] if module.relative_name in self.first_layer else
                                self.last_layer[module.relative_name], np.concatenate(weights_data, axis=0), training_context['steps'])
                        if len(grads_data) > 0:
                            grads_data = np.concatenate(grads_data, axis=0).mean()
                        else:
                            grads_data = None
                        if module.relative_name in self.first_layer:
                            training_context['grads_state']['first_layer'] = grads_data
                        elif module.relative_name in self.last_layer:
                            training_context['grads_state'][self.last_layer[module.relative_name]] = grads_data

                if len(training_context['grads_state']) > 0:
                    grad_str = yellow_color('{0:<16s}'.format(training_context['current_model'].name) + '|'.join(
                        ['{0} gradients: {1:<8.3e} '.format(k, v) for k, v in training_context['grads_state'].items() if isinstance(v, numbers.Number)]))
                    self.lines.append(grad_str + '\n')

    def on_overall_batch_end(self, training_context):
        if len(self.lines) > 0:
            for line in self.lines:
                print(line)
            self.lines = []

