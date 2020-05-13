from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
import os
import sys
import uuid
from collections import *
from collections import deque
from copy import copy, deepcopy
from functools import partial
from itertools import product as product
from itertools import repeat
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch.nn import init
from torch.nn.parameter import Parameter

from trident.backend.common import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential
from trident.backend.pytorch_ops import *
from trident.data.bbox_common import xywh2xyxy, xyxy2xywh
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity, Relu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *
from trident.models.pytorch_ssd import *

__all__ = ['Mobile_RFBnet',  'generate_priors']

_session = get_session()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_epsilon = _session.epsilon
_trident_dir = _session.trident_dir

dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


image_mean_test = image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2

min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
shrinkage_list = []
image_size = [640, 480]  # default input size 320*240
feature_map_w_h_list = [[40, 20, 10, 5], [30, 15, 8, 4]]  # default feature map size
priors = []




class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        #self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

        for ii in range(4):
            if(self.steps[ii] != pow(2,(ii+3))):
                print("steps must be [8,16,32,64]")
                sys.exit()

        self.feature_map_2th = [int(int((self.image_size[0] + 1) / 2) / 2),
                                int(int((self.image_size[1] + 1) / 2) / 2)]
        self.feature_map_3th = [int(self.feature_map_2th[0] / 2),
                                int(self.feature_map_2th[1] / 2)]
        self.feature_map_4th = [int(self.feature_map_3th[0] / 2),
                                int(self.feature_map_3th[1] / 2)]
        self.feature_map_5th = [int(self.feature_map_4th[0] / 2),
                                int(self.feature_map_4th[1] / 2)]
        self.feature_map_6th = [int(self.feature_map_5th[0] / 2),
                                int(self.feature_map_5th[1] / 2)]

        self.feature_maps = [self.feature_map_3th, self.feature_map_4th,
                             self.feature_map_5th, self.feature_map_6th]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]

                    cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                    cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True) -> torch.Tensor:
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
    print("priors nums:{}".format(len(priors)))
    priors = to_tensor(priors)#.view(-1, 4)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    '''Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    '''
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
                      torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]], dim=locations.dim() - 1)


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2, locations[..., :2] + locations[..., 2:] / 2],
                     locations.dim() - 1)


def basic_rfb(num_filters, scale=0.1):
    return ShortCut2d(Sequential(ShortCut2d(Sequential(
        Conv2d_Block((1, 1), depth_multiplier=0.125, strides=1, groups=1, auto_pad=True, use_bias=False, activation=None,
                     normalization='batch'),
        Conv2d_Block((3, 3), depth_multiplier=2, strides=1, groups=1, auto_pad=True, use_bias=False, activation='relu',
                     normalization='batch'),
        Conv2d_Block((3, 3), depth_multiplier=1, strides=1, groups=1, auto_pad=True, use_bias=False, dilation=2,
                     activation=None, normalization='batch'), name='branch1'),

        Sequential(
            Conv2d_Block((1, 1), depth_multiplier=0.125, strides=1, groups=1, auto_pad=True, use_bias=False, activation=None,
                         normalization='batch'),
            Conv2d_Block((3, 3), depth_multiplier=2, strides=1, groups=1, auto_pad=True, use_bias=False, activation='relu',
                         normalization='batch'),
            Conv2d_Block((3, 3), depth_multiplier=1, strides=1, groups=1, auto_pad=True, use_bias=False, dilation=3,
                         activation=None, normalization='batch'), name='branch1'), Sequential(
            Conv2d_Block((1, 1), depth_multiplier=0.125, strides=1, groups=1, auto_pad=True, use_bias=False, activation=None,
                         normalization='batch'),
            Conv2d_Block((3, 3), depth_multiplier=1.5, strides=1, groups=1, auto_pad=True, use_bias=False, activation='relu',
                         normalization='batch'),
            Conv2d_Block((3, 3), depth_multiplier=1.34, strides=1, groups=1, auto_pad=True, use_bias=False,
                         activation='relu', normalization='batch'),
            Conv2d_Block((3, 3), depth_multiplier=1, strides=1, groups=1, auto_pad=True, use_bias=False, dilation=5,
                         activation=None, normalization='batch'), name='branch2'), mode='concate'),
        Conv2d_Block((1, 1), num_filters=num_filters, strides=1, groups=1, auto_pad=True, use_bias=False,
                     activation=None, normalization='batch')),
        Conv2d_Block((1, 1), num_filters=num_filters, strides=1, groups=1, auto_pad=True, use_bias=False,
                     activation=None, normalization='batch'), mode='add', activation='relu')


def conv_dw(num_filters, strides):
    return Sequential(
        DepthwiseConv2d_Block((3, 3), depth_multiplier=1, strides=strides, use_bias=False, activation='relu',
                              normalization='batch'),
        Conv2d_Block((1, 1), num_filters=num_filters, strides=1, groups=1, auto_pad=True, use_bias=False,
                     activation='relu', normalization='batch'),

    )


def tiny_mobile_rfbnet(filter_base=16, num_classes=4):
    return Sequential(Conv2d_Block((3, 3), num_filters=filter_base, strides=2, groups=1, auto_pad=True, use_bias=False,
                                   activation='relu', normalization='batch'),
                      conv_dw(filter_base * 2, 1),
                      conv_dw(filter_base * 2, 2),  # 80*60
                      conv_dw(filter_base * 2, 1),
                      conv_dw(filter_base * 4, 2),  # 40*30
                      conv_dw(filter_base * 4, 1),
                      conv_dw(filter_base * 4, 1),
                      basic_rfb(filter_base * 4, scale=1.0),

                      conv_dw(filter_base * 8, 2),  # 20*15
                      conv_dw(filter_base * 8, 1),
                      conv_dw(filter_base * 8, 1),

                      conv_dw(filter_base * 16, 2),  # 10*8
                      conv_dw(filter_base * 16, 1))


class RFBnet(Layer):
    def __init__(self, *args, base_filters=16, num_classes=4, num_regressors=14,iou_threshold=0.3, center_variance=0.1, size_variance=0.2,
                 name='tiny_mobile_rfbnet', **kwargs):
        '''

        Parameters
        ----------
        layer_defs : object
        '''
        super(RFBnet, self).__init__(name=name)
        self.base_filters = base_filters
        backbond = tiny_mobile_rfbnet(self.base_filters)
        self.backbond1 = backbond[:8]
        self.backbond2 = backbond[8:11]
        self.backbond3 = backbond[11:13]
        self.iou_threshold = iou_threshold
        self.center_variance = center_variance
        self.size_variance = size_variance

        self.num_classes = num_classes
        self.num_regressors=num_regressors
        self.priors = []
        self.define_img_size(640)

        self.extra = Sequential(Conv2d((1, 1), num_filters=64, strides=1, activation='relu', use_bias=True),
                                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=2, auto_pad=True, activation='relu',
                                                use_bias=True),
                                Conv2d((1, 1), num_filters=256, strides=1, activation=None, use_bias=True), Relu())
        self.regression_headers = nn.ModuleList([Sequential(
            DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
            Conv2d((1, 1), num_filters=3 * self.num_regressors, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
                Conv2d((1, 1), num_filters=2 *self.num_regressors, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_regressors, strides=1, activation=None, use_bias=True)),

            Conv2d((3, 3), num_filters=3*self.num_regressors, strides=1, auto_pad=True, activation=None), ])
        self.classification_headers = nn.ModuleList([Sequential(
            DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
            Conv2d((1, 1), num_filters=3 * self.num_classes, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_classes, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_classes, strides=1, activation=None, use_bias=True)),

            Conv2d((3, 3), num_filters=3 * self.num_classes, strides=1, auto_pad=True, activation=None,
                   use_bias=True), ])

    def define_img_size(self, size=640):
        global image_size, feature_map_w_h_list, priors
        img_size_dict = {128: [128, 96], 160: [160, 120], 320: [320, 240], 480: [480, 360], 640: [640, 480],
                         1280: [1280, 960]}
        image_size = img_size_dict[size]

        feature_map_w_h_list_dict = {128: [[16, 8, 4, 2], [12, 6, 3, 2]], 160: [[20, 10, 5, 3], [15, 8, 4, 2]],
                                     320: [[40, 20, 10, 5], [30, 15, 8, 4]], 480: [[60, 30, 15, 8], [45, 23, 12, 6]],
                                     640: [[80, 40, 20, 10], [60, 30, 15, 8]],
                                     1280: [[160, 80, 40, 20], [120, 60, 30, 15]]}
        feature_map_w_h_list = feature_map_w_h_list_dict[size]

        for i in range(0, len(image_size)):
            item_list = []
            for k in range(0, len(feature_map_w_h_list[i])):
                item_list.append(image_size[i] / feature_map_w_h_list[i][k])
            shrinkage_list.append(item_list)
        self.priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def forward(self, *x):
        x = enforce_singleton(x)
        confidences = []
        locations = []

        x = self.backbond1(x)
        confidence, location = self.compute_header(0, x)
        confidences.append(confidence)
        locations.append(location)

        x = self.backbond2(x)
        confidence, location = self.compute_header(1, x)
        confidences.append(confidence)
        locations.append(location)

        x = self.backbond3(x)
        confidence, location = self.compute_header(2, x)
        confidences.append(confidence)
        locations.append(location)

        x = self.extra(x)
        confidence, location = self.compute_header(3, x)
        confidences.append(confidence)
        locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.training:
            return confidences, locations
        else:
            confidences_class = F.softmax(confidences[:,:,2],dim=-1)
            confidences_attr=confidences[:,:,2].sigmoid()
            boxes = convert_locations_to_boxes(locations, self.priors, self.center_variance, self.size_variance)
            boxes = xywh2xyxy(boxes)
            return confidences_class, boxes,confidences_attr






def Mobile_RFBnet(base_filters=16, num_classes=4,num_regressors=4):
    model = SsdDetectionModel(input_shape=(3, 480, 640),
                              output=RFBnet(base_filters=base_filters, num_classes=num_classes,num_regressors=num_regressors))
    model.signature = get_signature(model.model.forward)
    model.preprocess_flow = [resize((480, 640), keep_aspect=True, align_corner=True), normalize(127.5, 127.5)]
    return model







