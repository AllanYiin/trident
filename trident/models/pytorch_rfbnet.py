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
from collections import abc
from torch.nn import init
from torch.nn.parameter import Parameter

from trident.backend.common import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential, ModuleList, fix_layer, load, \
    get_device
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
from trident.data.vision_transforms import Resize, Normalize
import torchvision

__all__ = ['RfbNet', 'generate_priors']

_session = get_session()
_device = get_device()
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
        # self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        # self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

        for ii in range(4):
            if (self.steps[ii] != pow(2, (ii + 3))):
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
    priors = to_tensor(priors).to(get_device())  # .view(-1, 4)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

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
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
                      torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]], dim=locations.dim() - 1)


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2, locations[..., :2] + locations[..., 2:] / 2],
                     locations.dim() - 1)


def basic_rfb(num_filters, scale=0.1):
    return ShortCut2d(Sequential(
        ShortCut2d(
            Sequential(
                Conv2d_Block((1, 1), depth_multiplier=0.125, strides=1, groups=1, auto_pad=False, use_bias=False,
                             activation=None,
                             normalization='batch'),
                Conv2d_Block((3, 3), depth_multiplier=2, strides=1, groups=1, padding=(1, 1), use_bias=False,
                             activation=Relu(inplace=True),
                             normalization='batch'),
                Conv2d_Block((3, 3), depth_multiplier=1, strides=1, groups=1, padding=(2, 2), use_bias=False,
                             dilation=2,
                             activation=None, normalization='batch'), name='branch1'),

            Sequential(
                Conv2d_Block((1, 1), depth_multiplier=0.125, strides=1, groups=1, auto_pad=False, use_bias=False,
                             activation=None,
                             normalization='batch'),
                Conv2d_Block((3, 3), depth_multiplier=2, strides=1, groups=1, padding=(1, 1), use_bias=False,
                             activation=Relu(inplace=True),
                             normalization='batch'),
                Conv2d_Block((3, 3), depth_multiplier=1, strides=1, groups=1, padding=(3, 3), use_bias=False,
                             dilation=3,
                             activation=None, normalization='batch'), name='branch2'),
            Sequential(
                Conv2d_Block((1, 1), depth_multiplier=0.125, strides=1, groups=1, auto_pad=False, use_bias=False,
                             activation=None,
                             normalization='batch'),
                Conv2d_Block((3, 3), depth_multiplier=1.5, strides=1, groups=1, padding=(1, 1), use_bias=False,
                             activation=Relu(inplace=True),
                             normalization='batch'),
                Conv2d_Block((3, 3), depth_multiplier=1.33, strides=1, groups=1, padding=(1, 1), use_bias=False,
                             activation=Relu(inplace=True), normalization='batch'),
                Conv2d_Block((3, 3), depth_multiplier=1, strides=1, groups=1, padding=(5, 5), use_bias=False,
                             dilation=5,
                             activation=None, normalization='batch'), name='branch3')
            , mode='concate'),

        Conv2d_Block((1, 1), num_filters=num_filters, strides=1, groups=1, auto_pad=True, use_bias=False,
                     activation=None, normalization='batch')),
        Conv2d_Block((1, 1), num_filters=num_filters, strides=1, groups=1, auto_pad=True, use_bias=False,
                     activation=None, normalization='batch'), mode='add', activation='relu')


def conv_dw(num_filters, strides):
    return Sequential(
        DepthwiseConv2d_Block((3, 3), depth_multiplier=1, strides=strides, use_bias=False,
                              activation=Relu(inplace=True),
                              normalization='batch'),
        Conv2d_Block((1, 1), num_filters=num_filters, strides=1, groups=1, auto_pad=True, use_bias=False,
                     activation=Relu(inplace=True), normalization='batch'),
    )


def tiny_mobile_rfbnet(filter_base=16, num_classes=2):
    return Sequential(Conv2d_Block((3, 3), num_filters=filter_base, strides=2, groups=1, auto_pad=True, use_bias=False,
                                   activation=Relu(inplace=True), normalization='batch'),
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
    def __init__(self, *args, base_filters=16, num_classes=2, num_regressors=4, detection_threshold=0.4,
                 nms_threshold=0.3, center_variance=0.1, size_variance=0.2,
                 name='tiny_mobile_rfbnet', **kwargs):
        """

        Parameters
        ----------
        layer_defs : object
        """
        super(RFBnet, self).__init__(name=name)
        self.base_filters = base_filters
        backbond = tiny_mobile_rfbnet(self.base_filters)
        self.backbond1 = Sequential(*backbond[:8], name='backbond1')
        self.backbond2 = Sequential(*backbond[8:11], name='backbond2')
        self.backbond3 = Sequential(*backbond[11:13], name='backbond3')
        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.variance = (center_variance, size_variance)
        self.softmax = SoftMax(-1)

        self.num_classes = num_classes
        self.num_regressors = num_regressors
        self.register_buffer("priors", None)
        self.define_img_size(640)

        self.extra = Sequential(Conv2d((1, 1), num_filters=64, strides=1, activation=Relu(inplace=True), use_bias=True),
                                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=2, auto_pad=True,
                                                activation=Relu(inplace=True),
                                                use_bias=True),
                                Conv2d((1, 1), num_filters=256, strides=1, activation=None, use_bias=True),
                                Relu(inplace=True),
                                name='extra')
        self.regression_headers = ModuleList([
            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation=Relu(inplace=True),
                                use_bias=True),
                Conv2d((1, 1), num_filters=3 * self.num_regressors, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation=Relu(inplace=True),
                                use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_regressors, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation=Relu(inplace=True),
                                use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_regressors, strides=1, activation=None, use_bias=True)),

            Conv2d((3, 3), num_filters=3 * self.num_regressors, strides=1, auto_pad=True, activation=None)],
            name='regression_headers')
        self.classification_headers = ModuleList([
            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation=Relu(inplace=True),
                                use_bias=True),
                Conv2d((1, 1), num_filters=3 * self.num_classes, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation=Relu(inplace=True),
                                use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_classes, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation=Relu(inplace=True),
                                use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_classes, strides=1, auto_pad=True, activation=None,
                       use_bias=True)),

            Conv2d((3, 3), num_filters=3 * self.num_classes, strides=1, auto_pad=True, activation=None, use_bias=True)],
            name='classification_headers')

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
        self.register_buffer("priors", generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes))

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, self.num_regressors)
        return confidence, location

    def area_of(self, left_top, right_bottom) -> torch.Tensor:
        """Compute the areas of rectangles given two corners.

        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.

        Returns:
            area (N): return the area.
        """
        hw = clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        """Return intersection-over-union (Jaccard index) of boxes.
        Args:
            boxes0 (N, 4): ground truth boxes.
            boxes1 (N or 1, 4): predicted boxes.
            eps: a small number to avoid 0 as denominator.
        Returns:
            iou (N): IoU values.
        """
        overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = torch.min(boxes0[..., 2:4], boxes1[..., 2:4])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:4])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:4])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def hard_nms(self, box_scores, nms_threshold=0.3, top_k=-1, candidate_size=200):
        """
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        _, indexes = scores.sort(descending=True)
        indexes = indexes[:candidate_size]
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                current_box.unsqueeze(0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]

    # def predict(self, confidences, decode_boxes):
    #     all_picked_box_probs = []
    #
    #     for idx in range(len(confidences)):
    #         boxes = decode_boxes[idx]
    #         scores = confidences[idx]
    #
    #
    #         # this version of nms is slower on GPU, so we move data to CPU.
    #         boxes = boxes.cpu()
    #         scores = scores.cpu()
    #         picked_box_probs = []
    #         picked_labels = []
    #         for class_index in range(1, scores.size(1)):
    #             probs = scores[:, class_index]
    #
    #             #print(class_index, to_numpy(probs).max())
    #             mask = probs > self.prob_threshold
    #             subset_probs = probs[mask]
    #             if len()== 0:
    #                 print('max_conf:',class_index,to_numpy( scores[:, class_index]).max())
    #                 continue
    #             subset_boxes = boxes[mask, :]
    #             box_probs = torch.cat([subset_boxes, subset_probs.reshape(-1, 1)], dim=1)
    #             keep = torchvision.ops.nms(subset_boxes, subset_probs, iou_threshold=self.iou_threshold)
    #             box_probs = box_probs[keep, :]
    #             picked_box_probs.append(box_probs)
    #             picked_labels.extend([class_index] * box_probs.size(0))
    #         if picked_box_probs and len(picked_box_probs) >0:
    #
    #             picked_box_probs = concate(picked_box_probs, 0)
    #             picked_box_probs[:, 0] *= 640
    #             picked_box_probs[:, 1] *= 480
    #             picked_box_probs[:, 2] *= 640
    #             picked_box_probs[:, 3] *= 480
    #
    #             picked_labels = to_tensor(picked_labels).unsqueeze(-1).to(picked_box_probs.device).to(picked_box_probs.dtype)
    #
    #             all_picked_box_probs.append(concate([picked_box_probs, picked_labels], axis=-1))
    #
    #     return all_picked_box_probs
    #
    #     #
    #     # boxes = boxes
    #     # confidences = confidences
    #     # if detection_threshold is not None:
    #     #     self.detection_threshold = detection_threshold
    #     # picked_box_probs = []
    #     # picked_labels = []
    #     # for class_index in range(1, confidences.shape[-1]):
    #     #     probs = confidences[..., class_index]
    #     #     mask = probs > self.detection_threshold
    #     #     probs = probs[mask]
    #     #     if probs.shape[0] == 0:
    #     #         continue
    #     #     subset_boxes = boxes[mask, :]
    #     #     box_probs = concate([subset_boxes, probs.reshape(-1, 1)], axis=1)
    #     #     keep = torchvision.ops.nms(subset_boxes, probs,iou_threshold=iou_threshold)
    #     #
    #     #     picked_box_probs.append(box_probs[keep,:])
    #     #     picked_labels.extend([class_index] * len(keep))
    #     # if not picked_box_probs:
    #     #     try:
    #     #         shp=list(boxes.shape)
    #     #         shp[-1]=2
    #     #         boxes=concate([boxes, zeros(shp)], axis=-1)
    #     #     except Exception as e:
    #     #         print(e)
    #     #         PrintException()
    #     #     return boxes[boxes[..., -1] >0].unsqueeze(0)
    #     # elif len(picked_box_probs)==1:
    #     #     try:
    #     #         picked_box_probs = picked_box_probs[0]
    #     #         picked_labels = to_tensor(picked_labels)
    #     #     except Exception as e:
    #     #         print(e)
    #     #         PrintException()
    #     # else:
    #     #     try:
    #     #         picked_box_probs = concate(picked_box_probs,axis=0)
    #     #         picked_labels = to_tensor(picked_labels)
    #     #     except Exception as e:
    #     #         print(e)
    #     #         PrintException()
    #     # _, _, height, width = self.signature.inputs.value_list[0].shape
    #     # picked_box_probs[...,  0] *= width
    #     # picked_box_probs[...,  1] *= height
    #     # picked_box_probs[...,  2] *= width
    #     # picked_box_probs[...,  3] *= height
    #     # return concate([picked_box_probs[..., :4], picked_labels.unsqueeze(-1),picked_box_probs[...,  4].unsqueeze(-1)],axis=-1).unsqueeze(0)

    def predict(self, confidences, boxes):
        all_picked_box_probs = []


        boxes =boxes.cpu()
        scores = confidences.cpu()

        # this version of nms is slower on GPU, so we move data to CPU.

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(-1)):
            probs = scores[..., class_index]
            mask = probs > self.detection_threshold
            subset_probs = probs[mask]
            if subset_probs.shape[0] == 0:

                print('max_conf:', class_index, to_numpy(probs).max())
                continue

            subset_boxes = boxes[mask, :]

            box_probs = torch.cat([subset_boxes, subset_probs.reshape(-1, 1)], dim=1)
            keep = torchvision.ops.nms(subset_boxes, subset_probs, iou_threshold=self.nms_threshold)
            box_probs=box_probs[keep, :]
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        #if not picked_box_probs or len(picked_box_probs)== 0:

        if picked_box_probs and len(picked_box_probs) > 0:
            picked_box_probs = concate(picked_box_probs, 0)
            picked_box_probs[:, 0] *= 640
            picked_box_probs[:, 1] *= 480
            picked_box_probs[:, 2] *= 640
            picked_box_probs[:, 3] *= 480

            picked_labels = to_tensor(picked_labels).unsqueeze(-1).to(picked_box_probs.device).to(picked_box_probs.dtype)
            return concate([picked_box_probs, picked_labels], axis=-1)
        else:

            boxes=to_tensor([[0,0,0,0,0,0]]).float().view(1,6)
            return boxes[boxes[..., 4]>0].unsqueeze(0)

        #
        # boxes = boxes
        # confidences = confidences
        # if detection_threshold is not None:
        #     self.detection_threshold = detection_threshold
        # picked_box_probs = []
        # picked_labels = []
        # for class_index in range(1, confidences.shape[-1]):
        #     probs = confidences[..., class_index]
        #     mask = probs > self.detection_threshold
        #     probs = probs[mask]
        #     if probs.shape[0] == 0:
        #         continue
        #     subset_boxes = boxes[mask, :]
        #     box_probs = concate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        #     keep = torchvision.ops.nms(subset_boxes, probs,iou_threshold=iou_threshold)
        #
        #     picked_box_probs.append(box_probs[keep,:])
        #     picked_labels.extend([class_index] * len(keep))
        # if not picked_box_probs:
        #     try:
        #         shp=list(boxes.shape)
        #         shp[-1]=2
        #         boxes=concate([boxes, zeros(shp)], axis=-1)
        #     except Exception as e:
        #         print(e)
        #         PrintException()
        #     return boxes[boxes[..., -1] >0].unsqueeze(0)
        # elif len(picked_box_probs)==1:
        #     try:
        #         picked_box_probs = picked_box_probs[0]
        #         picked_labels = to_tensor(picked_labels)
        #     except Exception as e:
        #         print(e)
        #         PrintException()
        # else:
        #     try:
        #         picked_box_probs = concate(picked_box_probs,axis=0)
        #         picked_labels = to_tensor(picked_labels)
        #     except Exception as e:
        #         print(e)
        #         PrintException()
        # _, _, height, width = self.signature.inputs.value_list[0].shape
        # picked_box_probs[...,  0] *= width
        # picked_box_probs[...,  1] *= height
        # picked_box_probs[...,  2] *= width
        # picked_box_probs[...,  3] *= height
        # return concate([picked_box_probs[..., :4], picked_labels.unsqueeze(-1),picked_box_probs[...,  4].unsqueeze(-1)],axis=-1).unsqueeze(0)
    def rerec(self, box, img_shape):
        """Convert box to square."""
        h = box[:, 3] - box[:, 1]
        w = box[:, 2] - box[:, 0]
        max_len = np.maximum(w, h)

        box[:, 0] = box[:, 0] - 0.5 * (max_len - w)
        box[:, 1] = box[:, 1] - 0.5 * (max_len - h)
        box[:, 2] = box[:, 0] + max_len
        box[:, 3] = box[:, 1] + max_len
        box1 = box.copy()
        box[:, 0] = np.clip(box[:, 0], 0, img_shape[3])
        box[:, 1] = np.clip(box[:, 1], 0, img_shape[2])
        box[:, 2] = np.clip(box[:, 2], 0, img_shape[3])
        box[:, 3] = np.clip(box[:, 3], 0, img_shape[2])
        pad = np.abs(box1 - box)
        return box[0], pad[0]

    def forward(self, x):
        self.priors.to(self.device)
        confidences = []
        locations = []

        x = self.backbond1(x)
        confidence0, location0 = self.compute_header(0, x)
        confidences.append(confidence0)
        locations.append(location0)

        x = self.backbond2(x)
        confidence1, location1 = self.compute_header(1, x)
        confidences.append(confidence1)
        locations.append(location1)

        x = self.backbond3(x)
        confidence2, location2 = self.compute_header(2, x)
        confidences.append(confidence2)
        locations.append(location2)

        x = self.extra(x)
        confidence3, location3 = self.compute_header(3, x)
        confidences.append(confidence3)
        locations.append(location3)

        confidences = torch.cat(confidences, 1)
        if not hasattr(self, 'softmax') or self.softmax is None:
            self.softmax = SoftMax(-1)

        locations = torch.cat(locations, 1)

        if self.training:
            confidences = self.softmax(confidences)
            result = OrderedDict()
            result['confidences'] = confidences
            result['locations'] = locations
            return result

        else:
            confidences = softmax(confidences,axis=-1)
            locations = decode(locations, self.priors, self.variance)
            return confidences, locations


def RfbNet(include_top=True,
           pretrained=True,
           input_shape=(3, 480, 640),
           base_filters=16, num_classes=2, num_regressors=4,detection_threshold=0.4,nms_threshold=0.3,
           **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 480, 640)
    if num_classes != 2 or num_regressors != 4:
        pretrained = False
    rfbnet_model = SsdDetectionModel(input_shape=input_shape,
                                     output=RFBnet(base_filters=base_filters, num_classes=num_classes,
                                                   num_regressors=num_regressors))

    rfbnet_model.detection_threshold = detection_threshold
    rfbnet_model.nms_threshold = nms_threshold
    rfbnet_model.palette[0] = (0, 0, 0)
    rfbnet_model.palette[1] = (128, 255, 128)
    rfbnet_model.preprocess_flow = [
        Resize((480, 640), True, align_corner=True),
        Normalize(127.5, 127.5)
    ]

    if pretrained == True:
        rfbnet_model.class_names = ['background', 'face']
        download_model_from_google_drive('1T_0VYOHaxoyuG1fAxY-6g0C7pfXiujns', dirname, 'version-RFB-640.pth')
        recovery_model = fix_layer(load(os.path.join(dirname, 'version-RFB-640.pth')))
        recovery_model.softmax = SoftMax(axis=-1)
        # priors = recovery_model.priors.clone()
        # recovery_model.__delattr__("priors")
        # recovery_model.register_buffer("priors", priors)
        recovery_model.name = 'rfb640'
        recovery_model.eval()
        recovery_model.to(_device)
        rfbnet_model.load_state_dict(recovery_model.state_dict())
    return rfbnet_model

