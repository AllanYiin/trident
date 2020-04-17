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

from ..backend.common import *
from ..backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential
from ..backend.pytorch_ops import *
from ..data.bbox_common import xywh_to_xyxy, xyxy_to_xywh
from ..data.image_common import *
from ..data.utils import download_model_from_google_drive
from ..layers.pytorch_activations import get_activation, Identity, Relu
from ..layers.pytorch_blocks import *
from ..layers.pytorch_layers import *
from ..layers.pytorch_normalizations import get_normalization
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *

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

cfg = {'feature_maps': [38, 19, 10, 5, 3, 1], 'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300],
       'min_sizes': [30, 60, 111, 162, 213, 264], 'max_sizes': [60, 111, 162, 213, 264, 315],
       'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, }

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
                                   activation='relu', normalization='batch'), conv_dw(filter_base * 2, 1),
                      conv_dw(filter_base * 2, 2),  # 80*60
                      conv_dw(filter_base * 2, 1), conv_dw(filter_base * 4, 2),  # 40*30
                      conv_dw(filter_base * 4, 1), conv_dw(filter_base * 4, 1), basic_rfb(filter_base * 4, scale=1.0),

                      conv_dw(filter_base * 8, 2),  # 20*15
                      conv_dw(filter_base * 8, 1), conv_dw(filter_base * 8, 1),

                      conv_dw(filter_base * 16, 2),  # 10*8
                      conv_dw(filter_base * 16, 1))


class RFBnet(Layer):
    def __init__(self, *args, base_filters=16, num_classes=4, num_regressors=14,iou_threshold=0.3, center_variance=0.1, size_variance=0.2,
                 name='tiny_mobile_rfbnet', **kwargs):
        """

        Parameters
        ----------
        layer_defs : object
        """
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
            confidences_class = F.softmax(confidences[:,:,2],dim=2)
            confidences_attr=confidences[:,:,2].sigmoid()
            boxes = convert_locations_to_boxes(locations, self.priors, self.center_variance, self.size_variance)
            boxes = xywh_to_xyxy(boxes)
            return confidences_class, boxes,confidences_attr




class SsdDetectionModel(ImageDetectionModel):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(SsdDetectionModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = []
        self.detection_threshold = 0.7
        self.iou_threshold = 0.3

    def area_of(self, left_top, right_bottom):
        """Compute the areas of rectangles given two corners.

        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.

        Returns:
            area (N): return the area.
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
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
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        """

        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        if box_scores is None or len(box_scores)==0:
            return None, None
        scores = box_scores[:, -1]
        boxes = box_scores[:, :4]
        picked = []
        # _, indexes = scores.sort(descending=True)
        indexes = np.argsort(scores)
        # indexes = indexes[:candidate_size]
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            # current = indexes[0]
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            # indexes = indexes[1:]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(rest_boxes, np.expand_dims(current_box, axis=0), )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :],picked


    def predict(self, width, height, confidences_class, boxes, confidences_attr, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences_class = confidences_class[0]
        confidences_attr=confidences_attr[0]
        picked_box_probs = None
        picked_labels =None

        probs = confidences_class[:, 1]
        labels=np.argmax(confidences_class,-1)
        mask = probs > prob_threshold

        probs = probs[mask]
        labels = labels[mask]
        confidences_attr= confidences_attr[mask]
        if probs.shape[0] > 0:
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs, keep = self.hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k, )
            if box_probs is not None:
                picked_box_probs = box_probs
                picked_labels = labels[keep]
                confidences_attr = confidences_attr[keep]
            else:
                return None, None, None

        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        if picked_box_probs.shape[-1]>=14:
            picked_box_probs[:, 4] *= width
            picked_box_probs[:, 5] *= height
            picked_box_probs[:, 6] *= width
            picked_box_probs[:, 7] *= height
            picked_box_probs[:, 8] *= width
            picked_box_probs[:, 9] *= height
            picked_box_probs[:, 10] *= width
            picked_box_probs[:, 11] *= height
            picked_box_probs[:, 12] *= width
            picked_box_probs[:, 13] *= height

        return picked_box_probs, picked_labels,confidences_attr


    def nms(self,boxes, threshold=0.3):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list, add the index
            # value to the list of picked indexes, then initialize
            # the suppression list (i.e. indexes that will be deleted)
            # using the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]

            # loop over all indexes in the indexes list
            for pos in range(0, last):
                # grab the current index
                j = idxs[pos]

                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / area[j]

                # if there is sufficient overlap, suppress the
                # current bounding box
                if overlap >threshold:
                    suppress.append(pos)

            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
        return boxes[pick],pick

    def infer_single_image(self, img, scale=None):
        if self._model.built:
            try:
                self._model.to(self.device)
                self._model.eval()
                img = image2array(img)
                if img.shape[-1] == 4:
                    img = img[:, :, :3]
                img_orig = img.copy()

                for func in self.preprocess_flow:
                    if inspect.isfunction(func):
                        img = func(img)
                        if func.__qualname__ == 'resize.<locals>.img_op':
                            scale = func.scale

                img = image_backend_adaptive(img)
                inp = to_tensor(np.expand_dims(img, 0)).to(
                    torch.device("cuda" if self._model.weights[0].data.is_cuda else "cpu")).to(
                    self._model.weights[0].data.dtype)
                confidences_class, boxes,confidences_attr = self._model(inp)

                boxes, labels,confidences_attr= self.predict(640, 480, to_numpy(confidences_class), to_numpy(boxes),to_numpy(confidences_attr),
                                                    prob_threshold=self.detection_threshold,
                                                    iou_threshold=0.3)
                if boxes is not None and len(boxes)>0:
                    #boxes,keep=self.nms(boxes,threshold=0.3)
                    boxes,probs=boxes[:,:-1],boxes[:,-1]
                    boxes=boxes*(1 / scale[0])
                    #labels=labels[keep].astype(np.uint8)
                    #fn = rescale(1 / scale[0])
                    #img, boxes = fn(img[0], boxes)
                    return img_orig, boxes, labels.astype(np.uint8), probs,confidences_attr
                else:
                    return img_orig, None, None, None,None
            except:
                PrintException()


        else:
            raise ValueError('the model is not built yet.')

    def generate_bboxes(self, *outputs, threshould=0.5, scale=1):
        raise NotImplementedError



def Mobile_RFBnet(base_filters=16, num_classes=4,num_regressors=4):
    model = SsdDetectionModel(input_shape=(3, 480, 640),
                              output=RFBnet(base_filters=base_filters, num_classes=num_classes,num_regressors=num_regressors))
    model.signature = get_signature(model.model.forward)
    model.preprocess_flow = [resize((480, 640), keep_aspect=True, align_corner=True), normalize(127.5, 127.5)]
    return model







