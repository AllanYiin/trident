from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from itertools import product as product
import cv2
from ..data.dataset import BboxDataset
from ..backend.common import *
from ..backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential
from ..backend.pytorch_ops import *
from ..data.bbox_common import xywh2xyxy, xyxy2xywh,bbox_giou,bbox_giou_numpy
from ..data.image_common import *
from ..data.utils import download_model_from_google_drive
from ..layers.pytorch_activations import get_activation, Identity, Relu, softmax
from ..layers.pytorch_blocks import *
from ..layers.pytorch_layers import *
from ..layers.pytorch_normalizations import get_normalization
from ..layers.pytorch_pooling import *
from ..optims.pytorch_trainer import *
from ..optims.pytorch_losses import *

image_size = [640, 480]
cfg = {'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]], 'steps': [8, 16, 32, 64],
       'variance': [0.1, 0.2], 'clip': False, }

__all__ = ['Ssd', 'encode', 'decode', 'SsdBboxDataset', 'SsdBboxDatasetV2', 'SsdDetectionModel', 'MultiBoxLoss',
           'MultiBoxLossV2','IouLoss']


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2), box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, 0:2].unsqueeze(1).expand(A, B, 2), box_b[:, 0:2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(truths, priors, variances, labels, threshold=0.3):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    priors2 = priors.clone()
    num_priors = len(priors)
    overlaps = jaccard(truths, xywh2xyxy(priors.clone()))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        return np.zeros((num_priors, 4)).astype(np.float32), np.zeros(num_priors).astype(np.int64)

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # Shape: [num_priors,14]
    conf = labels[best_truth_idx]  # Shape: [num_priors]

    conf[best_truth_overlap < threshold] = 0  # label as background

    loc = encode(matches, priors2, variances)
    return loc, conf


def encode(matched, priors, variances):
    """Encode is the process let groundtruth convert to prior
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in xyxy
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes and landmarks (tensor), Shape: [num_priors, 14]
    """

    # dist b/t match center and prior's center
    priors = priors.clone()
    g_cxcy = (matched[:, 0:2] + matched[:, 2:4]) / 2 - priors[:, 0:2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:4])
    # match wh / prior wh
    g_wh = (matched[:, 2:4] - matched[:, 0:2]) / priors[:, 2:4]
    g_wh = torch.log(g_wh) / variances[1]

    # # landmarks
    # g_xy1 = (matched[:, 4:6] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    # g_xy2 = (matched[:, 6:8] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    # g_xy3 = (matched[:, 8:10] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    # g_xy4 = (matched[:, 10:12] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])
    # g_xy5 = (matched[:, 12:14] - priors[:, 0:2]) / (variances[0] * priors[:, 2:4])

    # return target for loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,14]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat([priors.unsqueeze(0)[:, :, 0:2] + loc[:, :, 0:2] * variances[0] * priors.unsqueeze(0)[:, :, 2:4],
                       priors.unsqueeze(0)[:, :, 2:4] * torch.exp(loc[:, :, 2:4] * variances[1])], -1)
    boxes[:, :, 0:2] -= boxes[:, :, 2:4] / 2
    boxes[:, :, 2:4] += boxes[:, :, 0:2]
    return boxes


class SsdBboxDataset(BboxDataset):
    def __init__(self, boxes=None, image_size=None, priors=None, center_variance=0.1, size_variance=0.2,
                 gt_overlap_tolerance=0.5, expect_data_type=ExpectDataType.absolute_bbox, class_names=None,
                 symbol='bbox', name=''):
        super().__init__(boxes=boxes, image_size=image_size, expect_data_type=expect_data_type, class_names=class_names,
                         symbol=symbol, name=name)
        self.priors = priors
        self.label_transform_funcs = []
        self.gt_overlap_tolerance = gt_overlap_tolerance
        self.bbox_post_transform_funcs = []

    def binding_class_names(self, class_names=None, language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = 'en-us'
            self.class_names[language] = list(class_names)
            self.__default_language__ = language
            self._lab2idx = {v: k for k, v in enumerate(self.class_names[language])}
            self._idx2lab = {k: v for k, v in enumerate(self.class_names[language])}

    def bbox_transform(self, bbox):
        num_priors = self.priors.shape[0]
        if bbox is None or len(bbox) == 0:
            return np.zeros((num_priors, 4)).astype(np.float32), np.zeros(num_priors).astype(np.int64)
        elif isinstance(bbox, np.ndarray):
            height, width = self.image_size
            gt_label = None
            gt_box = bbox
            if bbox.shape[-1] % 2 == 1:
                gt_box = bbox[:, :-1]
                gt_label = bbox[:, -1]

            gt_box[:, 0::2] /= width
            gt_box[:, 1::2] /= height

            # match priors (default boxes) and ground truth boxes
            if gt_box is not None and len(gt_box) > 0:
                truths = to_tensor(gt_box).float()
                labels = to_tensor(gt_label).long()
                loc_t, conf_t = match(truths, self.priors.data, (0.1, 0.2), labels, self.gt_overlap_tolerance)

                return to_numpy(loc_t).astype(np.float32), to_numpy(conf_t).astype(np.int64)
            return np.zeros((num_priors, 4)).astype(np.float32), np.zeros(num_priors).astype(np.int64)


class SsdBboxDatasetV2(BboxDataset):
    def __init__(self, boxes=None, image_size=None, priors=None, center_variance=0.1, size_variance=0.2,
                 gt_overlap_tolerance=0.5, expect_data_type=ExpectDataType.absolute_bbox, class_names=None,
                 symbol='bbox', name=''):
        super().__init__(boxes=boxes, image_size=image_size, expect_data_type=expect_data_type, class_names=class_names,
                         symbol=symbol, name=name)
        self.priors = priors
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.label_transform_funcs = []
        self.gt_overlap_tolerance = gt_overlap_tolerance
        self.bbox_post_transform_funcs = []

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

    def convert_boxes_to_locations(self, center_form_boxes, center_form_priors):
        if len(center_form_priors.shape) + 1 == len(center_form_boxes.shape):
            center_form_priors = np.expand_dims(center_form_priors, 0)
        return np.concatenate([(center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[...,
                                                                                            2:] / self.center_variance,
                               np.log(np.clip(center_form_boxes[..., 2:] / center_form_priors[..., 2:],1e-8,np.inf)) / self.size_variance],
                              axis=len(center_form_boxes.shape) - 1)

    def assign_priors(self, gt_boxes, gt_labels, center_form_priors, iou_threshold):
        # """Assign ground truth boxes and targets to priors.
        #
        # Args:
        #     gt_boxes (num_targets, 4): ground truth boxes.
        #     gt_labels (num_targets): labels of targets.
        #     priors (num_priors, 4): corner form priors
        # Returns:
        #     boxes (num_priors, 4): real values for priors.
        #     labels (num_priros): labels for priors.
        # """
        # # size: num_priors x num_targets
        # # if gt_boxes is not None :
        #
        # ious = self.iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
        # # size: num_priors
        # best_target_per_prior, best_target_per_prior_index = ious.max(1)
        # # size: num_targets
        # best_prior_per_target, best_prior_per_target_index = ious.max(0)
        #
        # for target_index, prior_index in enumerate(best_prior_per_target_index):
        #     best_target_per_prior_index[prior_index] = target_index
        # # 2.0 is used to make sure every target has a prior assigned
        # best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
        # # size: num_priors
        # labels = gt_labels[best_target_per_prior_index]
        # labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
        # boxes = gt_boxes[best_target_per_prior_index]
        # return boxes, labels
        corner_form_priors = xywh2xyxy(center_form_priors)
        ious = self.iou_of(np.expand_dims(gt_boxes, 0), np.expand_dims(corner_form_priors, 1))
        # size: num_priors
        best_target_per_prior, best_target_per_prior_index = np.max(ious, axis=1), np.argmax(ious, axis=1)
        # size: num_targets
        best_prior_per_target, best_prior_per_target_index = np.max(ious, axis=0), np.argmax(ious, axis=0)
        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        best_prior_per_target_index_list = best_prior_per_target_index.tolist()
        for i in range(best_target_per_prior.shape[0]):
            if i in best_prior_per_target_index_list:
                best_target_per_prior[i] = 2
        labels = gt_labels[best_target_per_prior_index]
        labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
        boxes = gt_boxes[best_target_per_prior_index]
        return boxes, labels

    def binding_class_names(self, class_names=None, language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = 'en-us'
            self.class_names[language] = list(class_names)
            self.__default_language__ = language
            self._lab2idx = {v: k for k, v in enumerate(self.class_names[language])}
            self._idx2lab = {k: v for k, v in enumerate(self.class_names[language])}

    def bbox_transform(self, bbox):
        if bbox is None or len(bbox) == 0:
            return np.zeros((self.priors.shape[0], 4)).astype(np.float32), np.zeros((self.priors.shape[0])).astype(
                np.int64)
        elif isinstance(bbox, np.ndarray):
            height, width = self.image_size
            bbox[:, 0] = bbox[:, 0] / width
            bbox[:, 2] = bbox[:, 2] / width
            bbox[:, 1] = bbox[:, 1] / height
            bbox[:, 3] = bbox[:, 3] / height
            if bbox.shape[-1] == 5:
                gt_box = bbox[:, :4]
                gt_label = bbox[:, 4]
                boxes, labels = self.assign_priors(gt_box, gt_label, to_numpy(self.priors), 0.3)
                boxes = xyxy2xywh(boxes)
                locations = self.convert_boxes_to_locations(boxes, to_numpy(self.priors))
                return boxes.astype(np.float32), labels.astype(np.int64)
            else:
                return bbox


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0

    num_pos = pos_mask.long().sum()
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(descending=True)
    _, orders = indexes.sort()
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


class MultiBoxLoss(nn.Module):
    def __init__(self, priors, neg_pos_ratio, center_variance, size_variance):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = to_tensor(priors)

    def forward(self, confidence, locations, target_confidence, target_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)

        # derived from cross_entropy=sum(log(p))
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, target_confidence, self.neg_pos_ratio)
        weight = to_tensor(np.array([0.05, 1, 5, 20, 10]))
        classification_loss = F.cross_entropy(confidence[mask, :].reshape(-1, num_classes), target_confidence[mask],
                                              weight=weight, reduction='sum')
        # classification_loss += 0.1*F.cross_entropy(confidence.reshape(-1, num_classes), target_confidence.reshape(
        # -1), weight=weight, reduction='sum')

        pos_mask = target_confidence > 0
        locations = locations[pos_mask, :].reshape(-1, 4)
        target_locations = target_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.mse_loss(locations, target_locations, reduction='sum')  # smooth_l1_loss
        smooth_l1_loss += F.l1_loss(locations[:, 2:4].exp(), target_locations[:, 2:4].exp(), reduction='sum')
        num_pos = target_locations.size(0)
        return (smooth_l1_loss + classification_loss) / num_pos


class MultiBoxLossV2(nn.Module):
    def __init__(self, priors, neg_pos_ratio, center_variance, size_variance):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLossV2, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = to_tensor(priors)

    def forward(self, confidence, locations, target_confidence, target_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            target_confidence (batch_size, num_priors): real labels of all the priors.
            target_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        # derived from cross_entropy=sum(log(p))
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, target_confidence, self.neg_pos_ratio)

        classification_loss = CrossEntropyLabelSmooth(weight=np.array([1,1,5,25,15]),axis=-1,num_classes=5,reduction='sum')(confidence[mask, :].reshape(-1, num_classes), target_confidence[mask].reshape(-1))

        pos_mask = target_confidence > 0
        target_locations = target_locations[pos_mask, :].reshape(-1, 4)

        num_pos = target_locations.size(0)
        return   classification_loss / num_pos

class IouLoss(nn.Module):
    def __init__(self, priors,  center_variance, size_variance):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(IouLoss, self).__init__()
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = to_tensor(priors)

    def forward(self, confidence, locations, target_confidence, target_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            target_confidence (batch_size, num_priors): real labels of all the priors.
            target_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        num_batch= confidence.size(0)

        confidence_logit = softmax(confidence, -1)
        confidence_logit_probs, confidence_logit_idxs = confidence_logit.max(-1)
        probs_mask = confidence_logit_probs > 0.5
        label_mask = confidence_logit_idxs > 0

        pos_target_mask_all = target_confidence > 0
        pos_infer_mask_all = (pos_target_mask_all.float() +probs_mask.float() + label_mask.float()==3)



        decode_locations_all = decode(locations, self.priors, (self.center_variance, self.size_variance))
        decode_target_locations_all = decode(target_locations, self.priors, (self.center_variance, self.size_variance))
        giou_np=0.0
        giou=0.0
        overlaps=0.0
        num_boxes=0
        for i in range(num_batch):
            pos_target_mask=pos_target_mask_all[i]
            pos_infer_mask=pos_infer_mask_all[i]
            decode_locations=decode_locations_all[i][pos_infer_mask,:]
            decode_target_locations=decode_target_locations_all[i][pos_target_mask,:]
            num_boxes+=decode_target_locations.shape[0]
            if decode_target_locations.shape[0]>0 and decode_locations.shape[0]>0:
                giou=giou+(1-(bbox_giou(decode_locations, decode_target_locations).sum(0)/decode_target_locations.shape[0])).sum()
                overlaps=overlaps+(-log(clip(jaccard(decode_locations, decode_target_locations),min=1e-8)).sum(0)/decode_target_locations.shape[0]).sum()
            elif decode_target_locations.shape[0]==0 and decode_locations.shape[0]==0:
                pass
            else:
                giou=giou+1
                overlaps=overlaps-log(to_tensor(1e-8))

        giou=giou/num_boxes
        overlaps=overlaps/num_boxes
        return giou


class Ssd(Layer):
    def __init__(self, backbond, base_filters=16, num_classes=5, num_regressors=14, iou_threshold=0.3,
                 variance=(0.1, 0.2), name='tiny_mobile_rfbnet', **kwargs):
        """

        Parameters
        ----------
        layer_defs : object
        """
        super(Ssd, self).__init__(name=name)
        self.base_filters = base_filters

        idxes = []
        if isinstance(backbond, Sequential):
            for i in range(len(backbond._modules)):
                layer = backbond[i]
                if hasattr(layer, 'strides') or isinstance(layer, Sequential):
                    if isinstance(layer, Sequential):
                        layer = layer[0]
                    if isinstance(layer.strides, int) and layer.strides == 2:
                        idxes.append(i)
                    elif isinstance(layer.strides, tuple) and layer.strides[0] == 2:
                        idxes.append(i)

            self.backbond1 = Sequential(backbond[:idxes[-2]], name='backbond1')
            self.backbond2 = Sequential(backbond[idxes[-2]:idxes[-1]], name='backbond2')
            self.backbond3 = Sequential(backbond[idxes[-1]:], name='backbond3')
        else:
            raise ValueError('{0} is not a valid backbond...'.format(backbond.__class__.__name__))

        self.iou_threshold = iou_threshold
        self.variance = variance

        self.num_classes = num_classes
        self.num_regressors = num_regressors

        self.extra = Sequential(Conv2d((1, 1), num_filters=64, strides=1, activation='relu', use_bias=True),
                                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=2, auto_pad=True, activation='relu',
                                                use_bias=True),
                                Conv2d((1, 1), num_filters=256, strides=1, activation=None, use_bias=True), Relu())
        self.regression_headers = nn.ModuleList([Sequential(
            DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
            Conv2d((1, 1), num_filters=3 * self.num_regressors, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_regressors, strides=1, activation=None, use_bias=True)),

            Sequential(
                DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, auto_pad=True, activation='relu', use_bias=True),
                Conv2d((1, 1), num_filters=2 * self.num_regressors, strides=1, activation=None, use_bias=True)),

            Conv2d((3, 3), num_filters=3 * self.num_regressors, strides=1, auto_pad=True, activation=None), ])

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

        self.priors = []

        # generate priors
        self.define_img_size(640)

    def generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True) -> torch.Tensor:
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
        priors = to_tensor(priors)  # .view(-1, 4)
        if clamp:
            torch.clamp(priors, 0.0, 1.0, out=priors)
        return priors

    def define_img_size(self, size=640):
        global image_size, feature_map_w_h_list, priors
        img_size_dict = {128: [128, 96], 160: [160, 120], 320: [320, 240], 480: [480, 360], 640: [640, 480],
                         1280: [1280, 960]}

        min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        shrinkage_list = []
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
        self.priors = self.generate_priors(feature_map_w_h_list_dict[size], shrinkage_list, img_size_dict[size],
                                           min_boxes)

    def init_from_pretrained_ssd(self, model):
        def _xavier_init_(m: Layer):
            if isinstance(m, (Conv2d, DepthwiseConv2d)):
                nn.init.xavier_uniform_(m.weight)

        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = state_dict['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict, strict=False)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

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
        locations = torch.cat(locations, 1)

        if self.training:
            return confidences, locations
        else:
            confidences = softmax(confidences, -1)

            locations = decode(locations, self.priors, self.variance)
            return confidences, locations


class SsdDetectionModel(ImageDetectionModel):
    def __init__(self, inputs=None, output=None, input_shape=None):
        super(SsdDetectionModel, self).__init__(inputs, output, input_shape)
        self.preprocess_flow = []
        self.detection_threshold = 0.7
        self.iou_threshold = 0.3
        self.palette = {}

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
        if box_scores is None or len(box_scores) == 0:
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

        return box_scores[picked, :], picked

    def nms(self, boxes, threshold=0.3):
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
                if overlap > threshold:
                    suppress.append(pos)

            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
        return boxes[pick], pick

    def infer_single_image(self, img, scale=1):
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

                confidence, boxes = self._model(inp)
                boxes = boxes[0]
                confidence = confidence[0]
                probs, label = confidence.data.max(-1)
                mask = probs > self.detection_threshold
                probs = probs[mask]
                label = label[mask]
                boxes = boxes[mask, :]
                mask = label > 0
                probs = probs[mask]
                label = label[mask]
                boxes = boxes[mask, :]

                if boxes is not None and len(boxes) > 0:
                    boxes = to_numpy(boxes)
                    label = to_numpy(label)
                    probs = to_numpy(probs)
                    box_probs = np.concatenate([boxes, label.reshape(-1, 1), probs.reshape(-1, 1)], axis=1)
                    if len(boxes) > 1:
                        box_probs, keep = self.hard_nms(box_probs, iou_threshold=self.iou_threshold, top_k=-1, )
                    boxes = box_probs[:, :4]
                    boxes[:, 0::2] *= 640
                    boxes[:, 1::2] *= 480
                    boxes[:, :4] /= scale

                    # boxes = boxes * (1 / scale[0])
                    return img_orig, boxes, box_probs[:, 4].astype(np.int32), box_probs[:, 5]
                else:
                    return img_orig, None, None, None
            except:
                PrintException()
        else:
            raise ValueError('the model is not built yet.')

    def infer_then_draw_single_image(self, img, scale=1):
        rgb_image, boxes, labels, probs = self.infer_single_image(img, scale)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if boxes is not None and len(boxes) > 0:
            boxes = np.round(boxes).astype(np.int32)
            if boxes.ndim == 1:
                boxes = np.expand_dims(boxes, 0)
            if labels.ndim == 0:
                labels = np.expand_dims(labels, 0)
            for m in range(len(boxes)):
                this_box = boxes[m]
                this_label = labels[m]
                cv2.rectangle(bgr_image, (this_box[0], this_box[1]), (this_box[2], this_box[3]),
                              self.palette[this_label],
                              1 if bgr_image.shape[1] < 480 else 2 if bgr_image.shape[1] < 640 else 3 if
                              bgr_image.shape[1] < 960 else 4)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        return rgb_image

