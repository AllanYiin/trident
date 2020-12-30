from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import inspect
import math
import os
import uuid
from collections import *
from collections import deque
from copy import copy, deepcopy
from functools import partial
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch.nn import init
from torch.nn.parameter import Parameter

from trident.backend.common import *
from trident.backend.pytorch_backend import Layer, Sequential, get_device
from trident.backend.pytorch_ops import *
from trident.data.image_common import *
from trident.data.bbox_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity, Mish, LeakyRelu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization, BatchNorm2d
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *
from trident.misc.visualization_utils import generate_palette,plot_bbox

__all__ = [ 'yolo4_body', 'YoloDetectionModel', 'DarknetConv2D', 'DarknetConv2D_BN_Mish',
           'DarknetConv2D_BN_Leaky', 'YoloLayer']

_session = get_session()
_device =get_device()
_epsilon = _session.epsilon
_trident_dir = _session.trident_dir



anchors1 = to_tensor(np.array([12, 16, 19, 36, 40, 28]).reshape(-1, 2),requires_grad=False)
anchors2 = to_tensor(np.array([36, 75, 76, 55, 72, 146]).reshape(-1, 2),requires_grad=False)
anchors3 = to_tensor(np.array([142, 110, 192, 243, 459, 401]).reshape(-1, 2),requires_grad=False)
anchors=(anchors1,anchors2,anchors3)



def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'use_bias': True}
    darknet_conv_kwargs['auto_pad'] = False if kwargs.get('strides')==(2,2) else True
    darknet_conv_kwargs['use_bias'] = True
    darknet_conv_kwargs.update(kwargs)
    return Conv2d(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    darknet_conv_kwargs = {'use_bias': False,'normalization':BatchNorm2d(momentum=0.03,eps=1e-4)}
    darknet_conv_kwargs['activation']=LeakyRelu(alpha=0.1)
    darknet_conv_kwargs['auto_pad'] = False if kwargs.get('strides')==(2,2) else True
    darknet_conv_kwargs.update(kwargs)
    return Conv2d_Block(*args, **darknet_conv_kwargs)



def DarknetConv2D_BN_Mish(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    darknet_conv_kwargs = {'use_bias': False, 'normalization':BatchNorm2d(momentum=0.03,eps=1e-4), 'activation': Mish()}
    darknet_conv_kwargs['auto_pad'] = False if kwargs.get('strides')==(2,2) else True
    darknet_conv_kwargs.update(kwargs)
    return Conv2d_Block(*args, **darknet_conv_kwargs)


def resblock_body(num_filters, num_blocks, all_narrow=True,keep_output=False,name=''):
    return Sequential(
        DarknetConv2D_BN_Mish((3, 3),num_filters ,strides=(2,2), auto_pad=False, padding=((1, 0), (1, 0)),name=name+'_preconv1'),
        ShortCut2d(
            {
            1:DarknetConv2D_BN_Mish((1, 1), num_filters // 2 if all_narrow else num_filters, name=name + '_shortconv'),
            0:Sequential(
                DarknetConv2D_BN_Mish((1, 1), num_filters // 2 if all_narrow else num_filters,name=name+'_mainconv'),
                For(range(num_blocks), lambda i:
                    ShortCut2d(
                        Identity(),
                        Sequential(
                            DarknetConv2D_BN_Mish((1, 1),num_filters // 2,name=name+'_for{0}_1'.format(i)),
                            DarknetConv2D_BN_Mish((3, 3),num_filters // 2 if all_narrow else num_filters,name=name+'_for{0}_2'.format(i))
                        ),
                        mode='add')
                ),
                DarknetConv2D_BN_Mish( (1, 1),num_filters // 2 if all_narrow else num_filters,name=name+'_postconv')
            )},
            mode='concate',name=name+'_route'),

        DarknetConv2D_BN_Mish((1,1),num_filters,name=name+'_convblock5')
        )


def yolo4_body(num_classes=80,image_size=608,anchors=anchors):
    anchors1=anchors[0]
    anchors2 = anchors[1]
    anchors3 = anchors[2]
    num_anchors=len(anchors1)
    """Create YOLO_V4 model CNN body in Pytorch."""
    return Sequential(
            DarknetConv2D_BN_Mish((3, 3), 32,name='first_layer'),
            resblock_body(64, 1, all_narrow=False,name='block64'),
            resblock_body(128, 2,name='block128'),
            resblock_body(256, 8,name='block256'),
            ShortCut2d(
                {
                    1:Sequential(
                        resblock_body(512, 8,name='block512'),
                        ShortCut2d(
                            {
                                1:Sequential(
                                    resblock_body(1024, 4, name='block1024'),
                                    DarknetConv2D_BN_Leaky( (1,1), 512,name='pre_maxpool1'),
                                    DarknetConv2D_BN_Leaky( (3, 3),1024,name='pre_maxpool2'),
                                    DarknetConv2D_BN_Leaky((1,1),512,name='pre_maxpool3'),
                                    ShortCut2d(
                                        MaxPool2d((13,13),strides=(1,1), auto_pad=True),
                                        MaxPool2d((9,9), strides=(1, 1), auto_pad=True),
                                        MaxPool2d((5,5), strides=(1, 1), auto_pad=True),
                                        Identity(),
                                        mode='concate'
                                    ),
                                    DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_1'),
                                    DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_2'),
                                    DarknetConv2D_BN_Leaky((1, 1), 512,name='y_19',keep_output=True),
                                    DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y19_upsample'),
                                    Upsampling2d(scale_factor=2,name='y19_upsample'),
                                ),
                                0:DarknetConv2D_BN_Leaky((1, 1), 256)
                            },mode='concate'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y38_1'),
                        DarknetConv2D_BN_Leaky((3, 3),512,name='pre_y38_2'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y38_3'),
                        DarknetConv2D_BN_Leaky((3, 3),512,name='pre_y38_4'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='y_38',keep_output=True),
                        DarknetConv2D_BN_Leaky((1, 1),128,name='pre_y_38_upsample'),
                        Upsampling2d(scale_factor=2,name='y_38_upsample'),
                    ),
                    0:DarknetConv2D_BN_Leaky((1, 1), 128)
                },
                mode='concate'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate1'),
            DarknetConv2D_BN_Leaky((3, 3), 256,name='pre_y76_concate2'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate3'),
            DarknetConv2D_BN_Leaky((3, 3), 256,name='pre_y76_concate4'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate5'),
            ShortCut2d(
                #y76_output
                Sequential(
                    DarknetConv2D_BN_Leaky( (3, 3),256,name='pre_y76_output'),
                    DarknetConv2D( (1, 1),num_anchors * (num_classes + 5),use_bias=True,name='y76_output'),
                    YoloLayer(anchors=anchors1,num_classes=num_classes,grid_size=76, img_dim=image_size),
                name='y76_output'),
                # y38_output
                Sequential(
                    ShortCut2d(
                        DarknetConv2D_BN_Leaky((3, 3), 256, strides=(2, 2), auto_pad=False,padding=((1,0),(1,0)),name='y76_downsample'),
                        branch_from='y_38',mode='concate'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate1'),
                    DarknetConv2D_BN_Leaky((3, 3), 512,name='pre_y38_concate2'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate3'),
                    DarknetConv2D_BN_Leaky((3, 3), 512,name='pre_y38_concate4'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate5'),
                    ShortCut2d(
                        Sequential(
                            DarknetConv2D_BN_Leaky((3, 3), 512, name='pre_y38_output'),
                            DarknetConv2D((1, 1), num_anchors * (num_classes + 5), use_bias=True, name='y38_output'),
                            YoloLayer(anchors=anchors2, num_classes=num_classes,grid_size=38,  img_dim=image_size),
                            name='y38_output'),

                        Sequential(
                            ShortCut2d(
                                DarknetConv2D_BN_Leaky((3, 3), 512, strides=(2, 2),auto_pad=False,padding=((1,0),(1,0)),name='y38_downsample'),
                                branch_from='y_19', mode='concate'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate1'),
                            DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_concate2'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate3'),
                            DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_concate4'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate5'),
                            Sequential(
                                DarknetConv2D_BN_Leaky((3, 3),1024,name='pre_y19_output'),
                                DarknetConv2D((1, 1), num_anchors * (num_classes + 5),use_bias=True,name='y19_output'),
                                YoloLayer(anchors=anchors3,num_classes=num_classes,grid_size=19, img_dim=image_size),
                            name='y19_output')),

                        mode='concate')
                )
                ,mode = 'concate')
    )


class YoloLayer(Layer):
    """Detection layer"""

    def __init__(self, anchors, num_classes,grid_size, img_dim=608,small_item_enhance=False):
        super(YoloLayer, self).__init__()

        self.register_buffer('anchors', to_tensor(anchors, requires_grad=False).to(get_device()))
        self.small_item_enhance = small_item_enhance
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        # self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size=grid_size

        #self.grid_size = to_tensor(grid_size) # grid size
        #yv, xv = torch.meshgrid([torch.arange(grid_size), torch.arange(grid_size)])
        #self.register_buffer('grid',  torch.stack((xv.detach(), yv.detach()), 2).view((1, 1, grid_size, grid_size, 2)).float().detach())

        self.stride = self.img_dim / grid_size

        self.compute_grid_offsets(grid_size)

    def compute_grid_offsets(self, grid_size):

        self.register_buffer('grid', meshgrid(grid_size,grid_size,requires_grad=False).view((1, 1, grid_size, grid_size, 2)).float().detach())
        #self.grid=meshgrid(grid_size,grid_size,requires_grad=False).view((1, 1, grid_size, grid_size, 2)).float().detach()

        # Calculate offsets for each grid

    def forward(self, x, targets=None):
        num_batch = x.size(0)
        grid_size = x.size(-1)

        if self.training and (self.grid_size!=x.size(-1) or self.grid_size!=x.size(-2)):
            self.compute_grid_offsets(grid_size)

        self.grid.to(get_device())
        self.anchors.to(get_device())
        anchor_vec = self.anchors/ self.stride
        anchor_wh = anchor_vec.view(1, self.num_anchors, 1, 1, 2)


        prediction = x.view(num_batch, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        # if self.training:
        #     return reshape(prediction,(num_batch, -1, self.num_classes + 5))

            # Get outputs
        xy = sigmoid(prediction[..., 0:2])  # Center x
        wh = prediction[..., 2:4]  # Width

        xy = reshape((xy + self.grid) * self.stride, (num_batch, -1, 2))
        wh = reshape((exp(wh) * anchor_wh) * self.stride, (num_batch, -1, 2))

        pred_conf = sigmoid(prediction[..., 4])  # Conf
        pred_class = sigmoid(prediction[..., 5:])  # Cls pred.

        pred_conf = reshape(pred_conf, (num_batch, -1, 1))
        pred_class = reshape(pred_class, (num_batch, -1, self.num_classes))
        cls_probs = reduce_max(pred_class, -1, keepdims=True)

        if self.small_item_enhance and self.stride == 8:
            pred_conf = (pred_conf * cls_probs).sqrt()

        output = torch.cat([xy, wh, pred_conf, pred_class], -1)


        return output



class YoloDetectionModel(ImageDetectionModel):
    def __init__(self, inputs=None, input_shape=None,output=None):
        super(YoloDetectionModel, self).__init__(inputs, input_shape,output)
        self.preprocess_flow = [resize((input_shape[-2], input_shape[-1]), True), normalize(0, 255)]
        self.detection_threshold = 0.7
        self.iou_threshold = 0.3
        self.class_names = None
        self.palette = generate_palette(80)

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

    def infer_single_image(self, img, scale=1, verbose=False):
        time_time =None
        if verbose:
            time_time = time.time()
            print("==-----  starting infer {0} -----=======".format(img))
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

                img = image_backend_adaption(img)
                inp = to_tensor(np.expand_dims(img, 0)).to(self.device).to(self._model.weights[0].data.dtype)

                if verbose:
                    print("======== data preprocess time:{0:.5f}".format((time.time() - time_time)))
                    time_time = time.time()

                boxes = self._model(inp)[0]
                if verbose:
                    print("======== infer  time:{0:.5f}".format((time.time() - time_time)))
                    time_time = time.time()

                mask = boxes[:, 4] > self.detection_threshold
                boxes = boxes[mask]
                if verbose:
                    print('         detection threshold:{0}'.format(self.detection_threshold))
                    print('         {0} bboxes keep!'.format(len(boxes)))
                if boxes is not None and len(boxes) > 0:
                    boxes = concate([xywh2xyxy(boxes[:, :4]), boxes[:, 4:]], axis=-1)
                    boxes = to_numpy(boxes)
                    if len(boxes) > 1:
                        box_probs, keep = self.hard_nms(boxes[:, :5], iou_threshold=self.iou_threshold, top_k=-1, )
                        boxes = boxes[keep]
                        print('         iou threshold:{0}'.format(self.iou_threshold))
                        print('         {0} bboxes keep!'.format(len(boxes)))
                    boxes[:, :4] /=scale
                    boxes[:, :4]=np.round(boxes[:, :4],0)

                    if verbose:
                        print("======== bbox postprocess time:{0:.5f}".format((time.time() - time_time)))
                        time_time = time.time()
                    # boxes = boxes * (1 / scale[0])
                    locations= boxes[:, :4]
                    probs = boxes[:, 4]
                    labels=np.argmax(boxes[:, 5:], -1).astype(np.int32)

                    if verbose and locations is not None:
                        for i in range(len(locations)):
                            print('         box{0}: {1} prob:{2:.2%} class:{3}'.format(i, [np.round(num, 4) for num in
                                                                                           locations[i].tolist()], probs[i],
                                                                                       labels[i] if self.class_names is None or int(labels[i]) >= len(self.class_names) else self.class_names[int(labels[i])]))

                    return img_orig,locations,labels,probs

                else:
                    return img_orig, None, None, None
            except:
                PrintException()
        else:
            raise ValueError('the model is not built yet.')

    def infer_then_draw_single_image(self, img, scale=1, verbose=False):
        rgb_image, boxes, labels, probs = self.infer_single_image(img, scale, verbose)
        time_time = None
        if verbose:
            time_time = time.time()

        if boxes is not None and len(boxes) > 0:
            if boxes.ndim == 1:
                boxes = np.expand_dims(boxes, 0)
            if labels.ndim == 0:
                labels = np.expand_dims(labels, 0)
            pillow_img = array2image(rgb_image.copy())
            for m in range(len(boxes)):
                this_box = boxes[m]
                this_label = labels[m]
                thiscolor=tuple([int(c) for c in self.palette[this_label][:3]])
                pillow_img=plot_bbox(this_box, pillow_img, thiscolor,self.class_names[int(this_label)], line_thickness=2)
            rgb_image = np.array(pillow_img.copy())
        if verbose:
            print("======== draw image time:{0:.5f}".format((time.time() - time_time)))
        return rgb_image


def YoLoV4(pretrained=True,
            freeze_features=False,
            input_shape=(3, 608, 608),
             classes=80,
             **kwargs):
    detector = YoloDetectionModel(input_shape=input_shape, output=yolo4_body(classes, input_shape[-1]))
    detector.load_model('Models/pretrained_yolov4_mscoco.pth.tar')


def darknet_body():
    return Sequential(
        DarknetConv2D_BN_Mish((3, 3), 32),
        resblock_body(64, 1, all_narrow=False),
        resblock_body(128, 2),
        resblock_body(256, 8),
        resblock_body(512, 8),
        resblock_body(1024, 4)
    )

def yolo4_body(num_classes=80,image_size=608):
    anchors1 = to_tensor(np.array([12, 16, 19, 36, 40, 28]).reshape(-1, 2),requires_grad=False)
    anchors2 = to_tensor(np.array([36, 75, 76, 55, 72, 146]).reshape(-1, 2),requires_grad=False)
    anchors3 = to_tensor(np.array([142, 110, 192, 243, 459, 401]).reshape(-1, 2),requires_grad=False)
    num_anchors=len(anchors1)
    """Create YOLO_V4 model CNN body in Keras."""
    return Sequential(
            DarknetConv2D_BN_Mish((3, 3), 32,name='first_layer'),
            resblock_body(64, 1, all_narrow=False,name='block64'),
            resblock_body(128, 2,name='block128'),
            resblock_body(256, 8,name='block256'),
            ShortCut2d(
                {
                    1:Sequential(
                        resblock_body(512, 8,name='block512'),
                        ShortCut2d(
                            {
                                1:Sequential(
                                    resblock_body(1024, 4, name='block1024'),
                                    DarknetConv2D_BN_Leaky( (1,1), 512,name='pre_maxpool1'),
                                    DarknetConv2D_BN_Leaky( (3, 3),1024,name='pre_maxpool2'),
                                    DarknetConv2D_BN_Leaky((1,1),512,name='pre_maxpool3'),
                                    ShortCut2d(
                                        MaxPool2d((13,13),strides=(1,1),auto_pad=True),
                                        MaxPool2d((9,9), strides=(1, 1), auto_pad=True),
                                        MaxPool2d((5,5), strides=(1, 1), auto_pad=True),
                                        Identity(),
                                        mode='concate'
                                    ),
                                    DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_1'),
                                    DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_2'),
                                    DarknetConv2D_BN_Leaky((1, 1), 512,name='y_19',keep_output=True),
                                    DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y19_upsample'),
                                    Upsampling2d(scale_factor=2,name='y19_upsample'),
                                ),
                                0:DarknetConv2D_BN_Leaky((1, 1), 256)
                            },mode='concate'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y38_1'),
                        DarknetConv2D_BN_Leaky((3, 3),512,name='pre_y38_2'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='pre_y38_3'),
                        DarknetConv2D_BN_Leaky((3, 3),512,name='pre_y38_4'),
                        DarknetConv2D_BN_Leaky((1, 1),256,name='y_38',keep_output=True),
                        DarknetConv2D_BN_Leaky((1, 1),128,name='pre_y_38_upsample'),
                        Upsampling2d(scale_factor=2,name='y_38_upsample'),
                    ),
                    0:DarknetConv2D_BN_Leaky((1, 1), 128)
                },
                mode='concate'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate1'),
            DarknetConv2D_BN_Leaky((3, 3), 256,name='pre_y76_concate2'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate3'),
            DarknetConv2D_BN_Leaky((3, 3), 256,name='pre_y76_concate4'),
            DarknetConv2D_BN_Leaky((1, 1), 128,name='pre_y76_concate5'),
            ShortCut2d(
                #y76_output
                Sequential(
                    DarknetConv2D_BN_Leaky( (3, 3),256,name='pre_y76_output'),
                    DarknetConv2D( (1, 1),num_anchors * (num_classes + 5),use_bias=True,name='y76_output'),
                    YoloLayer(anchors=anchors1,num_classes=num_classes,grid_size=76, img_dim=image_size),
                name='y76_output'),
                # y38_output
                Sequential(
                    ShortCut2d(
                        DarknetConv2D_BN_Leaky((3, 3), 256, strides=(2, 2), auto_pad=False, padding=((1, 0), (1, 0)),name='y76_downsample'),
                        branch_from='y_38',mode='concate'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate1'),
                    DarknetConv2D_BN_Leaky((3, 3), 512,name='pre_y38_concate2'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate3'),
                    DarknetConv2D_BN_Leaky((3, 3), 512,name='pre_y38_concate4'),
                    DarknetConv2D_BN_Leaky((1, 1), 256,name='pre_y38_concate5'),
                    ShortCut2d(
                        Sequential(
                            DarknetConv2D_BN_Leaky((3, 3), 512, name='pre_y38_output'),
                            DarknetConv2D((1, 1), num_anchors * (num_classes + 5), use_bias=True, name='y38_output'),
                            YoloLayer(anchors=anchors2, num_classes=num_classes,grid_size=38,  img_dim=image_size),
                            name='y38_output'),

                        Sequential(
                            ShortCut2d(
                                DarknetConv2D_BN_Leaky((3, 3), 512, strides=(2, 2), auto_pad=False, padding=((1, 0), (1, 0)),name='y38_downsample'),
                                branch_from='y_19', mode='concate'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate1'),
                            DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_concate2'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate3'),
                            DarknetConv2D_BN_Leaky((3, 3), 1024,name='pre_y19_concate4'),
                            DarknetConv2D_BN_Leaky((1, 1), 512,name='pre_y19_concate5'),
                            Sequential(
                                DarknetConv2D_BN_Leaky((3, 3),1024,name='pre_y19_output'),
                                DarknetConv2D((1, 1), num_anchors * (num_classes + 5),use_bias=True,name='y19_output'),
                                YoloLayer(anchors=anchors3,num_classes=num_classes,grid_size=19, img_dim=image_size),
                            name='y19_output')),

                        mode='concate')
                )
                ,mode = 'concate')
    )


