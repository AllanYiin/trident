from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
import gc
import os
import time

import numpy as np
import torch

from trident.backend.common import *
from trident.backend.opencv_backend import image2array
from trident.backend.pytorch_backend import *
from trident.backend.pytorch_backend import Layer, Sequential, load, get_device, fix_layer
from trident.backend.pytorch_ops import *
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.data.vision_transforms import Normalize
from trident.layers.pytorch_activations import PRelu
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import ImageDetectionModel

__all__ = ['Pnet', 'Rnet', 'Onet', 'Mtcnn']

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


def p_net():
    return Sequential(
        Conv2d((3, 3), 10, strides=1, auto_pad=False, use_bias=True, name='conv1'),
        PRelu(num_parameters=1),
        MaxPool2d((2, 2), strides=2, auto_pad=False),
        Conv2d((3, 3), 16, strides=1, auto_pad=False, use_bias=True, name='conv2'),
        PRelu(num_parameters=1),
        Conv2d((3, 3), 32, strides=1, auto_pad=False, use_bias=True, name='conv3'),
        PRelu(num_parameters=1),
        ModuleDict(
            {'confidence': Conv2d((1, 1), 1, strides=1, auto_pad=False, use_bias=True, activation='sigmoid',
                                  name='conv4_1'),
             'box': Conv2d((1, 1), 4, strides=1, auto_pad=False, use_bias=True, name='conv4_2'),
             'landmark': Conv2d((1, 1), 10, strides=1, auto_pad=False, use_bias=True, name='conv4_3')},
            is_multicasting=True)
        , name='pnet')


def r_net():
    return Sequential(
        Conv2d((3, 3), 28, strides=1, auto_pad=False, use_bias=True, name='conv1'),
        PRelu(num_parameters=1),
        MaxPool2d((3, 3), strides=2, auto_pad=False),
        Conv2d((3, 3), 48, strides=1, auto_pad=False, use_bias=True, name='conv2'),
        PRelu(num_parameters=1),
        MaxPool2d((3, 3), strides=2, auto_pad=False),
        Conv2d((2, 2), 64, strides=1, auto_pad=False, use_bias=True, name='conv3'),
        PRelu(num_parameters=1),
        Flatten(),
        Dense(128, activation=None, use_bias=True, name='conv4'),
        PRelu(num_parameters=1),
        ModuleDict({
            'confidence': Dense(1, activation='sigmoid', use_bias=True, name='conv5_1'),
            'box': Dense(4, activation=None, use_bias=True, name='conv5_2'),
            'landmark': Dense(10, activation=None, use_bias=True, name='conv5_3')}, is_multicasting=True)
        , name='rnet')


def o_net():
    return Sequential(
        Conv2d((3, 3), 32, strides=1, auto_pad=False, use_bias=True, name='conv1'),
        PRelu(num_parameters=1),
        MaxPool2d((3, 3), strides=2, auto_pad=False),
        Conv2d((3, 3), 64, strides=1, auto_pad=False, use_bias=True, name='conv2'),
        PRelu(num_parameters=1),
        MaxPool2d((3, 3), strides=2, auto_pad=False),
        Conv2d((3, 3), 64, strides=1, auto_pad=False, use_bias=True, name='conv3'),
        PRelu(num_parameters=1),
        MaxPool2d((2, 2), strides=2, auto_pad=False),
        Conv2d((2, 2), 128, strides=1, auto_pad=False, use_bias=True, name='conv4'),
        PRelu(num_parameters=1),
        Flatten(),
        Dense(256, activation=None, use_bias=True, name='conv5'),
        PRelu(num_parameters=1),
        ModuleDict({
            'confidence': Dense(1, activation='sigmoid', use_bias=True, name='conv6_1'),
            'box': Dense(4, activation=None, use_bias=True, name='conv6_2'),
            'landmark': Dense(10, activation=None, use_bias=True, name='conv6_3')}, is_multicasting=True)
        , name='onet')


def Pnet(pretrained=True,
         input_shape=(3, 12, 12),
         freeze_features=True,
         **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 12, 12)
    pnet = ImageDetectionModel(input_shape=input_shape, output=p_net())
    pnet.preprocess_flow = [Normalize(0, 255), image_backend_adaption]
    if pretrained == True:
        download_model_from_google_drive('1w9ahipO8D9U1dAXMc2BewuL0UqIBYWSX', dirname, 'pnet.pth')
        recovery_model = fix_layer(load(os.path.join(dirname, 'pnet.pth')))
        pnet.model = recovery_model

    pnet.model.input_shape = input_shape
    pnet.model.to(_device)
    return pnet


def Rnet(pretrained=True,
         input_shape=(3, 24, 24),
         **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 24, 24)
    rnet = ImageDetectionModel(input_shape=input_shape, output=r_net())
    rnet.preprocess_flow = [Normalize(0, 255), image_backend_adaption]
    if pretrained == True:
        download_model_from_google_drive('1CH7z133_KrcWMx9zXAblMCV8luiQ3wph', dirname, 'rnet.pth')
        recovery_model = load(os.path.join(dirname, 'rnet.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model.to(_device)
        rnet.model = recovery_model
    return rnet


def Onet(pretrained=True,
         input_shape=(3, 48, 48),
         **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 48, 48)
    onet = ImageDetectionModel(input_shape=(3, 48, 48), output=o_net())
    onet.preprocess_flow = [Normalize(0, 255), image_backend_adaption]
    if pretrained == True:
        download_model_from_google_drive('1a1dAlSzJOAfIz77Ic38JMQJYWDG_b7-_', dirname, 'onet.pth')
        recovery_model = load(os.path.join(dirname, 'onet.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model.to(_device)
        onet.model = recovery_model
    return onet


class DetectorHead(Layer):
    def __init__(self, cellsize=12, threshold=0.5, min_size=5, **kwargs):
        super(DetectorHead, self).__init__(**kwargs)
        self.cellsize = cellsize
        self.detection_threshold = threshold
        self.min_size = min_size

        self._built = True

    def forward(self, input, **kwargs):
        boxprobs, boxregs, landscape = input.value_list
        boxprobs = boxprobs[0]
        height, width = boxprobs.shape[1:]
        if boxprobs.size(0) == 2:
            boxprobs = boxprobs[1:, :, :]
        strides = 2
        boxregs = boxregs[0]
        input_shape = boxprobs.size()
        grid = meshgrid(boxprobs.size(1), boxprobs.size(2))
        grid = grid.view(2, -1)
        score = boxprobs[0]
        y, x = torch.where(score >= self.detection_threshold)
        boxregs = boxregs.permute(1, 2, 0)

        score = score[(y, x)]
        reg = boxregs[(y, x)].transpose(1, 0)
        bb = torch.stack([x, y], dim=0)

        q1 = (strides * bb + 1)
        q2 = (strides * bb + self.cellsize - 1 + 1)

        w = q2[0, :] - q1[0, :] + 1
        h = q2[1, :] - q1[1, :] + 1

        b1 = q1[0, :] + reg[0, :] * w
        b2 = q1[1, :] + reg[1, :] * h
        b3 = q2[0, :] + reg[2, :] * w
        b4 = q2[1, :] + reg[3, :] * h

        boxs = torch.stack([b1, b2, b3, b4, score], dim=-1)
        # keep =torchvision.ops.boxes.remove_small_boxes(boxs[:,:4],min_size=self.min_size)
        # boxs=boxs[keep]
        # print('total {0} boxes cutoff={1} '.format(len(x), cutoff))
        if boxs is None or len(boxs.size()) == 0:
            return None
        elif len(boxs.size()) == 1:
            boxs = boxs.unsqueeze(0)
        return boxs


def remove_useless_boxes(boxes, image_size=None, min_size=5):
    height, width = image_size if image_size is not None else (None, None)

    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxes = boxes[area > min_size * min_size]
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    greater0 = x1.gt(0).float() * x2.gt(0).float() * y1.gt(0).float() * y1.gt(0).float()
    boxes = boxes[greater0 > 0]
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    w = (x2 - x1)
    boxes = boxes[w > 1]
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    h = (y2 - y1)
    boxes = boxes[h > 1]

    return boxes


class Mtcnn(ImageDetectionModel):
    def __init__(self, pretrained=True, min_size=10, detection_threshold=(0.4, 0.7, 0.9), nms_threshold=(0.9, 0.8, 0.5),
                 **kwargs):

        self.pnet = Pnet(pretrained=pretrained, input_shape=(3, 12, 12)).model
        self.rnet = Rnet(pretrained=pretrained, input_shape=(3, 24, 24)).model
        self.onet = Onet(pretrained=pretrained, input_shape=(3, 48, 48)).model
        super(Mtcnn, self).__init__(input_shape=(3, 12, 12), output=self.pnet)
        self.min_size = min_size
        self.detection_threshold = detection_threshold
        self.nms_threshold = nms_threshold
        self.preprocess_flow = [Normalize(0, 255), image_backend_adaption]

    def get_image_pyrimid(self, img, min_size=None, factor=0.709):
        if min_size is None:
            min_size = self.min_size
        min_face_area = (min_size, min_size)
        h = img.shape[0]
        w = img.shape[1]
        minl = np.amin([h, w])
        m = 12.0 / min_size
        minl = minl * m
        # 收集縮放尺度以及對應縮圖
        scales = []
        images = []
        factor_count = 0
        while minl >= 12:
            scales += [m * np.power(factor, factor_count)]
            scaled_img = rescale(scales[-1])(img.copy())
            images.append(scaled_img)
            minl = minl * factor
            factor_count += 1
        return images, scales

    def generate_bboxes(self, probs, offsets, scale, threshold):
        """
           基於Pnet產生初始的候選框
        """
        stride = 2
        cell_size = 12

        # 透過np.where挑出符合基於門檻值的特徵圖位置(xy座標)
        inds = where(probs > threshold)

        '''
        >>> a =np.array([[1,2,3],[4,5,6]])
        >>> np.where(a>1)
        (array([0, 0, 1, 1, 1]), array([1, 2, 0, 1, 2]))
        '''
        # 如果沒有區域滿足機率門檻值，則傳回空array
        if inds[0].size == 0:
            return np.array([])

        # 根據pnet輸出的offset區域產生對應的x1,y1,x2,y2座標
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

        offsets = stack([tx1, ty1, tx2, ty2], axis=-1)
        # 以及抓出對應的機率值
        score = probs[inds[0], inds[1]]

        # 由於Pnet輸入的是基於圖像金字塔縮放尺度對應的圖片，因此需要根據縮放尺度來調整候選框座標，以還原成真實圖片的尺度
        # 根據 候選框、機率值、offset來排列
        bounding_boxes = concate([
            round((stride * inds[1] + 1.0) / scale).expand_dims(-1),
            round((stride * inds[0] + 1.0) / scale).expand_dims(-1),
            round((stride * inds[1] + 1.0 + cell_size) / scale).expand_dims(-1),
            round((stride * inds[0] + 1.0 + cell_size) / scale).expand_dims(-1),
            score.expand_dims(-1), offsets
        ], axis=-1)
        print(bounding_boxes.shape)
        # 將bounding_boxes由原本[框屬性數量,框個數]的形狀轉置為[框個數，框屬性數量]

        return bounding_boxes

    def convert_to_square(self, bboxes):
        """Convert bounding boxes to a square form.
        Arguments:
            bboxes: a float numpy array of shape [n, 5].
        Returns:
            a float numpy array of shape [n, 5],
                squared bounding boxes.
        """

        square_bboxes = zeros_like(bboxes)
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0
        max_side = maximum(h, w)
        square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
        square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
        return square_bboxes

    # 校準候選框座標
    # 將offset對應至圖片長寬的線性縮放來獲得更新的候選框精調後座標
    def calibrate_box(self, bboxes, offsets):
        """Transform bounding boxes to be more like true bounding boxes.
        'offsets' is one of the outputs of the nets.
        Arguments:
            bboxes: a float numpy array of shape [n, 5].
            offsets: a float numpy array of shape [n, 4].
        Returns:
            a float numpy array of shape [n, 5].
        """

        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        # w [w_len, 1]
        w = expand_dims(w, 1)
        # h [h_len, 1]
        h = expand_dims(h, 1)

        translation = concate([w, h, w, h], axis=-1) * offsets
        bboxes[:, 0:4] = bboxes[:, 0:4] + translation
        return bboxes

    # 基於tensor計算nms
    def nms(self, box_scores, overlap_threshold=0.5, top_k=-1):
        """Non-maximum suppression.
        Arguments:
            box_scores: a float numpy array of shape [n, 5],
                where each row is (xmin, ymin, xmax, ymax, score).
            overlap_threshold: a float number.
        Returns:
            list with indices of the selected boxes
        """

        # 計算面積
        def area_of(left_top, right_bottom):
            """Compute the areas of rectangles given two corners.

            Args:
                left_top (N, 2): left top corner.
                right_bottom (N, 2): right bottom corner.

            Returns:
                area (N): return the area.
            """
            hw = right_bottom - left_top
            return clip(hw[..., 0], min=0) * clip(hw[..., 1], min=0)

        # 計算IOU(交集/聯集)
        def iou_of(boxes0, boxes1, eps=1e-5):
            """Return intersection-over-union (Jaccard index) of boxes.

            Args:
                boxes0 (N, 4): ground truth boxes.
                boxes1 (N or 1, 4): predicted boxes.
                eps: a small number to avoid 0 as denominator.
            Returns:
                iou (N): IoU values.
            """
            overlap_left_top = maximum(boxes0[..., :2], boxes1[..., :2])
            overlap_right_bottom = minimum(boxes0[..., 2:], boxes1[..., 2:])

            overlap_area = area_of(overlap_left_top, overlap_right_bottom)
            area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
            area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
            return overlap_area / (area0 + area1 - overlap_area + eps)

        # 如果沒有有效的候選區域則回傳空的清單
        box_scores = to_tensor(box_scores)
        if len(box_scores) == 0:
            return []
        score = box_scores[:, 4]
        boxes = box_scores[:, :4]
        # 存放過關的索引值
        picked = []
        # 依照機率信心水準升冪排序
        indexes = argsort(score, descending=False)

        while len(indexes) > 0:
            # 如此一來，最後一筆即是信心水準最高值
            # 加入至過關清單中
            current = indexes[-1]
            picked.append(current.item())

            # 計算其餘所有候選框與此當前框之間的IOU

            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            current_score = score[current]
            # 除了最後一筆以外的都是其餘框
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = iou_of(
                rest_boxes,
                expand_dims(current_box, axis=0),
            )
            # IOU未超過門檻值的表示未與當前框重疊，則留下，其他排除
            indexes = indexes[iou <= overlap_threshold]
        return picked

    def detect(self, image):
        """
        Arguments:
            image: 基於RGB排列的圖像(可以是路徑或是numpy向量)

        Returns:
            輸出為候選框以及對應的五官特徵點

        """
        # 暫存此原圖
        image = image2array(image)
        self.image = image

        self.height, self.width = image.shape[:2]
        min_length = min(self.height, self.width)

        # 第一階段: 候選 pnet
        bounding_boxes = []

        # 先計算圖像金字塔的各個縮放比率
        images, scales = self.get_image_pyrimid(image, min_size=self.min_size, factor=0.707)

        # 每個縮放比率各執行一次Pnet(全卷積網路)
        for img, scale in zip(images, scales):
            # 生成該尺度下的候選區域
            # 透過機率值門檻做篩選後再透過nms去重複
            boxes = self.run_first_stage(img, scale)
            print('Scale:', builtins.round(scale * 10000) / 10000.0, 'Scaled Images:', img.shape, 'bboxes:', len(boxes),
                  flush=True)
            if boxes.ndim == 1:
                boxes.expand_dims(0)
            bounding_boxes.append(boxes)

        # 將各個尺度所檢測到的候選區域合併後
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = concate(bounding_boxes, axis=0)
        print('totl bboxes:', len(bounding_boxes))

        # 將候選框的座標做一下校準後再進行nms
        bounding_boxes = self.calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        keep = self.nms(bounding_boxes[:, 0:5], self.nms_threshold[0])
        bounding_boxes = bounding_boxes[keep]

        # 將框盡可能調整成正方形
        bounding_boxes = self.convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = round(bounding_boxes[:, 0:4])
        print('totl bboxes after nms:', len(bounding_boxes))

        # # 將這階段的候選框圖輸出
        # pnet_img = self.image.copy()
        # for box in bounding_boxes[:, :4]:
        #     pnet_img = plot_one_box(box, pnet_img, (255, 128, 128), None, 1)

        # plt.figure(figsize=(16, 16))
        # plt.axis('off')
        # plt.imshow(pnet_img.astype(np.uint8))
        if is_gpu_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        # 第二階段: 精調 rnet
        # 將第一階段留下來的候選框區域挖下來，縮放成24*24大小，交給rnet做確認以及框座標精調
        img_boxes = self.get_image_boxes(bounding_boxes, size=24)
        print('RNet!')

        probs = []
        offsets = []

        if len(img_boxes) > 16:
            for i in range(len(img_boxes) // 16 + 1):
                if i * 16< len(img_boxes):
                    output = self.rnet(to_tensor(img_boxes[i * 16:(i + 1) * 16, :, :, :]))
                    probs.append(to_numpy(output['confidence']))
                    offsets.append(to_numpy(output['box']))
                    del output
            probs = np.concatenate(probs, axis=0)
            offsets =np.concatenate(offsets, axis=0)

        else:

            output = self.rnet(to_tensor(img_boxes))
            probs = to_numpy(output['confidence'])  # 形狀為 [n_boxes, 1]
            offsets = to_numpy(output['box'])  # 形狀為 [n_boxes, 4]

        # 根據機率門檻值排除機率值較低的框
        keep = np.where(probs[:, 0] > self.detection_threshold[1])[0]
        bounding_boxes = to_numpy(bounding_boxes)[keep]
        bounding_boxes=np.concatenate([bounding_boxes[:,:4],probs[keep, 0].reshape((-1,1))],axis=1)
        #bounding_boxes[:, 4] = probs[keep, 0].reshape((-1,))
        offsets = offsets[keep]
        print('totl bboxes:', len(bounding_boxes))

        # 將框的座標做精調後再進行nms
        bounding_boxes = self.calibrate_box(bounding_boxes, offsets)
        keep = self.nms(bounding_boxes, self.nms_threshold[1])
        bounding_boxes = bounding_boxes[keep]

        # 將框盡可能調整成正方形
        bounding_boxes = self.convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = round(bounding_boxes[:, 0:4]).copy()
        print('totl bboxes after nms:', len(bounding_boxes))

        # # 將這階段的候選框圖輸出
        # rnet_img = self.image.copy()
        # for i in range(bounding_boxes.shape[0]):
        #     box = bounding_boxes[i, :4]
        #     rnet_img = plot_one_box(box, rnet_img, (255, 128, 128), None, 2)

        # plt.figure(figsize=(16, 16))
        # plt.axis('off')
        # plt.imshow(rnet_img.astype(np.uint8))
        if is_gpu_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        # 第三階段: 輸出 onet

        img_boxes = self.get_image_boxes(bounding_boxes, size=48)
        if len(img_boxes) == 0:
            return [], []
        print('ONet!')


        probs = []
        offsets = []
        landmarks = []

        if len(img_boxes) > 16:
            for i in range(len(img_boxes) //16 + 1):
                if i * 16 < len(img_boxes):
                    output = self.onet(to_tensor(img_boxes[i * 16:(i + 1) * 16, :, :, :]))
                    probs.append(output['confidence'].copy())
                    offsets.append(output['box'].copy())
                    landmarks.append(output['landmark'].copy())
                    del output
            probs = concate(probs, axis=0)
            offsets = concate(offsets, axis=0)
            landmarks = concate(landmarks, axis=0)

        else:

            output = self.onet(to_tensor(img_boxes))
            probs = output['confidence']  # 形狀為 [n_boxes, 1]
            offsets = output['box']  # 形狀為 [n_boxes, 4]
            # 只有這一階段需要檢視人臉特徵點
            landmarks = output['landmark']  # 形狀為 [n_boxes, 10]

        # 根據機率門檻值排除機率值較低的框
        keep = where(probs[:, 0] > self.detection_threshold[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 0].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]
        print('totl bboxes:', len(bounding_boxes))

        # 將框的座標做精調後計算對應的臉部特徵點位置，然後再進行nms
        bounding_boxes = self.calibrate_box(bounding_boxes, offsets)

        # 根據模型輸出計算人臉特徵點
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = expand_dims(xmin, 1) + expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = expand_dims(ymin, 1) + expand_dims(height, 1) * landmarks[:, 5:10]

        # 做最後一次nms
        keep = self.nms(bounding_boxes, self.nms_threshold[2])
        print('totl bboxes after nms:', len(bounding_boxes))
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]
        probs = probs[keep]

        # # 將這階段的候選框圖輸出
        # onet_img = self.image.copy()
        # for i in range(bounding_boxes.shape[0]):
        #     box = bounding_boxes[i, :4]
        #     onet_img = plot_one_box(box, onet_img, (255, 128, 128), None, 2)
        # for landmark in landmarks:
        #     landmarks_x = landmark[:5]
        #     landmarks_y = landmark[5:]
        #     for i in range(5):
        #         cv2.circle(onet_img, (int(landmarks_x[i]), int(landmarks_y[i])), 2, (255, 128, 255), 1)

        # plt.figure(figsize=(16, 16))
        # plt.axis('off')
        # plt.imshow(onet_img.astype(np.uint8))
        gc.collect()

        return self.image.copy(), bounding_boxes, probs, landmarks

        # 執行第一階段

    def run_first_stage(self, img, scale):
        """Run P-Net, generate bounding boxes, and do NMS.

        Arguments:
            img: an instance of PIL.Image.
            scale: a float number,
                scale width and height of the image by this number.


        Returns:
            a float numpy array of shape [n_boxes, 9],
                bounding boxes with scores and offsets (4 + 1 + 4).
        """

        sh, sw = img.shape[:2]
        width, height = self.width, self.height
        threshold = self.detection_threshold[0]

        # 將圖像做基礎處理後送入pnet
        for transform in self.preprocess_flow:
            img = transform(img)

        output = self.pnet(expand_dims(to_tensor(img), 0))

        probs = output['confidence'][0, 0, :, :]
        offsets = output['box']

        # 根據全卷積網路輸出結果計算對應候選框座標
        boxes = self.generate_bboxes(probs, offsets, scale, threshold)

        # 在此尺度的候選框先做一次nms已有效減少候選框數量，這樣後續rnet, onet才不會GPU爆掉。
        keep = self.nms(boxes[:, 0:5], overlap_threshold=self.nms_threshold[0])
        boxes = boxes[keep].copy()
        del output
        return boxes

        # 根據候選框座標至原圖挖取人臉圖像，已進行後續階段

    def get_image_boxes(self, bounding_boxes, size=24):
        """Cut out boxes from the image.
        Arguments:
            bounding_boxes: a float numpy array of shape [n, 5].
            size: an integer, size of cutouts.
        Returns:
            a float numpy array of shape [n, 3, size, size].
        """

        num_boxes = len(bounding_boxes)
        height, width = self.image.shape[:2]

        # 宣告空白的img_boxes物件用來存放挖取的人臉圖像區域
        img_boxes = np.zeros((num_boxes, 3, size, size), "float32")
        n = 0
        for i in range(num_boxes):
            x1, y1, x2, y2 = bounding_boxes[i][:4]
            try:
                # 根據x1,y1,x2,y2座標，且座標必須大於零且小於等於圖像長寬的原則來挖取人臉區域
                yy1 = int(builtins.max(y1, 0))
                yy2 = int(builtins.min(y2, self.height))
                xx1 = int(builtins.max(x1, 0))
                xx2 = int(builtins.min(x2, self.width))
                img_box = self.image[yy1:yy2, xx1:xx2, :]
                if img_box.shape[0] != img_box.shape[1]:
                    # 若挖出非正方形則補滿為正方形
                    max_length = builtins.max(list(img_box.shape[:2]))
                    new_img_box = np.zeros((max_length, max_length, 3))
                    new_img_box[0:img_box.shape[0], 0:img_box.shape[1], :] = img_box
                    img_box = new_img_box
                # 將正方形區域縮放後，經過預處理self.preprocess_flow後再塞入img_boxes
                img_box = resize((size, size), keep_aspect=True)(img_box)

                for transform in self.preprocess_flow:
                    img_box = transform(img_box)
                img_boxes[i, :, :, :] = img_box
                n += 1
            except:
                pass
        # 列印一下成功挖取的區域數量(有可能座標本身不合理造成無法成功挖取)
        print(n, 'image generated')
        return img_boxes

    def infer_single_image(self, img, **kwargs):
        if self.model.built:
            self.model.to(self.device)
            self.model.eval()
        image, boxes, probs, landmarks = self.detect(img)
        return image, to_numpy(boxes), to_numpy(probs).astype(np.int32), to_numpy(landmarks)

    def infer_then_draw_single_image(self, img):
        start_time = time.time()
        rgb_image, boxes, probs, landmark = self.infer_single_image(img)
        if boxes is not None and len(boxes) > 0:
            boxes = np.round(boxes).astype(np.int32)
            if boxes.ndim == 1:
                boxes = np.expand_dims(boxes, 0)

            print(img, time.time() - start_time)
            pillow_img = array2image(rgb_image.copy())

            print(boxes, labels, flush=True)
            if len(boxes) > 0:
                for m in range(len(boxes)):
                    this_box = boxes[m]
                    this_label = 1
                    if int(this_label) > 0:
                        thiscolor = self.palette[1]
                        print('face', this_box, probs[m], flush=True)
                        pillow_img = plot_bbox(this_box, pillow_img, thiscolor, self.class_names[
                            int(this_label)] if self.class_names is not None else '', line_thickness=2)
            rgb_image = np.array(pillow_img.copy())

        return rgb_image, boxes, probs, landmark
