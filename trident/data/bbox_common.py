from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# import pyximport; pyximport.install()
# import cython_bbox
# import cython_nms
from trident.backend.common import *

_session=get_session()
_backend=_session.backend
_image_backend=_session.image_backend

if _image_backend=='opencv':
    from trident.backend.opencv_backend import *
else:
    from trident.backend.pillow_backend import *

if _backend=='pytorch':
    from trident.backend.pytorch_backend import *
    from  trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *



__all__ = ['nms', 'xywh2xyxy', 'xyxy2xywh','bbox_iou','bbox_diou','bbox_giou','bbox_giou_numpy','plot_one_box']


def plot_one_box(box, img, color=None, label=None, line_thickness=None):
    import cv2
    # Plots one bounding box on image img
    tl = line_thickness if  line_thickness is not None else  round(0.15*(box[2]-box[0]) )# round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color=color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,  [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def xywh2xyxy(boxes,image_size=None):
    """
    Args:
        boxes (tensor or ndarray):
            boxes  with xywh  (centerx,centery,width, height) format
            boxes shape should be [n,m] m>=4
        image_size (size): (height, width)
    Returns
        xyxy (x1,y1,x2,y2)
    """
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(boxes, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(boxes) == 4
        cx, cy ,w,h= boxes[0], boxes[1], boxes[2], boxes[3]
        x1=cx-0.5*w
        y1=cy-0.5*h
        x2 = cx+0.5*w
        y2=cy+0.5*h
        if len(boxes)>4:
            boxlist=[x1, y1, x2, y2]
            boxlist.extend(boxes[4:])
            return np.array(boxlist)
        return np.array([x1, y1, x2, y2])
    elif isinstance(boxes, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        boxes[:, 0:2] =boxes[:, 0:2]- boxes[:, 2:4] / 2
        boxes[:, 2:4] =boxes[:, 2:4] + boxes[:, 0:2]

        height, width = np.inf, np.inf
        if image_size is not None:
            height, width = image_size
        boxes[:, :4] = np.round(boxes[:, :4], 0)
        boxes[:, 0] = np.clip(boxes[:, 0], a_min=0, a_max=width)
        boxes[:, 1] = np.clip(boxes[:, 1], a_min=0, a_max=height)
        boxes[:, 2] = np.clip(boxes[:, 2], a_min=0, a_max=width)
        boxes[:, 3] = np.clip(boxes[:, 3], a_min=0, a_max=height)
        return boxes
    elif is_tensor(boxes) :
        x1y1= clip(round(boxes[:, 0:2] -boxes[:, 2:4] /2,0),0)
        x2y2=clip(round(x1y1+ boxes[:, 2:4],0),0)
        boxes=concate([x1y1,x2y2],axis=-1)
        return boxes

    else:
        raise TypeError('Argument xywh must be a list, tuple, numpy array or tensor.')


def xyxy2xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(boxes, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(boxes) == 4 or len(boxes) == 5
        x1, y1 = boxes[0], boxes[1]
        w = boxes[2] - x1 + 1
        h = boxes[3] - y1 + 1
        return np.array([x1, y1, w, h])
    elif isinstance(boxes, np.ndarray):
        if boxes.ndim==1:
            boxes=np.expand_dims(boxes,0)
        return np.concatenate([(boxes[:, 2:4] + boxes[:, 0:2]) / 2,  # cx, cy
                        boxes[:, 2:4] - boxes[:, 0:2]], 1)  # w, h
    elif is_tensor(boxes):
        if boxes.ndim==1:
            boxes=expand_dims(boxes,0)
        return concate([(boxes[:, 2:4] + boxes[:, 0:2])/2,  # cx, cy
                     boxes[:, 2:4] - boxes[:, 0:2]], 1)  # w, h
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def clip_boxes_to_image(boxes, size):

    """
    Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image
    Returns:
        clipped_boxes (Tensor[N, 4])

    """
    height,width=size
    if len(boxes)>0:
        boxes[:,0]= clip(boxes[:,0],min=0, max=width)
        boxes[:,1]= clip(boxes[:,1],min=0, max=height)
        boxes[:,2]= clip(boxes[:,2],min=0, max=width)
        boxes[:,3]= clip(boxes[:,3],min=0,max=height)
    return boxes


def nms(boxes, threshold):
    """
        non max suppression

    Args
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'

    Returns:
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list


    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []
    box_len = len(boxes)
    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    sorted_index = np.argsort(score)
    # keep looping while some indexes still remain in the indexes list
    adjust_bbox = []
    while len(sorted_index) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(sorted_index) - 1
        i = sorted_index[-1]
        pick.append(int(i))
        sorted_index = sorted_index[:-1]
        if len(sorted_index) == 1:
            break
        # sorted_index = sorted_index[:-1]
        xx1 = np.max(x1[i], x1[sorted_index[:last]])
        yy1 = np.max(y1[i], y1[sorted_index[:last]])
        xx2 = np.min(x2[i], x2[sorted_index[:last]])
        yy2 = np.min(y2[i], y2[sorted_index[:last]])
        # compute the width and height of the bounding box
        xc = ((xx1 + xx2).abs_() / 2 - (x1[sorted_index[:last]] + x2[sorted_index[:last]]) / 2)
        yc = ((yy1 + yy2).abs_() / 2 - (y1[sorted_index[:last]] + y2[sorted_index[:last]]) / 2)

        w = (xx2 - xx1 + 1).clamp_(0)
        h = (yy2 - yy1 + 1).clamp_(0)
        inter = w * h

        # any of happends will dedup
        IoU = inter / (area[i] + area[sorted_index[:last]] - inter)
        sorted_index = sorted_index[IoU <= threshold]

    if boxes is not None and len(pick) > 0:
        try:
            out = boxes[pick]
            if len(out.size()) == 1:
                out = out.unsqueeze(0)
                if len(out) == 0:
                    return None
            return out,pick
        except Exception as e:
            print(e)
            print('box_len', box_len)
            print('pick', len(pick))
            print('boxes', len(boxes))

    return None



def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, 0:2], b[:, 0:2])
    rb = np.minimum(a[:, np.newaxis, 2:4], b[:, 2:4])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, 0:2], axis=1)
    area_b = np.prod(b[:, 2:4] - b[:, 0:2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, 0:2], b[:, 0:2])
    rb = np.minimum(a[:, np.newaxis, 2:4], b[:, 2:4])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:4] - a[:, 0:2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)



def bbox_iou(bboxes1, bboxes2, mode='iou', allow_neg=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        allow_neg ():
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if (bboxes1 is None or len(bboxes1) == 0) and (bboxes2 is None or len(bboxes2) == 0):
        return np.ones(1)

    elif bboxes1 is None or len(bboxes1)==0 or bboxes2 is None or len(bboxes2)==0:
        return np.zeros(1)

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        if not allow_neg:
            overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0)
        else:
            overlap = (x_end - x_start + 1) * (y_end - y_start + 1)
            flag = np.ones(overlap.shape)
            flag[x_end - x_start + 1 < 0] = -1.
            flag[y_end - y_start + 1 < 0] = -1.
            overlap = flag * np.abs(overlap)

        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def bbox_giou_numpy(bboxes1, bboxes2):
    """Calculate the gious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)

    Returns:
        gious(ndarray): shape (n, k)
    """


    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        x_min = np.minimum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        y_min = np.minimum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        x_max = np.maximum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        y_max = np.maximum(bboxes1[i, 3], bboxes2[:, 3])

        overlap = np.clip(np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0),1e-8,np.inf)
        closure = np.clip(np.maximum(x_max - x_min + 1, 0) * np.maximum(y_max - y_min + 1, 0),1e-8,np.inf)

        union =np.clip( area1[i] + area2 - overlap,1e-8,np.inf)

        ious[i, :] = overlap / union - (closure - union) / closure
    if exchange:
        ious = ious.T
    return ious

def bbox_giou(bboxes1, bboxes2):
    """
        Calculate GIoU loss on anchor boxes
        Reference Paper:
            "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
            https://arxiv.org/abs/1902.09630

    Args:
        bboxes1: tensor, shape=(n, 4), xyxy
        bboxes2: tensor, shape=(n, 4), xyxy

    Returns:
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)

    """
    bboxes1 = bboxes1.float()
    bboxes2 = bboxes2.float()
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    # if (bboxes1 is None or len(bboxes1) == 0) and (bboxes2 is None or len(bboxes2) == 0):
    #     return ones((rows, cols))
    #
    # elif bboxes1 is None or len(bboxes1)==0 or bboxes2 is None or len(bboxes2)==0:
    #     return zeros((rows, cols))
    #

    ious = zeros((rows, cols))
    ious.requires_grad=True
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = zeros((cols, rows),requires_grad=True)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = maximum(bboxes1[i, 0], bboxes2[:, 0])
        x_min = minimum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = maximum(bboxes1[i, 1], bboxes2[:, 1])
        y_min = minimum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = minimum(bboxes1[i, 2], bboxes2[:, 2])
        x_max = maximum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = minimum(bboxes1[i, 3], bboxes2[:, 3])
        y_max = maximum(bboxes1[i, 3], bboxes2[:, 3])

        overlap =clip( maximum(x_end - x_start + 1, 0) * maximum(y_end - y_start + 1, 0),min=1e-8)
        closure = clip(maximum(x_max - x_min + 1, 0) * maximum(y_max - y_min + 1, 0),min=1e-8)

        union = clip(area1[i] + area2 - overlap,min=1e-8)

        ious[i, :] = overlap / union - (closure - union) / closure
    if exchange:
        ious = ious.T
    return ious

def bbox_diou(bboxes1, bboxes2):
    """
    Calculate DIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Args:
        bboxes1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        bboxes2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    Returns:
        diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)

    """

    b1_mins =  bboxes1[..., :2]
    b1_maxes =  bboxes1[..., 2:4]
    b1_wh=b1_maxes-b1_mins


    b2_mins =  bboxes2[..., :2]
    b2_maxes = bboxes2[..., 2:4]
    b2_wh=b2_maxes-b2_mins

    intersect_mins = maximum(b1_mins, b2_mins)
    intersect_maxes = minimum(b1_maxes, b2_maxes)
    intersect_wh = maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + epsilon())

    # box center distance
    center_distance = reduce_sum(square((b1_maxes+b1_mins)/2 - (b2_maxes+b2_mins)/2), axis=-1)
    # get enclosed area
    enclose_mins = minimum(b1_mins, b2_mins)
    enclose_maxes = maximum(b1_maxes, b2_maxes)
    enclose_wh =maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = reduce_sum(square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * center_distance/ (enclose_diagonal + epsilon())

    # calculate param v and alpha to extend to CIoU
    #v = 4*K.square(tf.math.atan2(b1_wh[..., 0], b1_wh[..., 1]) - tf.math.atan2(b2_wh[..., 0], b2_wh[..., 1])) / (math.pi * math.pi)
    #alpha = v / (1.0 - iou + v)
    #diou = diou - alpha*v

    diou = expand_dims(diou, -1)
    return diou

#
# def soft_nms(boxes, sigma=0.5, overlap_threshold=0.3, score_threshold=0.001, method='linear'):
#     """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
#     if boxes.shape[0] == 0:
#         return boxes, []
#
#     methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
#     assert method in methods, 'Unknown soft_nms method: {}'.format(met