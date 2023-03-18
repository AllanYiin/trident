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
    from  trident.backend.pytorch_ops import *
else:
    from trident.backend.tensorflow_ops import *



__all__ = ['xywh2xyxy', 'xyxy2xywh','box_area','plot_one_box','convert_to_square']


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
        class_info = None
        if boxes.shape[-1] >4:
            class_info = boxes[:, 4:]
            boxes = boxes[:, :4]
        # Multiple boxes given as a 2D ndarray
        boxes[:, 0:2] =boxes[:, 0:2]- boxes[:, 2:4] / 2
        boxes[:, 2:4] =boxes[:, 2:4] + boxes[:, 0:2]

        height, width = np.inf, np.inf
        if image_size is not None:
            height, width = image_size
        boxes[:, :4] = boxes[:, :4]
        boxes[:, 0] = np.clip(boxes[:, 0], a_min=0, a_max=width)
        boxes[:, 1] = np.clip(boxes[:, 1], a_min=0, a_max=height)
        boxes[:, 2] = np.clip(boxes[:, 2], a_min=0, a_max=width)
        boxes[:, 3] = np.clip(boxes[:, 3], a_min=0, a_max=height)
        if class_info is not None:
            boxes = np.concatenate([boxes, class_info], axis=-1)
        return boxes
    elif is_tensor(boxes) :
        class_info = None
        if boxes.shape[-1] >4:
            class_info = boxes[:, 4:]
            boxes = boxes[:, :4]
        x1y1= clip(boxes[:, 0:2] -boxes[:, 2:4] /2,0)
        x2y2=clip(x1y1+ boxes[:, 2:4],0)
        if class_info is not None:
            boxes = concate([x1y1,x2y2, class_info], axis=-1)
        else:
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
        if boxes.shape[-1]>4:
            return np.concatenate([(boxes[:, 2:4] + boxes[:, 0:2]) / 2.0,  # cx, cy
                                   boxes[:, 2:4] - boxes[:, 0:2],boxes[:, 4:]], 1)  # w, h
        elif boxes.shape[-1]==4:
            return np.concatenate([(boxes[:, 2:4] + boxes[:, 0:2]) / 2.0,  # cx, cy
                            boxes[:, 2:4] - boxes[:, 0:2]], 1)  # w, h
    elif is_tensor(boxes):
        if boxes.ndim==1:
            boxes=expand_dims(boxes,0)
        if boxes.shape[-1] > 4:
            return concate([(boxes[:, 2:4] + boxes[:, 0:2]) / 2,  # cx, cy
                            boxes[:, 2:4] - boxes[:, 0:2],boxes[:, 4:]], 1)  # w, h
        else:
            return concate([(boxes[:, 2:4] + boxes[:, 0:2])/2,  # cx, cy
                         boxes[:, 2:4] - boxes[:, 0:2]], 1)  # w, h
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

#
# def box_iou(boxes1, boxes2):
#     """Calculate the ious between each bbox of bboxes1 and bboxes2.
#
#     Args:
#         boxes1(ndarray/ tensor): shape (n, 4)
#         boxes2(ndarray/ tensor): shape (k, 4)
#
#     Returns:
#         iou(ndarray/ tensor): shape (n, k)
#         union (ndarray/ tensor): shape (n, k)
#
#     Examples:
#         >>> box_iou(to_tensor(np.array([[104, 85, 200, 157]])).cpu(),to_tensor(np.array([[110, 80, 195, 153]])).cpu())
#         (tensor([[0.7878]]), tensor([[7337]]))
#         >>> box_iou(np.array([[104, 85, 200, 157]]),np.array([[110, 80, 195, 153]]))
#         (array([[7.8779e-01]]), array([[7.3370e+03]]))
#        >>> box_iou(to_tensor(np.array([[104, 85, 200, 157]])).cpu(),to_tensor(np.array([[10, 20, 45, 73]])).cpu())
#         (tensor([[0.]]), tensor([[8767]]))
#         >>> box_iou(np.array([[104, 85, 200, 157]]),np.array([[10, 20, 45, 73]]))
#         (array([[0.0000e+00]]), array([[8.7670e+03]]))
#
#     """
#     area1 = box_area(boxes1)
#     area2 = box_area(boxes2)
#
#     lt = maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
#
#     wh = clip(rb- lt,min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1] # [N,M]
#
#     union = area1[:, None] + area2 - inter
#     iou = inter / union
#     return iou, union
#
#
#
#
# def box_giou(boxes1, boxes2):
#     """
#     Generalized IoU from https://giou.stanford.edu/
#     The boxes should be in [x0, y0, x1, y1] format
#     Returns a [N, M] pairwise matrix, where N = len(boxes1)
#     and M = len(boxes2)
#
#     Examples:
#         >>> box_giou(to_tensor(np.array([[104, 85, 200, 157]])).cpu(),to_tensor(np.array([[110, 80, 195, 153]])).cpu())
#         tensor([[0.7803]])
#         >>> box_giou(np.array([[104, 85, 200, 157]]),np.array([[110, 80, 195, 153]]))
#         array([[7.8035e-01]])
#         >>> box_giou(to_tensor(np.array([[104, 85, 200, 157]])).cpu(),to_tensor(np.array([[10, 20, 45, 73]])).cpu())
#         tensor([[-0.6632]])
#         >>> box_giou(np.array([[104, 85, 200, 157]]),np.array([[10, 20, 45, 73]]))
#         array([[-6.6320e-01]])
#
#     """
#     # degenerate boxes gives inf / nan results
#     # so do an early check
#     assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
#     assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
#     iou, union = box_iou(boxes1, boxes2)
#
#     lt = minimum(boxes1[:, None, :2], boxes2[:, :2])
#     rb =maximum(boxes1[:, None, 2:], boxes2[:, 2:])
#
#     wh = clip(rb - lt,min=0)  # [N,M,2]
#     area = wh[:, :, 0] * wh[:, :, 1]
#
#     return iou - (area - union) / area
#
# bbox_giou=box_giou
# bbox_giou_numpy=box_giou

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

        xx1 = np.maximum(x1[i], x1[sorted_index[:last]])
        yy1 = np.maximum(y1[i], y1[sorted_index[:last]])
        xx2 = np.minimum(x2[i], x2[sorted_index[:last]])
        yy2 = np.minimum(y2[i], y2[sorted_index[:last]])
        # compute the width and height of the bounding box
        xc = (np.abs(xx1 + xx2) / 2 - (x1[sorted_index[:last]] + x2[sorted_index[:last]]) / 2)
        yc = (np.abs(yy1 + yy2) / 2 - (y1[sorted_index[:last]] + y2[sorted_index[:last]]) / 2)

        w = np.clip(xx2 - xx1 + 1,a_min=0,a_max=None)
        h = np.clip(yy2 - yy1 + 1,a_min=0,a_max=None)
        inter = w * h

        # any of happends will dedup
        IoU = inter / (area[i] + area[sorted_index[:last]] - inter)
        sorted_index = sorted_index[IoU <= threshold]

    if boxes is not None and len(pick) > 0:
        return pick
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

def landmark106_to_68(landmarks):
    landmark106to68 =to_tensor( [1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17,  # face17
                       43, 48, 49, 51, 50,  # left brow点
                       102, 103, 104, 105, 101,  # right brow 5
                       72, 73, 74, 86, 78, 79, 80, 85, 84,  # nose 9
                       35, 41, 42, 39, 37, 36,  # left eye 6
                       89, 95, 96, 93, 91, 90,  # right eye 6
                       52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54  # mouth 20
                       ]).long()
    return landmarks[landmark106to68]



def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.
    Args:
        bboxes: a float numpy array of shape [n, 5].
    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.

    Examples:
        >>> convert_to_square(to_tensor(np.array([[104, 85, 200, 157]]))).cpu()
        tensor([[104,  73, 200, 169]])
        >>> convert_to_square(np.array([[104, 85, 200, 157]]))
        array([[104,  73, 200, 169]])
    """

    h = bboxes[:, 3] - bboxes[:, 1]
    w = bboxes[:, 2] - bboxes[:, 0]
    max_len = maximum(w, h)

    bboxes[:, 0] = round(bboxes[:, 0] - 0.5 * (max_len - w))
    bboxes[:, 1] = round(bboxes[:, 1] - 0.5 * (max_len - h))
    bboxes[:, 2] = bboxes[:, 0] + max_len
    bboxes[:, 3] = bboxes[:, 1] + max_len
    return bboxes

