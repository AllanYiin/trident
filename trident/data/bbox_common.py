import numpy as np

# import pyximport; pyximport.install()
# import cython_bbox
# import cython_nms
from ..backend.common import *

_session=get_session()
_backend=_session.backend
_image_backend=_session.image_backend

if _image_backend=='opencv':
    from ..backend.opencv_backend import *
else:
    from ..backend.pillow_backend import *


__all__ = ['nms','xywh_to_xyxy','xyxy_to_xywh']



def xywh_to_xyxy(xywh):
    '''

    Parameters
    ----------
    xywh (centerx,centery,width, height)

    Returns
    -------
    xyxy (x1,y1,x2,y2)

    '''
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def clip_boxes_to_image(boxes, size):

    """
    Clip boxes so that they lie inside an image of size `size`.
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image
    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    height,width=size
    boxes[:,0]= boxes[:,0].clamp(min=0, max=width)
    boxes[:,1]= boxes[:,1].clamp(min=0, max=height)
    boxes[:,2]= boxes[:,2].clamp(min=0, max=width)
    boxes[:,3]= boxes[:,3].clamp(min=0, max=height)
    return boxes


def nms(boxes, threshold):
    """
        non max suppression

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
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

    sorted_index = np.argsort(score, descending=False)
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

#
# def soft_nms(boxes, sigma=0.5, overlap_threshold=0.3, score_threshold=0.001, method='linear'):
#     """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
#     if boxes.shape[0] == 0:
#         return boxes, []
#
#     methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
#     assert method in methods, 'Unknown soft_nms method: {}'.format(method)
#
#     boxes, keep = cython_nms.soft_nms(
#         np.ascontiguousarray(boxes, dtype=np.float32),
#         np.float32(sigma),
#         np.float32(overlap_threshold),
#         np.float32(score_threshold),
#         np.uint8(methods[method])
#     )
#     return boxes, keep
