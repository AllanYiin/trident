import builtins
import inspect
import math

from PIL import Image
import numbers
import random
import copy
from typing import Sequence, Tuple, Dict, Union, Optional

import cv2
import numpy as np
from skimage.filters import threshold_minimum, threshold_local, threshold_isodata, threshold_yen

from trident.backend.common import *
from trident.data.bbox_common import box_area
from trident.backend.tensorspec import ObjectType, get_signature
from trident.backend.tensorspec import TensorSpec
import matplotlib.pyplot as plt

np.long=np.int_
np.bool=np.bool_
np.uint8=np.ubyte
np.float32=np.single


if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *
from trident.data.transform import VisionTransform

__all__ = ['Resize', 'Unresize', 'ShortestEdgeResize', 'Rescale', 'RandomCrop', 'RandomRescaleCrop', 'RandomCenterCrop',
           'RandomMultiScaleImage', 'RandomTransform', 'RandomTransformAffine',
           'AdjustBrightness', 'AdjustContrast', 'AdjustSaturation', 'AddNoise', 'AdjustHue', 'RandomAdjustHue',
           'RandomAdjustBrightness', 'RandomAdjustContrast',
           'RandomAdjustSaturation', 'AutoLevel', 'RandomLighting', 'GrayMixRGB',
           'Normalize', 'Unnormalize', 'CLAHE', 'Lighting', 'HorizontalFlip', 'RandomMirror', 'AdjustGamma',
           'RandomBlur', 'RandomAdjustGamma', 'Blur', 'InvertColor',
           'RandomInvertColor', 'GrayScale', 'RandomGrayScale', 'RandomGridMask', 'GridMask', 'ToLowResolution',
           'ImageDilation', 'ImageErosion', 'ErosionThenDilation', 'DilationThenErosion', 'AdaptiveBinarization',
           'SaltPepperNoise', 'RandomErasing', 'ToRGB', 'ImageMosaic', 'DetectionMixup']

def _check_float_dtype(arr):
    return arr.astype(np.float32)

def _check_pixel_value_range(arr):
    return np.clip(arr,0,255)


def _get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def _get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final

def _check_range_tuple(value):
    return isinstance(value, (tuple, list)) and len(value) == 2 and all(
        [isinstance(v, numbers.Number) for v in value])


def randomize_with_validate(keep_prob=0.5, valid_range=None, effectless_value=None, **kwargs):
    def randomize_wrapper(cls):
        class Wrapper:
            def __init__(self, **kwargs):
                self._args = None
                self.rangs = OrderedDict()
                self.other_args = None
                self.keep_prob = keep_prob
                self.valid_range = None
                if isinstance(valid_range, (tuple, list)) and len(valid_range) == 2 and all(
                        [isinstance(t, numbers.Number) for t in valid_range]):
                    self.valid_range = valid_range

                self.effectless_value = None
                if isinstance(effectless_value, numbers.Number):
                    self.effectless_value = float(effectless_value)

                # if 'scale' in kwargs and len(kwargs[ 'scale'])==2 and len([k for k,v in kwargs.items() if len(v)==2 ])==1 :
                #     self.valid_range=kwargs[ 'scale']
                #     kwargs.pop('scale')
                #

                self._args = get_signature(cls.__init__, name=cls.__name__)

                candidate_ranges = dict(
                    [(k.replace('_range', '').replace('_scale', ''), v) for k, v in kwargs.items() if
                     isinstance(v, tuple) and len(v) == 2])
                for k, v in candidate_ranges.items():
                    value_range = list(v)
                    if value_range[0] < valid_range[0]:
                        value_range[0] = valid_range[0]
                    if value_range[1] > valid_range[1]:
                        value_range[1] = valid_range[1]
                    if value_range[1] == value_range[0]:
                        value_range = valid_range

                    if k in self._args.inputs:
                        self.rangs[k] = tuple(value_range)
                    elif len(candidate_ranges) == 1 and 'value' in self._args.inputs:
                        self.rangs['value'] = tuple(value_range)
                    elif len(candidate_ranges) == 1 and 'scale' in self._args.inputs:
                        self.rangs['scale'] = tuple(value_range)
                    elif len(candidate_ranges) == 1 and 'ksize' in self._args.inputs:
                        self.rangs['ksize'] = tuple(value_range)
                    else:
                        keys = [k for k in self._args.inputs.key_list if k not in ['self', 'name',
                                                                                   'keep_ratio'] and 'size' not in k and 'shape' not in k and isinstance(
                            self._args.inputs[k].default, numbers.Number)]
                        if len(keys) > 0:
                            self.rangs[keys[0]] = tuple(value_range)

                rangs = OrderedDict([(k, random.uniform(*v)) for k, v in self.rangs.items()])
                other_args = OrderedDict([(k, v.default) for k, v in self._args.inputs.items() if k not in self.rangs])
                if 'name' in other_args:
                    other_args.pop('name')

                self.other_args = other_args

                if len(rangs.values()) == 0 and self.valid_range is not None:
                    if len(self.rangs) == 0 and len(kwargs) == 0:
                        self.wrap = cls()
                    else:
                        self.wrap = cls(random.uniform(*self.valid_range), **other_args)
                else:
                    self.wrap = cls(*list(rangs.values()), **other_args)

                self.rn = random.random()

            def __call__(self, inputs: Union[Dict[TensorSpec, np.ndarray], np.ndarray], spec: TensorSpec = None,
                         **kwargs):
                keep_prob = kwargs.get('keep_prob', kwargs.get('keep_ratio', self.keep_prob))
                self.keep_prob = keep_prob
                self.rn = random.random()
                if self.rn > keep_prob:
                    rangs, other_args = self.set_random()

                    for k, v in rangs.items():
                        setattr(self.wrap, k, v)
                    for k, v in other_args.items():
                        setattr(self.wrap, k, v)

                    # if len(rangs) > 0 and len(self.kwargs) > 0:
                    #     for k, v in rangs:
                    #         setattr(self.wrap, k, v)
                    # else:
                    #     if len(self._args) > 0:
                    #         setattr(self.wrap, self._args[0], random.uniform(*self.valid_range))
                    return self.wrap(inputs, spec=spec, **kwargs)
                else:
                    return inputs

            def _apply_image(self, image, spec: TensorSpec):
                return self.wrap._apply_image(image, spec)

            def _apply_coords(self, coords, spec: TensorSpec):
                return self.wrap._apply_coords(coords, spec)

            def _apply_mask(self, mask, spec: TensorSpec):
                return self.wrap._apply_mask(mask, spec)

            def _get_shape(self, image):
                return self.wrap._get_shape(image)

            def set_random(self):
                self.rn = random.random()
                rangs = dict([(k, random.uniform(*v)) for k, v in self.rangs.items() if
                              isinstance(v, (tuple, list)) and len(v) == 2])
                other_args = dict(
                    [(k, v) for k, v in self.other_args.items() if not (isinstance(v, (tuple, list)) and len(v) == 2)])
                return rangs, other_args

        Wrapper.__name__ = Wrapper.__qualname__ = cls.__name__
        Wrapper.__doc__ = cls.__doc__
        return Wrapper

    return randomize_wrapper


def randomize(keep_prob=0.5, **kwargs):
    def randomize_wrapper(cls):
        class Wrapper:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.wrap = cls()
                self.keep_prob = keep_prob
                self.rn = 0

            def __call__(self, inputs: Union[Dict[TensorSpec, np.ndarray], np.ndarray], spec: TensorSpec = None,
                         **kwargs):
                keep_prob = kwargs.get('keep_prob', kwargs.get('keep_ratio', self.keep_prob))
                self.keep_prob = keep_prob
                if self.rn > keep_prob:
                    return self.wrap(inputs, spec=spec, **kwargs)
                else:
                    return inputs

            def set_random(self):
                self.rn = random.random()

        Wrapper.__name__ = Wrapper.__qualname__ = cls.__name__
        Wrapper.__doc__ = cls.__doc__
        return Wrapper

    return randomize_wrapper


class Resize(VisionTransform):
    r"""
    Resize the input data.
    param output_size: target size of image, with (height, width) shape.
    param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4– a Lanczos interpolation over 8×8 pixel neighborhood.
    param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, output_size, keep_aspect=True, align_corner=True, interpolation=cv2.INTER_AREA,
                 background_color=(0, 0, 0), name='resize', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        if isinstance(self.output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        self.keep_aspect = keep_aspect
        self.align_corner = align_corner
        self.interpolation = interpolation
        self.background_color = background_color
        self.scale = 1

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        h, w, eh, ew, th, tw, pad_vert, pad_horz, scale = self._shape_info

        image=_check_float_dtype(_check_pixel_value_range(image))
        if h == eh and w == ew:
            return image
        elif not self.keep_aspect:
            return cv2.resize(image.copy(), (ew, eh), interpolation=self.interpolation if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)
        else:

            if image.shape[-1]== 1:
                image=image.squeeze(-1)
                image=np.stack([image,image,image],axis=-1)
            resized_image = cv2.resize(image.copy(), (tw, th), interpolation=self.interpolation if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)

            shp = list(int_shape(resized_image))
            output = np.ones((eh, ew, 3),dtype=np.float32) * self.background_color

            output[pad_vert:pad_vert+shp[0], pad_horz:pad_horz+shp[1],:] = resized_image
            output = _check_float_dtype(_check_pixel_value_range(output))
            return output

    def _apply_coords(self, coords, spec: TensorSpec):
        # 原圖尺寸、預期尺寸、縮放後尺寸
        h, w, eh, ew, th, tw, pad_vert, pad_horz, scale = self._shape_info
        if h == eh and w == ew:
            return coords
        elif not self.keep_aspect:
            coords[..., 0] = coords[..., 0] * (ew / w)
            coords[..., 1] = coords[..., 1] * (eh / h)
        else:
            coords[..., 0] = coords[..., 0] * scale+ pad_horz
            coords[..., 1] = coords[..., 1] * scale+pad_vert
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        h, w, eh, ew, th, tw, pad_vert, pad_horz, scale = self._shape_info
        mask_dtype = mask.dtype
        _dtype=np.int32 if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else np.float32

        mask=mask.astype(_dtype)
        if mask.shape[-1]==1:
            mask=np.squeeze(mask,-1)

        if h == eh and w == ew:
            return mask

        elif not self.keep_aspect:
            resized_mask = cv2.resize(mask, (ew, eh), interpolation=cv2.INTER_NEAREST if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else cv2.INTER_AREA)
            return resized_mask.astype(mask_dtype)
        else:

            resized_mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else cv2.INTER_AREA)
            shp = list(int_shape(resized_mask))

            if mask.ndim == 3:
                output = np.zeros((eh, ew,3)).astype(_dtype)
                output[pad_vert:pad_vert + shp[0], pad_horz:pad_horz + shp[1], :] = resized_mask
            else:
                output = np.zeros((eh, ew)).astype(_dtype)
                output[pad_vert:pad_vert + shp[0], pad_horz:pad_horz + shp[1]] = resized_mask
            return output.astype(_dtype)

    def _get_shape(self, image):
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        h, w = image.shape[:2]
        eh, ew = self.output_size

        if not self.keep_aspect:
            th = eh
            tw = ew
            self.scale = min(float(eh) / h, float(ew) / w)
            return h, w, eh, ew,  eh,ew, 0, 0, 1
        else:
            self.scale = min(float(eh) / h, float(ew) / w)
            th = int(h * self.scale)
            tw = int(w * self.scale)
            if self.align_corner:
                pad_vert=0
                pad_horz=0
            else:
                pad_vert=(eh - th)//2
                pad_horz=(ew - tw)//2
            return h, w, eh, ew, th, tw, pad_vert, pad_horz, self.scale


class Unresize(VisionTransform):
    r"""
    Resize the input data.
    param output_size: target size of image, with (height, width) shape.
    param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, output_size, keep_aspect=True, align_corner=True, interpolation=cv2.INTER_AREA, name='resize',
                 **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        if isinstance(self.output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        self.keep_aspect = keep_aspect
        self.align_corner = align_corner
        self.interpolation = interpolation
        self.scale = 1

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        h, w, th, tw, pad_vert, pad_horz, scale = self._shape_info

        if not self.keep_aspect:
            return cv2.resize(image.copy(), (tw, th), interpolation=self.interpolation if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)
        else:
            if self.align_corner:
                if ndim(image) == 2:
                    image = image[:th, :tw]
                elif ndim(image) == 3:
                    image = image[:th, :tw, :]
            else:
                if ndim(image) == 2:
                    image = image[pad_vert // 2:th + pad_vert // 2, pad_horz // 2:tw + pad_horz // 2]
                elif ndim(image) == 3:
                    image = image[pad_vert // 2:th + pad_vert // 2, pad_horz // 2:tw + pad_horz // 2, :]

            resized_image = cv2.resize(image.copy(), (w, h), interpolation=self.interpolation if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)

            return resized_image

    def _apply_coords(self, coords, spec: TensorSpec):
        h, w, th, tw, pad_vert, pad_horz, scale = self._shape_info
        if h == th and w == tw:
            return coords

        coords[:, 0] = np.round((coords[:, 0]+0.5) * scale-0.5)
        coords[:, 1] = np.round((coords[:, 1]+0.5) * scale-0.5)
        if not self.align_corner:
            coords[:, 0] += pad_vert // 2
            coords[:, 1] += pad_horz // 2
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        h, w, th, tw, pad_vert, pad_horz, scale = self._shape_info
        if h == th and w == tw:
            return mask
        mask_dtype = mask.dtype
        mask = mask.astype(np.float32)
        if not self.keep_aspect:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return mask.astype(mask_dtype)
        else:
            if self.align_corner:
                if mask.ndim == 2:
                    mask = mask[:th, :tw]
                elif mask.ndim == 3:
                    mask = mask[:th, :tw, :]
            else:
                if mask.ndim == 2:
                    mask = mask[pad_vert // 2:th + pad_vert // 2, pad_horz // 2:tw + pad_horz // 2]
                elif mask.ndim == 3:
                    mask = mask[pad_vert // 2:th + pad_vert // 2, pad_horz // 2:tw + pad_horz // 2, :]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            return mask.astype(mask_dtype)

    def _get_shape(self, image):
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        h, w = image.shape[:2]
        eh, ew = self.output_size

        if not self.keep_aspect:
            self.scale = 1
            return eh, ew, eh, ew, 0, 0, 1
        else:
            self.scale = min(float(eh) / h, float(ew) / w)
            th = int(builtins.round(h * self.scale, 0))
            tw = int(builtins.round(w * self.scale, 0))
            pad_vert = eh - th
            pad_horz = ew - tw
            return eh, ew, th, tw, pad_vert, pad_horz, self.scale


class ShortestEdgeResize(VisionTransform):
    def __init__(self, output_size, keep_aspect=True, interpolation=cv2.INTER_AREA, name='short_edge_resize'):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        if isinstance(self.output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        self.keep_aspect = keep_aspect

        self.interpolation = interpolation

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        h, w, th, tw, eh, ew, offsetx, offsety = self._shape_info
        if h == eh and w == ew:
            return image
        image = cv2.resize(image, (tw, th), self.interpolation if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)
        if ndim(image) == 2:
            return image.copy()[offsety:offsety + eh, offsetx:offsetx + ew]
        elif ndim(image) == 3:
            return image.copy()[offsety:offsety + eh, offsetx:offsetx + ew, :]

    def _apply_coords(self, coords, spec: TensorSpec):
        h, w, th, tw, eh, ew, offsetx, offsety = self._shape_info
        if h == eh and w == ew:
            return coords
        coords[:, 0] = clip(coords[:, 0] * self.scale - offsetx, 0, ew)
        coords[:, 1] = clip(coords[:, 1] * self.scale - offsety, 0, eh)
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        h, w, th, tw, eh, ew, offsetx, offsety = self._shape_info
        if h == eh and w == ew:
            return mask
        mask_dtype = mask.dtype
        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else cv2.INTER_AREA)
        mask = mask.astype(mask_dtype)
        if ndim(mask) == 2:
            return mask.copy()[offsety:offsety + eh, offsetx:offsetx + ew]
        elif ndim(mask) == 3:
            return mask.copy()[offsety:offsety + eh, offsetx:offsetx + ew, :]

    def _get_shape(self, image):
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        h, w = image.shape[:2]
        eh, ew = self.output_size

        self.scale = builtins.max(eh / h, ew / w)
        th = int(builtins.round(h * self.scale, 0))
        tw = int(builtins.round(w * self.scale, 0))

        offsetx = int(random.randint(0, int(tw - ew)) if tw - ew >= 1 else 0)
        offsety = int(random.randint(0, int(th - eh)) if th - eh >= 1 else 0)
        return h, w, th, tw, eh, ew, offsetx, offsety


class Rescale(VisionTransform):
    r"""
    Resize the input data.
    param output_size: target size of image, with (height, width) shape.
    param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, scale, interpolation=cv2.INTER_AREA, name='rescale', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.scale = scale
        self.output_size = None
        self.interpolation = interpolation

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        h, w, _ = image.shape
        self.output_size = (int(w * self.scale), int(h * self.scale))
        return cv2.resize(image, (int(w * self.scale), int(h * self.scale)), self.interpolation if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)

    def _apply_coords(self, coords, spec: TensorSpec):
        coords[:, 0] = coords[:, 0] * self.scale
        coords[:, 1] = coords[:, 1] * self.scale
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        h, w, _ = mask.shape
        mask_dtype = mask.dtype
        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, self.output_size, interpolation=cv2.INTER_NEAREST if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else cv2.INTER_AREA)
        mask = mask.astype(mask_dtype)
        return mask


class RandomRescaleCrop(VisionTransform):
    r"""
    Resize the input data.

    1. expect size=(eh,ew) get the random scale from scale_range
    2. (th,tw)=(eh*scale_range,ew*scale_range) will be the target crop area size
    3. if crop area size is larger than image ,then padding to target crop size, otherwise crop from offset


    param output_size: target size of image, with (height, width) shape.
    param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, output_size, scale_range=(0.5, 2.0), background_color=(0, 0, 0), interpolation=cv2.INTER_AREA,
                 name='random_rescale_crop', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        self.background_color = background_color
        if isinstance(self.output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        scale_range = kwargs.get('scale', scale_range)
        if isinstance(scale_range, numbers.Number):
            self.scale_range = (scale_range, scale_range)
        else:
            self.scale_range = scale_range

        self.interpolation = interpolation

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):

        if self._shape_info is None:
            self._shape_info = self._get_shape(image)

        height, width, offset_x, offset_y, th, tw, eh, ew, target_scale = self._shape_info
        try:
            if image.shape[-1] == 1:
                image = image.squeeze(-1)
                image = np.stack([image, image, image], axis=-1)

            cropped_img = image[offset_y: offset_y + th, offset_x: offset_x + tw, :]
            cropped_img = cv2.resize(cropped_img, None, fx=1 / target_scale, fy=1 / target_scale,
                                     interpolation=self.interpolation if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)

            if cropped_img.shape[0] == eh and cropped_img.shape[1] == ew:
                return cropped_img
            else:
                background = np.ones((eh, ew, 3)) * self.background_color

                background[:builtins.min(cropped_img.shape[0], eh), :builtins.min(cropped_img.shape[1], ew),
                :] = cropped_img[:builtins.min(cropped_img.shape[0], eh), :builtins.min(cropped_img.shape[1], ew), :]
                return background
        except Exception as e:
            print(e, self._shape_info)
            PrintException()

    def _apply_coords(self, coords, spec: TensorSpec):
        height, width, offset_x, offset_y, th, tw, eh, ew, target_scale = self._shape_info
        try:
            coords[:, 0] = coords[:, 0] - offset_x
            coords[:, 1] = coords[:, 1] - offset_y
            coords = coords / target_scale

            return coords
        except Exception as e:
            print(e, self._shape_info)
            PrintException()

    def _apply_mask(self, mask, spec: TensorSpec):
        height, width, offset_x, offset_y, th, tw, eh, ew, target_scale = self._shape_info
        try:

            cropped_mask = mask[offset_y: offset_y + th, offset_x: offset_x + tw]
            cropped_mask = cv2.resize(cropped_mask, None, fx=1 / target_scale, fy=1 / target_scale,
                                      interpolation=cv2.INTER_NEAREST if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else cv2.INTER_AREA)

            if cropped_mask.shape[0] == eh and cropped_mask.shape[1] == ew:
                return cropped_mask
            else:
                background = np.zeros((eh, ew)).astype(mask.dtype)
                background[:builtins.min(cropped_mask.shape[0], eh),
                :+builtins.min(cropped_mask.shape[1], ew)] = cropped_mask[:builtins.min(cropped_mask.shape[0], eh),
                                                             :+builtins.min(cropped_mask.shape[1], ew)]
                return background
        except Exception as e:
            print(e, self._shape_info)
            PrintException()

    def _get_shape(self, image):
        height, width = image.shape[:2]
        # area = height * width
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        eh, ew = self.output_size

        target_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        tw, th = builtins.min(int(round(ew * target_scale)), width), builtins.min(int(round(eh * target_scale)), height)

        offset_x, offset_y = 0, 0

        if 0 < tw < ew - 2:
            offset_x = random.uniform(0, (ew - tw) // 2)
        else:
            offset_x = 0
        if 0 < th < eh - 2:
            offset_y = random.uniform(0, (eh - th) // 2)
        else:
            offset_y = 0
        return int(height), int(width), int(offset_x), int(offset_y), int(th), int(tw), int(eh), int(ew), target_scale


class RandomCenterCrop(VisionTransform):
    r"""
    Resize the input data.
    param output_size: target size of image, with (height, width) shape.
    param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, output_size, scale_range=(0.8, 1.2), background_color=(0, 0, 0), interpolation=cv2.INTER_AREA,
                 name='random_center_crop', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        self.background_color = background_color
        if isinstance(self.output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        scale_range = kwargs.get('scale', scale_range)
        if isinstance(scale_range, numbers.Number):
            self.scale_range = (scale_range, scale_range)
        else:
            self.scale_range = scale_range
        self.interpolation = interpolation

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        x, y, th, tw, eh, ew, h, w = self._shape_info

        if image.shape[-1] == 1:
            image = image.squeeze(-1)
            image = np.stack([image, image, image], axis=-1)

        resized_image = cv2.resize(image.copy(), (tw, th), interpolation=self.interpolation if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)

        crop_image = resized_image[y: builtins.min(y+eh,th), x:builtins.min(x+ew, tw)]
        #crop_image = resized_image[y: builtins.min(y + eh, th), x:builtins.min(x + ew, tw)]
        if crop_image.shape[0] < eh or crop_image.shape[1] < ew:
            background = np.ones((eh, ew, 3)) * self.background_color
            if ndim(crop_image) == 2:
                background[
                builtins.max(eh - crop_image.shape[0], 0) // 2:builtins.max(eh - crop_image.shape[0], 0) // 2 +
                                                               crop_image.shape[0],
                builtins.max(ew - crop_image.shape[1], 0) // 2:builtins.max(ew - crop_image.shape[1], 0) // 2 +
                                                               crop_image.shape[1], 0] = crop_image
            else:
                background[
                builtins.max(eh - crop_image.shape[0], 0) // 2:builtins.max(eh - crop_image.shape[0], 0) // 2 +
                                                               crop_image.shape[0],
                builtins.max(ew - crop_image.shape[1], 0) // 2:builtins.max(ew - crop_image.shape[1], 0) // 2 +
                                                               crop_image.shape[1], :] = crop_image
            return background
        else:
            return crop_image

    def _apply_coords(self, coords, spec: TensorSpec):
        x, y, th, tw, eh, ew, h, w = self._shape_info
        scale = tw / w
        coords[:, 0] = coords[:, 0] * scale
        coords[:, 1] = coords[:, 1] * scale
        coords[:, 0] -= x
        coords[:, 1] -= y
        coords[:, 0] += builtins.max(ew - tw, 0) // 2
        coords[:, 1] += builtins.max(eh - th, 0) // 2
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        x, y, th, tw, eh, ew, h, w = self._shape_info
        mask_dtype = mask.dtype
        _dtype=np.int32 if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else np.float32
        mask = mask.astype(_dtype)
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, -1)

        resized_mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else cv2.INTER_LANCZOS4)
        crop_mask = resized_mask[y: builtins.min(y+eh,th), x:builtins.min(x+ew, tw)]
        if ndim(crop_mask) == 3:
            crop_mask = crop_mask[:, :, 0]
        if crop_mask.shape[0] < eh or crop_mask.shape[1] < ew:
            background = np.zeros((eh, ew)).astype(mask.dtype)

            background[builtins.max(eh - crop_mask.shape[0], 0) // 2:builtins.max(eh - crop_mask.shape[0], 0) // 2 +crop_mask.shape[0],builtins.max(ew - crop_mask.shape[1], 0) // 2:builtins.max(ew - crop_mask.shape[1], 0) // 2 +crop_mask.shape[1]] = crop_mask
            return background
        else:
            return crop_mask

    def _get_shape(self, image):
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        eh, ew = self.output_size
        base_scale = builtins.min(eh / h, ew / w)
        current_scale = np.random.uniform(*self.scale_range)
        current_scale = base_scale * current_scale

        th, tw = int(h * current_scale), int(w * current_scale)

        x = builtins.max(int((tw - ew) / 2.0), 0)
        y = builtins.max(int((th - eh) / 2.0), 0)
        return x, y, th, tw, eh, ew, h, w

#
# class RandomCrop(VisionTransform):
#     r"""
#     Crop the input data randomly. Before applying the crop transform,
#     pad the image first. If target size is still bigger than the size of
#     padded image, pad the image size to target size.
#     param output_size: target size of image, with (height, width) shape.
#
#     """
#
#     def __init__(self, output_size, name='random_crop', **kwargs):
#         super().__init__(name)
#         self.is_spatial = True
#         self.output_size = output_size
#         if isinstance(self.output_size, numbers.Number):
#             self.output_size = (output_size, output_size)
#
#     def apply(self, input: Tuple, spec: TensorSpec):
#         return super().apply(input, spec)
#
#     def _apply_image(self, image, spec: TensorSpec):
#         if self._shape_info is None:
#             self._shape_info = self._get_shape(image)
#         h, w, eh, ew, offset_x, offset_y, offset_x1, offset_y1 = self._shape_info
#         if image.ndim == 2 or (image.ndim == 3 and int_shape(image)[-1] == 1):
#             origin_ndim = image.ndim
#             if origin_ndim == 3:
#                 image = image[:, :, 0]
#             output = np.zeros(self.output_size, dtype=image.dtype)
#             crop_im = image[offset_y:min(offset_y + eh, h), offset_x:min(offset_x + ew, w)]
#             output[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1]] = crop_im
#             # 如果目标尺寸大于原图，则左上角填充0
#             if eh > h or ew > w:
#                 output[h:eh, :] = 0
#                 output[:, w:ew] = 0
#             if origin_ndim == 3:
#                 output = np.expand_dims(output, -1)
#             return output
#         elif image.ndim == 3:
#             output_shape = self.output_size + (
#                 1,) if spec is not None and spec.object_type == ObjectType.gray else self.output_size + (3,)
#             output = np.zeros(output_shape, dtype=image.dtype)
#             crop_im = image[offset_y:min(offset_y + eh, h), offset_x:min(offset_x + ew, w), :]
#             output[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1], :] = crop_im
#             if eh > h or ew > w:
#                 output[h:eh, :, :] = 0
#                 output[:, w:ew, :] = 0
#             return output
#
#     def _apply_coords(self, coords, spec: TensorSpec):
#         h, w, eh, ew, offset_x, offset_y, offset_x1, offset_y1 = self._shape_info
#         coords[:, 0] = coords[:, 0] - offset_x + offset_x1
#         coords[:, 1] = coords[:, 1] - offset_y + offset_y1
#
#         return coords
#
#     def _apply_mask(self, mask, spec: TensorSpec):
#         h, w, eh, ew, offset_x, offset_y, offset_x1, offset_y1 = self._shape_info
#         if mask.ndim == 2:
#             output = np.zeros(self.output_size, dtype=mask.dtype)
#             crop_mask = mask[offset_y:min(offset_y + eh, h), offset_x:min(offset_x + ew, w)]
#             output[offset_y1:offset_y1 + crop_mask.shape[0], offset_x1:offset_x1 + crop_mask.shape[1]] = crop_mask
#             return output
#         elif mask.ndim == 3:
#             output = np.zeros((*self.output_size, 3), dtype=mask.dtype)
#             crop_mask = mask[offset_y:min(offset_y + eh, h), offset_x:min(offset_x + ew, w), :]
#             output[offset_y1:offset_y1 + crop_mask.shape[0], offset_x1:offset_x1 + crop_mask.shape[1], :] = crop_mask
#             return output
#
#     def _get_shape(self, image):
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             self.output_size = (self.output_size, self.output_size)
#         eh, ew = self.output_size
#
#         offset_x = 0
#         offset_y = 0
#
#         if w > ew:
#             offset_x = random.randint(0, w - ew)
#         if h > eh:
#             offset_y = random.randint(0, h - eh)
#
#         offset_x1 = random.randint(0, ew - w) if ew > w else 0
#         offset_y1 = random.randint(0, eh - h) if eh > h else 0
#         return h, w, eh, ew, offset_x, offset_y, offset_x1, offset_y1


class RandomCrop(VisionTransform):
    """
    隨機裁剪輸入數據。在應用裁剪變換之前，先對圖像進行填充。
    如果目標尺寸仍然大於填充後圖像的尺寸，則將圖像尺寸填充到目標尺寸。

    Args:
        output_size: 目標圖像尺寸，格式為 (height, width)
        name: 變換名稱，默認為 'random_crop'
    """

    def __init__(self, output_size, name='random_crop', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = self._normalize_output_size(output_size)

    def _normalize_output_size(self, output_size):
        """標準化輸出尺寸為 (height, width) 格式"""
        if isinstance(output_size, numbers.Number):
            return (output_size, output_size)
        return output_size

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        """對圖像應用隨機裁剪"""
        if self._shape_info is None:
            self._shape_info = self._calculate_crop_params(image)

        crop_params = self._shape_info

        if self._is_grayscale_image(image):
            return self._crop_grayscale_image(image, crop_params, spec)
        elif image.ndim == 3:
            return self._crop_color_image(image, crop_params, spec)
        else:
            raise ValueError(f"不支援的圖像維度: {image.ndim}")

    def _is_grayscale_image(self, image):
        """判斷是否為灰度圖像"""
        return (image.ndim == 2 or
                (image.ndim == 3 and int_shape(image)[-1] == 1))

    def _crop_grayscale_image(self, image, crop_params, spec):
        """裁剪灰度圖像"""
        original_ndim = image.ndim

        # 確保圖像為 2D
        if original_ndim == 3:
            image = image[:, :, 0]

        # 執行裁剪操作
        output = self._perform_crop_operation(image, crop_params, is_grayscale=True)

        # 恢復原始維度
        if original_ndim == 3:
            output = np.expand_dims(output, -1)

        return output

    def _crop_color_image(self, image, crop_params, spec):
        """裁剪彩色圖像"""
        channels = (1 if spec is not None and spec.object_type == ObjectType.gray
                    else 3)
        output_shape = self.output_size + (channels,)

        return self._perform_crop_operation(image, crop_params,
                                            output_shape=output_shape,
                                            is_grayscale=False)

    def _perform_crop_operation(self, image, crop_params, output_shape=None, is_grayscale=True):
        """執行實際的裁剪操作"""
        (original_h, original_w, target_h, target_w,
         crop_start_x, crop_start_y, paste_start_x, paste_start_y) = crop_params

        # 設定輸出形狀
        if output_shape is None:
            output_shape = self.output_size

        # 初始化輸出數組
        output = np.zeros(output_shape, dtype=image.dtype)

        # 計算裁剪區域
        crop_end_x = min(crop_start_x + target_w, original_w)
        crop_end_y = min(crop_start_y + target_h, original_h)

        if is_grayscale:
            cropped_region = image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        else:
            cropped_region = image[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :]

        # 將裁剪區域貼到輸出數組
        paste_end_x = paste_start_x + cropped_region.shape[1]
        paste_end_y = paste_start_y + cropped_region.shape[0]

        if is_grayscale:
            output[paste_start_y:paste_end_y, paste_start_x:paste_end_x] = cropped_region
        else:
            output[paste_start_y:paste_end_y, paste_start_x:paste_end_x, :] = cropped_region

        return output

    def _apply_coords(self, coords, spec: TensorSpec):
        """對座標應用變換"""
        (_, _, _, _, crop_start_x, crop_start_y,
         paste_start_x, paste_start_y) = self._shape_info

        # 調整座標位置
        coords[:, 0] = coords[:, 0] - crop_start_x + paste_start_x
        coords[:, 1] = coords[:, 1] - crop_start_y + paste_start_y

        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        """對遮罩應用變換"""
        crop_params = self._shape_info

        if mask.ndim == 2:
            return self._perform_crop_operation(mask, crop_params,
                                                output_shape=self.output_size,
                                                is_grayscale=True)
        elif mask.ndim == 3:
            output_shape = (*self.output_size, 3)
            return self._perform_crop_operation(mask, crop_params,
                                                output_shape=output_shape,
                                                is_grayscale=False)
        else:
            raise ValueError(f"不支援的遮罩維度: {mask.ndim}")

    def _calculate_crop_params(self, image):
        """計算裁剪參數"""
        original_h, original_w = image.shape[:2]
        target_h, target_w = self.output_size

        # 計算裁剪起始位置（當原圖大於目標尺寸時）
        crop_start_x = (random.randint(0, original_w - target_w)
                        if original_w > target_w else 0)
        crop_start_y = (random.randint(0, original_h - target_h)
                        if original_h > target_h else 0)

        # 計算貼上起始位置（當目標尺寸大於原圖時）
        paste_start_x = (random.randint(0, target_w - original_w)
                         if target_w > original_w else 0)
        paste_start_y = (random.randint(0, target_h - original_h)
                         if target_h > original_h else 0)

        return (original_h, original_w, target_h, target_w,
                crop_start_x, crop_start_y, paste_start_x, paste_start_y)
class RandomTransformAffine(VisionTransform):
    r"""Apply Random affine transformation to the input PIL image.
    degrees (Union[int, float, sequence]): Range of the rotation degrees.
            If degrees is a number, the range will be (-degrees, degrees).
            If degrees is a sequence, it should be (min, max).
    translate (sequence, optional): Sequence (tx, ty) of maximum translation in
        x(horizontal) and y(vertical) directions (default=None).
        The horizontal shift and vertical shift are selected randomly from the range:
        (-tx*width, tx*width) and (-ty*height, ty*height), respectively.
        If None, no translations are applied.
    scale (sequence, optional): Scaling factor interval (default=None, original scale is used).
    shear (Union[int, float, sequence], optional): Range of shear factor (default=None).
        If shear is an integer, then a shear parallel to the X axis in the range of (-shear, +shear) is applied.
        If shear is a tuple or list of size 2, then a shear parallel to the X axis in the range of
        (shear[0], shear[1]) is applied.
        If shear is a tuple of list of size 4, then a shear parallel to X axis in the range of
        (shear[0], shear[1]) and a shear parallel to Y axis in the range of (shear[2], shear[3]) is applied.
        If shear is None, no shear is applied.
    param output_size: target size of image, with (height, width) shape.
    param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, rotation_range=15, zoom_range=0.02, shift_range=0.02, shear_range=0.2, random_flip=0,
                 border_mode='random_color', background_color=None, interpolation=cv2.INTER_AREA, keep_prob=0.5,
                 name='transform_affine', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = None
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.interpolation = interpolation
        self.random_flip = random_flip
        self.background_color = background_color
        if border_mode not in ['random_color', 'constant', 'replicate', 'zero', 'reflect', 'wrap']:
            print(
                'Only {0} are valid items'.format(['random_color', 'constant', 'replicate', 'zero', 'reflect', 'wrap']))
            self.border_mode = 'random_color'
        else:
            self.border_mode = border_mode
        self.keep_prob = keep_prob

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        image = unpack_singleton(image)
        H, W, C = int_shape(image)
        self.output_size = (H, W)

        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        mat_img, height, width, angle, is_flip, rr, shear_factor, background_color = self._shape_info

        if rr > self.keep_prob:
            image = np.clip(image.astype(np.float32), 0, 255)[:, :, :3]
            _borderMode = cv2.BORDER_CONSTANT
            _borderValue = background_color
            if self.border_mode == 'replicate':
                _borderMode = cv2.BORDER_REPLICATE
            elif self.border_mode == 'reflect':
                _borderMode = cv2.BORDER_REFLECT
            elif self.border_mode == 'wrap':
                _borderMode = cv2.BORDER_WRAP
            elif self.border_mode == 'zero':
                _borderValue = (0, 0, 0)

            image = cv2.warpAffine(image.copy(), mat_img, dsize=(width, height), borderMode=_borderMode,borderValue=_borderValue, flags=cv2.INTER_AREA  if spec.object_type !=ObjectType.densepose else cv2.INTER_NEAREST)  # , borderMode=cv2.BORDER_REPLICATE

            # if shear_factor>0:
            #     image = cv2.warpAffine(image, mat_shear, dsize=(int(nW), image.shape[0]), borderMode=_borderMode,borderValue=_borderValue,flags=cv2.INTER_AREA)
            #     image = cv2.resize(image,(width, height))
            if is_flip:
                image = cv2.flip(image, 1)
                return image


            else:
                return image
        else:
            return image

    def _apply_boxes(self, boxes,spec:TensorSpec):
        mat_img, height, width,angle, is_flip, rr, shear_factor, background_color = self._shape_info
        if rr > self.keep_prob:
            if isinstance( self.output_size,numbers.Number):
                self.output_size=( self.output_size, self.output_size)
            eh, ew = self.output_size
            if ndim(boxes)==0:
                return boxes
            else:
                if ndim(boxes) == 1:
                    boxes=np.expand_dims(boxes,0)
                location= boxes[:, :4]
                class_info = boxes[:, 4:5] if boxes.shape[-1]>4 else None
                keypoints = boxes[:, 5:] if boxes.shape[-1]>5 else None

                corners = _get_corners(np.asarray(location))

                corners = corners.reshape(-1, 2)
                corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
                corners = np.dot(mat_img, corners.T).T

                new_corners = corners.reshape(-1, 8)

                new_bbox = _get_enclosing_box(new_corners)

                if is_flip:
                    new_bbox = np.stack([width - new_bbox[:, 2], new_bbox[:, 1], width - new_bbox[:, 0], new_bbox[:, 3]], axis=1)

                if keypoints is not None:
                    coords_keypoints = np.asarray(keypoints).reshape(-1, 2)
                    keypoints = self._apply_keypoints(coords_keypoints, spec).reshape((-1, keypoints.shape[-1]))

                trans_boxes = new_bbox
                if class_info is not None  and class_info.shape[-1]>0 and keypoints is not None and len(keypoints)>0:
                    trans_boxes = np.concatenate((trans_boxes, class_info,keypoints), axis=1)
                elif class_info is not None  and class_info.shape[-1]>0:
                    trans_boxes = np.concatenate((trans_boxes, class_info), axis=1)
                return trans_boxes
        else:
            return boxes

    def _apply_coords(self, coords, spec: TensorSpec):
        mat_img, height, width, angle, is_flip, rr, shear_factor, background_color = self._shape_info
        outlier_mask = coords[:, 1] < 2
        # if len(coords[outlier_mask,:])3 and coords[~outlier_mask,1].min()>(coords[60,1]-((coords[66,1]+coords[79,1])/2)):
        #     print('landmark異常數據')
        #     [print(y,coords[y,1]) for y in coords[:,1] if coords[y,1]<2 ]
        #     print(np.concatenate([np.expand_dims(np.arange(len(coords)),0),coords.copy()],axis=0).astype(np.int32).tolist())

        if rr > self.keep_prob:

            coords = coords.transpose([1, 0])
            coords = np.insert(coords, 2, 1, axis=0)
            # # print(coords)
            # # print(transform_matrix)
            coords_result = np.matmul(mat_img, coords)
            coords_result = coords_result[0:2, :].transpose([1, 0])

            if is_flip:
                if coords_result.shape[-1] == 4:
                    coords_result = np.stack(
                        [width - coords_result[..., 2], coords_result[..., 1], width - coords_result[..., 0],
                         coords_result[..., 3]], axis=-1)
                elif coords_result.shape[-1] == 2:
                    coords_result[..., 0] = width - coords_result[..., 0]

            return coords_result
        else:
            return coords

    def _apply_mask(self, mask, spec: TensorSpec):

        mat_img, height, width, angle, is_flip, rr, shear_factor, background_color = self._shape_info
        if rr > self.keep_prob:
            mask_dtype = mask.dtype
            mask = mask.astype(np.float32)

            mask = cv2.warpAffine(mask, mat_img, dsize=(width, height), borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0), flags=cv2.INTER_NEAREST if spec.object_type in [ObjectType.binary_mask,ObjectType.label_mask,ObjectType.color_mask] else cv2.INTER_AREA)  # , borderMode=cv2.BORDER_REPLICATE
            if spec.object_type == ObjectType.binary_mask:
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
            mask = mask.astype(mask_dtype)
            if is_flip:
                return np.fliplr(mask)
            else:
                return mask
        else:
            return mask

    def _get_shape(self, image):
        self.rn = random.randint(0, 10)
        h, w = image.shape[:2]
        self.output_size = (h, w)


        angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1]) if _check_range_tuple(
            self.rotation_range) else np.random.uniform(-self.rotation_range,
                                                        self.rotation_range) if self.rotation_range > 0 else 0
        if self.rotation_range == 0:
            angle = 0
        if _check_range_tuple(self.zoom_range):
            scale = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        elif isinstance(self.zoom_range, numbers.Number) and self.zoom_range > 0:
            scale = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        else:
            scale = 1
        if isinstance(self.shift_range,numbers.Number):
            tx = np.random.uniform(-self.shift_range, self.shift_range) * w
            ty = np.random.uniform(-self.shift_range, self.shift_range) * h
        elif isinstance(self.shift_range,tuple) and len(self.shift_range)==2:

            tx = np.random.uniform(-self.shift_range[1], self.shift_range[1]) * w if self.shift_range[1]>0 else 0
            ty = np.random.uniform(-self.shift_range[0], self.shift_range[0]) * h if self.shift_range[0]>0 else 0
        else:
            print('self.shift_range:',self.shift_range)
        M = np.eye(3)



        rotation_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        #
        # cos = np.abs(rotation_mat[0, 0])
        # sin = np.abs(rotation_mat[0, 1])
        # nW = int((h * sin) + (w * cos))
        # nH = int((h * cos) + (w * sin))
        #
        # rotation_mat[0, 2] += (nW / 2) - w /2
        # rotation_mat[1, 2] += (nH / 2) - h /2

        rotation_mat[:, 2] += (tx, ty)
        M[:2] = rotation_mat



        # c, s = np.cos(angle*(pi()/180)), np.sin(angle*(pi()/180))
        # M=np.array([
        #     [c, -s, 0],
        #     [s, c, 0],
        #     [0, 0, 1.],
        # ])

        # Shear
        shear_factor = random.uniform(0, self.shear_range) if self.shear_range > 0 else 0
        # mat_shear= np.array([[1, abs(shear_factor), 0], [0, 1, 0]])
        #
        # nW = image.shape[1] + abs(shear_factor * image.shape[0])

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear_range, self.shear_range) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear_range, self.shear_range) * math.pi / 180)  # y shear (deg)

        mat = S @ M
        rr_flip = np.random.random()
        rr = np.random.random()

        background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if self.border_mode != 'random_color' and self.background_color is not None:
            background_color = self.background_color

        return  np.array(mat[:2]).reshape((2, 3)), h, w, angle, rr_flip < self.random_flip, rr, shear_factor, background_color


RandomTransform = RandomTransformAffine


class RandomMultiScaleImage(VisionTransform):
    def __init__(self, output_size, scale_range=(0.8, 1.2), background_color=(0, 0, 0), interpolation=cv2.INTER_AREA,
                 keep_aspect=True, name='random_multiscale_image', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.keep_aspect = keep_aspect
        self.background_color = background_color
        self.output_size = output_size
        if isinstance(self.output_size, numbers.Number):
            self.output_size = (output_size, output_size)
        scale_range = kwargs.get('scale', scale_range)
        if isinstance(scale_range, numbers.Number):
            self.scale_range = (scale_range, scale_range)
        else:
            self.scale_range = scale_range
        self.interpolation = interpolation
        self.idx = 0
        # self.tmp_fun=None
        self.resize_funs = [Resize(output_size, True, align_corner=True, interpolation=interpolation),
                            Resize(output_size, True, align_corner=False, background_color=background_color,
                                   interpolation=interpolation),
                            Resize(output_size, False, background_color=background_color, interpolation=interpolation),
                            ShortestEdgeResize(output_size=output_size, keep_aspect=True, interpolation=interpolation),
                            ShortestEdgeResize(output_size=output_size, keep_aspect=True, interpolation=interpolation),
                            RandomRescaleCrop(output_size=output_size, scale_range=scale_range,
                                              background_color=background_color, interpolation=interpolation),
                            RandomRescaleCrop(output_size=output_size,
                                              scale_range=((scale_range[0] + 1) / 2, (scale_range[1] + 1) / 2),
                                              background_color=background_color, interpolation=interpolation),
                            RandomCrop(output_size=output_size),
                            RandomCrop(output_size=output_size),
                            RandomCenterCrop(output_size=output_size, scale_range=scale_range,
                                             background_color=background_color, interpolation=interpolation)]
        if self.keep_aspect:
            self.resize_funs.pop(2)

    def _apply_image(self, image, spec: TensorSpec):
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        idx = self._shape_info
        return self.resize_funs[idx]._apply_image(image, spec)

    def _apply_mask(self, mask, spec: TensorSpec):
        idx = self._shape_info
        return self.resize_funs[idx]._apply_mask(mask, spec)

    def _apply_coords(self, coords, spec: TensorSpec):
        idx = self._shape_info
        return self.resize_funs[idx]._apply_coords(coords, spec)

    def _get_shape(self, image=None):
        idx = random.choice(range(len(self.resize_funs)))
        return idx


class HorizontalFlip(VisionTransform):
    def __init__(self, name='horizontal_flip', **kwargs):
        super().__init__(name)
        self.is_spatial = True

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        image = unpack_singleton(image)
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        height, width = self._shape_info
        return np.fliplr(image)

    def _apply_coords(self, coords, spec: TensorSpec):
        height, width = self._shape_info
        if coords.shape[-1] == 4:
            coords = np.stack([width - coords[..., 2], coords[..., 1], width - coords[..., 0], coords[..., 3]], axis=-1)
        elif coords.shape[-1] == 2:
            coords[..., 0] = width - coords[..., 0]
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        mask = unpack_singleton(mask)
        height, width = self._shape_info
        return mask[:, ::-1]

    def _get_shape(self, image):
        height, width, _ = image.shape
        return height, width


class VerticalFlip(VisionTransform):
    def __init__(self, name='vertical_flip', **kwargs):
        super().__init__(name)
        self.is_spatial = True

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        image = unpack_singleton(image)
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        height, width = self._shape_info
        return np.flipud(image)

    def _apply_coords(self, coords, spec: TensorSpec):
        height, width = self._shape_info
        if coords.shape[-1] == 4:
            coords = np.stack([coords[..., 0], height - coords[..., 3], coords[..., 2], height - coords[..., 1]],
                              axis=-1)
        elif coords.shape[-1] == 2:
            coords[..., 1] = height - coords[..., 1]
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        mask = unpack_singleton(mask)
        height, width = self._shape_info
        return mask[:, ::-1]

    def _get_shape(self, image):
        height, width, _ = image.shape
        return height, width


@randomize(keep_prob=0.8)
class RandomMirror(HorizontalFlip):
    pass


class Normalize(VisionTransform):
    r"""
    Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    param mean: sequence of means for each channel.
    param std: sequence of standard deviations for each channel.

    """

    def __init__(self, mean=0.0, std=1.0, name='normalize', **kwargs):
        super().__init__(name)
        self.mean = mean
        self.std = std

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if image is None:
            pass
        image = image.astype(np.float32)
        norm_mean = self.mean
        norm_std = self.std
        if isinstance(self.mean, numbers.Number) and image.ndim == 3:
            norm_mean = np.ones((1, 1, image.shape[-1]), dtype=np.float32) * self.mean
        elif isinstance(self.mean, (list, tuple)) and len(self.mean) == image.shape[-1] and image.ndim == 3:
            norm_mean = np.expand_dims(np.expand_dims(to_numpy(self.mean), 0), 0).astype(np.float32)

        if isinstance(self.std, numbers.Number) and image.ndim == 3:
            norm_std = np.ones((1, 1, image.shape[-1]), dtype=np.float32) * self.std
        elif isinstance(self.std, (list, tuple)) and len(self.std) == image.shape[-1] and image.ndim == 3:
            norm_std = np.expand_dims(np.expand_dims(to_numpy(self.std), 0), 0).astype(np.float32)

        if image.ndim == 3:
            image -= norm_mean
            image /= norm_std
            return image
        elif image.ndim == 2:
            if isinstance(norm_mean, numbers.Number) and isinstance(norm_std, numbers.Number):
                image -= norm_mean
                image /= norm_std
                return image
        return image

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class Unnormalize(VisionTransform):
    r"""
    Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    param mean: sequence of means for each channel.
    param std: sequence of standard deviations for each channel.

    """

    def __init__(self, mean=0.0, std=1.0, name='normalize', **kwargs):
        super().__init__(name)
        self.mean = mean
        self.std = std

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        # image = image.astype(np.float32)
        if image.ndim == 3 and image.shape[0] <= 4:
            image = image.transpose([1, 2, 0])

        norm_mean = self.mean
        norm_std = self.std
        if isinstance(self.mean, numbers.Number) and image.ndim == 3:
            norm_mean = np.array([self.mean] * image.shape[-1]).astype(np.float32)
            norm_mean = np.expand_dims(np.expand_dims(norm_mean, 0), 0)

        elif isinstance(self.mean, (list, tuple)) and image.ndim == 3:
            norm_mean = np.array([self.mean]).astype(np.float32)
            # norm_mean = np.expand_dims(np.expand_dims(norm_mean, 0), 0)

        if isinstance(self.std, numbers.Number) and image.ndim == 3:
            norm_std = np.array([self.std] * image.shape[-1]).astype(np.float32)
            norm_std = np.expand_dims(np.expand_dims(norm_std, 0), 0)

        elif isinstance(self.std, (list, tuple)) and image.ndim == 3:
            norm_std = np.array([self.std]).astype(np.float32)
            # norm_std = np.expand_dims(norm_std, 0)

        image *= norm_std
        image += norm_mean
        return image

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class AddNoise(VisionTransform):
    r"""
    Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    param mean: sequence of means for each channel.
    param std: sequence of standard deviations for each channel.

    """

    def __init__(self, intensity=0.1, name='add_noise', **kwargs):
        super().__init__(name)
        self.intensity = intensity

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        rr = random.randint(0, 10)
        orig_mean = image.mean()
        orig_std = np.std(image)
        noise = self.intensity * np.random.normal(0, orig_std, image.shape)
        if rr % 2 == 0:
            noise = self.intensity * np.random.uniform(orig_mean - orig_std, orig_mean + orig_std, image.shape)
        image = np.clip(image + noise, 0, 255)
        return image

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class AdjustBrightness(VisionTransform):
    """Adjust brightness of an Image.
        Args:
            value (float):  How much to adjust the brightness. Can be
                any non-negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.
        Returns:
            np.ndarray: Brightness adjusted image.
    """

    def __init__(self, value=0, name='adjust_brightness', **kwargs):
        super().__init__(name)
        if value < 0:
            raise ValueError("brightness value should be non-negative")
        self.value = value

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self.value == 0:
            return image
        image = image.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - self.value), 1 + self.value)
        image = image * alpha
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class AdjustContrast(VisionTransform):
    """Adjust contrast of an Image.
    Args:
        value (float): How much to adjust the contrast. Can be any
            non-negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        np.ndarray: Contrast adjusted image.
    """

    def __init__(self, value=0, name='adjust_contrast', **kwargs):
        super().__init__(name)
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = value

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self.value == 0:
            return image
        image = image.astype(np.float32)
        mean_value = -1
        if ndim(image) == 2 or (ndim(image) == 3 and image.shape[-1] == 1):
            mean_value = image.mean()
        else:
            mean_value = round(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())
        image = (1 - self.value) * mean_value + self.value * image
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class AdjustSaturation(VisionTransform):
    r"""
    Adjust saturation of the input data.
    Args:
        value (float):  How much to adjust the saturation. 0 will
            give a gray image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        output image array

    """

    def __init__(self, value=0, name='adjust_saturation', **kwargs):
        super().__init__(name)
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = value

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self.value == 0:
            return image

        dtype = image.dtype
        image = image.astype(np.float32)
        degenerate = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        image = (1 - self.value) * degenerate + self.value * image
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class AdjustHue(VisionTransform):
    r"""
       Adjust hue of the input data.
       param value: how much to adjust the hue. Can be any number
           between 0 and 0.5, 0 gives the original image.

    """

    def __init__(self, value=0, name='adjust_hue', **kwargs):
        super().__init__(name)
        if value < -0.5 or value > 0.5:
            raise ValueError("hue value should be in [0.0, 0.5]")
        self.value = value

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self.value == 0:
            return image
        image_dtype = image.dtype
        image = np.clip(image, 0, 255).astype(np.uint8)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        h, s, v = np.split(hsv_image, 3, axis=-1)
        # uint8 addition take cares of rotation across boundaries
        # handle negative hue shift correctly without overflow error
        shift = int(self.value * 255)
        h = (h.astype(np.int16) + shift) % 256
        h = h.astype(np.uint8)
        hsv_image = cv2.merge([h, s, v])

        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB_FULL)
        return image.clip(0, 255).astype(image_dtype)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class AdjustGamma(VisionTransform):
    """Perform gamma correction on an image.
        Also known as Power Law Transform. Intensities in RGB mode are adjusted
        based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
        See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """

    def __init__(self, gamma=1, gain=1, name='adjust_gamma', **kwargs):
        super().__init__(name)
        if gamma < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.gamma = gamma
        self.gain = gain

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self.gamma == 1:
            return image
        image = image.astype(np.float32)
        image = 255. * self.gain * np.power(np.clip(image / 255., 0, 1), self.gamma)
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


@randomize_with_validate(valid_range=(0., 3.), effectless_value=1.)
class RandomAdjustBrightness(AdjustBrightness):
    pass


@randomize_with_validate(valid_range=(0., 3.), no_change_value=0.)
class RandomAdjustContrast(AdjustContrast):
    pass


@randomize_with_validate(valid_range=(0., 3.), no_change_value=0.)
class RandomAdjustSaturation(AdjustSaturation):
    pass


@randomize_with_validate(valid_range=(-0.5, 0.5), no_change_value=0.)
class RandomAdjustHue(AdjustHue):
    pass


@randomize_with_validate(valid_range=(0., 3.), no_change_value=1.)
class RandomAdjustGamma(AdjustGamma):
    pass


class GrayMixRGB(VisionTransform):

    def __init__(self, keep_prob=0.5, name='gray_mix_rgb', **kwargs):
        super().__init__(name)

        self.keep_prob = keep_prob

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if random.random() <= self.keep_prob:
            return image

        dtype = image.dtype
        image = image.astype(np.float32)
        gray = cv2.cvtColor(cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        # gray0 = gray[:, :, 0]

        min_rgb = image.mean(axis=-1, keepdims=True)
        mask = np.greater(gray, min_rgb)
        image = mask * gray + (1 - mask) * image

        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class Blur(VisionTransform):
    def __init__(self, ksize=5, name='blur', **kwargs):
        super().__init__(name)
        if ksize <= 0:
            raise ValueError("lighting scale should be positive")
        self.ksize = int(ksize)

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if int(self.ksize) % 2 == 0:
            self.ksize = int(self.ksize) + 1
        else:
            self.ksize = int(self.ksize)
        blur = cv2.GaussianBlur(image, (int(self.ksize), int(self.ksize)), cv2.BORDER_DEFAULT)
        return np.clip(blur.clip(0, 255).astype(np.float32), 0, 255)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class InvertColor(VisionTransform):
    def __init__(self, name='color', **kwargs):
        super().__init__(name)

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        return np.clip(255 - image, 0, 255)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class GrayScale(VisionTransform):
    def __init__(self, keepdims=True, name='gray_scale', **kwargs):
        super().__init__(name)
        self.is_spatial = False
        self.keepdims = keepdims

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if image.ndim == 4 and self.keepdims:
            return cv2.cvtColor(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGBA2GRAY), cv2.COLOR_GRAY2RGB)
        elif image.ndim == 4 and not self.keepdims:
            return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGBA2GRAY)
        elif image.ndim == 3 and image.shape[-1] == 1 and self.keepdims:
            return image
        elif image.ndim == 3 and image.shape[-1] == 1 and not self.keepdims:
            return image[:, :, 0]
        elif image.ndim == 3 and self.keepdims:
            return cv2.cvtColor(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and not self.keepdims:
            return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2 and self.keepdims:
            return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_GRAY2RGB)
        else:
            return np.clip(image, 0, 255)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class ToRGB(VisionTransform):
    def __init__(self, name='to_rgb', **kwargs):
        super().__init__(name)
        self.is_spatial = False

    def apply(self, input: Tuple, spec: TensorSpec):
        return self._apply_image(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if image.ndim == 3 and int_shape(image)[-1] == 1:
            image = image.copy()[:, :, 0]
        if image.ndim == 3:
            pass
        elif image.ndim == 2:
            image = np.clip(cv2.cvtColor(image.astype(np.float32), cv2.COLOR_GRAY2RGB), 0, 255)
        return image

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


@randomize_with_validate(valid_range=(1, 31), no_change_value=1)
class RandomBlur(Blur):
    pass


@randomize(keep_prob=0.8)
class RandomInvertColor(InvertColor):
    pass


@randomize(keep_prob=0.8)
class RandomGrayScale(GrayScale):
    pass


class ImageErosion(VisionTransform):
    """ Erosion operation
    Erosion is a mathematical morphology operation that uses a structuring element for shrinking the shapes in an image. The binary erosion of an image by a structuring element
    is the locus of the points where a superimposition of the structuring element centered on the point is entirely contained in the set of non-zero elements of the image.

    Args:
        filter_size (int): the size of the structuring element .
        repeat (int): the number of repeating operation.

    Returns:
        output image array

    """

    def __init__(self, filter_size=3, repeat=1, name='image_erosion', **kwargs):
        super().__init__(name)
        if filter_size < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.filter_size = filter_size
        self.repeat = repeat

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        # Creating kernel
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)

        # Using cv2.erode() method
        image = cv2.erode(image, kernel, iterations=self.repeat)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class ImageDilation(VisionTransform):
    """ Dilation operation
    Dilation is a mathematical morphology operation that uses a structuring element for expanding the shapes in an image. The binary dilation of an image by a structuring
    element is the locus of the points covered by the structuring element, when its center lies within the non-zero points of the image.

    Args:
        filter_size (int): the size of the structuring element .
        repeat (int): the number of repeating operation.

    Returns:
        output image array

    """

    def __init__(self, filter_size=3, repeat=1, name='image_dilation', **kwargs):
        super().__init__(name)
        if filter_size < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.filter_size = filter_size
        self.repeat = repeat

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        # Creating kernel
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)

        # Using cv2.erode() method
        image = cv2.dilate(image, kernel, iterations=self.repeat)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class DilationThenErosion(VisionTransform):
    r"""
       Adjust hue of the input data.
       param value: how much to adjust the hue. Can be any number
           between 0 and 0.5, 0 gives the original image.

    """

    def __init__(self, filter_size=3, repeat=1, name='dilation_then_erosion', **kwargs):
        super().__init__(name)
        if filter_size < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.filter_size = filter_size
        self.repeat = repeat

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        # Creating kernel
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)

        # Using cv2.erode() method
        for i in range(self.repeat):
            image = cv2.dilate(image, kernel, iterations=1)
            image = cv2.erode(image, kernel, iterations=1)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class ErosionThenDilation(VisionTransform):
    r"""
       Adjust hue of the input data.
       param value: how much to adjust the hue. Can be any number
           between 0 and 0.5, 0 gives the original image.

    """

    def __init__(self, filter_size=3, repeat=1, name='erosion_then_dilation', **kwargs):
        super().__init__(name)
        if filter_size < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.filter_size = filter_size
        self.repeat = repeat

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)

        # Using cv2.erode() method
        for i in range(self.repeat):
            image = cv2.erode(image, kernel, iterations=1)
            image = cv2.dilate(image, kernel, iterations=1)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class AdaptiveBinarization(VisionTransform):
    """Perform gamma correction on an image.
        Also known as Power Law Transform. Intensities in RGB mode are adjusted
        based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
        See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """

    def __init__(self, threshold_type='otsu', gaussian_filtering=True, name='adaptive_binarization', **kwargs):
        super().__init__(name)
        valid_item = ['otsu' 'percentile', 'isodata', 'local', 'minimum']
        self.threshold_type = threshold_type
        self.gaussian_filtering = gaussian_filtering

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if image.ndim == 3 and image.shape[-1] != 1:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        if gray.min() == gray.max():
            return image
        if self.gaussian_filtering:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret = None
        th = 127.5
        if self.threshold_type == 'otsu':
            # th = threshold_otsu(gray)
            th, ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold_type == 'minimum':
            th = threshold_minimum(gray)
        elif self.threshold_type == 'yen':
            th = threshold_yen(gray)
        elif self.threshold_type == 'local':
            th = threshold_local(gray, block_size=35, offset=10)
        elif self.threshold_type == 'isodata':
            th = threshold_isodata(gray, nbins=256)
        elif self.threshold_type == 'percentile':
            p10 = np.percentile(gray.copy(), 10)
            p90 = np.percentile(gray.copy(), 90)
            if abs(gray.mean() - p90) < abs(gray.mean() - p10) and p90 - p10 > 80:  # white background
                gray[gray < p10] = 0
                gray[gray > p90] = 255
            elif abs(gray.mean() - p90) > abs(gray.mean() - p10) and p90 - p10 > 80:  # white background
                gray[gray > p90] = 255
                gray[gray < p10] = 0

        gray = ret if ret is not None else (gray > th).astype(np.float32) * 255.0
        if gray.max() - gray.min() < 20:
            return clip(image, 0, 255).astype(np.float32)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return clip(image, 0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class Lighting(VisionTransform):

    def __init__(self, value=0.1, name='lighting', **kwargs):
        super().__init__(name)
        if value < 0:
            raise ValueError("lighting value should be non-negative")
        self.value = value
        self.eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203]])  # reverse the first dimension for BGR
        self.eigval = np.array([55.4625, 4.7940, 1.1475])

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self.value == 0:
            return image

        dtype = image.dtype
        image = image.astype(np.float32)
        alpha = np.random.normal(0, self.value, size=(3,))

        alter = (self.eigvec * np.expand_dims(alpha, 0) * np.expand_dims(self.eigval, 0)).sum(axis=1).reshape(1, 1, 3)
        image += alter

        return image.clip(0, 255).astype(dtype)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


@randomize_with_validate(valid_range=(0, 1.), effectless_value=0.)
class RandomLighting(Lighting):
    pass


class CLAHE(VisionTransform):
    def __init__(self, clipLimit=2, gridsize=3, name='clahe', **kwargs):
        super().__init__(name)
        self.gridsize = gridsize
        self.clipLimit = clipLimit

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        image = image.astype(np.uint8)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.gridsize, self.gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class AutoLevel(VisionTransform):

    def __init__(self, name='autoleveling', **kwargs):
        super().__init__(name)

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        minv = np.percentile(image, 5)
        maxv = np.percentile(image, 95)
        if minv == maxv:
            return image
        elif maxv - minv < 40:
            minv = image.min()
            maxv = image.max()

        image = np.clip((image - minv) * (255.0 / (maxv - minv)), 0, 255)
        return image.astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class SaltPepperNoise(VisionTransform):
    """Perform gamma correction on an image.
        Also known as Power Law Transform. Intensities in RGB mode are adjusted
        based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
        See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """

    def __init__(self, prob=0.05, keep_prob=0.5, name='saltpepper', **kwargs):
        super().__init__(name)
        self.prob = prob
        self.keep_prob = keep_prob

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        rr = random.random()
        if rr > self.keep_prob:
            imgtype = image.dtype
            rnd = np.random.uniform(0, 1, size=(image.shape[0], image.shape[1]))
            # noisy = image.copy()
            image[rnd < self.prob] = 0.0
            image[rnd > 1 - self.prob] = 255.0
            return clip(image, 0, 255).astype(np.float32)
        else:
            return image

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask


class RandomErasing(VisionTransform):
    """Perform gamma correction on an image.
        Also known as Power Law Transform. Intensities in RGB mode are adjusted
        based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
        See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """

    def __init__(self, size_range=(0.05, 0.4), transparency_range=(0.4, 0.8), transparancy_ratio=0.5, keep_prob=0.5,
                 name='random_erasing', **kwargs):
        super().__init__(name)
        self.size_range = size_range
        self.transparency_range = transparency_range
        self.transparancy_ratio = transparancy_ratio
        self.keep_prob = keep_prob

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        rr, p1,left,top,w1,h1,c1,transparancy= self._shape_info

        if rr> self.keep_prob:
            if p1<= self.transparancy_ratio:
                mask = np.ones_like(image)
                mask[top:top + h1, left:left + w1, :] = 0
                image = image * (mask) + image * (1 - mask) * (transparancy)
                return image
            else:
                image[top:top + h1, left:left + w1, :] = c1
            return image
        else:
            return image

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        rr, p1, left, top, w1, h1, c1, transparancy = self._shape_info
        if rr> self.keep_prob:
            if p1>= self.transparancy_ratio:
                mask[top:top + h1, left:left + w1] = 0
            return mask
        else:
            return mask

    def _get_shape(self, image):
        self.rr = np.random.rand()
        if self.rr > self.keep_prob:
            s_l, s_h = self.size_range
            r_1 = 0.3
            r_2 = 1 / 0.3
            h, w, c = image.shape
            c1=255
            while True:
                s = np.random.uniform(s_l, s_h) * h * w / 4.0
                r = np.random.uniform(r_1, r_2)
                w1 = int(np.sqrt(s / r))
                h1 = int(np.sqrt(s * r))
                left = np.random.randint(0, w)
                top = np.random.randint(0, h)

                if left + w1 <= w and top + h1 <= h:
                    break
            p1 = np.random.uniform(0, 1)
            if p1 <= self.transparancy_ratio:
                transparancy = np.random.uniform(*self.transparency_range)
                mask = np.ones_like(image)
                mask[top:top + h1, left:left + w1, :] = 0
                image = image * (mask) + image * (1 - mask) * (transparancy)
            else:
                transparancy=0
                c1 = np.random.uniform(0, 255)
                image[top:top + h1, left:left + w1, :] = c1
            return self.rr, p1, left, top, w1, h1, c1, transparancy
        else:
            return self.rr, None, None, None, None, None, None, None


class GridMask(VisionTransform):
    """GridMask augmentation for image classification and object detection.

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
        https://arxiv.org/abs/2001.04086
         https://github.com/akuxcw/GridMask
    """

    def __init__(self, d1=96, d2=None, rotate=1, ratio=0.5, mode=0, keep_prob=0.5, name='gridmask', **kwargs):
        super().__init__(name)
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.keep_prob = keep_prob
        self.mode = mode

    def set_prob(self, epoch, max_epoch):
        self.prob = self.keep_prob * min(1, epoch / max_epoch)

    def get_grid(self, image,d,st_h,st_w,r):
        h, w = image.shape[:2]
        hh = math.ceil((math.sqrt(h * h + w * w)))

        if self.d2 is None:
            self.d2 = maximum(h, w)


        # d = self.d

        # maybe use ceil? but I guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)

        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0

        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(image)
        image = image * mask

        return image

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        self._shape_info = self._get_shape(image)
        rr, grid_paras=self._shape_info

        n, c, h, w = image.size()
        if rr > self.keep_prob:
            y = []
            for i in range(n):
                d,st_h,st_w,r = grid_paras[i]
                y.append(self.get_grid(image[i],d,st_h,st_w,r))
            y = concate(y).view(n, c, h, w)
            return y
        else:
            return image

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        rr, grid_paras = self._shape_info

        n, c, h, w = mask.size()
        if rr > self.keep_prob and spec.object_type in [ObjectType.alpha_mask]:
            y = []
            for i in range(n):
                d, st_h, st_w, r = grid_paras[i]
                y.append(self.get_grid(image[i], d, st_h, st_w, r))
            y = concate(y).view(n, c, h, w)
            return y
        else:
            return mask
    def _get_shape(self, image):
        rr = random.random()
        n, c, h, w = image.size()
        grid_paras = []
        if rr > self.keep_prob:
            for i in range(n):
                d = np.random.randint(self.d1, self.d2)
                st_h = np.random.randint(d)
                st_w = np.random.randint(d)
                r = np.random.randint(self.rotate)

                grid_paras.append((d,st_h,st_w,r))

        return rr ,grid_paras


class RandomGridMask(VisionTransform):
    """Perform gamma correction on an image.
        Also known as Power Law Transform. Intensities in RGB mode are adjusted
        based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
        See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """

    def __init__(self, output_size=None, d1=None, d2=None, max_d1=None, rotate_range=(0.2, 1), ratio=0.2, mode=0,
                 keep_prob=0.5, name='gridmask', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        self.d1 = d1
        self.d2 = d2
        self.max_d1 = max_d1
        self.rotate_range = rotate_range
        self.ratio = ratio
        self.keep_prob = keep_prob
        self.mode = mode
        self.current_mask = None

    def set_prob(self, epoch, max_epoch):
        self.prob = self.keep_prob * min(1, epoch / max_epoch)

    def get_grid(self, image):
        h, w = image.shape[:2]
        if self.output_size is None:
            self.output_size = (h, w)
        hh = math.ceil((math.sqrt(h * h + w * w)))
        d2 = self.d2
        d1 = self.d1
        if self.d2 is None:
            d2 = minimum(h, w)
        elif isinstance(self.d2, numbers.Number):
            pass
        elif isinstance(self.d2, (tuple, list)):
            d2 = random.choice(list(self.d2))

        if self.d1 is None:
            divisors = get_divisors(minimum(h, w))

            divisors = list(sorted(set(divisors)))
            divisors = [d for d in divisors if d <= minimum(h, w) / 2.0]
            if self.max_d1 is not None:
                divisors = [d for d in divisors if d <= self.max_d1]
            if minimum(h, w) in divisors:
                divisors.remove(minimum(h, w))
            d1 = random.choice(divisors)
        elif isinstance(self.d1, (tuple, list)):
            d1 = random.choice(list(self.d1))

        d = np.random.randint(d1, d2)
        # d = self.d

        # maybe use ceil? but I guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate_range[0], self.rotate_range[1], )
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask
        # 暫存mask以處理bbox
        self.current_mask = mask.copy()
        image = image * np.expand_dims(mask, -1)

        return image

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        h, w, rr, keep = self._shape_info
        height, width = self.output_size
        if keep:
            return image
        else:
            return self.get_grid(image)

    def _apply_boxes(self, boxes, spec: TensorSpec):
        if keep:
            return boxes
        elif boxes is None:
            return boxes
        else:
            keep_boxes = []
            location = boxes[:, :4]
            class_info = boxes[:, 4:5] if boxes.shape[-1] > 4 else None
            keypoints = boxes[:, 5:] if boxes.shape[-1] > 5 else None
            # 如果剛好格線遮蔽了小框，則排除小框定義
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
            for i in range(len(boxes)):
                box = np.round(location[i]).astype(np.int32)
                if self.current_mask[box[1]:box[3] + 1, box[0]:box[2] + 1].sum() == 0:
                    pass
                else:
                    keep_boxes.append(boxes[i])
            if len(keep_boxes) > 1:
                return np.stack(keep_boxes, 0)
            elif len(keep_boxes) == 1:
                return np.array(keep_boxes)
            else:
                return None

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask

    def _get_shape(self, image):
        h, w = image.shape[:2]

        rr = random.random()
        keep = rr < self.keep_prob
        if self.output_size is None:
            self.output_size = (h, w)
        return h, w, rr, keep


class DetectionMixup(VisionTransform):
    def __init__(self, output_size=None, keep_prob=0.8, name='detection_mixup', **kwargs):
        super().__init__(name)
        self.alpha = 1
        self.output_size = output_size
        self.keep_prob = keep_prob
        self.memory_cache = []

    def _apply_image(self, image, spec: TensorSpec):
        image = unpack_singleton(image)

        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        keep, other_idx, lam, offsetx, offsety = self._shape_info
        if keep:
            return image
        height, width = self.output_size
        background = np.zeros((height, width, 3))
        if spec in self.memory_cache[other_idx]:
            p1 = self.memory_cache[other_idx][spec].copy()
            # p1 = p1[offsety:builtins.min(height, offsety+p1.shape[0]), offsetx:builtins.min(width, offsetx+p1.shape[1]):]
            background[offsety:builtins.min(height, offsety + p1.shape[0]),
            offsetx:builtins.min(width, offsetx + p1.shape[1]), :] = p1[:builtins.min(height,
                                                                                      offsety + p1.shape[0]) - offsety,
                                                                     :builtins.min(width,
                                                                                   offsetx + p1.shape[1]) - offsetx, :]
            mixed_x = lam * image + (1 - lam) * background

            return clip(mixed_x, 0, 255).astype(np.float32)
        else:
            return image

    def _apply_coords(self, coords, spec: TensorSpec):

        return coords

    def _apply_boxes(self, boxes, spec: TensorSpec):
        keep, other_idx, lam, offsetx, offsety = self._shape_info
        if keep:
            return boxes

        height, width = self.output_size
        if spec in self.memory_cache[other_idx]:
            box1 = self.memory_cache[other_idx][spec].copy() if self.memory_cache[other_idx][spec] is not None else None
            box0 = boxes.copy() if boxes is not None else None
            if boxes is None or len(boxes) == 0 or np.array_equal(np.unique(boxes), -1 * np.ones(1)):
                box0 = None
            if box1 is None or len(box1) == 0 or np.array_equal(np.unique(box1), -1 * np.ones(1)):
                box1 = None
        else:
            return boxes

        merge_boxes = []
        if box0 is not None:
            # B = box0.shape[0]
            # location = box0[:, :4]
            # class_info = box0[:, 4:5] if box0.shape[-1] > 4 else None
            # keypoints = box0[:, 5:] if box0.shape[-1] > 5 else None
            merge_boxes.append(box0)
        if box1 is not None:
            # B1 = box1.shape[0]
            # location1 = box1[:, :4].reshape((B1, 2, 2))
            # class_info1 = box1[:, 4:5] if box1.shape[-1] > 4 else None
            # keypoints1 = box1[:, 5:].reshape((B1, -1, 2)) if box1.shape[-1] > 5 else None
            #
            # location1[:, :, 0] = np.clip(location1[:, :, 0], 0, width)
            # location1[:, :, 1] = np.clip(location1[:, :, 1], 0, height)
            # if keypoints1 is not None:
            #     keypoints1[:, :, 0] = np.clip(keypoints1[:, :, 0], 0, width)
            #     keypoints1[:, :, 1] = np.clip(keypoints1[:, :, 1], 0, height)
            #
            # merge_list=[location1.reshape((B1,4)), class_info1]
            # if keypoints1 is not None and len(keypoints1)>0:
            #     merge_list.append(keypoints1.reshape((B1, -1)))
            #
            # box1 = np.concatenate(merge_list, axis=-1)
            box1[:, 0::2] += offsetx
            box1[:, 1::2] += offsety
            merge_boxes.append(box1)

        if len(merge_boxes) > 0:
            trans_boxes = np.concatenate(merge_boxes, axis=0)
            return trans_boxes
        else:
            return None

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask

    def _get_shape(self, image):
        if self.output_size is None:
            self.output_size = image.shape[:2]
        height, width = image.shape[:2]
        rr = random.random()
        keep = rr < self.keep_prob or len(self.memory_cache) - 1 <= 3
        other_idxes = None
        other_idx = None
        lam = 0
        offsetx = int(random.uniform(0, width / 2))
        offsety = int(random.uniform(0, height / 2))
        if not keep:
            other_idxes = list(range(len(self.memory_cache) - 1))
            random.shuffle(other_idxes)
            other_idx = other_idxes[0]

            height, width = self.output_size
            lam = builtins.min(builtins.max(np.random.beta(self.alpha, self.alpha), 0.3), 0.7)

        return keep, other_idx, lam, offsetx, offsety


class ImageMosaic(VisionTransform):
    """Perform gamma correction on an image.
        Also known as Power Law Transform. Intensities in RGB mode are adjusted
        based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
        See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """

    def __init__(self, output_size=None, keep_prob=0.7, name='image_mosaic', **kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        self.keep_prob = keep_prob
        self.memory_cache = []

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        image = unpack_singleton(image)

        self._shape_info = self._get_shape(image)
        keep, other_idxes, center, img1_offsetx, img1_offsety, img2_offsetx, img2_offsety, img3_offsetx, img3_offsety = self._shape_info
        if keep:
            return image
        height, width = self.output_size
        p1 = self.memory_cache[other_idxes[0]][spec].copy()[img1_offsety:img1_offsety + center[0],
             img1_offsetx:img1_offsetx + width - center[1], :]
        p2 = self.memory_cache[other_idxes[1]][spec].copy()[img2_offsety:img2_offsety + height - center[0],
             img2_offsetx:img2_offsetx + center[1], :]
        p3 = self.memory_cache[other_idxes[2]][spec].copy()[img3_offsety:img3_offsety + height - center[0],
             img3_offsetx:img3_offsetx + width - center[1], :]
        # print('p2',p2.shape)
        # print('p3',p3.shape)

        base_img = image.copy()
        base_img[:p1.shape[0], center[1]:center[1] + p1.shape[1], :] = p1
        base_img[center[0]:center[0] + p2.shape[0], :p2.shape[1], :] = p2
        base_img[center[0]:center[0] + p3.shape[0], center[1]:center[1] + p3.shape[1]:] = p3

        return clip(base_img, 0, 255).astype(np.float32)

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_boxes(self, boxes, spec: TensorSpec):
        keep, other_idxes, center, img1_offsetx, img1_offsety, img2_offsetx, img2_offsety, img3_offsetx, img3_offsety = self._shape_info
        if keep:
            return boxes
        height, width = self.output_size
        concate_list_all = []
        box0 = boxes.copy() if boxes is not None else None
        if box0 is None or len(box0) == 0:
            pass
        else:
            location0 = box0[:, :4].reshape(box0.shape[0], 2, 2)
            area0_old = clip(box_area(clip(box0[:, :4], min=0)), min=1)
            location0[:, :, 0] = np.clip(location0[:, :, 0], 0, center[1])
            location0[:, :, 1] = np.clip(location0[:, :, 1], 0, center[0])
            area0_new = box_area(location0.reshape(-1, 4)) / area0_old
            location0 = location0[area0_new >= 0.05, :, :]
            box0 = box0[area0_new >= 0.05, :]

            if len(box0) > 0:
                class_info0 = box0[:, 4:5] if box0.shape[-1] > 4 else None
                keypoints0 = box0[:, 5:] if box0.shape[-1] > 5 else None
                if box0.shape[-1] > 5:
                    keypoints0 = box0[:, 5:].reshape(box0.shape[0], -1, 2)
                    invisible_keypoints0 = np.less_equal(keypoints0, 0).astype(np.bool)
                    keypoints0[:, :, 0] = np.clip(keypoints0[:, :, 0], 0, center[1])
                    keypoints0[:, :, 1] = np.clip(keypoints0[:, :, 1], 0, center[0])
                    keypoints0[invisible_keypoints0] = -1

                else:
                    pass
                concate_list0 = None
                if keypoints0 is None:
                    # print('concate_list0',location0.reshape((box0.shape[0], 4)).shape,class_info0.shape)
                    concate_list0 = [location0.reshape((box0.shape[0], 4)), class_info0]
                else:
                    concate_list0 = [location0.reshape((box0.shape[0], 4)), class_info0,
                                     keypoints0.reshape((box0.shape[0], -1))]

                box0 = np.concatenate(concate_list0, axis=-1)
                concate_list_all.append(box0)

        box1 = self.memory_cache[other_idxes[0]][spec].copy() if self.memory_cache[other_idxes[0]][
                                                                     spec] is not None else None
        if box1 is None or len(box1) == 0:
            pass
        else:
            location1 = box1[:, :4].reshape(box1.shape[0], 2, 2)
            area1_old = clip(box_area(clip(box1[:, :4], min=0)), min=1)
            location1[:, :, 0] = np.clip(location1[:, :, 0] - img1_offsetx + center[1], center[1], width)
            location1[:, :, 1] = np.clip(location1[:, :, 1] - img1_offsety, 0, center[0])
            area1_new = box_area(location1.reshape(-1, 4)) / area1_old
            location1 = location1[area1_new >= 0.05, :, :]
            box1 = box1[area1_new >= 0.05, :]

            if len(box1) > 0:
                class_info1 = box1[:, 4:5] if box1.shape[-1] > 4 else None
                keypoints1 = None
                if box1.shape[-1] > 5:
                    keypoints1 = box1[:, 5:].reshape(box1.shape[0], -1, 2)
                    invisible_keypoints1 = np.less_equal(keypoints1, 0).astype(np.bool)
                    keypoints1[:, :, 0] = np.clip(keypoints1[:, :, 0] - img1_offsetx + center[1], center[1], width)
                    keypoints1[:, :, 1] = np.clip(keypoints1[:, :, 1] - img1_offsety, 0, center[0])
                    keypoints1[invisible_keypoints1] = -1

                else:
                    pass
                concate_list1 = None
                if keypoints1 is None:
                    # print('concate_list1',location1.reshape((box1.shape[0], 4)).shape,class_info1.shape)
                    concate_list1 = [location1.reshape((box1.shape[0], 4)), class_info1]
                else:
                    concate_list1 = [location1.reshape((box1.shape[0], 4)), class_info1,
                                     keypoints1.reshape((box1.shape[0], -1))]
                box1 = np.concatenate(concate_list1, axis=-1)
                concate_list_all.append(box1)

        box2 = self.memory_cache[other_idxes[1]][spec].copy() if self.memory_cache[other_idxes[1]][
                                                                     spec] is not None else None
        if box2 is None or len(box2) == 0:
            pass
        else:
            location2 = box2[:, :4].reshape(box2.shape[0], 2, 2)
            area2_old = clip(box_area(clip(box2[:, :4], min=0)), min=1)
            location2[:, :, 0] = np.clip(location2[:, :, 0] - img2_offsetx, 0, center[1])
            location2[:, :, 1] = np.clip(location2[:, :, 1] - img2_offsety + center[0], center[0], height)
            area2_new = box_area(location2.reshape(-1, 4)) / area2_old
            location2 = location2[area2_new >= 0.05, :, :]
            box2 = box2[area2_new >= 0.05, :]

            if len(box2) > 0:

                class_info2 = box2[:, 4:5] if box2.shape[-1] > 4 else None
                keypoints2 = None
                if box2.shape[-1] > 5:
                    keypoints2 = box2[:, 5:].reshape(box2.shape[0], -1, 2)
                    invisible_keypoints2 = np.less_equal(keypoints2, 0).astype(np.bool)
                    keypoints2[:, :, 0] = np.clip(keypoints2[:, :, 0] - img2_offsetx, 0, center[1])
                    keypoints2[:, :, 1] = np.clip(keypoints2[:, :, 1] - img2_offsety + center[0], center[0], height)
                    keypoints2[invisible_keypoints2] = -1


                else:
                    pass
                concate_list2 = None
                if keypoints2 is None:
                    # print('concate_list2',location2.reshape((box2.shape[0], 4)).shape,class_info2.shape)
                    concate_list2 = [location2.reshape((box2.shape[0], 4)), class_info2]
                else:
                    concate_list2 = [location2.reshape((box2.shape[0], 4)), class_info2,
                                     keypoints2.reshape((box2.shape[0], -1))]

                box2 = np.concatenate(concate_list2, axis=-1)
                concate_list_all.append(box2)

        box3 = self.memory_cache[other_idxes[2]][spec].copy() if self.memory_cache[other_idxes[2]][
                                                                     spec] is not None else None
        if box3 is None or len(box3) == 0:
            pass
        else:
            # print('box3',box3)
            location3 = box3[:, :4].reshape(box3.shape[0], 2, 2)
            area3_old = clip(box_area(clip(box3[:, :4], min=0)), min=1)
            location3[:, :, 0] = np.clip(location3[:, :, 0] - img3_offsetx + center[1], center[1], width)
            location3[:, :, 1] = np.clip(location3[:, :, 1] - img3_offsety + center[0], center[0], height)
            area3_new = box_area(location3.reshape(-1, 4)) / area3_old
            location3 = location3[area3_new >= 0.05, :, :]
            box3 = box3[area3_new >= 0.05, :]

            if len(box3) > 0:

                class_info3 = box3[:, 4:5] if box3.shape[-1] > 4 else None
                keypoints3 = None
                if box3.shape[-1] > 5:
                    keypoints3 = box3[:, 5:].reshape(box3.shape[0], -1, 2)
                    invisible_keypoints3 = np.less_equal(keypoints3, 0).astype(np.bool)
                    keypoints3[:, :, 0] = np.clip(keypoints3[:, :, 0] - img3_offsetx + center[1], center[1], width)
                    keypoints3[:, :, 1] = np.clip(keypoints3[:, :, 1] - img3_offsety + center[0], center[0], height)
                    keypoints3[invisible_keypoints3] = -1


                else:
                    pass
                concate_list3 = None
                if keypoints3 is None:
                    # print('concate_list3',location3.reshape((box3.shape[0], 4)).shape,class_info3.shape)
                    concate_list3 = [location3.reshape((box3.shape[0], 4)), class_info3]
                else:
                    concate_list3 = [location3.reshape((box3.shape[0], 4)), class_info3,
                                     keypoints3.reshape((box3.shape[0], -1))]
                box3 = np.concatenate(concate_list3, axis=-1)
                concate_list_all.append(box3)
        if len(concate_list_all) > 0:
            trans_boxes = np.concatenate(concate_list_all, axis=0)
            hw = clip(trans_boxes[:, :4][..., 2:] - trans_boxes[:, :4][..., :2], 0.0, None)
            area = hw[..., 0] * hw[..., 1]
            area_mask = area >= 1
            trans_boxes = trans_boxes[area_mask, :]
            return trans_boxes
        else:
            return None

    def _apply_mask(self, mask, spec: TensorSpec):
        keep, other_idxes, center, img1_offsetx, img1_offsety, img2_offsetx, img2_offsety, img3_offsetx, img3_offsety = self._shape_info
        height, width = self.output_size
        if keep:
            return mask

        mp1 = self.memory_cache[other_idxes[0]][spec].copy()[img1_offsety:img1_offsety + center[0],
              img1_offsetx:img1_offsetx + width - center[1]]
        mp2 = self.memory_cache[other_idxes[1]][spec].copy()[img2_offsety:img2_offsety + height - center[0],
              img2_offsetx:img2_offsetx + center[1]]
        mp3 = self.memory_cache[other_idxes[2]][spec].copy()[img3_offsety:img3_offsety + height - center[0],
              img3_offsetx:img3_offsetx + width - center[1]]

        base_msk = mask.copy()
        base_msk[:mp1.shape[0], center[1]:center[1] + mp1.shape[1]] = mp1
        base_msk[center[0]:center[0] + mp2.shape[0], :mp2.shape[1]] = mp2
        base_msk[center[0]:center[0] + mp3.shape[0], center[1]:center[1] + mp3.shape[1]] = mp3
        return base_msk

    def _get_shape(self, image):
        if self.output_size is None:
            self.output_size = image.shape[:2]
        rr = random.random()
        keep = rr < self.keep_prob or len(self.memory_cache[:-1]) < 3
        other_idxes = None
        center = None
        img1_offsetx = img1_offsety = img2_offsetx = img2_offsety = img3_offsetx = img3_offsety = 0
        if not keep:
            other_idxes = list(range(len(self.memory_cache[:-1])))
            random.shuffle(other_idxes)
            other_idxes = other_idxes[:3]

            height, width = self.output_size
            rs = np.random.uniform(0.5, 1.5, [2])  # random shift
            center = (int(height * rs[0] / 2), int(width * rs[1] / 2))

            img1 = self.memory_cache[other_idxes[0]].value_list[0]
            img1_offsetx = img1.shape[1] if img1.shape[1] - 1 < (width - center[1]) else np.random.choice(
                np.arange(0, img1.shape[1] - (width - center[1]), 1))
            img1_offsety = img1.shape[0] if img1.shape[0] - 1 < center[0] else np.random.choice(
                np.arange(0, img1.shape[0] - center[0], 1))

            img2 = self.memory_cache[other_idxes[1]].value_list[0]
            img2_offsetx = img2.shape[1] if img2.shape[1] - 1 < center[1] else np.random.choice(
                np.arange(0, img2.shape[1] - center[1], 1))
            img2_offsety = img2.shape[0] if img2.shape[0] - 1 < (height - center[0]) else np.random.choice(
                np.arange(0, img2.shape[0] - (height - center[0]), 1))

            img3 = self.memory_cache[other_idxes[2]].value_list[0]
            img3_offsetx = img3.shape[1] if img3.shape[1] - 1 < (width - center[1]) else np.random.choice(
                np.arange(0, img3.shape[1] - (width - center[1]), 1))
            img3_offsety = img3.shape[0] if img3.shape[0] - 1 < (height - center[0]) else np.random.choice(
                np.arange(0, img3.shape[0] - (height - center[0]), 1))

        return (
            keep, other_idxes, center, img1_offsetx, img1_offsety, img2_offsetx, img2_offsety, img3_offsetx,
            img3_offsety)


class ToLowResolution(VisionTransform):
    def __init__(self, scale=1 / 2, name='to_low_resolution', **kwargs):
        super().__init__(name)
        self.is_spatial = False
        self.scale = scale

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image, spec: TensorSpec):
        rnd = random.randint(0, 10)
        if rnd == 0:
            image = Rescale(scale=self.scale)(image)
        elif rnd % 3 == 0:
            image = Rescale(scale=math.pow(2, 3))(image)
            image = Rescale(scale=self.scale * math.pow(2, -3))(image)
        elif rnd % 3 == 1:
            image = RandomBlur(ksize_range=(1, 5))(image)
            image = Rescale(scale=self.scale)(image)
        elif rnd % 3 == 2:
            image = Rescale(scale=math.pow(2, 2))(image)
            image = Rescale(scale=math.pow(2, -2))(image)
            image = RandomBlur(ksize_range=(1, 5))(image)
            image = Rescale(scale=math.pow(2, 1))(image)
            image = Rescale(scale=math.pow(2, -1))(image)
            image = Rescale(scale=self.scale)(image)
        return image

    def _apply_coords(self, coords, spec: TensorSpec):
        return coords

    def _apply_mask(self, mask, spec: TensorSpec):
        return mask
