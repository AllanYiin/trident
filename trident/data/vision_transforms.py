import math
import numbers
import random
from typing import Sequence, Tuple, Dict, Union, Optional
import collections
import  numpy as np
import cv2
from trident.data.image_common import object_type_inference

from trident.backend.pytorch_ops import tensor_to_shape

from trident.backend.common import OrderedDict

from trident.backend.tensorspec import TensorSpec
from  trident.data.transform import VisionTransform
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec
if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *

__all__ = ['Resize', 'ShortestEdgeResize', 'Rescale','RandomCrop','RandomRescaleCrop','RandomCenterCrop','RandomTransform']

class Resize(VisionTransform):
    r"""
    Resize the input data.
    :param output_size: target size of image, with (height, width) shape.
    :param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    :param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, output_size,keep_aspect=True, align_corner=True, interpolation=cv2.INTER_LINEAR,name='resize',**kwargs):
        super().__init__(name)
        self.output_size = output_size
        self.keep_aspect=keep_aspect
        self.align_corner=align_corner
        self.interpolation = interpolation

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        self._shape_info = self._get_shape(image)
        h, w, th, tw,pad_vert,pad_horz = self._shape_info

        if h == th and w == tw:
            return image
        if self.keep_aspect == False:
            return cv2.resize(image, (tw,th), self.interpolation)
        else:

            image=cv2.resize(image, (tw,th), self.interpolation)
            output=np.zeros((*self.output_size,3))
            if self.align_corner==True:
                output[:th,:tw,:]=image
            else:
                output[pad_vert//2:th+pad_vert, pad_horz//2:tw+pad_horz, :] = image
            return output

    def _apply_coords(self, coords,spec:TensorSpec):
        h, w, th, tw,pad_vert,pad_horz = self._shape_info
        if h == th and w == tw:
            return coords
        coords[:, 0] = coords[:, 0] * (tw / w)
        coords[:, 1] = coords[:, 1] * (th / h)
        if self.align_corner == False:
            coords[:, 0] +=pad_horz//2
            coords[:, 1] +=pad_vert//2
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        h, w, th, tw,pad_vert,pad_horz = self._shape_info
        if h == th and w == tw:
            return mask

        if self.keep_aspect == False:
            return cv2.resize(mask, (tw, th), cv2.INTER_NEAREST)
        else:

            mask = cv2.resize(mask, (tw, th),cv2.INTER_NEAREST)
            output = np.zeros((*self.output_size, 3))
            if mask.ndim==2:
                output = np.zeros((*self.output_size, 1))
            elif mask.ndim==3 :
                output = np.zeros((*self.output_size, mask.shape[-1]))

            if self.align_corner == True:
                output[:th, :tw, :] = mask
            else:
                output[pad_vert // 2:th + pad_vert , pad_horz // 2:tw + pad_horz , :] = mask
            return np.squeeze(output)

    def _get_shape(self, image):
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        h, w,c = image.shape
        eh, ew = self.output_size

        if self.keep_aspect==False:
            return h, w,  eh, ew,0,0
        else:
            scale = min(eh / h, ew / w)
            th =  int( h*scale)
            tw=  int( w*scale)
            pad_vert = eh - th
            pad_horz = ew-tw
            return h,w,th,tw,pad_vert,pad_horz

class ShortestEdgeResize(VisionTransform):
    def __init__(
        self,
        min_size,
        max_size,
        sample_style="range",
        interpolation=cv2.INTER_LINEAR,
        *,
        order=None
    ):
        super().__init__(order)
        if sample_style not in ("range", "choice"):
            raise NotImplementedError(
                "{} is unsupported sample style".format(sample_style)
            )
        self.sample_style = sample_style
        if isinstance(min_size, int):
            min_size = (min_size, min_size)
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def apply(self, input: Tuple,spec:TensorSpec):

        return super().apply(input)

    def _apply_image(self, image,spec:TensorSpec):
        self._shape_info = self._get_shape(input)
        h, w, th, tw = self._shape_info
        if h == th and w == tw:
            return image
        return cv2.resize(image, (tw,th), self.interpolation)

    def _apply_coords(self, coords,spec:TensorSpec):
        h, w, th, tw = self._shape_info
        if h == th and w == tw:
            return coords
        coords[:, 0] = coords[:, 0] * (tw / w)
        coords[:, 1] = coords[:, 1] * (th / h)
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        h, w, th, tw = self._shape_info
        if h == th and w == tw:
            return mask
        return cv2.resize(mask,(tw,th), cv2.INTER_NEAREST)

    def _get_shape(self, image):
        h, w = image.shape[:2]
        if self.sample_style == "range":
            size = np.random.randint(self.min_size[0], self.min_size[1] + 1)
        else:
            size = np.random.choice(self.min_size)

        scale = size / min(h, w)
        if h < w:
            th, tw = size, scale * w
        else:
            th, tw = scale * h, size
        if max(th, tw) > self.max_size:
            scale = self.max_size / max(th, tw)
            th = th * scale
            tw = tw * scale
        th = int(round(th))
        tw = int(round(tw))
        return h, w, th, tw

class Rescale(VisionTransform):
    r"""
    Resize the input data.
    :param output_size: target size of image, with (height, width) shape.
    :param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    :param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, scale, interpolation=cv2.INTER_LINEAR,name='rescale',**kwargs):
        super().__init__(name)
        self.scale = scale
        self.output_size = None
        self.interpolation = interpolation

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        h, w, _ = image.shape
        self.output_size =(int(w*self.scale),int(h*self.scale))
        return cv2.resize(image, (int(w*self.scale),int(h*self.scale)), self.interpolation)

    def _apply_coords(self, coords,spec:TensorSpec):
        coords[:, 0] = coords[:, 0] *self.scale
        coords[:, 1] = coords[:, 1] *self.scale
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return cv2.resize(mask, self.output_size,  cv2.INTER_NEAREST)


class RandomRescaleCrop(VisionTransform):
    r"""
    Resize the input data.
    :param output_size: target size of image, with (height, width) shape.
    :param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    :param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, output_size,scale_range=(0.1, 3.0),ratio_range=(3.0 / 4, 4.0 / 3), interpolation=cv2.INTER_LINEAR,name='random_rescale_crop',**kwargs):
        super().__init__(name)
        self.output_size = output_size
        self.scale_range=scale_range
        self.ratio_range=ratio_range
        self.interpolation = interpolation

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        self._shape_info = self._get_shape(image)
        x, y, w, h ,eh,ew = self._shape_info

        cropped_img = image[y: y + h, x: x + w]
        return cv2.resize(cropped_img, (ew,eh), self.interpolation)

    def _apply_coords(self, coords,spec:TensorSpec):
        x, y, w, h ,eh,ew = self._shape_info
        coords[:, 0] = (coords[:, 0] - x) * ew / w
        coords[:, 1] = (coords[:, 1] - y) * eh / h
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        x, y, w, h, eh, ew = self._shape_info

        cropped_mask = mask[y: y + h, x: x + w]
        return cv2.resize(cropped_mask, (ew, eh), cv2.INTER_NEAREST)


    def _get_shape(self, image):
        height, width, _ = image.shape
        area = height * width
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        eh, ew = self.output_size

        for _ in range(10):
            target_area = np.random.uniform(*self.scale_range) * area
            log_ratio = tuple(math.log(x) for x in self.ratio_range)
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                x = np.random.randint(0, width - w + 1)
                y = np.random.randint(0, height - h + 1)
                return x, y, w, h,eh, ew

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio_range):
            w = width
            h = int(round(w / min(self.ratio_range)))
        elif in_ratio > max(self.ratio_range):
            h = height
            w = int(round(h * max(self.ratio_range)))
        else:  # whole image
            w = width
            h = height
        x = (width - w) // 2
        y = (height - h) // 2
        return x, y, w, h,eh,ew

class RandomCenterCrop(VisionTransform):
    r"""
    Resize the input data.
    :param output_size: target size of image, with (height, width) shape.
    :param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    :param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, output_size,scale_range=(0.1, 0.99), interpolation=cv2.INTER_LINEAR,name='random_center_crop',**kwargs):
        super().__init__(name)
        self.output_size = output_size
        self.scale_range=scale_range
        self.interpolation = interpolation

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        self._shape_info = self._get_shape(image)
        x, y,th,tw,eh, ew= self._shape_info

        crop_image= image[y: y + th, x: x + tw]
        return cv2.resize(crop_image, (ew, eh), self.interpolation)

    def _apply_coords(self, coords,spec:TensorSpec):
        x, y, th, tw, eh, ew = self._shape_info
        coords[:, 0] -= x
        coords[:, 1] -= y
        coords[:, 0] = (coords[:, 0]*true_divide(ew,tw)).astype(np.int)
        coords[:, 1] =(coords[:, 1]*(true_divide(eh,th))).astype(np.int)
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        x, y, th, tw, eh, ew = self._shape_info

        crop_imask = mask[y: y + th, x: x + tw]
        return cv2.resize(crop_imask, (ew, eh),cv2.INTER_NEAREST)


    def _get_shape(self, image):
        current_scale=np.random.uniform(*self.scale_range)
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        eh, ew = self.output_size



        h, w, _ = image.shape
        th, tw = int(h * current_scale), int(w * current_scale)
        assert th <= h and tw <= w, "output size is bigger than image size"
        x = int(round((w - tw) / 2.0))
        y = int(round((h - th) / 2.0))
        return x, y,th,tw,eh, ew


class RandomCrop(VisionTransform):
    r"""
    Crop the input data randomly. Before applying the crop transform,
    pad the image first. If target size is still bigger than the size of
    padded image, pad the image size to target size.
    :param output_size: target size of image, with (height, width) shape.

    """

    def __init__(self, output_size,name='random_crop',**kwargs):
        super().__init__(name)
        self.output_size = output_size



    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        self._shape_info = self._get_shape(image)
        h,w,eh, ew,offset_x,offset_y,offset_x1,offset_y1 = self._shape_info
        if image.ndim == 2:
            output = np.zeros(self.output_size)
            crop_im = image[offset_y:min(offset_y + eh,h), offset_x:min(offset_x + ew,w)]
            output[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1]] = crop_im
            return output
        elif image.ndim == 3:
            output=np.zeros((*self.output_size,3))
            crop_im = image[offset_y:min(offset_y + eh,h), offset_x:min(offset_x + ew,w),:]
            output[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1],:] = crop_im
            return output

    def _apply_coords(self, coords,spec:TensorSpec):
        h,w,eh, ew,offset_x,offset_y,offset_x1,offset_y1 = self._shape_info
        coords[:, 0] = coords[:, 0]- offset_x+offset_x1
        coords[:, 1] = coords[:, 1]  - offset_y+offset_y1

        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        h, w, eh, ew, offset_x, offset_y, offset_x1, offset_y1 = self._shape_info
        if mask.ndim == 2:
            output = np.zeros(self.output_size)
            crop_mask = mask[offset_y:min(offset_y + eh,h), offset_x:min(offset_x + ew,w)]
            output[offset_y1:offset_y1 + crop_mask.shape[0], offset_x1:offset_x1 + crop_mask.shape[1]] = crop_mask
            return output
        elif mask.ndim == 3:
            output = np.zeros((*self.output_size, 3))
            crop_mask = mask[offset_y:min(offset_y + eh,h), offset_x:min(offset_x + ew,w), :]
            output[offset_y1:offset_y1 + crop_mask.shape[0], offset_x1:offset_x1 + crop_mask.shape[1], :] = crop_mask
            return output

    def _get_shape(self, image):
        h, w, _ = image.shape
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        eh, ew = self.output_size


        offset_x = 0
        offset_y = 0

        if w > ew:
            offset_x = random.choice(range(w - ew))
        if h > eh:
            offset_y = random.choice(range(h - eh))
        offset_x1 = random.choice(range(ew - w)) if ew > w else 0
        offset_y1 = random.choice(range(eh - h)) if eh > h else 0
        return h,w,eh, ew,offset_x,offset_y,offset_x1,offset_y1

class RandomTransform(VisionTransform):
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
    :param output_size: target size of image, with (height, width) shape.
    :param interpolation: interpolation method. All methods are listed below:
        * cv2.INTER_NEAREST – a nearest-neighbor interpolation.
        * cv2.INTER_LINEAR – a bilinear interpolation (used by default).
        * cv2.INTER_AREA – resampling using pixel area relation.
        * cv2.INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood.
        * cv2.INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
    :param order: the same with :class:`VisionTransform`.
    """

    def __init__(self, rotation_range=15,zoom_range=0.02, shift_range=0.02,  shear_range= 0.2,random_flip= 0.15, interpolation=cv2.INTER_LINEAR,name='resize',**kwargs):
        super().__init__(name)
        self.output_size =None
        self.rotation_range = rotation_range
        self.shift_range=shift_range
        self.zoom_range=zoom_range
        self.shear_range=shear_range
        self.interpolation = interpolation
        self.random_flip=random_flip

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        image=unpack_singleton(image)
        self._shape_info = self._get_shape(image)
        mat_img, height, width,is_flip = self._shape_info
        self.output_size=(height, width)
        image=np.clip(image.astype(np.float32),0,255)[:,:,:3]
        image= cv2.warpAffine(image.copy(), mat_img, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))  # , borderMode=cv2.BORDER_REPLICATE
        if is_flip:
            return image[:,::-1]
        else:
            return image
    def _apply_coords(self, coords,spec:TensorSpec):
        mat_img, height, width,is_flip = self._shape_info
        #
        coords = coords.transpose([1, 0])
        coords = np.insert(coords, 2, 1, axis=0)
        # # print(coords)
        # # print(transform_matrix)
        coords_result = np.matmul(mat_img, coords)
        coords_result = coords_result[0:2, :].transpose([1, 0])
        if is_flip:
            coords_result[:, 0::2]=width-coords_result[:, 0::2]
        return coords_result

    def _apply_mask(self, mask,spec:TensorSpec):
        mat_img, height, width,is_flip = self._shape_info
        mask =cv2.warpAffine(mask, mat_img, (width, height), borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))  # , borderMode=cv2.BORDER_REPLICATE
        if is_flip:
            return mask[:, ::-1]
        else:
            return mask
    def _get_shape(self, image):
        h, w, c = image.shape
        self.output_size =(h, w)

        # rotation
        angle = random.uniform(-1*self.rotation_range, self.rotation_range)
        scale = 1.0
        shear=0
        translations = (0, 0)
        # translation
        if self.shift_range is not None:
            if isinstance(self.shift_range, numbers.Number):
                self.shift_range = (-1 * self.shift_range, self.shift_range)
            max_dx = self.shift_range[0] * image.shape[0]
            max_dy = self.shift_range[1] * image.shape[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        # scale
        if self.zoom_range is not None:
            scale = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        else:
            scale = 1.0

        # shear
        if self.shear_range is not None:
            if isinstance(self.shear_range, numbers.Number):
                shear = [random.uniform(-1 * self.shear_range, self.shear_range), 0.]
            elif len(self.shear_range) == 2:
                shear = [random.uniform(self.shear_range[0], self.shear_range[1]), 0.]
            elif len(self.shear_range) == 4:
                shear = [random.uniform(self.shear_range[0], self.shear_range[1]),
                         random.uniform(self.shear_range[2], self.shear_range[3])]
        else:
            shear = 0.0

        self.output_size =  image.shape[0]
        center = ( image.shape[0] * 0.5 + 0.5,  image.shape[1] * 0.5 + 0.5)

        angle = math.radians(angle)
        # if isinstance(shear, (tuple, list)) and len(shear) == 2:
        #     shear = [math.radians(s) for s in shear]
        # elif isinstance(shear, numbers.Number):
        #     shear = math.radians(shear)
        #     shear = [shear, 0]
        # else:
        #     raise ValueError(
        #         "Shear should be a single value or a tuple/list containing " +
        #         "two values. Got {}.".format(shear))

        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
            math.sin(angle + shear[0]) * math.sin(angle + shear[1])
        matrix = [
            math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
            -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translations[0]) + matrix[1] * (-center[1] - translations[1])
        matrix[5] += matrix[3] * (-center[0] - translations[0]) + matrix[4] * (-center[1] - translations[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        rr=np.random.random()
        return np.array(matrix).reshape((2,3)),h, w,rr< self.random_flip
