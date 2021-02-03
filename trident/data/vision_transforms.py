import builtins
import math
import numbers
import random
import inspect
from functools import wraps
from typing import Sequence, Tuple, Dict, Union, Optional
import collections
import  numpy as np
import cv2
from trident.backend.tensorspec import TensorSpec,object_type_inference


from trident.backend.pytorch_ops import tensor_to_shape

from trident.backend.common import OrderedDict
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec,object_type_inference
if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *
from trident.data.transform import VisionTransform

__all__ = ['Resize', 'ShortestEdgeResize', 'Rescale','RandomCrop','RandomRescaleCrop','RandomCenterCrop','RandomTransform','RandomTransformAffine',
           'AdjustBrightness','AdjustContrast','AdjustSaturation','AddNoise','AdjustHue','RandomAdjustHue','RandomAdjustBrightness','RandomAdjustContrast','RandomAdjustSaturation',
           'Normalize','Unnormalize','CLAHE','Lighting','HorizontalFlip','RandomMirror','AdjustGamma','RandomBlur','RandomAdjustGamma','Blur']




def randomize_with_validate(valid_range=None,no_change_value=None,**kwargs):
    def randomize_wrapper(cls):
        class Wrapper:
            def __init__(self, **kwargs):
                self.valid_range = None
                self.no_change_value = None
                if isinstance(valid_range,(tuple,list)) and len(valid_range)==2 and all([isinstance(t,numbers.Number) for t in valid_range]):
                    self.valid_range=valid_range
                if isinstance(no_change_value,numbers.Number):
                    self.no_change_value=float(no_change_value)

                rangs= dict([(k.replace( '_range',''),random.uniform(*v))   for k,v in kwargs.items() if  isinstance(v,tuple) and len(v)==2 ])
                other_rangs = dict([(k, v) for k, v in kwargs.items() if not( isinstance(v,tuple) and len(v)==2)])


                self.kwargs=kwargs

                argspec = inspect.getfullargspec( cls.__init__)
                _args=argspec.args
                _args.remove('self')
                _args.remove('name')

                for k,v in self.kwargs.items():
                    if k in _args:
                        _args.remove(k)
                self._args=_args
                if len(rangs.values())==0  and self.valid_range is not None:
                    if len(_args)==0 and  len(self.kwargs)==0:
                        self.wrap = cls()
                    else:
                        self.wrap = cls(random.uniform(*self.valid_range),**other_rangs)
                else:

                    self.wrap = cls(*list(rangs.values()),**other_rangs)
                self.rn=0

            def __call__(self,inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray]):
                rangs,other_rangs=self.set_random()
                if len(rangs)>0 and len(self.kwargs)>0:
                    for k,v in rangs:
                        setattr(self.wrap,k,v)
                else:
                    if len(self._args) > 0:
                        setattr(self.wrap,self._args[0],random.uniform(*self.valid_range))

                if self.rn % 5 > 0:
                    return self.wrap(inputs)
                else:
                    return inputs

            def set_random(self):
                self.rn = random.randint(0, 10)
                rangs = dict([(k.replace('_range', ''), random.uniform(*v)) for k, v in kwargs.items() if isinstance(v, tuple) and len(v) == 2])
                other_rangs = dict([(k, v) for k, v in kwargs.items() if not (isinstance(v, tuple) and len(v) == 2)])
                return rangs,other_rangs

        Wrapper.__name__ =  cls.__name__
        return Wrapper
    return randomize_wrapper


def randomize(cls):
    class Wrapper:
        def __init__(self, **kwargs):
            self.kwargs=kwargs
            self.wrap = cls()
            self.rn=0
        def __call__(self,inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray]):
            if self.rn % 5 > 0:
                return self.wrap(inputs)
            else:
                return inputs

        def set_random(self):
            self.rn = random.randint(0, 10)

    Wrapper.__name__ =  cls.__name__
    return Wrapper



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
        self.is_spatial=True
        self.output_size = output_size
        self.keep_aspect=keep_aspect
        self.align_corner=align_corner
        self.interpolation = interpolation
        self.scale=1

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        self._shape_info = self._get_shape(image)
        h, w, th, tw,pad_vert,pad_horz = self._shape_info

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
            self.scale = min(float(eh) / h, float(ew) / w)
            th =  int(builtins.round(h*self.scale,0))
            tw=  int( builtins.round(w*self.scale,0))
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
        self.is_spatial = True
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
        self.is_spatial = True
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
        self.is_spatial = True
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

    def __init__(self, output_size,scale_range=(0.8, 1.2), interpolation=cv2.INTER_LINEAR,name='random_center_crop',**kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        self.scale_range=scale_range
        self.interpolation = interpolation

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        self._shape_info = self._get_shape(image)
        x, y, th, tw, eh, ew = self._shape_info

        crop_image = image[y: y + th, x: x + tw]
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
        current_scale = np.random.uniform(*self.scale_range)
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        eh, ew = self.output_size

        h, w, _ = image.shape
        th, tw = builtins.min(int(h * current_scale),eh), builtins.min(int(w * current_scale),ew)

        x = int(round((w - tw) / 2.0))
        y = int(round((h - th) / 2.0))
        return x, y, th, tw, eh, ew


class RandomCrop(VisionTransform):
    r"""
    Crop the input data randomly. Before applying the crop transform,
    pad the image first. If target size is still bigger than the size of
    padded image, pad the image size to target size.
    :param output_size: target size of image, with (height, width) shape.

    """

    def __init__(self, output_size,name='random_crop',**kwargs):
        super().__init__(name)
        self.is_spatial = True
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
        self.is_spatial = True
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

        image=np.clip(image.astype(np.float32),0,255)[:,:,:3]
        image= cv2.warpAffine(image.copy(), mat_img,dsize= (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))  # , borderMode=cv2.BORDER_REPLICATE
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
        mask = unpack_singleton(mask)
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

RandomTransform=RandomTransformAffine


class HorizontalFlip(VisionTransform):
    def __init__(self,name='horizontal_flip',**kwargs):
        super().__init__(name)
        self.is_spatial = True

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        image=unpack_singleton(image)
        self._shape_info = self._get_shape(image)
        height, width = self._shape_info
        return image[:,::-1]

    def _apply_coords(self, coords,spec:TensorSpec):
        height, width = self._shape_info
        coords[:, 0::2]=width-coords[:, 0::2]
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        mask = unpack_singleton(mask)
        height, width= self._shape_info
        return mask[:, ::-1]

    def _get_shape(self, image):
        height, width, _ = image.shape
        return height, width

@randomize
class RandomMirror(HorizontalFlip):
    pass

class Normalize(VisionTransform):
    r"""
    Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    :param mean: sequence of means for each channel.
    :param std: sequence of standard deviations for each channel.

    """

    def __init__(self, mean=0.0, std=1.0,name='normalize',**kwargs):
        super().__init__(name)
        self.mean = mean
        self.std=std

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        image = image.astype(np.float32)
        norm_mean=self.mean
        norm_std=self.std
        if isinstance(self.mean, numbers.Number) and image.ndim == 3:
            norm_mean = np.array([self.mean, self.mean, self.mean]).astype(np.float32)
            norm_mean = np.expand_dims(norm_mean, 0)
            norm_mean = np.expand_dims(norm_mean, 0)
        if isinstance(self.std, numbers.Number) and image.ndim == 3:
            norm_std = np.array([self.std, self.std, self.std]).astype(np.float32)
            norm_std = np.expand_dims(norm_std, 0)
            norm_std = np.expand_dims(norm_std, 0)
        if image.ndim == 3:
            image -= norm_mean
            image /= norm_std
            return image
        elif image.ndim == 2:
            if isinstance(norm_mean, numbers.Number) and isinstance(norm_std, numbers.Number):
                image -= norm_mean
                image /=norm_std
                return image
        return image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask

class Unnormalize(VisionTransform):
    r"""
    Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    :param mean: sequence of means for each channel.
    :param std: sequence of standard deviations for each channel.

    """

    def __init__(self, mean=0.0, std=1.0,name='normalize',**kwargs):
        super().__init__(name)
        self.mean = mean
        self.std=std

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        image = image.astype(np.float32)
        norm_mean=self.mean
        norm_std=self.std
        if isinstance(self.mean, numbers.Number) and image.ndim == 3:
            norm_mean = np.array([self.mean, self.mean, self.mean]).astype(np.float32)
            norm_mean = np.expand_dims(norm_mean, 0)
            norm_mean = np.expand_dims(norm_mean, 0)
        if isinstance(self.std, numbers.Number) and image.ndim == 3:
            norm_std = np.array([self.std, self.std, self.std]).astype(np.float32)
            norm_std = np.expand_dims(norm_std, 0)
            norm_std = np.expand_dims(norm_std, 0)
        if image.ndim == 3:
            image *= norm_std
            image += norm_mean

            return image
        elif image.ndim == 2:
            if isinstance(norm_mean, numbers.Number) and isinstance(norm_std, numbers.Number):
                image *= norm_std
                image += norm_mean
                return image
        return image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask

class AddNoise(VisionTransform):
    r"""
    Normalize the input data with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input data.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    :param mean: sequence of means for each channel.
    :param std: sequence of standard deviations for each channel.

    """

    def __init__(self, intensity=0.1,name='add_noise',**kwargs):
        super().__init__(name)
        self.intensity = intensity


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        rr = random.randint(0, 10)
        orig_min = image.min()
        orig_max = image.max()
        noise = np.random.standard_normal(image.shape) * (self.intensity * (orig_max - orig_min))
        if rr % 2 == 0:
            noise = np.random.uniform(-1, 1, image.shape) * (self.intensity * (orig_max - orig_min))
        image = np.clip(image + noise, orig_min, orig_max)
        return image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


class AdjustBrightness(VisionTransform):
    """Adjust brightness of an Image.
        Args:
            value (float):  How much to adjust the brightness. Can be
                any non negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.
        Returns:
            np.ndarray: Brightness adjusted image.
    """
    def __init__(self, value=0,name='adjust_brightness',**kwargs):
        super().__init__(name)
        if value < 0:
            raise ValueError("brightness value should be non-negative")
        self.value = value


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if self.value == 0:
            return image
        image = image.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - self.value), 1 + self.value)
        image = image * alpha
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask

class AdjustContrast(VisionTransform):
    """Adjust contrast of an Image.
    Args:
        value (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        np.ndarray: Contrast adjusted image.
    """

    def __init__(self, value=0,name='adjust_contrast',**kwargs):
        super().__init__(name)
        if value < 0:
            raise ValueError("contrast value should be non-negative")
        self.value = value


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if self.value == 0:
            return image
        image = image.astype(np.float32)
        mean = round(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean())
        image = (1 -  self.value) * mean +  self.value * image
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self,mask,spec:TensorSpec):
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

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if self.value == 0:
            return image

        dtype = image.dtype
        image = image.astype(np.float32)
        degenerate = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        image = (1 - self.value) * degenerate + self.value * image
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask

class AdjustHue(VisionTransform):
    r"""
       Adjust hue of the input data.
       :param value: how much to adjust the hue. Can be any number
           between 0 and 0.5, 0 gives the original image.

    """

    def __init__(self, value=0,name='adjust_hue',**kwargs):
        super().__init__(name)
        if value < -0.5 or value > 0.5:
            raise ValueError("hue value should be in [0.0, 0.5]")
        self.value = value


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if self.value == 0:
            return image
        image = image.astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        hsv[..., 0] += np.uint8(self.value * 255)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
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

    def __init__(self, gamma=1, gain=1,name='adjust_gamma', **kwargs):
        super().__init__(name)
        if gamma < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.gamma=gamma
        self.gain=gain
    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if self.gamma == 1:
            return image
        image = image.astype(np.float32)
        image = 255. * self.gain * np.power(image / 255., self.gamma)
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


@randomize_with_validate(valid_range=(0., 3.), no_change_value=1.)
class RandomAdjustBrightness(AdjustBrightness):
    pass

@randomize_with_validate(valid_range=(0., 3.), no_change_value=1.)
class RandomAdjustContrast(AdjustContrast):
    pass

@randomize_with_validate(valid_range=(0., 3.), no_change_value=1.)
class RandomAdjustSaturation(AdjustSaturation):
    pass


@randomize_with_validate(valid_range=(-0.5, 0.5), no_change_value=0.)
class RandomAdjustHue(AdjustHue):
    pass

@randomize_with_validate(valid_range=(0., 3.), no_change_value=1.)
class RandomAdjustGamma(AdjustGamma):
    pass


class Blur(VisionTransform):
    def __init__(self, ksize=5,name='blur',**kwargs):
        super().__init__(name)
        if ksize <= 0:
            raise ValueError("lighting scale should be positive")
        self.ksize = int(ksize)


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if self.ksize == 0:
            return image
        blur = cv2.GaussianBlur(image, (int(self.ksize), int(self.ksize)),cv2.BORDER_DEFAULT)
        return blur.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


@randomize_with_validate(valid_range=(1, 20), no_change_value=1)
class RandomBlur(Blur):
    pass

class Lighting(VisionTransform):


    def __init__(self, scale=0,name='lighting',**kwargs):
        super().__init__(name)
        if scale < 0:
            raise ValueError("lighting scale should be non-negative")
        self.scale = scale
        self.eigvec = np.array(
            [
                [-0.5836, -0.6948, 0.4203],
                [-0.5808, -0.0045, -0.8140],
                [-0.5675, 0.7192, 0.4009],
            ]
        )  # reverse the first dimension for BGR
        self.eigval = np.array([0.2175, 0.0188, 0.0045])


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if self.scale == 0:
            return image

        dtype = image.dtype
        image = image.astype(np.float32)
        alpha = np.random.normal(scale=self.scale, size=3)
        image = image + self.eigvec.dot(alpha * self.eigval)
        return image.clip(0, 255).astype(dtype)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


class CLAHE(VisionTransform):


    def __init__(self, clipLimit=5,gridsize=8,name='clahe',**kwargs):
        super().__init__(name)
        self.gridsize=gridsize
        self.clipLimit=clipLimit



    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.gridsize, self.gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask