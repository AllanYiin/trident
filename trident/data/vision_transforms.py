import builtins
import math
import numbers
import random
import inspect
from collections import Iterable
from functools import wraps
from typing import Sequence, Tuple, Dict, Union, Optional
import collections
import  numpy as np
import cv2
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_minimum, threshold_local, threshold_isodata, threshold_yen

from trident.backend.tensorspec import TensorSpec, object_type_inference, ObjectType


from trident.backend.common import OrderedDict
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec,object_type_inference

if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *
from trident.data.transform import VisionTransform

__all__ = ['Resize', 'ShortestEdgeResize', 'Rescale','RandomCrop','RandomRescaleCrop','RandomCenterCrop','RandomMultiScaleImage','RandomTransform','RandomTransformAffine',
           'AdjustBrightness','AdjustContrast','AdjustSaturation','AddNoise','AdjustHue','RandomAdjustHue','RandomAdjustBrightness','RandomAdjustContrast','RandomAdjustSaturation',
           'Normalize','Unnormalize','CLAHE','Lighting','HorizontalFlip','RandomMirror','AdjustGamma','RandomBlur','RandomAdjustGamma','Blur','InvertColor','RandomInvertColor','GrayScale','RandomGrayScale',
           'ImageDilation','ImageErosion','ErosionThenDilation','DilationThenErosion','AdaptiveBinarization','SaltPepperNoise','RandomErasing','ToRGB','ImageMosaic']




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

            def __call__(self,inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray],spec:TensorSpec=None,**kwargs):
                rangs,other_rangs=self.set_random()
                if len(rangs)>0 and len(self.kwargs)>0:
                    for k,v in rangs:
                        setattr(self.wrap,k,v)
                else:
                    if len(self._args) > 0:
                        setattr(self.wrap,self._args[0],random.uniform(*self.valid_range))

                if self.rn % 5 > 0:
                    return self.wrap(inputs,spec=spec,**kwargs)
                else:
                    return inputs

            def set_random(self):
                self.rn = random.randint(0, 10)
                rangs = dict([(k.replace('_range', ''), random.uniform(*v)) for k, v in kwargs.items() if isinstance(v, tuple) and len(v) == 2])
                other_rangs = dict([(k, v) for k, v in kwargs.items() if not (isinstance(v, tuple) and len(v) == 2)])
                return rangs,other_rangs

        Wrapper.__name__ = Wrapper.__qualname__ =  cls.__name__
        Wrapper.__doc__ = cls.__doc__
        return Wrapper
    return randomize_wrapper


def randomize(cls):
    class Wrapper:
        def __init__(self, **kwargs):
            self.kwargs=kwargs
            self.wrap = cls()
            self.rn=0
        def __call__(self,inputs: Union[Dict[TensorSpec,np.ndarray],np.ndarray],spec:TensorSpec=None,**kwargs):
            if self.rn % 5 > 0:
                return self.wrap(inputs,spec=spec,**kwargs)
            else:
                return inputs

        def set_random(self):
            self.rn = random.randint(0, 10)

    Wrapper.__name__ = Wrapper.__qualname__ = cls.__name__
    Wrapper.__doc__ = cls.__doc__
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

    def __init__(self, output_size,keep_aspect=True, align_corner=True, interpolation=cv2.INTER_LANCZOS4,name='resize',**kwargs):
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
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        h, w, th, tw,pad_vert,pad_horz = self._shape_info

        if not self.keep_aspect:
            return cv2.resize(image.copy(), (tw,th), interpolation=self.interpolation)
        else:

            resized_image=cv2.resize(image.copy(), (tw,th), interpolation=self.interpolation)
            shp=list(int_shape(resized_image))
            shp[:2] = self.output_size

            output=np.zeros(shp)
            if self.align_corner:
                if ndim(resized_image)==2 :
                    output[:th,:tw]=resized_image
                elif  ndim(resized_image) == 3 :
                    output[:th, :tw, :] = resized_image
            else:
                if ndim(resized_image) == 2 :
                    output[pad_vert//2:th+pad_vert//2, pad_horz//2:tw+pad_horz//2] = resized_image
                elif ndim(resized_image) == 3 :
                    output[pad_vert // 2:th + pad_vert // 2, pad_horz // 2:tw + pad_horz // 2,:] = resized_image
            return output

    def _apply_coords(self, coords,spec:TensorSpec):
        h, w, th, tw,pad_vert,pad_horz = self._shape_info
        if h == th and w == tw:
            return coords
        coords[:, 0] = coords[:, 0] * (tw / w)
        coords[:, 1] = coords[:, 1] * (th / h)
        if not self.align_corner:
            coords[:, 0] +=pad_horz//2
            coords[:, 1] +=pad_vert//2
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        h, w, th, tw,pad_vert,pad_horz = self._shape_info
        if h == th and w == tw:
            return mask
        mask_dtype=mask.dtype
        mask=mask.astype(np.float32)
        if not self.keep_aspect:
            mask=cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            return mask.astype(mask_dtype)
        else:

            mask = cv2.resize(mask, (tw, th),interpolation=cv2.INTER_NEAREST).astype(np.float32)
            output = np.zeros((*self.output_size, 3)).astype(np.float32)
            if mask.ndim==2:
                output = np.zeros(self.output_size).astype(np.float32)
            elif mask.ndim==3 :
                output = np.zeros((*self.output_size, mask.shape[-1])).astype(np.float32)

            if self.align_corner:
                if mask.ndim == 2:
                    output[:th, :tw] = mask
                elif mask.ndim == 3:
                    output[:th, :tw, :] = mask
            else:
                if mask.ndim == 2:
                    output[pad_vert // 2:th + pad_vert // 2 , pad_horz // 2:tw +pad_horz // 2] = mask
                elif mask.ndim == 3:
                    output[pad_vert // 2:th + pad_vert // 2 , pad_horz // 2:tw +pad_horz // 2 , :] = mask
            mask = np.squeeze(output)
            return mask.astype(mask_dtype)

    def _get_shape(self, image):
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        h, w = image.shape[:2]
        eh, ew = self.output_size

        if not self.keep_aspect:
            return h, w,  eh, ew,0,0
        else:
            self.scale = min(float(eh) / h, float(ew) / w)
            th =  int(builtins.round(h*self.scale,0))
            tw=  int( builtins.round(w*self.scale,0))
            pad_vert = eh - th
            pad_horz = ew-tw
            return h,w,th,tw,pad_vert,pad_horz



class ShortestEdgeResize(VisionTransform):
    def __init__(self,output_size,keep_aspect=True,interpolation=cv2.INTER_LANCZOS4,name='short_edge_resize',**kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size=output_size
        self.keep_aspect=keep_aspect

        self.interpolation = interpolation

    def apply(self, input: Tuple, spec: TensorSpec):
        return super().apply(input, spec)

    def _apply_image(self, image,spec:TensorSpec):
        if  self._shape_info is None:
            self._shape_info = self._get_shape(image)
        h, w, th, tw, eh, ew,offsetx,offsety = self._shape_info
        if h == eh and w == ew:
            return image
        image=cv2.resize(image, (tw,th), self.interpolation)
        if ndim(image)==2:
            return image.copy()[offsety:offsety+eh,offsetx:offsetx+ew]
        elif ndim(image)==3:
            return image.copy()[offsety:offsety+eh,offsetx:offsetx+ew,:]


    def _apply_coords(self, coords,spec:TensorSpec):
        h, w, th, tw, eh, ew,offsetx,offsety = self._shape_info
        if h == eh and w == ew:
            return coords
        coords[:, 0] = clip(coords[:, 0] *self.scale-offsetx,0,ew)
        coords[:, 1] = clip(coords[:, 1] *self.scale-offsety,0,eh)
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        h, w, th, tw, eh, ew, offsetx, offsety = self._shape_info
        if h == eh and w == ew:
            return mask
        mask = cv2.resize(mask, (tw, th), cv2.INTER_NEAREST)
        if ndim(mask) == 2:
            return mask.copy()[offsety:offsety + eh, offsetx:offsetx + ew]
        elif ndim(mask) == 3:
            return mask.copy()[offsety:offsety + eh, offsetx:offsetx + ew, :]

    def _get_shape(self, image):
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        h, w = image.shape[:2]
        eh,ew=self.output_size

        self.scale = builtins.max(eh/h,ew/w)
        th = int(builtins.round(h * self.scale, 0))
        tw = int(builtins.round(w * self.scale, 0))

        offsetx=int(random.randint(0,int(tw-ew)) if tw-ew>=1 else 0)
        offsety = int(random.randint(0, int(th - eh)) if th-eh>=1 else 0)
        return h, w, th, tw, eh, ew,offsetx,offsety






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

    def __init__(self, scale, interpolation=cv2.INTER_LANCZOS4,name='rescale',**kwargs):
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

    def __init__(self, output_size,scale_range=(0.1, 3.0), interpolation=cv2.INTER_LANCZOS4,name='random_rescale_crop',**kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        self.scale_range=scale_range
        self.interpolation = interpolation

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if  self._shape_info is None:
            self._shape_info = self._get_shape(image)
        x, y, th,tw,eh, ew,target_scale= self._shape_info
        image=cv2.resize(image, (tw, th), interpolation=self.interpolation)
        cropped_img = image[y: builtins.min(y + eh, th), x: builtins.min(x + ew, tw)]
        if cropped_img.shape[0]==eh and cropped_img.shape[1]==ew:
            return cropped_img
        else:
            background=np.zeros((eh,ew,1 if spec is not None and spec.object_type==ObjectType.gray else 3))
            background[builtins.max(eh-cropped_img.shape[0],0)//2:builtins.max(eh-cropped_img.shape[0],0)//2+cropped_img.shape[0],builtins.max(ew-cropped_img.shape[1],0)//2:builtins.max(ew-cropped_img.shape[1],0)//2+cropped_img.shape[1],:]=cropped_img
            return background


    def _apply_coords(self, coords,spec:TensorSpec):
        x, y, th,tw,eh, ew,target_scale= self._shape_info
        coords[:, 0] = np.clip(coords[:, 0]*target_scale -x,0,ew)
        coords[:, 1] = np.clip(coords[:, 1] *target_scale-y,0,eh)
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        x, y, th,tw,eh, ew,target_scale= self._shape_info
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        mask_shape=list(int_shape(mask))
        mask_shape[0]=eh
        mask_shape[1] = ew
        cropped_mask = mask[y: builtins.min(y + eh, th), x: builtins.min(x + ew, tw)]
        if cropped_mask.shape[0]==eh and cropped_mask.shape[1]==ew:
            return cropped_mask
        else:
            background=np.zeros(tuple(mask_shape))
            background[builtins.max(eh-cropped_mask.shape[0],0)//2:builtins.max(eh-cropped_mask.shape[0],0)//2+cropped_mask.shape[0],builtins.max(ew-cropped_mask.shape[1],0)//2:builtins.max(ew-cropped_mask.shape[1],0)//2+cropped_mask.shape[1]]=cropped_mask
            return background


    def _get_shape(self, image):
        height, width= image.shape[:2]
        area = height * width
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        eh, ew = self.output_size

        target_scale= np.random.uniform(self.scale_range[0],self.scale_range[1])
        th,tw=int(round(height*target_scale)),int(round(width*target_scale))
        x,y=0,0

        if 0 < ew < tw-1 and 0 < eh < th-1:
            x = np.random.randint(0, tw -ew )
            y = np.random.randint(0, th - eh)
        return x, y, th,tw,eh, ew,target_scale



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

    def __init__(self, output_size,scale_range=(0.8, 1.2), interpolation=cv2.INTER_LANCZOS4,name='random_center_crop',**kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.output_size = output_size
        self.scale_range=scale_range
        self.interpolation = interpolation

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if  self._shape_info is None:
            self._shape_info = self._get_shape(image)
        x, y, th, tw, eh, ew,h,w = self._shape_info
        resized_image=cv2.resize(image.copy(), (tw,th), interpolation=self.interpolation)
        crop_image = resized_image[y: builtins.min(y + eh, th), x:builtins.min( x + ew,tw)]
        if crop_image.shape[0]<eh or crop_image.shape[1]<ew:
            background=np.zeros((eh,ew,1 if spec is not None and spec.object_type==ObjectType.gray else 3))
            if ndim(crop_image)==2:
                background[builtins.max(eh - crop_image.shape[0], 0) // 2:builtins.max(eh - crop_image.shape[0], 0) // 2 + crop_image.shape[0],
                builtins.max(ew - crop_image.shape[1], 0) // 2:builtins.max(ew - crop_image.shape[1], 0) // 2 + crop_image.shape[1], 0] = crop_image
            else:
                background[builtins.max(eh-crop_image.shape[0],0)//2:builtins.max(eh-crop_image.shape[0],0)//2+crop_image.shape[0],builtins.max(ew-crop_image.shape[1],0)//2:builtins.max(ew-crop_image.shape[1],0)//2+crop_image.shape[1],:]=crop_image
            return background
        else:
            return crop_image

    def _apply_coords(self, coords,spec:TensorSpec):
        x, y, th, tw, eh, ew,h,w = self._shape_info
        coords[:, 0] = (coords[:, 0] * true_divide(tw, w)).astype(np.int)
        coords[:, 1] = (coords[:, 1] * (true_divide(th, h))).astype(np.int)
        coords[:, 0] -= builtins.max(int(round((tw - ew) / 2.0)),0)
        coords[:, 1] -= builtins.max(int(round((th - eh) / 2.0)),0)
        coords[:, 0] +=builtins.max(ew-tw,0)//2
        coords[:, 1] +=builtins.max(eh - th, 0) // 2
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        x, y, th, tw, eh, ew, h, w = self._shape_info
        mask = cv2.resize(mask, (tw,th), interpolation=cv2.INTER_NEAREST)
        crop_mask =  mask[y: builtins.min(y + eh, th), x:builtins.min( x + ew,tw)]
        if crop_mask.shape[0] < eh or crop_mask.shape[1] < ew:
            background = np.zeros((eh, ew))
            background[builtins.max(eh-crop_mask.shape[0],0)//2:builtins.max(eh-crop_mask.shape[0],0)//2+crop_mask.shape[0],builtins.max(ew-crop_mask.shape[1],0)//2:builtins.max(ew-crop_mask.shape[1],0)//2+crop_mask.shape[1]]=crop_mask
            return background
        else:
            return crop_mask


    def _get_shape(self, image):
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        eh, ew = self.output_size
        base_scale =builtins.min(eh/h,ew/w)
        current_scale = np.random.uniform(*self.scale_range)
        current_scale=base_scale*current_scale

        th, tw = int(h * current_scale), int(w * current_scale)

        x = builtins.max(int(round((tw - ew) / 2.0)),0)
        y = builtins.max(int(round((th - eh) / 2.0)),0)
        return x, y, th, tw, eh, ew,h,w


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
        if  self._shape_info is None:
            self._shape_info = self._get_shape(image)
        h,w,eh, ew,offset_x,offset_y,offset_x1,offset_y1 = self._shape_info
        if image.ndim == 2 or (image.ndim == 3 and int_shape(image)[-1]==1):
            origin_ndim=image.ndim
            if origin_ndim==3:
                image=image[:,:,0]
            output = np.zeros(self.output_size)
            crop_im = image[offset_y:min(offset_y + eh,h), offset_x:min(offset_x + ew,w)]
            output[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1]] = crop_im
            if origin_ndim == 3:
                output=np.expand_dims(output,-1)
            return output
        elif image.ndim == 3:
            output=np.zeros(self.output_size+(1,) if spec is not None and spec.object_type==ObjectType.gray else self.output_size+(3,))
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
        h, w= image.shape[:2]
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
        if self._shape_info is None:
            self._shape_info = self._get_shape(image)
        mat_img, height, width,is_flip = self._shape_info

        if self.rn % 5 > 0:
            image=np.clip(image.astype(np.float32),0,255)[:,:,:3]
            image= cv2.warpAffine(image.copy(), mat_img,dsize= (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))  # , borderMode=cv2.BORDER_REPLICATE
            if is_flip:
                return image[:,::-1]
            else:
                return image
        else:
            return image

    def _apply_coords(self, coords,spec:TensorSpec):
        mat_img, height, width,is_flip = self._shape_info
        if self.rn % 5 > 0:
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
        else:
            return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        mask = unpack_singleton(mask)
        mat_img, height, width,is_flip = self._shape_info
        if self.rn % 5 > 0:
            mask =cv2.warpAffine(mask, mat_img, (width, height), borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))  # , borderMode=cv2.BORDER_REPLICATE
            if is_flip:
                return mask[:, ::-1]
            else:
                return mask
        else:
            return mask
    def _get_shape(self, image):
        self.rn = random.randint(0, 10)
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




class RandomMultiScaleImage(VisionTransform):
    def __init__(self, output_size,scale_range=(0.8, 1.2), interpolation=cv2.INTER_LANCZOS4,keep_aspect=True,name='random_multiscale_image',**kwargs):
        super().__init__(name)
        self.is_spatial = True
        self.keep_aspect=keep_aspect
        self.output_size = output_size
        self.scale_range=scale_range
        self.interpolation = interpolation
        self.idx = 0
        self.resize_funs=[Resize(output_size,True,align_corner=True,interpolation=interpolation),
                          Resize(output_size, True,align_corner=False, interpolation=interpolation),
                          Resize(output_size, False, interpolation=interpolation),
                          ShortestEdgeResize(output_size=output_size,keep_aspect=True,interpolation=interpolation),
                          ShortestEdgeResize(output_size=output_size, keep_aspect=True, interpolation=interpolation),
                          RandomRescaleCrop(output_size=output_size,scale_range=scale_range,interpolation=interpolation),
                          RandomRescaleCrop(output_size=output_size, scale_range=((scale_range[0]+1)/2,(scale_range[1]+1)/2), interpolation=interpolation),
                          RandomCrop(output_size=output_size),
                          RandomCrop(output_size=output_size),
                          RandomCenterCrop(output_size=output_size,scale_range=scale_range,interpolation=interpolation)]
        if self.keep_aspect:
            self.resize_funs.pop(2)


    def __call__(self, inputs: Union[Dict[TensorSpec, np.ndarray], np.ndarray], **kwargs):
        fn = random_choice(self.resize_funs)
        spec = kwargs.get('spec')
        return fn.apply_batch(inputs, spec)







class HorizontalFlip(VisionTransform):
    def __init__(self,name='horizontal_flip',**kwargs):
        super().__init__(name)
        self.is_spatial = True

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        image=unpack_singleton(image)
        if self._shape_info is None:
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
            norm_mean=np.ones((1,1,image.shape[-1]),dtype=np.float32)*self.mean
        elif  isinstance(self.mean,(list,tuple))  and len(self.mean)==image.shape[-1]  and image.ndim == 3:
            norm_mean=np.expand_dims(np.expand_dims(to_numpy(self.mean),0),0).astype(np.float32)

        if isinstance(self.std, numbers.Number) and image.ndim == 3:
            norm_std = np.ones((1, 1, image.shape[-1]),dtype=np.float32) * self.std
        elif isinstance(self.std, (list, tuple)) and len(self.std) == image.shape[-1] and image.ndim == 3:
            norm_std = np.expand_dims(np.expand_dims(to_numpy(self.std), 0), 0).astype(np.float32)

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
        #image = image.astype(np.float32)
        if image.ndim == 3 and image.shape[0] <=4:
            image=image.transpose([1,2,0])

        norm_mean=self.mean
        norm_std=self.std
        if isinstance(self.mean, numbers.Number) and image.ndim == 3:
            norm_mean = np.array([self.mean]*image.shape[-1]).astype(np.float32)
            norm_mean = np.expand_dims(np.expand_dims(norm_mean, 0), 0)

        elif isinstance(self.mean, (list,tuple)) and image.ndim == 3:
            norm_mean = np.array([self.mean]).astype(np.float32)
            #norm_mean = np.expand_dims(np.expand_dims(norm_mean, 0), 0)

        if isinstance(self.std, numbers.Number) and image.ndim == 3:
            norm_std = np.array([self.std]*image.shape[-1]).astype(np.float32)
            norm_std = np.expand_dims(np.expand_dims(norm_std, 0), 0)

        elif isinstance(self.std, (list,tuple)) and image.ndim == 3:
            norm_std = np.array([self.std]).astype(np.float32)
            #norm_std = np.expand_dims(norm_std, 0)

        image *= norm_std
        image += norm_mean
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
        orig_mean = image.mean()
        orig_std = np.std(image)
        noise = self.intensity*np.random.normal(0,orig_std,image.shape)
        if rr % 2 == 0:
            noise =  self.intensity*np.random.uniform(orig_mean-orig_std, orig_mean+orig_std, image.shape)
        image = np.clip(image + noise,0, 255)
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
        image = 255. * self.gain * np.power(np.clip(image / 255.,0,1), self.gamma)
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
        if int(self.ksize)%2 == 0:
            self.ksize=int(self.ksize)+1
        else:
            self.ksize = int(self.ksize)
        blur = cv2.GaussianBlur(image, (int(self.ksize), int(self.ksize)),cv2.BORDER_DEFAULT)
        return blur.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


class InvertColor(VisionTransform):
    def __init__(self, name='color',**kwargs):
        super().__init__(name)



    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
            return 255 - image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


class GrayScale(VisionTransform):
    def __init__(self, keepdims=True,name='gray_scale',**kwargs):
        super().__init__(name)
        self.is_spatial = False
        self.keepdims=keepdims

    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if image.ndim == 4 and self.keepdims:
            return cv2.cvtColor(cv2.cvtColor(image.astype(np.float32),cv2.COLOR_RGBA2GRAY),cv2.COLOR_GRAY2RGB)
        elif image.ndim == 4 and not self.keepdims:
            return cv2.cvtColor(image.astype(np.float32),cv2.COLOR_RGBA2GRAY)
        elif image.ndim == 3 and self.keepdims:
            return cv2.cvtColor(cv2.cvtColor(image.astype(np.float32),cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and not self.keepdims:
            return cv2.cvtColor(image.astype(np.float32),cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2 and self.keepdims:
            return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_GRAY2RGB)
        else:
            return image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask

class ToRGB(VisionTransform):
    def __init__(self, name='to_rgb',**kwargs):
        super().__init__(name)
        self.is_spatial = False

    def apply(self, input: Tuple,spec:TensorSpec):
        return self._apply_image(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if  image.ndim == 3 and int_shape(image)[-1]==1:
            image=image.copy()[:,:,0]
        if image.ndim == 3:
            pass
        elif image.ndim == 2:
            image=cv2.cvtColor(image.astype(np.float32), cv2.COLOR_GRAY2RGB)
        return image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


@randomize_with_validate(valid_range=(1, 10), no_change_value=1)
class RandomBlur(Blur):
    pass

@randomize
class RandomInvertColor(InvertColor):
    pass

@randomize
class RandomGrayScale(GrayScale):
    pass




class ImageErosion(VisionTransform):
    """ Erosion operation
    Erosion is a mathematical morphology operation that uses a structuring element for shrinking the shapes in an image. The binary erosion of an image by a structuring element is the locus of the points where a superimposition of the structuring element centered on the point is entirely contained in the set of non-zero elements of the image.

    Args:
        filter_size (int): the size of the structuring element .
        repeat (int): the number of repeating operation.

    Returns:
        output image array

    """

    def __init__(self, filter_size=3,repeat=1,name='image_erosion',**kwargs):
        super().__init__(name)
        if filter_size < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.filter_size=filter_size
        self.repeat=repeat


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        # Creating kernel
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)

        # Using cv2.erode() method
        image = cv2.erode(image, kernel, iterations = self.repeat)
        if image.ndim==2:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask

class ImageDilation(VisionTransform):
    """ Dilation operation
    Dilation is a mathematical morphology operation that uses a structuring element for expanding the shapes in an image. The binary dilation of an image by a structuring element is the locus of the points covered by the structuring element, when its center lies within the non-zero points of the image.

    Args:
        filter_size (int): the size of the structuring element .
        repeat (int): the number of repeating operation.

    Returns:
        output image array

    """

    def __init__(self, filter_size=3,repeat=1,name='image_dilation',**kwargs):
        super().__init__(name)
        if filter_size < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.filter_size=filter_size
        self.repeat=repeat


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        # Creating kernel
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)

        # Using cv2.erode() method
        image = cv2.dilate(image, kernel, iterations = self.repeat)
        if image.ndim==2:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


class DilationThenErosion(VisionTransform):
    r"""
       Adjust hue of the input data.
       :param value: how much to adjust the hue. Can be any number
           between 0 and 0.5, 0 gives the original image.

    """

    def __init__(self, filter_size=3,repeat=1,name='dilation_then_erosion',**kwargs):
        super().__init__(name)
        if filter_size < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.filter_size=filter_size
        self.repeat=repeat


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        # Creating kernel
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)

        # Using cv2.erode() method
        for i in range(self.repeat):
            image = cv2.dilate(image, kernel, iterations=1)
            image = cv2.erode(image, kernel, iterations =1)
        if image.ndim==2:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask

class ErosionThenDilation(VisionTransform):
    r"""
       Adjust hue of the input data.
       :param value: how much to adjust the hue. Can be any number
           between 0 and 0.5, 0 gives the original image.

    """

    def __init__(self, filter_size=3,repeat=1,name='erosion_then_dilation',**kwargs):
        super().__init__(name)
        if filter_size < 0:
            raise ValueError('Gamma should be a non-negative real number')
        self.filter_size=filter_size
        self.repeat=repeat


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)

        # Using cv2.erode() method
        for i in range(self.repeat):
            image = cv2.erode(image, kernel, iterations=1)
            image = cv2.dilate(image, kernel, iterations=1)

        if image.ndim==2:
            image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        return image.clip(0, 255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
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

    def __init__(self,threshold_type='otsu',gaussian_filtering=True, name='adaptive_binarization', **kwargs):
        super().__init__(name)
        valid_item=[ 'otsu' 'percentile', 'isodata', 'local', 'minimum']
        self.threshold_type=threshold_type
        self.gaussian_filtering=gaussian_filtering



    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        if image.ndim==3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.uint8)
        else:
            gray=image.astype(np.uint8)
        if gray.min()==gray.max():
            return image
        if self.gaussian_filtering:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret=None
        th=127.5
        if self.threshold_type=='otsu':
            #th = threshold_otsu(gray)
            th,ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

        gray =ret if ret is not None else  (gray > th).astype(np.float32) * 255.0
        if gray.max() - gray.min() < 20:
            return clip(image,0,255).astype(np.float32)
        image=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return clip(image,0,255).astype(np.float32)

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


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

    def __init__(self, prob=0.005,keep_prob=0.5, name='saltpepper', **kwargs):
        super().__init__(name)
        self.prob=prob
        self.keep_prob=keep_prob


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        rr=random.random()
        if rr>self.keep_prob:
            imgtype = image.dtype
            rnd = np.random.rand(image.shape[0], image.shape[1])
            #noisy = image.copy()
            image[rnd < self.prob / 2] = 0.0
            image[rnd > 1 - self.prob / 2] = 255.0
            return clip(image,0,255).astype(np.float32)
        else:
            return image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
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

    def __init__(self, size_range=(0.05,0.4),transparency_range=(0.4,0.8),transparancy_ratio=0.5, keep_prob=0.5,name='random_erasing', **kwargs):
        super().__init__(name)
        self.size_range=size_range
        self.transparency_range=transparency_range
        self.transparancy_ratio=transparancy_ratio
        self.keep_prob=keep_prob


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

    def _apply_image(self, image,spec:TensorSpec):
        rr = random.random()
        if rr > self.keep_prob:
            s_l, s_h = self.size_range
            r_1 = 0.3
            r_2 = 1 / 0.3
            h, w, c = image.shape
            p_1 = np.random.rand()

            if p_1 > 0.5:
                return image

            while True:
                s = np.random.uniform(s_l, s_h) * h * w / 4.0
                r = np.random.uniform(r_1, r_2)
                w1 = int(np.sqrt(s / r))
                h1 = int(np.sqrt(s * r))
                left = np.random.randint(0, w)
                top = np.random.randint(0, h)

                if left + w1 <= w and top + h1 <= h:
                    break
            self.rr = np.random.uniform(0, 1)
            if self.rr  <= self.transparancy_ratio:
                transparancy = np.random.uniform(*self.transparency_range)
                mask = np.ones_like(image)
                mask[top:top + h1, left:left + w1, :] = 0
                image = image * (mask) + image * (1 - mask) * (transparancy)
            else:

                if self.rr % 2 == 1:
                    c1 = np.random.uniform(0, 255, (h1, w1, c))
                else:
                    c1 = np.random.uniform(0, 255)

                image[top:top + h1, left:left + w1, :] = c1
            return clip(image,0,255).astype(np.float32)
        else:
            return image

    def _apply_coords(self, coords,spec:TensorSpec):
        return coords

    def _apply_mask(self, mask,spec:TensorSpec):
        return mask


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

    def __init__(self,output_size, keep_prob=0.7,name='image_mosaic', **kwargs):
        super().__init__(name)
        self.output_size=output_size
        self.keep_prob=keep_prob


    def apply_batch(self, inputs: Sequence[Tuple],spec:Optional[TensorSpec]=None):
        r"""Apply transform on batch input data."""
        if not isinstance(inputs,OrderedDict) :
            if spec is None and self.is_spatial==True:
                self._shape_info =None
                spec = TensorSpec(shape=tensor_to_shape(inputs,need_exclude_batch_axis=True,is_singleton=True), object_type=ObjectType.rgb)
            return self.apply(inputs, spec)
        else:
            results=OrderedDict()
            imgs=None
            boxes=None
            masks=None
            imgs = inputs.value_list[0]

            if int_shape(imgs)[0]<=4:
                return inputs
            else:
                for i in range(len(inputs.values())):
                    k,v=inputs.item_list[i]
                    results[k] = []
                    if i==0:
                        imgs=v
                    elif  isinstance(k,TensorSpec) and 'box' in str(k.object_type).lower():
                            boxes =v
                    elif  isinstance(k,TensorSpec) and 'mask' in str(k.object_type).lower():
                            masks =v
                if get_backend() == 'pytorch':
                    imgs = imgs.transpose([0, 2, 3, 1])

                for i in range(int_shape(imgs)[0]):
                    rr=random.random()
                    if rr>self.keep_prob:

                        img=imgs[i]

                        other_idxes=list(range(int_shape(imgs)[0]))
                        other_idxes.remove(i)
                        random.shuffle(other_idxes)
                        other_idxes=other_idxes[:3]

                        height,width =self.output_size
                        rs = np.random.uniform(0.5, 1.5, [2])  # random shift
                        center = (int(height * rs[0] / 2), int(width * rs[1] / 2))

                        # crop each image
                        p1= imgs[other_idxes[0]].copy()[:center[0],center[1]:width,:]
                        p2 =imgs[other_idxes[1]].copy()[center[0]:height,:center[1],:]
                        p3=imgs[other_idxes[2]].copy()[center[0]:height,center[1]:width,:]

                        base_img=img.copy()
                        base_img[:center[0],center[1]:width,:]=p1
                        base_img[center[0]:height,:center[1],:] = p2
                        base_img[center[0]:height,center[1]:width,:] = p3

                        results[results.key_list[0]].append(base_img)

                        if masks is not None  and masks[i].any():
                            msk=masks[i]
                            mp1 = masks[other_idxes[0]].copy()[ :center[0], center[1]:width]
                            mp2 = masks[other_idxes[1]].copy()[center[0]:height, :center[1]]
                            mp3 = masks[other_idxes[2]].copy()[center[0]:height, center[1]:width]

                            base_msk = msk.copy()
                            base_msk[ :center[0], center[1]:width] = mp1
                            base_msk[center[0]:height, :center[1]] = mp2
                            base_msk[ center[0]:height, center[1]:width] = mp3

                            mask_key=[k for k in results.key_list if 'mask' in str(k.object_type).lower() ]
                            if mask_key:
                                results[mask_key[0]].append(base_msk)

                        mosaic_bboxes=[]
                        if boxes is not None  and boxes[i].any():
                            box=boxes[i]
                            box1 = boxes[other_idxes[0]]
                            box2 =boxes[other_idxes[1]]
                            box3 =boxes[other_idxes[2]]
                            box_list=[]
                            if box.any():
                                box[:, 0][box[:, 0] > center[1]] = center[1]
                                box[:, 2][box[:, 2] > center[1]] = center[1]
                                box[:, 1][box[:, 1] > center[0]] = center[0]
                                box[:, 3][box[:, 3] > center[0]] = center[0]
                                box_list.append(box)
                            if box1.any():
                                box1[:, 0][box1[:, 0] < center[1]] = center[1]
                                box1[:, 2][box1[:, 2] < center[1]] = center[1]
                                box1[:, 1][box1[:, 1] > center[0]] = center[0]
                                box1[:, 3][box1[:, 3] > center[0]] = center[0]
                                box_list.append(box1)

                            if box2.any():
                                box2[:, 0][box2[:, 0] > center[1]] = center[1]
                                box2[:, 2][box2[:, 2] > center[1]] = center[1]
                                box2[:, 1][box2[:, 1] < center[0]] = center[0]
                                box2[:, 3][box2[:, 3] < center[0]] = center[0]
                                box_list.append(box2)


                            if box3.any():
                                box3[:, 0][box3[:, 0] < center[1]] = center[1]
                                box3[:, 2][box3[:, 2] < center[1]] = center[1]
                                box3[:, 1][box3[:, 1] < center[0]] = center[0]
                                box3[:, 3][box3[:, 3] < center[0]] = center[0]
                                box_list.append(box3)


                            if len(box_list)>0:
                                mosaic_bboxes = np.concatenate(box_list)
                                # remove no area bbox
                                keep_x = mosaic_bboxes[:, 2] - mosaic_bboxes[:, 0] > 16
                                keep_y = mosaic_bboxes[:, 3] - mosaic_bboxes[:, 1] > 16
                                keep_mask = np.logical_and(keep_x, keep_y)
                                mosaic_bboxes = mosaic_bboxes[keep_mask]
                            else:
                                mosaic_bboxes = list()
                        box_key = [k for k in results.key_list if 'box' in str(k.object_type).lower()]
                        if box_key:
                            results[box_key[0]].append(mosaic_bboxes)
                    else:
                        results[results.key_list[0]].append(imgs[i])
                        mask_key = [k for k in results.key_list if 'mask' in str(k.object_type).lower()]
                        if mask_key:
                            results[mask_key[0]].append(masks[i])
                        box_key = [k for k in results.key_list if 'box' in str(k.object_type).lower()]
                        if box_key:
                            results[box_key[0]].append(boxes[i])

            for i in range(len(results)):
                if i==0:
                    result_imgs=np.stack(results.value_list[i],0)
                    if get_backend() == 'pytorch':
                        result_imgs = result_imgs.transpose([0, 3, 1, 2])

                    results[results.key_list[0]] =result_imgs
                elif not 'box' in str(results.key_list[i].object_type).lower():
                    results[results.key_list[i]]=np.array(results.value_list[i])
            return results


    def apply(self, input: Tuple,spec:TensorSpec):
        return super().apply(input,spec)

