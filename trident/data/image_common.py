from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import math
import numbers
import sys
import os
import random
import re
import builtins
import time
from collections import Counter
from itertools import repeat
from functools import partial
import inspect
from typing import Union, Dict

import cv2
import numpy as np
import six
from scipy import ndimage
from skimage import color
from skimage import exposure
from skimage import morphology
from skimage import transform, exposure
from skimage.filters import *
from skimage.morphology import square
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec, assert_input_compatibility, ObjectType

__all__ = ['transform_func','read_image', 'read_mask', 'save_image', 'save_mask', 'image2array', 'array2image', 'mask2array',
           'array2mask', 'list_pictures', 'normalize', 'unnormalize', 'channel_reverse', 'blur', 'random_blur',
           'random_crop', 'resize', 'rescale', 'downsample_then_upsample', 'add_noise', 'gray_scale', 'to_rgb',
           'to_bgr', 'auto_level', 'random_invert_color', 'image_backend_adaption', 'reverse_image_backend_adaption',
           'random_adjust_hue', 'random_channel_shift', 'random_cutout', 'random_rescale_crop', 'random_center_crop',
           'adjust_gamma','adjust_brightness_contrast', 'random_adjust_gamma', 'adjust_contrast', 'random_adjust_contrast', 'clahe',
           'erosion_then_dilation', 'dilation_then_erosion', 'image_erosion', 'image_dilation', 'adaptive_binarization',
           'random_transform', 'horizontal_flip', 'random_mirror', 'to_low_resolution','random_erasing']



if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *


if get_image_backend() == 'opencv':
    from trident.backend.opencv_backend import *
else:
    from trident.backend.pillow_backend import *

read_image = read_image
read_mask = read_mask
save_image = save_image
save_mask = save_mask
image2array = image2array
array2image = array2image
mask2array = mask2array
array2mask = array2mask

def distict_color_count(img):
    """Count for distinct colors in input image

    Returns:
        object:

    Examples:
        >>> img=read_image('../../trident_logo.png')[:,:,:3]
        >>> len(distict_color_count(img))
        1502
        >>> img=read_image('../../trident_logo.png')[:,:,:1]
        >>> len(distict_color_count(img))
        125

    """
    return Counter([tuple(colors) for i in img for colors in i])



def object_type_inference(data):
    if isinstance(data,np.ndarray):
        if data.ndim == 2 and data.shape[-1] == 2:
            return ObjectType.landmarks
        elif data.ndim == 2 and data.shape[-1] in (4, 5) and 0<=data.max().round(0)<=255 and 0<=data.min().round(0)<=255:
            return ObjectType.absolute_bbox
        elif data.ndim == 2 and data.shape[-1] in (4, 5) and 0<=data.max().round(0)<=1 and 0<=data.min().round(0)<=1:
            return ObjectType.relative_bbox
        elif data.ndim == 2 and len(distict_color_count(np.expand_dims(data,-1)))==2:
            return ObjectType.binary_mask
        elif data.ndim == 2 and (data.max()-data.min()+1)== len(distict_color_count(np.expand_dims(data,-1))):
            return ObjectType.label_mask
        elif data.ndim == 2 and 0<=data.max().round(0)<=255 and 0<=data.min().round(0)<=255:
            return ObjectType.gray
        elif data.ndim == 3 and data.shape[-1] == 1 and len(distict_color_count(data))==2:
            return ObjectType.binary_mask
        elif data.ndim == 3 and data.shape[-1] == 1 and (data.max()-data.min()+1)== len(distict_color_count(data)):
            return ObjectType.label_mask
        elif data.ndim == 3 and data.shape[-1] == 1 and 0<=data.max().round(0)<=255 and 0<=data.min().round(0)<=255:
            return ObjectType.gray
        elif data.ndim == 3 and data.shape[-1] == 3 and 0<=data.max().round(0)<=255 and 0<=data.min().round(0)<=255 and len(distict_color_count(data))<100:
            return ObjectType.color_mask
        elif data.ndim == 3 and data.shape[-1] == 3 and 0<=data.max().round(0)<=255 and 0<=data.min().round(0)<=255:
            return ObjectType.rgb
        elif data.ndim == 3 and data.shape[-1] == 4 and 0<=data.max().round(0)<=255 and 0<=data.min().round(0)<=255:
            return ObjectType.rgba
        elif data.ndim == 3 and data.dtype==np.int64 and 0<=data.max().round(0)<=1 and 0<=data.min().round(0)<=1:
            return ObjectType.binary_mask
        elif data.ndim == 3 and data.dtype in [np.float32,np.float16] and 0<=data.max()<=1 and 0<=data.min().round(0)<=1:
            return ObjectType.alpha_mask
        elif data.ndim <= 1 and data.dtype==np.int64 :
            return ObjectType.classification_label
        elif data.ndim == 2 and data.dtype==np.int64:
            return ObjectType.color_mask
        else:
            sys.stderr.write('Object type cannot be inferred: shape:{0} dtype:{1} min:{2} max:{3} .'.format(data.shape,data.dtype,data.min(),data.max())+'\n')
            return ObjectType.array_data


def transform_func(func):
    """

    Args:
        func ():

    Returns:



    """
    def wrapper(*args, **kwargs):
        argspec = inspect.getfullargspec(func)

        if len(args)>len(argspec.varargs)-1:
            raise ValueError('Beside image, there should be only {0} in {1} function, but you get {2}'.format(len(argspec.varargs)-1,func.__name__,len(args)))
        if len(kwargs) > len(argspec.kwonlyargs) - 1:
            raise ValueError('there should be only {0} in {1} function, but you get {2}'.format(len(argspec.kwonlyargs) , func.__name__, len(kwargs)))
        for i in range(len(args)):
            kwargs[argspec.args[i+1]]=args[i]
        ret = partial(func,**kwargs)
        return ret

    return wrapper

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|jfif'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if
            re.match(r'([\w]+\.(?:' + ext + '))', f)]


def check_same_size(*images):
    result = True
    height, width = images[0].shape[:2]
    # check same isze
    for img in images:
        hh, ww = images[0].shape[:2]
        if hh == height and ww == width:
            pass
        else:
            result = False
    return True


def random_augmentation(func):
    def wrapper(prob=0, *args, **kwargs):
        if random.random() <= prob:
            return func(*args, **kwargs)

    return wrapper


def add_noise(intensity=0.1):
    def img_op(image: np.ndarray,**kwargs):
        rr = random.randint(0, 10)
        orig_min = image.min()
        orig_max = image.max()
        noise = np.random.standard_normal(image.shape) * (intensity * (orig_max - orig_min))
        if rr % 2 == 0:
            noise = np.random.uniform(-1, 1, image.shape) * (intensity * (orig_max - orig_min))
        image = np.clip(image + noise, orig_min, orig_max)
        return image

    return img_op


def normalize(mean, std):
    def img_op(image: np.ndarray,**kwargs):
        norm_mean = mean
        norm_std = std
        if isinstance(norm_mean, numbers.Number) and image.ndim == 3:
            norm_mean = np.array([norm_mean, norm_mean, norm_mean])
            norm_mean = np.expand_dims(norm_mean, 0)
            norm_mean = np.expand_dims(norm_mean, 0)
        if isinstance(norm_std, numbers.Number) and image.ndim == 3:
            norm_std = np.array([norm_std, norm_std, norm_std])
            norm_std = np.expand_dims(norm_std, 0)
            norm_std = np.expand_dims(norm_std, 0)
        if image.ndim == 3:
            if int_shape(image)==(224,224,224):
                print('')
            return (image - norm_mean) / norm_std
        elif image.ndim == 2:
            if isinstance(norm_mean, numbers.Number) and isinstance(norm_std,numbers.Number):
                return (image - norm_mean) / norm_std
        return image

    img_op.mean = mean
    img_op.std = std
    return img_op


def unnormalize(mean, std):
    def img_op(image: np.ndarray,**kwargs):
        image = reverse_image_backend_adaption(image)
        norm_mean = mean
        norm_std = std
        if isinstance(norm_mean, tuple):
            norm_mean = list(norm_mean)

        if isinstance(norm_std, tuple):
            norm_std = list(norm_std)

        if isinstance(norm_mean, (float, int)) and isinstance(norm_std, (float, int)) and image.ndim == 3:
            return image * float(norm_std) + float(norm_mean)
        elif isinstance(norm_mean, list) and isinstance(norm_std, list) and len(norm_mean) == 1 and len(norm_std) == 1:
            return image * float(norm_std[0]) + float(norm_mean[0])
        elif isinstance(norm_mean, list) and isinstance(norm_std, list) and len(norm_mean) == 3 and len(norm_std) == 3:
            norm_mean = np.reshape(np.array(norm_mean), (1, 1, 3))
            norm_std = np.reshape(np.array(norm_std), (1, 1, 3))
            return image * norm_std + norm_mean
        return image

    return img_op


#
# 0: Nearest-neighbor
# 1: Bi-linear (default)
# 2: Bi-quadratic
# 3: Bi-cubic
# 4: Bi-quartic
# 5: Bi-quintic

# all size is HWC or (H,W)
def resize(size, keep_aspect=True, order=1, align_corner=True):
    def img_op(image: Union[np.ndarray,Dict[TensorSpec,np.ndarray]],**kwargs):
        results=None
        if isinstance(image,np.ndarray):
            imspec=kwargs.get("spec")
            if imspec is None:
                imspec=TensorSpec(shape=to_tensor(image.shape), object_type=object_type_inference(image))
            results = OrderedDict()
            results[imspec]=image
        elif isinstance(image,dict):
            results=image

        if keep_aspect:
            heigth,width  = size
            currentHeight = None
            currentWidth = None
            if isinstance(image, np.ndarray):
                currentHeight, currentWidth = image.shape[:2]
            elif isinstance(image, OrderedDict):
                currentHeight, currentWidth = image.value_list[0].shape[:2]

            scale=builtins.min(heigth/currentHeight,width/currentWidth)
            new_h=currentHeight*scale
            new_w=currentWidth*scale
            pad_vert = (heigth - new_h) / 2
            pad_horz = (width - new_w) / 2
            pad_top, pad_btm = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)

            img_op.scale = scale
            if align_corner:
                img_op.pad_top = 0
                img_op.pad_btm = int(pad_btm + pad_top)
                img_op.pad_left = 0
                img_op.pad_right = int(pad_right + pad_left)
            else:
                img_op.pad_top = int(pad_top)
                img_op.pad_btm = int(pad_btm)
                img_op.pad_left = int(pad_left)
                img_op.pad_right = int(pad_right)

            img_op.all_pad = (img_op.pad_top, img_op.pad_btm, img_op.pad_left, img_op.pad_right)
            for spec,im in results.items():
                if spec.is_spatial==True:
                    if im is None:
                        pass
                    elif spec.object_type in [ObjectType.absolute_bbox] :  # bbox [:,4]   [:,5]
                        class_info = None
                        if im.shape[-1] >4:
                            class_info = im[:, 4:]
                            im = im[:, :4]
                        im[:, :4]=im[:, :4]*img_op.scale
                        im[:, 0::2] +=img_op.pad_left
                        im[:, 1::2] +=img_op.pad_top

                        if class_info is not None:
                            im=np.concatenate([im,class_info],axis=-1)
                        results[spec] = im
                    elif spec.object_type in [ObjectType.landmarks]  :  # landmark [:,2]
                        im[:, :2]=im[:, :2]*img_op.scale
                        im[:, 0::2] += img_op.pad_left
                        im[:, 1::2] += img_op.pad_top
                        results[spec] = im
                    # elif spec.object_type in [ObjectType.color_mask]:  # landmark [:,2]
                    #     new_im = np.zeros((size[0], size[1],3),dtype=np.int64)
                    #     im = transform.rescale(im,  img_op.scale, clip=False, anti_aliasing=False, multichannel=True,preserve_range=True, order=0).astype(np.int64)
                    #     if align_corner:
                    #         new_im[:im.shape[0], :im.shape[1],:] = im
                    #     else:
                    #         new_im[pad_top:im.shape[0] + pad_top, pad_left:im.shape[1] + pad_left,:] = im
                    #     results[spec]= np.pad(im, img_op.all_pad, 'constant').astype(np.int64)
                    #

                    elif spec.object_type in [ObjectType.label_mask, ObjectType.color_mask, ObjectType.label_mask]:  # label_mask
                        im= transform.rescale(im, (scale, scale), clip=True, anti_aliasing=False, multichannel=False if im.ndim == 2 else True, preserve_range=True,order=0).astype(np.int64)
                        new_im = np.zeros((size[0], size[1], 3))
                        if align_corner:
                            if im.ndim==2:
                                new_im = np.zeros((size[0], size[1]))
                                new_im[:im.shape[0], :im.shape[1]] = im
                            elif im.ndim==3:
                                new_im[:im.shape[0], :im.shape[1], :] = im
                        else:
                            if im.ndim==2:
                                new_im = np.zeros((size[0], size[1]))
                                new_im[pad_top:im.shape[0] + pad_top, pad_left:im.shape[1] + pad_left] = im
                            elif im.ndim==3:
                                new_im[pad_top:im.shape[0] + pad_top, pad_left:im.shape[1] + pad_left, :] = im

                        results[spec] = new_im
                    elif spec.object_type in [ObjectType.rgb, ObjectType.rgba, ObjectType.gray]:  # image

                        im = im.astype(np.float32)
                        im= transform.rescale(im, (scale, scale), clip=True, anti_aliasing=True,multichannel=False if im.ndim == 2 else True,order=1)
                        new_im=None
                        if align_corner:
                            if im.ndim==2:
                                new_im = np.zeros((size[0], size[1]))
                                new_im[:im.shape[0], :im.shape[1]] = im
                            elif im.ndim==3:
                                new_im = np.zeros((size[0], size[1],im.shape[-1]))
                                new_im[:im.shape[0], :im.shape[1], :] = im
                        else:
                            if im.ndim==2:
                                new_im = np.zeros((size[0], size[1]))
                                new_im[pad_top:im.shape[0] + pad_top, pad_left:im.shape[1] + pad_left] = im
                            elif im.ndim==3:
                                new_im = np.zeros((size[0], size[1], im.shape[-1]))
                                new_im[pad_top:im.shape[0] + pad_top, pad_left:im.shape[1] + pad_left, :] = im

                        results[spec] = new_im
            if isinstance(image, np.ndarray):
                return results.value_list[0]
            elif isinstance(image, OrderedDict):
                return results


        else:
            img_op.scalex =np.true_divide(float(size[1]),float(image[0].shape[1]))
            img_op.scaley =np.true_divide(float(size[0]),float(image[0].shape[0]))
            img_op.scale = (img_op.scaley, img_op.scalex)
            for spec, im in results.items():
                if spec.is_spatial == True:
                    if im is None:
                        pass
                    elif spec.object_type in [ObjectType.absolute_bbox]:  # bbox
                        im[:, 0::2] /= img_op.w
                        im[:, 1::2] /= img_op.h
                        results[spec]=im
                    else:
                        im = im.astype(np.float32)
                        results[spec]= transform.resize(im, size, anti_aliasing=True, order=0 if im.ndim == 2 else order)
            return results
    return img_op


def rescale(scale, order=1):
    def img_op(image: Union[np.ndarray,Dict[TensorSpec,np.ndarray]],**kwargs):
        results = None
        if isinstance(image, np.ndarray):
            imspec = kwargs.get("spec")
            if imspec is None:
                imspec = TensorSpec(shape=to_tensor(image.shape), object_type=object_type_inference(image))

            results = OrderedDict()
            results[imspec] = image
        elif isinstance(image, dict):
            results = image

        height = None
        width = None
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        elif isinstance(image, OrderedDict):
            height, width = image.value_list[0].shape[:2]

        img_op.height = height
        img_op.width = width
        img_op.scale = scale
        for spec, im in results.items():
            if spec.is_spatial == True:
                if im is None:
                    pass
                elif spec.object_type in [ObjectType.absolute_bbox]:  # bbox [:,4]   [:,5]
                    class_info = None
                    if im.shape[-1] > 4:
                        class_info = im[:, 4:]
                        im = im[:, :4]
                    im[:, :4] = im[:, :4] * img_op.scale
                    if class_info is not None:
                        im = np.concatenate([im, class_info], axis=-1)
                    results[spec] = im
                elif spec.object_type in [ObjectType.landmarks]:  # landmark [:,2]
                    im[:, :2] = im[:, :2] * img_op.scale
                    results[spec] = im
                elif spec.object_type in [ObjectType.label_mask,ObjectType.color_mask,ObjectType.label_mask]:  # label_mask
                    results[spec] = transform.rescale(im, (scale, scale), clip=True, anti_aliasing=False, multichannel=False if im.ndim==2 else True,preserve_range=True, order=0).astype(np.int64)
                elif spec.object_type in [ObjectType.rgb, ObjectType.rgba, ObjectType.gray]:  # image
                    im = im.astype(np.float32)
                    results[spec] = transform.rescale(im, (scale, scale), clip=True, anti_aliasing=True,
                                                   multichannel=False if im.ndim==2 else True,
                                                   order=0 if im.ndim == 2 else order)
                else:
                    pass
        if isinstance(image, np.ndarray):
            return results.value_list[0]
        elif isinstance(image, OrderedDict):
            return results
    return img_op

def random_rescale_crop(h, w, scale=(0.5, 2), order=1):
    def img_op(image: Union[np.ndarray,Dict[TensorSpec,np.ndarray]],**kwargs):
        # start = time.time()
        scalemin, scalemax = scale
        current_scale = np.random.choice(np.random.uniform(scalemin, scalemax, 100))
        rescale_fn=rescale(current_scale)
        random_crop_fn=random_crop(h,w)

        results = None
        if isinstance(image, np.ndarray):
            imspec = kwargs.get("spec")
            if imspec is None:
                imspec = TensorSpec(shape=to_tensor(image.shape), object_type=object_type_inference(image))
            results = OrderedDict()
            results[imspec] = image
        elif isinstance(image, dict):
            results = image
        results = random_crop_fn(rescale_fn(results, **kwargs))

        if isinstance(image, np.ndarray):
            return results.value_list[0]
        elif isinstance(image, OrderedDict):
            return results

    return img_op


def random_center_crop(h,w, scale=(0.8, 1.2)):
    scalemin, scalemax = scale

    def img_op(image: Union[np.ndarray,Dict[TensorSpec,np.ndarray]],**kwargs):
        results = None
        # generare strong typed dict
        if isinstance(image, np.ndarray):
            imspec = kwargs.get("spec")
            if imspec is None:
                imspec = TensorSpec(shape=to_tensor(image.shape), object_type=object_type_inference(image))
            results = OrderedDict()
            results[imspec] = image
        elif isinstance(image, dict):
            results = image

        height = None
        width = None
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        elif isinstance(image, OrderedDict):
            height, width = image.value_list[0].shape[:2]

        max_value = max(height, width)
        i = int(round((max_value - height) / 2.))
        j = int(round((max_value - width) / 2.))

        scale = min(w / max_value, h / max_value) * np.random.choice(np.arange(scalemin, scalemax, 0.01))
        img_op.scale=scale
        resized_h = int(round(max_value * scale))
        resized_w = int(round(max_value * scale))
        i1 = int(round((max(resized_h, h) - resized_h) / 2.))
        j1 = int(round((max(resized_w, w) - resized_w) / 2.))

        i2 = int(round((max(resized_h, h) - h) / 2.))
        j2 = int(round((max(resized_w, w) - w) / 2.))
        for spec, im in results.items():
            if spec.is_spatial == True:
                if im is None:
                    pass
                elif spec.object_type in [ObjectType.absolute_bbox]:  # bbox [:,4]   [:,5]
                    class_info = None
                    if im.shape[-1] >4:
                        class_info = im[:, 4:]
                        im = im[:, :4]
                    im[:, 0] = np.clip((im[:, 0] + j) * scale + j1 - j2, 0, w)
                    im[:, 1] = np.clip((im[:, 1] + i) * scale + i1 - i2, 0, h)
                    im[:, 2] = np.clip((im[:, 2] + j) * scale + j1 - j2, 0, w)
                    im[:, 3] = np.clip((im[:, 3] + i) * scale + i1 - i2, 0, h)
                    area = (im[:, 3] - im[:, 1]) * (im[:, 2] - im[:, 0])
                    im = im[area > 0]
                    if len(im) > 0:
                        if class_info is not None:
                            class_info= class_info[area > 0]
                            im=np.concatenate([im,class_info],axis=-1)
                    results[spec] = im

                elif spec.object_type in [ObjectType.landmarks]:  # landmark [:,2]
                    im[:, 0] = (im[:, 0] + j) * scale + j1 - j2
                    im[:, 1] = (im[:, 1] + i) * scale + i1 - i2
                    results[spec] = im
                elif spec.object_type in [ObjectType.rgb, ObjectType.rgba, ObjectType.gray, ObjectType.binary_mask, ObjectType.color_mask, ObjectType.label_mask]:

                    if im.ndim == 3:
                        blank = np.zeros((max_value, max_value, im.shape[-1]))
                        blank[i:i + height, j:j + width, :] = im

                        resized_im = transform.rescale(blank, scale=scale, anti_aliasing=False if 'mask' in spec.object_type.value else True,clip=True, multichannel=False if im.ndim==2 else True,preserve_range=True,order=0 if 'mask' in spec.object_type.value else 1)
                        returnData = np.zeros((max(resized_h, h), max(resized_w, w), resized_im.shape[-1]))

                        returnData[i1:i1 + resized_im.shape[0], j1:j1 + resized_im.shape[1], :] = resized_im
                        results[spec] = returnData[i2:i2 + h, j2:j2 + w, :]
                    elif im.ndim == 2:
                        blank = np.zeros((max_value, max_value)).astype(im.dtype)
                        blank[i:i + height, j:j + width] = im

                        resized_im = np.round(transform.rescale(blank,scale=scale, anti_aliasing=False if 'mask' in spec.object_type.value else True, clip=True, multichannel=False if im.ndim==2 else True,preserve_range=True, order=0 if 'mask' in spec.object_type.value else 1)).astype(im.dtype)
                        returnData = np.zeros((max(resized_h, h), max(resized_w, w))).astype(im.dtype)

                        returnData[i1:i1 + resized_im.shape[0], j1:j1 + resized_im.shape[1]] = resized_im
                        results[spec] = returnData[i2:i2 + h, j2:j2 + w]
        if isinstance(image, np.ndarray):
            return results.value_list[0]
        elif isinstance(image, OrderedDict):
            return results

    return img_op


def random_crop(h, w):
    def img_op(image: Union[np.ndarray,Dict[TensorSpec,np.ndarray]],**kwargs):
        results = None
        #generare strong typed dict
        if isinstance(image, np.ndarray):
            imspec = kwargs.get("spec")
            if imspec is None:
                imspec = TensorSpec(shape=to_tensor(image.shape), object_type=object_type_inference(image))
            results = OrderedDict()
            results[imspec] = image
        elif isinstance(image, dict):
            results = image

        height = None
        width = None
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        elif isinstance(image, OrderedDict):
            height, width = image.value_list[0].shape[:2]

        img_op.h = h
        img_op.w = w
        offset_x = 0
        offset_y = 0

        if width > w:
            offset_x = random.choice(range(width - w))
        if height > h:
            offset_y = random.choice(range(height - h))
        offset_x1 = random.choice(range(w - width)) if w > width else 0
        offset_y1 = random.choice(range(h - height)) if h > height else 0
        for spec, im in results.items():
            if spec.is_spatial == True:
                if im is None:
                    pass
                elif spec.object_type in [ObjectType.absolute_bbox] :  # bbox
                    class_info=None
                    if im.shape[-1] ==5:
                        class_info=im[:,4:5]
                        im=im[:,:4]
                    im[:, 0::2] = np.clip(im[:, 0::2] - offset_x, 0, w)
                    im[:, 1::2] = np.clip(im[:, 1::2] - offset_y, 0, h)

                    area = (im[:, 3] - im[:, 1]) * (im[:, 2] - im[:, 0])
                    im = im[area > 0]

                    if len(im) > 0:
                        im[:, 0::2] += offset_x1
                        im[:, 1::2] += offset_y1
                        if class_info is not None:
                            class_info= class_info[area > 0]
                            im=np.concatenate([im,class_info],axis=-1)
                    results[spec] = im
                elif spec.object_type in [ObjectType.landmarks]:  # landmark [:,2]
                    im[:, 0] = im[:, 0] - offset_x+offset_x1
                    im[:, 1] = im[:, 1] - offset_y+offset_y1
                    results[spec] = im
                elif spec.object_type in [ObjectType.rgb,ObjectType.rgba, ObjectType.gray,ObjectType.binary_mask,ObjectType.color_mask,ObjectType.label_mask]:
                    if im.ndim == 2:
                        returnData = np.zeros((h, w), dtype=np.float32)
                        crop_im = im[offset_y:offset_y + h, offset_x:offset_x + w]
                        returnData[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1]] = crop_im
                        results[spec] = returnData
                    elif im.ndim == 3:
                        returnData = np.zeros((h, w, im.shape[-1]), dtype=np.float32)
                        crop_im = im[offset_y:offset_y + h, offset_x:offset_x + w, :]
                        returnData[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1], :] = crop_im
                        results[spec] = returnData
        if isinstance(image, np.ndarray):
            return results.value_list[0]
        elif isinstance(image, OrderedDict):
            return results
    return img_op



def random_transform(rotation_range= 15, zoom_range= 0.02, shift_range= 0.02,shear_range = 0.2,
                     random_flip= 0.15):
    # 把現有的圖片隨機擾亂
    coverage = 110
    rotation = np.random.uniform(-rotation_range, rotation_range) if rotation_range!=0 else 0
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)if zoom_range!=0 else 1
    shear= np.random.uniform( - shear_range,  shear_range)if shear_range!=0 else 0
    shift_x = np.random.uniform(-shift_range, shift_range) if shift_range != 0 else 0
    shift_y = np.random.uniform(-shift_range, shift_range)  if shift_range != 0 else 0
    rr = np.random.random()
    def img_op(image: Union[np.ndarray,Dict[TensorSpec,np.ndarray]],**kwargs):
        results = None
        if isinstance(image, np.ndarray):
            imspec = kwargs.get("spec")
            if imspec is None:
                imspec = TensorSpec(shape=to_tensor(image.shape), object_type=object_type_inference(image))
            results = OrderedDict()
            results[imspec] = image
        elif isinstance(image, dict):
            results = image

        height = None
        width = None
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        elif isinstance(image, OrderedDict):
            height, width = image.value_list[0].shape[:2]

        img_op.tx = int(shift_x* width)
        img_op.ty = int(shift_y* height )
        mat = cv2.getRotationMatrix2D((width // 2+img_op.tx, height // 2+img_op.ty), rotation,1)
        #mat[:, 2] += (tx, ty)

        cos = np.abs(mat[0, 0])
        sin = np.abs(mat[0, 1])
        new_W = int((height * sin) + (width * cos))
        new_H = int((height * cos) + (width * sin))
        mat[0, 2] += (new_W / 2) - width // 2
        mat[1, 2] += (new_H / 2) - height // 2
        mat_img = mat.copy()
        mat_box = mat.copy()

        for spec, im in results.items():
            if spec.is_spatial == True:
                if im is None:
                    pass
                elif spec.object_type in [ObjectType.absolute_bbox]:  # bbox [:,4]   [:,5]
                    class_info = None
                    if im.shape[-1] >4:
                        class_info = im[:, 4:]
                        im = im[:, :4]
                    # compute the new bounding dimensions of the image
                    box_w = (im[:, 2] - im[:, 0]).reshape(-1, 1)
                    box_h = (im[:, 3] - im[:, 1]).reshape(-1, 1)

                    x1 = im[:, 0].reshape(-1, 1)
                    y1 = im[:, 1].reshape(-1, 1)

                    x2 = x1 + box_w
                    y2 = y1

                    x3 = x1
                    y3 = y1 + box_h

                    x4 = im[:, 2].reshape(-1, 1)
                    y4 = im[:, 3].reshape(-1, 1)

                    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
                    corners = corners.reshape(-1, 2)

                    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
                    new_box = np.dot(mat_box, corners.T).T
                    new_box=new_box.reshape(-1, 8)

                    x_ = new_box[:, [0, 2, 4, 6]]
                    y_ = new_box[:, [1, 3, 5, 7]]

                    xmin = np.min(x_, 1).reshape(-1, 1)
                    ymin = np.min(y_, 1).reshape(-1, 1)
                    xmax = np.max(x_, 1).reshape(-1, 1)
                    ymax = np.max(y_, 1).reshape(-1, 1)

                    im = np.hstack((xmin, ymin, xmax, ymax, new_box[:, 8:]))

                    if im.ndim==1:
                        im=np.expand_dims(im,0)

                    if rr< random_flip:
                        im[:, 0::2] =img_op.flip_width - im[:, 2::-2]
                    im[:, 0] = np.clip(im[:,0],0,width)
                    im[:, 1] = np.clip(im[:, 1], 0, height)
                    im[:, 2] = np.clip(im[:,2], 0, width)
                    im[:, 3] = np.clip(im[:,3], 0, height)

                    area = (im[:, 3] - im[:, 1]) * (im[:, 2] - im[:, 0])
                    im = im[area > 0]
                    if len(im) > 0:
                        if class_info is not None:
                            class_info = class_info[area > 0]
                            im=np.concatenate([im,class_info],axis=-1)
                        results[spec] = im
                elif spec.object_type in [ObjectType.landmarks]:  # landmark [:,2]
                    new_n=[]
                    for i in range(len(im)):
                        pts = []
                        pts.append(np.squeeze(np.array(mat_img[0]))[0] * im[i][0] + np.squeeze(np.array(mat_img[0]))[1] * im[i][1] + np.squeeze(np.array(mat_img[0]))[2])
                        pts.append(np.squeeze(np.array(mat_img[1]))[0] * im[i][0] + np.squeeze(np.array(mat_img[1]))[1] * im[i][1] + np.squeeze(np.array(mat_img[1]))[2])
                        new_n.append(pts)
                    results[spec] = np.array(new_n)
                elif im.ndim == 3:
                    new_image = cv2.warpAffine(im.copy(), mat_img, (width, height), borderMode=cv2.BORDER_CONSTANT,borderValue=(255, 255, 255))  # , borderMode=cv2.BORDER_REPLICATE
                    if rr< random_flip:
                        img_op.flip_width=new_image.shape[1]
                        new_image = new_image[:, ::-1]
                    results[spec] =  new_image
                elif im.ndim ==2:
                    image=np.concatenate([np.expand_dims(im.copy(),-1),np.expand_dims(im.copy(),-1),np.expand_dims(im.copy(),-1)],axis=-1)
                    new_image = cv2.warpAffine(im.copy(), mat_img, (width, height), borderMode=cv2.BORDER_CONSTANT,borderValue=(255, 255, 255))  # , borderMode=cv2.BORDER_REPLICATE
                    if np.random.random() < random_flip:
                        new_image = new_image[:, ::-1]
                    new_image=cv2.cvtColor(new_image,cv2.COLOR_RGB2GRAY)

                    results[spec] = new_image
        if isinstance(image, np.ndarray):
            return results.value_list[0]
        elif isinstance(image, OrderedDict):
            return results
    return img_op



def horizontal_flip():
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    def img_op(image: Union[np.ndarray,Dict[TensorSpec,np.ndarray]],**kwargs):
        results = None
        if isinstance(image, np.ndarray):
            imspec = kwargs.get("spec")
            if imspec is None:
                imspec = TensorSpec(shape=to_tensor(image.shape), object_type=object_type_inference(image))
            results = OrderedDict()
            results[imspec] = image
        elif isinstance(image, dict):
            results = image

        height = None
        width = None
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        elif isinstance(image, OrderedDict):
            height, width = image.value_list[0].shape[:2]

        for spec, im in results.items():
            if spec.is_spatial == True:
                if im is None:
                    pass
                elif spec.object_type in [ObjectType.absolute_bbox]:  # bbox [:,4]   [:,5]
                    class_info = None
                    if im.shape[-1] >4:
                        class_info = im[:, 4:]
                        im = im[:, :4]
                    im[:, 0::2] = width - im[:, 2::-2]
                    if len(im) > 0:
                        if class_info is not None:
                            im=np.concatenate([im,class_info],axis=-1)
                    results[spec] = im
                elif spec.object_type in [ObjectType.landmarks]:  # landmark [:,2]
                    im[:, 0::2] = width - im[:, 2::-2]
                    results[spec] = im
                elif spec.object_type in [ObjectType.rgb, ObjectType.rgba, ObjectType.gray]:  # image
                    if im.ndim == 3:
                        results[spec] = im[:, ::-1]
                    elif im.ndim == 2:
                        results[spec] = im[::-1]
        if isinstance(image, np.ndarray):
            return results.value_list[0]
        elif isinstance(image, OrderedDict):
            return results

    return img_op


def random_mirror():
    fn=horizontal_flip()
    def img_op(image: Union[np.ndarray,Dict[TensorSpec,np.ndarray]],**kwargs):
        img_op.rnd = random.randint(0, 10)
        if img_op.rnd % 2 == 0:
            return fn(image,**kwargs)
        else:
            return image

    return img_op



def downsample_then_upsample(scale=4, order=1, repeat=1):
    def img_op(image: np.ndarray,**kwargs):
        for i in range(repeat):
            image = rescale(scale=1.0 / scale)(image)
            image = rescale(scale=scale)(image)
        return image

    return img_op


def invert_color():
    def img_op(image: np.ndarray,**kwargs):
        if np.min(image) >= 0 and np.max(image) <= 1:
            return 1 - image
        elif np.min(image) >= -1 and np.min(image) < 0 and np.max(image) <= 1:
            return 1 - (image * 0.5 + 0.5)
        else:
            return 255 - image

    return img_op


def random_invert_color():
    def img_op(image: np.ndarray,**kwargs):
        if random.random() < 0.7:
            if np.min(image) >= 0 and np.max(image) <= 1:
                return 1 - image
            elif np.min(image) >= -1 and np.min(image) < 0 and np.max(image) <= 1:
                return 1 - (image * 0.5 + 0.5)
            else:
                return 255 - image
        else:
            return image

    return img_op


def gray_scale():
    def img_op(image: np.ndarray,**kwargs):
        if image.shape[-1] == 3:
            image = color.rgb2gray(image.astype(np.float32))
            if image.ndim==2 or (image.ndim==3 and  image.shape[-1] == 1):
                if image.ndim == 2:
                    image=np.expand_dims(image,-1)
                image=np.concatenate([image,image,image],axis=-1)
        return image

    return img_op


def to_rgb():
    def img_op(image: np.ndarray,**kwargs):

        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
            image = np.concatenate([image, image, image], -1)
            return image

        elif len(image.shape) == 3:
            if image.shape[0] in (1,3, 4) and image.shape[-1] >4:
                image = image.transpose([1, 2, 0])
            if image.shape[-1] in (3, 4):
                image = image[:,:, :3]
            elif  image.shape[-1] ==1:
               image = np.concatenate([image, image, image], -1)
            return image

        return image

    return img_op


def to_bgr():
    def img_op(image: np.ndarray,**kwargs):
        if len(image.shape) == 2 :
            image = np.expand_dims(image, -1)
            image = np.concatenate([image, image, image], -1)
            return image
        elif image.ndim == 3:
            if image.shape[0] in (1,3, 4) and image.shape[-1] >4:
                image = image.transpose([1, 2, 0])
            if image.shape[-1] in (3, 4):
                if image.shape[0] == 4:
                    image = image[:, :,:3]
                image = image[..., ::-1]
            elif  image.shape[-1] ==1:
               image = np.concatenate([image, image, image], -1)
            return image

        return image

    return img_op


def adjust_brightness(src, x):
    alpha = 1.0 + random.uniform(-x, x)
    src *= alpha
    return src


# def adjust_contrast(src, x):
#     alpha = 1.0 + random.uniform(-x, x)
#     coef = np.array([[[0.299, 0.587, 0.114]]])
#     gray = src * coef
#     gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
#     src *= alpha
#     src += gray
#     return src


def adjust_saturation(src, x):
    alpha = 1.0 + random.uniform(-x, x)
    coef = np.array([[[0.299, 0.587, 0.114]]])
    gray = src * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    gray *= (1.0 - alpha)
    src *= alpha
    src += gray
    return src


def adjust_brightness_contrast(brightness=255, contrast=127):
    """

    Args:
        brightness ():       -255~255
        contrast (float):   -127~127

    Returns:

    """


    def map(x, in_min, in_max, out_min, out_max):
        return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    brightness = map(brightness, 0, 510.0, -255.0, 255.0)
    contrast = map(contrast, 0.0, 254.0, -127.0, 127.0)
    def img_op(image: np.ndarray,**kwargs):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            image = image * alpha_b + image * 0 + gamma_b
        else:
            image = image.copy()

        if contrast != 0:
            f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            image = image * alpha_c + image * 0 + gamma_c

        return image
    return img_op


def adjust_gamma(gamma=1):
    def img_op(image: np.ndarray,**kwargs):
        return exposure.adjust_gamma(image/255.0, gamma)*255.0

    return img_op


def random_adjust_gamma(gamma=(0.6, 1.4)):
    gamma_range=gamma
    def img_op(image: np.ndarray,**kwargs):
        image=np.clip(image,0.0,255.0)
        gammamin, gammamax = gamma_range
        avg_pix=image.mean()
        if avg_pix>220:
            gammamax=builtins.max(gammamin,1)
        elif avg_pix<30:
            gammamin = builtins.min(1, gammamax)

        gamma = np.random.choice(np.arange(gammamin, gammamax, 0.01))
        return exposure.adjust_gamma(image/255.0, gamma)*255.0

    return img_op


def adjust_contrast(alpha=1):
    beta = 0

    def img_op(image: np.ndarray,**kwargs):
        image = image.astype(np.float32) * alpha + beta
        if image.max() > 255.0:
            image=image * 255 / (image.max())
        return image.astype(np.float32)

    return img_op


def random_adjust_contrast(scale=(0.5, 1.5)):
    beta = 0
    scalemin, scalemax = scale

    def img_op(image: np.ndarray,**kwargs):
        image=np.clip(image,0.0,255.0)
        alpha = random.uniform(scalemin, scalemax)
        image = image.astype(np.float32) * alpha + beta
        if image.max()>255.0:
            image=image*255.0/(image.max())
        return image.astype(np.float32)

    return img_op


def random_adjust_hue(hue_range=(-20, 20), saturation_range=(0.5, 1.5), lightness_range=(-50, 50)):
    def img_op(image: np.ndarray,**kwargs):
        # hue is mapped to [0, 1] from [0, 360]
        # if hue_offset not in range(-180, 180):
        #     raise ValueError('Hue should be within (-180, 180)')
        # if saturation not in range(-100, 100):
        #     raise ValueError('Saturation should be within (-100, 100)')
        # if lightness not in range(-100, 100):
        #     raise ValueError('Lightness should be within (-100, 100)')
        image = np.clip(image, 0.0, 255.0)
        hue_offset = random.uniform(*hue_range)
        saturation_offset = random.uniform(*saturation_range)
        lightness_offset = random.uniform(*lightness_range)

        image =cv2.cvtColor(image.astype(np.float32),cv2.COLOR_RGB2HSV)

        image[:, :, 0] +=hue_offset
        image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        image[:, :, 1] *=saturation_offset

        image[:, :, 2] = image[:, :, 2] +lightness_offset

        image =cv2.cvtColor(image,cv2.COLOR_HSV2RGB).astype(np.float32)

        if image.max()>255.0:
            image=image* 255.0/image.max()
        return np.clip(image.astype(np.float32),0,255)

    return img_op


def auto_level():
    def img_op(image: np.ndarray,**kwargs):
        minv = np.percentile(image, 5)
        maxv = np.percentile(image, 95)
        if maxv - minv < 40:
            minv = image.min()
            maxv = image.max()

        image = np.clip((image - minv) * (255.0 / (maxv - minv)), 0, 255)
        return image.astype(np.float32)

    return img_op



def image_backend_adaption(image):
    if  get_backend() == 'tensorflow':
        if image.ndim==2: #gray-scale image
            image=np.expand_dims(image,-1).astype(np.float32)
        elif image.ndim in (3,4):
            image=image.astype(np.float32)
    else:
        if image.ndim==2: #gray-scale image
            image=np.expand_dims(image,0).astype(np.float32)
        elif image.ndim==3:
            image = np.transpose(image, [2, 0, 1]).astype(np.float32)
        elif image.ndim==4:
            image = np.transpose(image, [0, 3, 1, 2]).astype(np.float32)
    return image


def reverse_image_backend_adaption(image):
    if get_backend() in ['pytorch', 'cntk'] and image.ndim == 3 and image.shape[0] in [3, 4]:
        image = np.transpose(image, [1, 2, 0]).astype(np.float32)
    elif get_backend() in ['pytorch', 'cntk'] and image.ndim == 4 and image.shape[1] in [3, 4]:
        image = np.transpose(image, [0, 2, 3, 1]).astype(np.float32)
    return image


def random_channel_shift(intensity=0.15):
    channel_axis = -1
    inten = intensity
    def img_op(image: np.ndarray,**kwargs):
        image = np.rollaxis(image, channel_axis, channel_axis)
        min_x, max_x = np.min(image), np.max(image)
        intensity = max_x / 255 * inten
        channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x) for x_channel in
                          image.shape[-1]]
        x = np.stack(channel_images, axis=channel_axis)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    return img_op



def random_erasing(size_range=(0.02,0.3),transparency_range=(0.4,0.8),transparancy_ratio=0.5):
    def img_op(image):
        s_l ,s_h= size_range
        r_1 = 0.3
        r_2 = 1 / 0.3
        h, w, c = image.shape
        p_1 = np.random.rand()

        if p_1 > 0.5:
            return image

        while True:
            s = np.random.uniform(s_l, s_h) * h * w/4.0
            r = np.random.uniform(r_1, r_2)
            w1 = int(np.sqrt(s / r))
            h1 = int(np.sqrt(s * r))
            left = np.random.randint(0, w)
            top = np.random.randint(0, h)

            if left + w1 <= w and top + h1 <= h:
                break
        rr=np.random.uniform(0,1)
        if rr<=transparancy_ratio:
            transparancy= np.random.uniform(*transparency_range)
            mask=np.ones_like(image)
            mask[top:top + h1, left:left + w1, :]=0
            image=image*(mask)+ image*(1-mask)*(transparancy)
        else:

            if rr%2==1:
                c1 = np.random.uniform(0, 255, (h1, w1, c))
            else:
                c1 = np.random.uniform(0, 255)

            image[top:top + h1, left:left + w1, :] = c1

        return image

    return img_op


def random_cutout(img, mask):
    h, w = img.shape[:2] if get_backend() == 'tensorflow' or len(img.shape) == 2 else img.shape[1:3]
    cutx = random.choice(range(0, w // 4))
    cuty = random.choice(range(0, h // 4))
    offsetx = random.choice(range(0, w))
    offsety = random.choice(range(0, h))
    block = np.zeros((min(offsety + cuty, h) - offsety, min(offsetx + cutx, w) - offsetx))
    if random.randint(0, 10) % 4 == 1:
        block = np.clip(np.random.standard_normal(
            (min(offsety + cuty, h) - offsety, min(offsetx + cutx, w) - offsetx)) * 127.5 + 127.5, 0, 255)
    elif random.randint(0, 10) % 4 == 2:
        block = np.random.uniform(0, 255, (min(offsety + cuty, h) - offsety, min(offsetx + cutx, w) - offsetx))
    if get_backend() == 'tensorflow':
        block = np.expand_dims(block, -1)
        block = np.concatenate([block, block, block], axis=-1)
        img[offsety:min(offsety + cuty, img.shape[0]), offsetx:min(offsetx + cutx, img.shape[1]), :] = block
        mask[offsety:min(offsety + cuty, mask.shape[0]), offsetx:min(offsetx + cutx, mask.shape[1])] = 0
    else:
        block = np.expand_dims(0, block)
        block = np.concatenate([block, block, block], axis=0)
        img[:, offsety:min(offsety + cuty, img.shape[0]), offsetx:min(offsetx + cutx, img.shape[1])] = block
        mask[offsety:min(offsety + cuty, mask.shape[0]), offsetx:min(offsetx + cutx, mask.shape[1])] = 0
    return img, mask



## denoise, smooth,
def clahe(clip_limit=0.1, nbins=16):
    def img_op(image: np.ndarray,**kwargs):
        image = image.astype(np.float32) / 255.0
        if image.max() - image.min() > 0.2:
            image = exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
        return image * 255.0

    return img_op


def image_erosion(filter_size=3, repeat=1):
    """ Erosion operation
    Erosion is a mathematical morphology operation that uses a structuring element for shrinking the shapes in an image. The binary erosion of an image by a structuring element is the locus of the points where a superimposition of the structuring element centered on the point is entirely contained in the set of non-zero elements of the image.

    Args:
        filter_size (int): the size of the structuring element .
        repeat (int): the number of repeating operation.

    Returns:
        output image array

    """
    def img_op(image: np.ndarray,**kwargs):
        structure_shape = [1] * image.ndim
        structure_shape[0]=filter_size
        structure_shape[1] = filter_size
        for i in range(repeat):
            image =ndimage.morphology.grey_erosion(image,size=(filter_size,filter_size),structure=np.ones(tuple(structure_shape)))
        return clip(image,0,255)

    return img_op


def image_dilation(filter_size=3, repeat=1):
    """ Dilation operation
    Dilation is a mathematical morphology operation that uses a structuring element for expanding the shapes in an image. The binary dilation of an image by a structuring element is the locus of the points covered by the structuring element, when its center lies within the non-zero points of the image.

    Args:
        filter_size (int): the size of the structuring element .
        repeat (int): the number of repeating operation.

    Returns:
        output image array

    """
    def img_op(image: np.ndarray,**kwargs):
        structure_shape = [1] * image.ndim
        structure_shape[0]=filter_size
        structure_shape[1] = filter_size
        for i in range(repeat):
            image = ndimage.morphology.grey_dilation(image, size=(filter_size,filter_size),structure=np.ones(tuple(structure_shape)))

        return clip(image,0,255)

    return img_op


def erosion_then_dilation(filter_size=3, repeat=1):
    def img_op(image: np.ndarray,**kwargs):
        structure_shape=[1]*image.ndim
        structure_shape[0]=filter_size
        structure_shape[1] = filter_size
        for i in range(repeat):
            image = ndimage.morphology.grey_erosion(image, size=(filter_size,filter_size), structure=np.ones(tuple(structure_shape)))
            image=clip(image, 0, 255)
            image = ndimage.morphology.grey_dilation(image, size=(filter_size,filter_size), structure=np.ones(tuple(structure_shape)))
            image = clip(image, 0, 255)
        return image

    return img_op


def dilation_then_erosion(filter_size=3, repeat=1):
    def img_op(image: np.ndarray,**kwargs):
        structure_shape = [1] * image.ndim
        structure_shape[0]=filter_size
        structure_shape[1] = filter_size
        for i in range(repeat):
            image = ndimage.morphology.grey_dilation(image, size=(filter_size,filter_size), structure=np.ones(tuple(structure_shape)))
            image = ndimage.morphology.grey_erosion(image, size=(filter_size,filter_size), structure=np.ones(tuple(structure_shape)))
        return image

    return img_op


def adaptive_binarization(threshold_type='otsu'):
    def img_op(image: np.ndarray,**kwargs):

        image = image.astype(np.float32)
        original_shape=image.shape
        local_thresh = None
        blur = gaussian(image, sigma=0.5, multichannel=True if image.ndim == 3 else None)
        try:
            if threshold_type == 'otsu':
                if len(blur.shape) > 2 and blur.shape[-1] in (3, 4):
                    blur = color.rgb2gray(blur.astype(np.float32))
                if blur.min() == blur.max():
                    return image
                else:
                    local_thresh = threshold_otsu(blur)
            elif threshold_type == 'minimum':
                local_thresh = threshold_minimum(blur)
            elif threshold_type == 'local':
                local_thresh = threshold_local(blur, block_size=35, offset=10)
            elif threshold_type == 'isodata':
                local_thresh = threshold_isodata(blur, nbins=256)
            elif threshold_type == 'percentile':
                p10 = np.percentile(image.copy(), 10)
                p90 = np.percentile(image.copy(), 90)
                if abs(image.mean() - p90) < abs(image.mean() - p10) and p90 - p10 > 80:  # white background
                    image[image < p10] = 0
                    image[image > p90] = 255
                elif abs(image.mean() - p90) > abs(image.mean() - p10) and p90 - p10 > 80:  # white background
                    image[image > p90] = 255
                    image[image < p10] = 0
                return image

            blur = (blur > local_thresh).astype(np.float32) * 255.0
            if blur.max() - blur.min() < 10:
                pass
            else:
                image = blur
        except Exception as e:
            print(e)

        if image.ndim==2:
            if len(original_shape)==2:
                pass
            elif  len(original_shape)==3:
                image=np.expand_dims(image,-1)
                if original_shape[-1] ==3:
                    image=to_rgb()(image)

        return image

    return img_op


def blur(sigma=0.3):
    def img_op(image: np.ndarray,**kwargs):
        if len(image.shape) > 2 and image.shape[-1] in (3, 4):
            image = gaussian(image.astype(np.float32), sigma=sigma, multichannel=True, preserve_range=True)
        else:
            image = gaussian(image.astype(np.float32), sigma=sigma, multichannel=False, preserve_range=True)
        return image

    return img_op


def random_blur(sigma=(0.1, 0.6)):
    def img_op(image: np.ndarray,**kwargs):
        n = random.randint(0, 3)
        if n == 0:
            return image
        else:
            image = gaussian(image.astype(np.float32), sigma=np.random.choice(np.arange(sigma[0], sigma[1], 0.05)),
                             multichannel=True)
        return image

    return img_op


def channel_reverse():
    def img_op(image: np.ndarray,**kwargs):
        if image.ndim == 4:
            image = image[..., ::-1]
        elif image.ndim == 3:
            image = image[:, :, ::-1]
        return image

    return img_op


def build_affine_matrix(s, c, rad, t=(0, 0)):
    if isinstance(s, float):
        s = [s, s]
    C = [[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1]]
    S = [[s[0], 0, 0], [0, s[1], 0], [0, 0, 1]]
    T = [[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]
    R = [[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]]
    C_prime = [[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]]
    M = np.dot(C_prime, np.dot(T, np.dot(R, np.dot(S, C))))
    return M




def to_low_resolution(scale=2):
    def img_op(image: np.ndarray,**kwargs):
        rnd = random.randint(0, 10)
        if rnd == 0:
            image = rescale(1 / scale)(image)
        elif rnd % 3 == 0:
            image = downsample_then_upsample(scale + 2)(image)
            image = add_noise(0.02)(image)
            image = rescale(1 / scale)(image)
        elif rnd % 3 == 1:
            image = add_noise(0.02)(image)
            image = random_blur((0.1, 0.5))(image)
            image = rescale(1 / scale)(image)
        elif rnd % 3 == 2:
            image = downsample_then_upsample(scale, repeat=2)(image)
            image = add_noise(0.02)(image)
            image = rescale(1 / scale)(image)
        return image

    return img_op



# def image_smoothening():
#     def img_op(image: np.ndarray,**kwargs):
#         ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
#         ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         blur = cv2.GaussianBlur(th2, (1, 1), 0)
#         ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         return th3

#
# def do_random_crop_rescale(image, mask, w, h):
#     height, width = image.shape[:2]
#     x,y=0,0
#     if width>w:
#         x = np.random.choice(width-w)
#     if height>h:
#         y = np.random.choice(height-h)
#     image = image[y:y+h,x:x+w]
#     mask  = mask [y:y+h,x:x+w]
#
#     #---
#     if (w,h)!=(width,height):
#         image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize( mask,  dsize=(width,height), interpolation=cv2.INTER_NEAREST)
#
#     return image, mask
#
# def do_random_crop_rotate_rescale(image, mask, w, h):
#     H,W = image.shape[:2]
#
#     #dangle = np.random.uniform(-2.5, 2.5)
#     #dscale = np.random.uniform(-0.10,0.10,2)
#     dangle = np.random.uniform(-8, 8)
#     dshift = np.random.uniform(-0.1,0.1,2)
#
#     dscale_x = np.random.uniform(-0.00075,0.00075)
#     dscale_y = np.random.uniform(-0.25,0.25)
#
#     cos = math.cos(dangle/180*math.pi)
#     sin = math.sin(dangle/180*math.pi)
#     sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
#     tx,ty = dshift*min(H,W)
#
#     src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
#     src = src*[sx,sy]
#     x = (src*[cos,-sin]).sum(1)+W/2
#     y = (src*[sin, cos]).sum(1)+H/2
#     # x = x-x.min()
#     # y = y-y.min()
#     # x = x + (W-x.max())*tx
#     # y = y + (H-y.max())*ty
#     #
#     # if 0:
#     #     overlay=image.copy()
#     #     for i in range(4):
#     #         cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)image_show('overlay',overlay)
#     #
#
#
#     src = np.column_stack([x,y])
#     dst = np.array([[0,0],[w,0],[w,h],[0,h]])
#     s = src.astype(np.float32)
#     d = dst.astype(np.float32)
#     transform = cv2.getPerspectiveTransform(s,d)
#
#     image = cv2.warpPerspective( image, transform, (W, H),
#         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
#
#     mask = cv2.warpPerspective( mask, transform, (W, H),
#         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
#
#     return image, mask
#
# def do_random_log_contast(image, gain=[0.70, 1.30] ):
#     gain = np.random.uniform(gain[0],gain[1],1)
#     inverse = np.random.choice(2,1)
#
#     image = image.astype(np.float32)/255
#     if inverse==0:
#         image = gain*np.log(image+1)
#     else:
#         image = gain*(2**image-1)
#
#     image = np.clip(image*255,0,255).astype(np.uint8)
#     return image
#
#

#
#







