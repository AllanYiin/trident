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
import warnings

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
from trident.backend.decorators import deprecated
from trident.backend.tensorspec import TensorSpec, assert_input_compatibility, ObjectType
from trident.data.vision_transforms import *

__all__ = [ 'list_pictures','list_images','check_same_size', 'normalize', 'unnormalize', 'channel_reverse', 'blur', 'random_blur',
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


from trident.backend.opencv_backend import *





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

@deprecated('0.7.0', 'list_images')
def list_pictures(directory, ext='jpg|jpeg|jpe|tiff|tif|bmp|png|ppm|jfif'):
    filelist = []
    for root, dirs, files in os.walk(directory):
        filelist = filelist + [os.path.join(root,f) for f in files if re.match( "([^\\s]+(\\.(?i)({0}))$)".format(ext), f)]
    return filelist



def list_images(directory, ext='jpg|jpeg|jpe|tiff|tif|bmp|png|ppm|jfif'):
    # return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if
    #         re.match(r'([\w]+\.(?:' + ext + '))', f)]
    filelist = []
    for root, dirs, files in os.walk(directory):
        filelist = filelist + [os.path.join(root,f) for f in files if re.match( "([^\\s]+(\\.(?i)({0}))$)".format(ext), f)]
    return filelist


def check_same_size(*images):
    result = True
    base_shape =None
    if isinstance(images[0],str):
        return True
    elif isinstance(images[0],numbers.Number):
        base_shape=0
    elif  isinstance(images[0],np.ndarray):
        base_shape = TensorShape(images[0].shape)
    # check same isze
    for img in images:
        if base_shape==0 and isinstance(img, numbers.Number):
            pass
        elif   isinstance(img,np.ndarray) and base_shape== TensorShape(img.shape):
            pass
        else:
            print(img.shape)
            result = False
    return True



def add_noise(intensity=0.1):
    return AddNoise(intensity=intensity)


def normalize(mean, std):
    return Normalize(mean=mean,std=std)


def unnormalize(mean, std):
    return Unnormalize(mean=mean,std=std)

#
# 0: Nearest-neighbor
# 1: Bi-linear (default)
# 2: Bi-quadratic
# 3: Bi-cubic
# 4: Bi-quartic
# 5: Bi-quintic

# all size is HWC or (H,W)
def resize(size, keep_aspect=True, order=1, align_corner=True):
    interpolation=cv2.INTER_LINEAR if order==1 else cv2.INTER_NEAREST
    return Resize(size, keep_aspect=keep_aspect,interpolation=interpolation, align_corner=align_corner)


def rescale(scale, order=1):
    interpolation = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
    return Rescale(scale=scale,interpolation=interpolation)

def random_rescale_crop(h, w, scale=(0.5, 2), order=1):
    interpolation = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
    return RandomRescaleCrop(output_size=(h,w),scale_range=scale,interpolation=interpolation)


def random_center_crop(h,w, scale=(0.8, 1.2), order=1):
    interpolation = cv2.INTER_LINEAR if order == 1 else cv2.INTER_NEAREST
    return RandomCenterCrop(output_size=(h,w),scale_range=scale,interpolation=interpolation)


def random_crop(h, w):
    return RandomCrop(output_size=(h,w))



def random_transform(rotation_range= 15, zoom_range= 0.02, shift_range= 0.02,shear_range = 0.2,   random_flip= 0.15):
    return RandomTransformAffine(rotation_range=rotation_range,zoom_range=zoom_range,shift_range=shift_range,shear_range=shear_range,random_flip=random_flip)



def horizontal_flip():
    return HorizontalFlip()


def random_mirror():
    return RandomMirror()



def downsample_then_upsample(scale=4, order=1, repeat=1):
    def img_op(image: np.ndarray,**kwargs):
        for i in range(repeat):
            image = rescale(scale=1.0 / scale)(image)
            image = rescale(scale=scale)(image)
        return image

    return img_op


def invert_color():
    return InvertColor()


def random_invert_color():
    return RandomInvertColor()


def gray_scale():
    return GrayScale()

def random_gray_scale():
    return RandomGrayScale()

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


def adjust_brightness(value):
    return AdjustBrightness(value=value)



def adjust_saturation(value):
    return AdjustSaturation(value=value)



def adjust_contrast(value):
    return AdjustContrast(value=value)


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



def random_adjust_contrast(scale=(0.5, 1.5)):
    return RandomAdjustContrast(value_range=scale)


def random_adjust_hue(hue_range=(-20, 20), saturation_range=(0.5, 1.5), lightness_range=(-50, 50)):
    return RandomAdjustHue(value_range=(0,0.5))


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
    image=to_numpy(image)
    if get_backend() =='pytorch' and len( int_shape(image))== 3 and int_shape(image)[0] in [3, 4]:
        image = np.transpose(image, [1, 2, 0]).astype(np.float32)
    elif get_backend() =='pytorch'  and len( int_shape(image))== 4 and int_shape(image)[1] in [3, 4]:
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
    return RandomErasing


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


    return ImageDilation(filter_size=filter_size,repeat=repeat)


def image_dilation(filter_size=3, repeat=1):
    """ Dilation operation
    Dilation is a mathematical morphology operation that uses a structuring element for expanding the shapes in an image. The binary dilation of an image by a structuring element is the locus of the points covered by the structuring element, when its center lies within the non-zero points of the image.

    Args:
        filter_size (int): the size of the structuring element .
        repeat (int): the number of repeating operation.

    Returns:
        output image array

    """
    return  ImageErosion(filter_size=filter_size, repeat=repeat)


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
    # def img_op(image: np.ndarray,**kwargs):
    #
    #     image = image.astype(np.float32)
    #     original_shape=image.shape
    #     local_thresh = None
    #     blur = gaussian(image, sigma=0.5, multichannel=True if image.ndim == 3 else None)
    #     try:
    #         if threshold_type == 'otsu':
    #             if len(blur.shape) > 2 and blur.shape[-1] in (3, 4):
    #                 blur = color.rgb2gray(blur.astype(np.float32))
    #             if blur.min() == blur.max():
    #                 return image
    #             else:
    #                 local_thresh = threshold_otsu(blur)
    #         elif threshold_type == 'minimum':
    #             local_thresh = threshold_minimum(blur)
    #         elif threshold_type == 'local':
    #             local_thresh = threshold_local(blur, block_size=35, offset=10)
    #         elif threshold_type == 'isodata':
    #             local_thresh = threshold_isodata(blur, nbins=256)
    #         elif threshold_type == 'percentile':
    #             p10 = np.percentile(image.copy(), 10)
    #             p90 = np.percentile(image.copy(), 90)
    #             if abs(image.mean() - p90) < abs(image.mean() - p10) and p90 - p10 > 80:  # white background
    #                 image[image < p10] = 0
    #                 image[image > p90] = 255
    #             elif abs(image.mean() - p90) > abs(image.mean() - p10) and p90 - p10 > 80:  # white background
    #                 image[image > p90] = 255
    #                 image[image < p10] = 0
    #             return image
    #
    #         blur = (blur > local_thresh).astype(np.float32) * 255.0
    #         if blur.max() - blur.min() < 10:
    #             pass
    #         else:
    #             image = blur
    #     except Exception as e:
    #         print(e)
    #
    #     if image.ndim==2:
    #         if len(original_shape)==2:
    #             pass
    #         elif  len(original_shape)==3:
    #             image=np.expand_dims(image,-1)
    #             if original_shape[-1] ==3:
    #                 image=to_rgb()(image)
    #
    #     return image

    return AdaptiveBinarization(threshold_type=threshold_type)


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







