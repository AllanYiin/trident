from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import re
import time
from itertools import repeat

import cv2
import numpy as np
import six
from scipy import ndimage
from skimage import color
from skimage import exposure
from skimage import morphology
from skimage import transform, exposure
from skimage.filters import *

from ..backend.common import *

__all__ = ['read_image', 'read_mask', 'save_image', 'save_mask', 'image2array', 'array2image', 'mask2array',
           'array2mask', 'list_pictures', 'normalize', 'unnormalize', 'channel_reverse', 'blur', 'random_blur',
           'random_crop', 'resize', 'rescale', 'downsample_then_upsample', 'add_noise', 'gray_scale', 'to_rgb',
           'to_bgr', 'auto_level', 'random_invert_color', 'image_backend_adaptive', 'reverse_image_backend_adaptive',
           'random_adjust_hue', 'random_channel_shift', 'random_cutout', 'random_rescale_crop', 'random_center_crop',
           'adjust_gamma', 'random_adjust_gamma', 'adjust_contrast', 'random_adjust_contrast', 'clahe',
           'erosion_then_dilation', 'dilation_then_erosion', 'image_erosion', 'image_dilation', 'adaptive_binarization',
           'random_transform', 'horizontal_flip', 'random_mirror', 'to_low_resolution']

_session = get_session()
_backend = _session.backend
_image_backend = _session.image_backend

if _image_backend == 'opencv':
    from ..backend.opencv_backend import *
else:
    from ..backend.pillow_backend import *

read_image = read_image
read_mask = read_mask
save_image = save_image
save_mask = save_mask
image2array = image2array
array2image = array2image
mask2array = mask2array
array2mask = array2mask


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
    def img_op(image: np.ndarray):
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
    def img_op(image: np.ndarray):
        norm_mean = mean
        norm_std = std
        if isinstance(norm_mean, (float, int)) and image.ndim == 3:
            norm_mean = np.array([norm_mean, norm_mean, norm_mean])
            norm_mean = np.expand_dims(norm_mean, -2)
            norm_mean = np.expand_dims(norm_mean, -2)
        if isinstance(norm_std, (float, int)) and image.ndim == 3:
            norm_std = np.array([norm_std, norm_std, norm_std])
            norm_std = np.expand_dims(norm_std, -2)
            norm_std = np.expand_dims(norm_std, -2)
        if image.ndim == 3:
            return (image - norm_mean) / norm_std
        elif image.ndim == 2:
            if isinstance(norm_mean, (float, int)) and isinstance(norm_std, (float, int)):
                return (image - norm_mean) / norm_std
        return image

    img_op.mean = mean
    img_op.std = std
    return img_op


def unnormalize(mean, std):
    def img_op(image: np.ndarray):
        image = reverse_image_backend_adaptive(image)
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
    def img_op(*image: np.ndarray):
        results = to_list(image)
        if keep_aspect:
            h, w = image[0].shape[:2]
            img_op.h=h
            img_op.w=w
            sh, sw = size

            # aspect ratio of image
            aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h
            target_aspect=sw/sh
            # compute scaling and pad sizing
            if aspect > target_aspect:  # horizontal image
                new_w = sw
                new_h = np.round(new_w / aspect).astype(int)
                pad_vert = (sh - new_h) / 2
                pad_top, pad_btm = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
                pad_left, pad_right = 0, 0
            elif aspect < target_aspect:  # vertical image
                new_h = sh
                new_w = np.round(new_h * aspect).astype(int)
                pad_horz = (sw - new_w) / 2
                pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
                pad_top, pad_btm = 0, 0
            else:  # square image
                new_h, new_w = sh, sw
                pad_left, pad_right, pad_top, pad_btm = 0, 0, 0, 0

            scalex = new_w / w
            scaley = new_h / h
            img_op.scale = min(scalex,scaley)
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
            for i in range(len(results)):
                im = results[i]
                if im is None:
                    pass
                elif im.ndim == 2 and im.shape[-1] in (4, 5):  # bbox
                    im[:, 0:4]=im[:, 0:4]*img_op.scale
                    results[i] = im
                elif im.ndim == 2 and im.dtype == np.int64:  # bbox
                    new_im = np.zeros((size[0], size[1]))
                    im = transform.rescale(im,  img_op.scale, clip=False, anti_aliasing=False, multichannel=False,
                                           preserve_range=True, order=0).astype(np.int64)
                    if align_corner:
                        new_im[:im.shape[0], :im.shape[1]] = im
                    else:
                        new_im[pad_top:im.shape[0] + pad_top, pad_left:im.shape[1] + pad_left] = im
                    results[i] = np.pad(im, img_op.all_pad, 'constant').astype(np.int64)

                else:
                    im = im.astype(np.float32)
                    if im.ndim==2:
                        new_im= np.zeros((size[0],size[1]))
                    elif im.ndim==3:
                        new_im = np.zeros((size[0], size[1],im.shape[-1]))

                    im = transform.rescale(im, img_op.scale, clip=False, anti_aliasing=True,
                                           multichannel=True if len(im.shape) == 3 else False,
                                           order=0 if im.ndim == 2 else order)
                    if im.shape[0]>size[0] or im.shape[1]>size[1]:
                        print('')
                    if align_corner:
                        if im.ndim == 2:
                            new_im[:im.shape[0],:im.shape[1]]=im
                        elif im.ndim == 3:
                            new_im[:im.shape[0],:im.shape[1],:]=im
                        results[i] = new_im
                    else:
                        if im.ndim == 2:
                            new_im[pad_top:im.shape[0]+pad_top,pad_left:im.shape[1]+pad_left]=im
                        elif im.ndim == 3:
                            new_im[pad_top:im.shape[0]+pad_top,pad_left:im.shape[1]+pad_left,:]=im

                        results[i] = new_im
            return unpack_singleton(tuple(results))

        else:
            img_op.scalex = size[1] / image[0].shape[1]
            img_op.scaley = size[0] / image[0].shape[0]
            img_op.scale = (img_op.scaley, img_op.scalex)
            for i in range(len(results)):
                im = results[i]
                if im is None:
                    pass
                elif im.ndim == 2 and im.shape[-1] in (4, 5):  # bbox
                    im[:, 0::2] /= img_op.w
                    im[:, 1::2] /= img_op.h
                    results[i] = im
                else:
                    im = im.astype(np.float32)
                    results[i] = transform.resize(im, size, anti_aliasing=True, order=0 if im.ndim == 2 else order)
            return unpack_singleton(tuple(results))

    return img_op


def rescale(scale, order=1):
    def img_op(*image: np.ndarray):
        results = to_list(image)
        h, w = image[0].shape[:2]
        img_op.h = h
        img_op.w = w
        img_op.scale = (scale, scale)
        for i in range(len(results)):
            im = results[i]
            if im is None or len(im) == 0:
                pass
            elif len(im.shape) == 2 and im.shape[-1] in (4, 5, 14, 15):  # bbox
                im[:, 0:4]=im[:, 0:4]* img_op.scale
                results[i] = im
            elif im.ndim == 2 and im.dtype == np.int64:  # bbox
                results[i] = transform.rescale(im, (scale, scale), clip=True, anti_aliasing=False, multichannel=False,
                                               preserve_range=True, order=0).astype(np.int64)

            else:
                im = im.astype(np.float32)
                results[i] = transform.rescale(im, (scale, scale), clip=False, anti_aliasing=True,
                                               multichannel=True if len(im.shape) == 3 else False,
                                               order=0 if im.ndim == 2 else order)
        return unpack_singleton(tuple(results))


    return img_op


def downsample_then_upsample(scale=4, order=1, repeat=1):
    def img_op(image: np.ndarray):
        for i in range(repeat):
            image = rescale(scale=1.0 / scale)(image)
            image = rescale(scale=scale)(image)
        return image

    return img_op


def invert_color():
    def img_op(image: np.ndarray):
        if np.min(image) >= 0 and np.max(image) <= 1:
            return 1 - image
        elif np.min(image) >= -1 and np.min(image) < 0 and np.max(image) <= 1:
            return 1 - (image * 0.5 + 0.5)
        else:
            return 255 - image

    return img_op


def random_invert_color():
    def img_op(image: np.ndarray):
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
    def img_op(image: np.ndarray):
        if image.shape[0] == 3:
            image = color.rgb2gray(image.astype(np.float32))
        return image

    return img_op


def to_rgb():
    def img_op(image: np.ndarray):

        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
            image = np.concatenate([image, image, image], -1)
            return image

        elif len(image.shape) == 3:
            if image.shape[0] == 3 and image.shape[0] == 4:
                image = image.transpose([1, 2, 0])
            if image.shape[-1] == 4:
                image = image[:, :, :3]
        return image

    return img_op


def to_bgr():
    def img_op(image: np.ndarray):
        if image.ndim == 3:
            if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
                if image[0].shape == 4:
                    image = image[:3, :, :]
                image = image.transpose([1, 2, 0])
            if image[-1].shape in (3, 4):
                if image[-1].shape == 4:
                    image = image[:, :, :3]
                image = image[..., ::-1]
        return image

    return img_op


def adjust_gamma(gamma=1):
    def img_op(image: np.ndarray):
        return exposure.adjust_gamma(image, gamma)

    return img_op


def random_adjust_gamma(gamma=(0.6, 1.4)):
    gammamin, gammamax = gamma

    def img_op(image: np.ndarray):
        gamma = np.random.choice(np.arange(gammamin, gammamax, 0.01))
        return exposure.adjust_gamma(image, gamma)

    return img_op


def adjust_contrast(alpha=1):
    beta = 0

    def img_op(image: np.ndarray):
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image.astype(np.float32)

    return img_op


def random_adjust_contrast(scale=(0.5, 1.5)):
    beta = 0
    scalemin, scalemax = scale

    def img_op(image: np.ndarray):
        alpha = random.uniform(scalemin, scalemax)
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image.astype(np.float32)

    return img_op


def random_adjust_hue(hue_range=(-20, 20), saturation_range=(-10, 10), lightness_range=(-20, 20)):
    def img_op(image: np.ndarray):
        # hue is mapped to [0, 1] from [0, 360]
        # if hue_offset not in range(-180, 180):
        #     raise ValueError('Hue should be within (-180, 180)')
        # if saturation not in range(-100, 100):
        #     raise ValueError('Saturation should be within (-100, 100)')
        # if lightness not in range(-100, 100):
        #     raise ValueError('Lightness should be within (-100, 100)')
        hue_offset = random.uniform(*hue_range)
        saturation = random.uniform(*saturation_range)
        lightness = random.uniform(*lightness_range)
        image = color.rgb2hsv(image.astype('uint8'))
        hue_offset = ((180 + hue_offset) % 180) / 360.0
        image[:, :, 0] = image[:, :, 0] + hue_offset
        image[:, :, 0][image[:, :, 0] > 1] -= 1
        image[:, :, 0][image[:, :, 0] < 0.0] += 1

        # image[:, :, 1] = image[:, :, 1] + saturation / 200.0
        # image[:, :, 2] = image[:, :, 2] + lightness / 200.0
        image = color.hsv2rgb(image) * 255.0
        return image.astype(np.float32)

    return img_op


def auto_level():
    def img_op(image: np.ndarray):
        minv = np.percentile(image, 5)
        maxv = np.percentile(image, 95)
        if maxv - minv < 40:
            minv = image.min()
            maxv = image.max()

        image = np.clip((image - minv) * (255.0 / (maxv - minv)), 0, 255)
        return image.astype(np.float32)

    return img_op


def random_crop(h, w):
    def img_op(*image: np.ndarray):
        results = to_list(image)
        height, width = image[0].shape[:2]
        img_op.h = height
        img_op.w = width
        offset_x = 0
        offset_y = 0

        if width > w:
            offset_x = random.choice(range(width - w))
        if height > h:
            offset_y = random.choice(range(height - h))
        offset_x1 = random.choice(range(w - width)) if w > width else 0
        offset_y1 = random.choice(range(h - height)) if h > height else 0
        for i in range(len(results)):
            im = results[i]
            if im is None:
                pass
            elif im.ndim == 2 and im.shape[-1] in (4, 5):  # bbox

                im[:, 0::2] = np.clip(im[:, 0::2] - offset_x, 0, w)
                im[:, 1::2] = np.clip(im[:, 1::2] - offset_y, 0, h)

                area = (im[:, 3] - im[:, 1]) * (im[:, 2] - im[:, 0])
                im = im[area > 0]
                if len(im) > 0:
                    im[:, 0::2] += offset_x1
                    im[:, 1::2] += offset_y1

                    results[i] = im
                else:
                    results[i] = None

            elif im.ndim == 2:
                returnData = np.zeros((h, w), dtype=np.float32)
                crop_im = im[offset_y:offset_y + h, offset_x:offset_x + w]
                returnData[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1]] = crop_im
                results[i] = returnData
            elif im.ndim == 3:
                returnData = np.zeros((h, w, 3), dtype=np.float32)
                crop_im = im[offset_y:offset_y + h, offset_x:offset_x + w, :]
                returnData[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1], :] = crop_im
                results[i] = returnData
        return unpack_singleton(tuple(results))

    return img_op


def random_rescale_crop(h, w, scale=(0.5, 2), order=1):
    scalemin, scalemax = scale

    def img_op(*image: np.ndarray):
        # start = time.time()
        results = to_list(image)
        scale = np.random.choice(np.random.uniform(scalemin, scalemax, 100))
        height, width = image[0].shape[:2]
        height, width = int(height * scale), int(width * scale)
        offset_x = 0
        offset_y = 0

        if width > w:
            offset_x = random.choice(range(width - w))
        if height > h:
            offset_y = random.choice(range(height - h))
        offset_x1 = random.choice(range(w - width)) if w > width else 0
        offset_y1 = random.choice(range(h - height)) if h > height else 0
        # stop = time.time()
        # print('prepare random crop:{0}'.format(stop - start))
        # start = stop
        for i in range(len(results)):
            im = results[i]
            if im is None:
                pass
            elif im.ndim == 2 and im.shape[-1] in (4, 5):  # bbox
                im[:, 0] *= scale
                im[:, 1] *= scale
                im[:, 2] *= scale
                im[:, 3] *= scale

                im[:, 0] = np.clip(im[:, 0] - offset_x, 0, w)
                im[:, 1] = np.clip(im[:, 1] - offset_y, 0, h)
                im[:, 2] = np.clip(im[:, 2] - offset_x, 0, w)
                im[:, 3] = np.clip(im[:, 3] - offset_y, 0, h)
                area = (im[:, 3] - im[:, 1]) * (im[:, 2] - im[:, 0])
                im = im[area > 0]
                if len(im) > 0:
                    im[:, 0] += offset_x1
                    im[:, 1] += offset_y1
                    im[:, 2] += offset_x1
                    im[:, 3] += offset_y1
                    results[i] = im
                else:
                    results[i] = None

            elif im.ndim == 2:

                im = np.round(transform.rescale(im, (scale, scale), clip=False, anti_aliasing=False, multichannel=False,
                                                preserve_range=True, order=0)).astype(im.dtype)
                returnData = np.zeros((h, w), dtype=im.dtype)
                crop_im = im[offset_y:offset_y + h, offset_x:offset_x + w]
                returnData[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1]] = crop_im
                results[i] = returnData
            elif im.ndim == 3:
                # im = transform.rescale(im, (scale, scale), clip=False, anti_aliasing=True,
                #                        multichannel=True if len(im.shape) == 3 else False,
                #                        order=0 if im.ndim == 2 else order)
                im = rescale(scale)(im)
                # stop = time.time()
                # print('resize:{0}'.format(stop - start))
                # start = stop
                returnData = np.zeros((h, w, 3), dtype=np.float32)
                crop_im = im[offset_y:offset_y + h, offset_x:offset_x + w, :]
                returnData[offset_y1:offset_y1 + crop_im.shape[0], offset_x1:offset_x1 + crop_im.shape[1], :] = crop_im
                results[i] = returnData  # stop = time.time()  # print('crop:{0}'.format(stop - start))  # start = stop
        return unpack_singleton(tuple(results))

    return img_op


def random_center_crop(h,w, scale=(0.8, 1.2)):
    scalemin, scalemax = scale

    def img_op(*image: np.ndarray):
        results = to_list(image)
        height, width = image[0].shape[:2]
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
        for k in range(len(results)):
            im = results[k]
            if im is None:
                pass
            elif im.ndim == 2 and im.shape[-1] in (4, 5):  # bbox
                im[:, 0] = np.clip((im[:, 0] + j) * scale + j1 - j2, 0, w)
                im[:, 1] = np.clip((im[:, 1] + i) * scale + i1 - i2, 0, h)
                im[:, 2] = np.clip((im[:, 2] + j) * scale + j1 - j2, 0, w)
                im[:, 3] = np.clip((im[:, 3] + i) * scale + i1 - i2, 0, h)
                area = (im[:, 3] - im[:, 1]) * (im[:, 2] - im[:, 0])
                im = im[area > 0]
                if len(im) > 0:
                    results[k] = im
                else:
                    results[k] = None

            elif im.ndim == 3:
                blank = np.zeros((max_value, max_value, im.shape[-1]))
                blank[i:i + height, j:j + width, :] = im

                resized_im = transform.resize(blank, (resized_h, resized_w, blank.shape[-1]), anti_aliasing=True,
                                              clip=False, order=0 if blank.ndim == 2 else 1)
                returnData = np.zeros((max(resized_h, h), max(resized_w, w), resized_im.shape[-1]))

                returnData[i1:i1 + resized_im.shape[0], j1:j1 + resized_im.shape[1], :] = resized_im
                results[k] = returnData[i2:i2 + h, j2:j2 + w, :]
            elif im.ndim == 2:
                blank = np.zeros((max_value, max_value)).astype(im.dtype)
                blank[i:i + height, j:j + width] = im

                resized_im = np.round(transform.resize(blank, (resized_h, resized_w), anti_aliasing=False, clip=False,
                                                       preserve_range=True, order=0)).astype(im.dtype)
                returnData = np.zeros((max(resized_h, h), max(resized_w, w))).astype(im.dtype)

                returnData[i1:i1 + resized_im.shape[0], j1:j1 + resized_im.shape[1]] = resized_im
                results[k] = returnData[i2:i2 + h, j2:j2 + w]
        return unpack_singleton(tuple(results))

    return img_op


def horizontal_flip():
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    def img_op(*image: np.ndarray):
        results = to_list(image)
        height, width = image[0].shape[:2]
        for k in range(len(results)):
            im = results[k]
            if im is None:
                pass
            elif im.ndim == 2 and im.shape[-1] in (4, 5):  # bbox
                im[:, 0::2] = width - im[:, 2::-2]
                if len(im) > 0:
                    results[k] = im
                else:
                    results[k] = None
            elif im.ndim == 3:
                results[k] = im[:, ::-1]
            elif im.ndim == 2:
                results[k] = im[::-1]
        return unpack_singleton(tuple(results))

    return img_op


def random_mirror():
    def img_op(*image: np.ndarray):
        results = to_list(image)
        height, width = image[0].shape[:2]
        img_op.rnd = random.randint(0, 10)
        for k in range(len(results)):
            im = results[k]
            if im is None:
                pass
            elif im.ndim == 2 and im.shape[-1] in (4, 5):  # bbox
                if img_op.rnd % 2 == 0:
                    im[:, 0::2] = width - im[:, 2::-2]
                if len(im) > 0:
                    results[k] = im
                else:
                    results[k] = None

            elif im.ndim == 3:
                if img_op.rnd % 2 == 0:
                    results[k] = im[:, ::-1]
            elif im.ndim == 2:
                if img_op.rnd % 2 == 0:
                    results[k] = im[:, ::-1]
        return unpack_singleton(tuple(results))

    return img_op


def image_backend_adaptive(image):
    if _session.backend == 'tensorflow' and image.ndim in (3, 4):
        image = image.astype(np.float32)
    elif _session.backend in ['pytorch', 'cntk'] and image.ndim == 3:
        image = np.transpose(image, [2, 0, 1]).astype(np.float32)
    elif _session.backend in ['pytorch', 'cntk'] and image.ndim == 4:
        image = np.transpose(image, [0, 3, 1, 2]).astype(np.float32)
    elif isinstance(image, np.ndarray):
        return image.astype(np.float32)
    elif isinstance(image, list):
        return np.array(image).astype(np.float32)
    return image


def reverse_image_backend_adaptive(image):
    if _session.backend in ['pytorch', 'cntk'] and image.ndim == 3 and image.shape[0] in [3, 4]:
        image = np.transpose(image, [1, 2, 0]).astype(np.float32)
    elif _session.backend in ['pytorch', 'cntk'] and image.ndim == 4 and image.shape[1] in [3, 4]:
        image = np.transpose(image, [0, 2, 3, 1]).astype(np.float32)
    return image


def random_channel_shift(intensity=15):
    channel_axis = -1
    inten = intensity

    def img_op(image: np.ndarray):
        image = np.rollaxis(image, channel_axis, channel_axis)
        min_x, max_x = np.min(image), np.max(image)
        intensity = max_x / 255 * inten
        channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x) for x_channel in
                          image[:, :, ]]
        x = np.stack(channel_images, axis=channel_axis)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    return img_op


def random_cutout(img, mask):
    h, w = img.shape[:2] if _backend == 'tensorflow' or len(img.shape) == 2 else img.shape[1:3]
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
    if _backend == 'tensorflow':
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
    def img_op(image: np.ndarray):
        image = image.astype(np.float32) / 255.0
        if image.max() - image.min() > 0.2:
            image = exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
        return image * 255.0

    return img_op


def image_erosion(filter_size=3, repeat=1):
    def img_op(image: np.ndarray):
        for i in range(repeat):
            image = morphology.erosion(image, morphology.square(filter_size))
        return image

    return img_op


def image_dilation(filter_size=3, repeat=1):
    def img_op(image: np.ndarray):
        for i in range(repeat):
            image = morphology.dilation(image, morphology.square(filter_size))
        return image

    return img_op


def erosion_then_dilation(filter_size=3, repeat=1):
    def img_op(image: np.ndarray):
        for i in range(repeat):
            image = morphology.erosion(image, morphology.square(filter_size))
            image = morphology.dilation(image, morphology.square(filter_size))
        return image

    return img_op


def dilation_then_erosion(filter_size=3, repeat=1):
    def img_op(image: np.ndarray):
        for i in range(repeat):
            image = morphology.dilation(image, morphology.square(filter_size))
            image = morphology.erosion(image, morphology.square(filter_size))
        return image

    return img_op


def adaptive_binarization(threshold_type='otsu'):
    def img_op(image: np.ndarray):
        image = image.astype(np.float32)
        local_thresh = None
        blur = gaussian(image, sigma=0.5, multichannel=True if image.ndim == 3 else None)
        try:
            if threshold_type == 'otsu':
                if len(blur.shape) > 2 and blur.shape[-1] in (3, 4):
                    image = color.rgb2gray(blur.astype(np.float32))
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
        return image

    return img_op


def blur(sigma=0.3):
    def img_op(image: np.ndarray):
        if len(image.shape) > 2 and image.shape[-1] in (3, 4):
            image = gaussian(image.astype(np.float32), sigma=sigma, multichannel=True, preserve_range=True)
        else:
            image = gaussian(image.astype(np.float32), sigma=sigma, multichannel=False, preserve_range=True)
        return image

    return img_op


def random_blur(sigma=(0.1, 0.6)):
    def img_op(image: np.ndarray):
        n = random.randint(0, 3)
        if n == 0:
            return image
        else:
            image = gaussian(image.astype(np.float32), sigma=np.random.choice(np.arange(sigma[0], sigma[1], 0.05)),
                             multichannel=True)
        return image

    return img_op


def channel_reverse():
    def img_op(image: np.ndarray):
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


def random_transform(rotation_range=12, zoom_range=0.05, shift_range=0.05, random_flip=0.3):
    def img_op(img):
        h, w = img.shape[0:2]
        rotation = np.random.uniform(-rotation_range, rotation_range)
        scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        tx = np.random.uniform(-shift_range, shift_range) * w
        ty = np.random.uniform(-shift_range, shift_range) * h
        mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
        mat[:, 2] += (tx, ty)

        img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_CONSTANT)  # , borderMode=cv2.BORDER_REPLICATE
        if np.random.random() < random_flip:
            img = img[:, ::-1]

        return img

    return img_op


def to_low_resolution(scale=2):
    def img_op(image: np.ndarray):
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
#     def img_op(image: np.ndarray):
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







