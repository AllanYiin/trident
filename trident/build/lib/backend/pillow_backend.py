from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import PIL

try:
    from PIL import ImageEnhance
    from PIL import ImageOps
    from PIL import ImageFilter
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None
    ImageFilter=None
import numpy as np
from .common import floatx
from .common import image_data_format

version=PIL.__version__
sys.stderr.write('Pillow version:{0}.\n'.format(version))


def image2array(img):
    if isinstance(img,str):
        if os.path.exists(img):
            img=pil_image.open(img)
        else:
            return None
    if isinstance(img,pil_image.Image):
        arr = np.array(img).astype(floatx())
        if len(arr.shape) > 2:
            if image_data_format()=='channels_first':
                #HWC->CHW  RGB->BGR
                arr = arr.transpose(2, 0, 1)
                arr = arr[::-1]
        if arr.flags['C_CONTIGUOUS'] == False:
            arr = np.ascontiguousarray(arr)
        return arr
    return None


def array2img(arr: np.ndarray):
    arr =np.squeeze(np.clip(arr,0,255))
    if len(arr.shape) > 2:
        if arr.shape[0]==3 or (arr.shape[0]<arr.shape[1] and arr.shape[0]<arr.shape[2]):
            arr = arr[::-1]
            arr=arr.transpose([1,2,0])
    img = pil_image.fromarray(arr.astype(np.int8))
    return img

#調整明暗
def adjust_brightness(image,gamma):
    if gamma is None:
        gamma = np.random.choice(np.arange(0.5, 1.5, 0.1))
    ImageEnhance.Brightness(image).enhance(gamma)
    return image

#模糊
def adjust_blur(image):
    image=image.filter(ImageFilter.BLUR)
    return image

def adjust_saturation(img,saturation):
    if saturation is None:
        saturation = np.random.choice(np.arange(0.5, 2, 0.2))
    img = ImageEnhance.Color(img).enhance(saturation)
    return img
