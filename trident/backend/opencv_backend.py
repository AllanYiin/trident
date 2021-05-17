from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import time
import urllib


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

import PIL

try:
    from PIL import ImageEnhance
    from PIL import ImageOps
    from PIL import ImageFilter
    from PIL import Image as pil_image
    from PIL.PngImagePlugin import PngImageFile
except ImportError:
    pil_image = None
    ImageEnhance = None
    ImageFilter=None
import cv2
import numpy as np

from trident.backend.common import get_session,floatx



__all__ = ['read_image','read_mask','save_image','save_mask','image2array','array2image','mask2array','array2mask','adjust_brightness','adjust_blur','adjust_saturation']


_session=get_session()
_backend=_session.backend
_image_backend=_session.image_backend

version=cv2.__version__
sys.stderr.write('Opencv version:{0}.\n'.format(version))


def read_image(im_path:str):
    """
    Read pillow image from the image file path

    Args:
        im_path (str):

    Returns:
        pillow image

    """
    try:
        if os.path.exists(im_path) and im_path.split('.')[-1] in ('jpg','jepg','png','bmp','tiff'):
            img=pil_image.open(im_path)
            img=np.array(img).astype(np.float32)
            if img.max()<=1:
                img=img*255.0
            return img
        else:
            if not os.path.exists(im_path):
                raise ValueError('{0} not exsit'.format(im_path))
            else:
                raise ValueError('extension {0} not support (jpg, jepg, png, bmp, tiff)'.format(im_path.split('.')[-1]))
    except :
        try:
            img = plt.imread(im_path)
            return img[::-1].astype(np.float32)
        except Exception as e:
            print(e)
            return None

def read_mask(im_path:str):
    """
    Read pillow image as mask from the image file path
    The result is the same as  pillow_image.convert('L')

    Args:
        im_path (str):

    Returns:
        pillow image

    """
    try:
        if os.path.exists(im_path) and im_path.split('.')[-1] in ('jpg','jepg','png','bmp','tiff'):
            img=pil_image.open(im_path).convert('L')
            return img
        else:
            if not os.path.exists(im_path):
                raise ValueError('{0} not exsit'.format(im_path))
            else:
                raise ValueError('extension {0} not support (jpg, jepg, png, bmp, tiff)'.format(im_path.split('.')[-1]))
    except Exception as e:
        print(e)
        return None

def save_image(arr,file_path):
    """
    Saving a ndarray/ pillow image as image file

    Args:
        arr (tensor, ndarray, pillow image):
        file_path ():

    Returns:

    """

    img=array2image(arr)
    img.save(file_path)

def save_mask(arr,file_path):
    img=array2mask(arr)
    img.save(file_path)





def image2array(img):
    """

    Args:
        img (string, pillow image or numpy.ndarray): Image to be converted to ndarray.

    Returns:
        ndarray  (HWC / RGB)
    """
    arr = None
    if isinstance(img, str):
        arr=file2array(img)
    elif isinstance(img, PngImageFile):
        arr = np.array(img.im).astype(np.float32)
    elif isinstance(img, pil_image.Image):
        arr = np.array(img).astype(np.float32)
    if isinstance(img, np.ndarray):
        arr = img.astype(_session.floatx)
        if arr.ndim not in [2, 3]:
            raise ValueError('image should be 2 or 3 dimensional. Got {} dimensions.'.format(arr.ndim))
        if arr.ndim == 3:
            if arr.shape[2] in [3, 4] and arr.shape[0] not in [3, 4]:
                pass
            elif arr.shape[0] in [1, 3, 4]:
                arr = arr.transpose([1, 2, 0])
            else:
                raise ValueError('3d image should be 1, 3 or 4 channel. Got {} channel.'.format(arr.shape[0]))

    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    return arr.astype(np.float32)


def array2image(arr:np.ndarray):
    """
    Args
        arr  (ndarry)  : array need to convert back to image

    Returns
        pillow image


    """
    # confirm back to numpy
    arr=np.clip(np.squeeze(arr),0,255)
    if arr.ndim not in [2, 3]:
        raise ValueError('image should be 2 or 3 dimensional. Got {} dimensions.'.format(arr.ndim))
    mode = None

    if arr.ndim == 2:
        mode = 'L'
    elif arr.ndim == 3:
        if (_backend=='tensorflow' and  arr.shape[2] in [3, 4]) or (arr.shape[2] in [3, 4] and arr.shape[0] not in [3, 4]):
            pass
        elif (_backend!='tensorflow' and  arr.shape[0] in [3, 4] and arr.shape[2] not in [3, 4]):
            arr = arr.transpose([1, 2, 0])
        elif _backend in ['pytorch', 'cntk'] and arr.ndim == 3 and arr.shape[0] in [3, 4] and arr.shape[2] not in [3, 4]:
            arr = arr.transpose([1, 2, 0])
        else:
            raise ValueError('3d image should be 3 or 4 channel. Got {} channel.'.format(arr.shape[0]))
        if arr.shape[2] == 3:
            mode = 'RGB'
        elif arr.shape[2] == 4:
            mode = 'RGBA'

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = pil_image.fromarray(arr, mode)
    return img

def mask2array(img):
    """

    Args
        img  (string), pillow image or numpy.ndarray): Image to be converted to ndarray.

    Returns
        ndarray  (HW / single channel)


    """
    if isinstance(img,str):
        if os.path.exists(img) and img.split('.')[-1] in ('jpg','jepg','png','bmp','tiff'):
            img=pil_image.open(img).convert('L')
        else:
            return None
    arr=None
    if isinstance(img,pil_image.Image):
        arr = np.array(img).astype(_session.floatx)
    elif isinstance(img, np.ndarray):
        if arr.ndim not in [2, 3]:
            raise ValueError('image should be 2 or 3 dimensional. Got {} dimensions.'.format(arr.ndim))
        if arr.ndim == 3:
            if arr.shape[3] in [3, 4] and arr.shape[0] not in [3, 4]:
                pass
            elif arr.shape[0] in [3, 4]:
                arr = arr.transpose([1, 2, 0])
            else:
                raise ValueError('3d image should be 3 or 4 channel. Got {} channel.'.format(arr.shape[0]))
        arr=img.astype(_session.floatx)
    if arr.flags['C_CONTIGUOUS'] == False:
        arr = np.ascontiguousarray(arr)
    return arr

def array2mask(arr:np.ndarray):
    """

    Args
        arr  ndarry  : array need to convert back to image

    Returns
        pillow image


    """
    # confirm back to numpy
    arr=np.squeeze(arr)
    if arr.ndim not in [2, 3]:
        raise ValueError('image should be 2 or 3 dimensional. Got {} dimensions.'.format(arr.ndim))
    mode = None
    if arr.max()==1:
        arr=arr*255
    if arr.ndim == 2:
        #arr = np.expand_dims(arr, 2)
        mode = 'L'
    elif arr.ndim == 3:
        if arr.shape[3] in [3, 4] and arr.shape[0] not in [3, 4]:
            pass
        elif arr.shape[0] in [3, 4]:
            arr = arr.transpose([1, 2, 0])
        else:
            raise ValueError('3d image should be 3 or 4 channel. Got {} channel.'.format(arr.shape[0]))
        if arr.shape[3] == 3:
            mode = 'RGB'
        elif arr.shape[3] == 4:
            mode = 'RGBA'

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = pil_image.fromarray(arr, mode)
    return img



def file2array(img_path,flag = cv2.IMREAD_COLOR):
    """

    Args:
        flag ():
        img_path ():

    Returns:

    Examples:
        >>> arr=file2array('../../trident.png')


    """

    with open(img_path, "rb") as f:
        img_np = np.frombuffer(f.read(), np.uint8)

        img = cv2.imdecode(img_np, flag)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB,dst= img)
        return img.astype(np.float32)

def file2array1(img_path):
    """

    Args:
        img_path ():

    Returns:

    Examples:
        >>> arr=file2array('../../trident.png')


    """
    arr = mpimg.imread(img_path)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = arr  # [::-1]
    if arr.max() <= 1 and arr.dtype != np.uint8:
        arr *= 255.0
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')
    plt.close()
    return arr

def file2array2(img_path):
    """

    Args:
        img_path ():

    Returns:

    Examples:
        >>> arr=file2array('../../trident.png')


    """
    img = pil_image.open(img_path)
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] in [3, 4] and arr.shape[0] not in [3, 4]:
            pass
        elif arr.shape[0] in [1, 3, 4]:
            arr = arr.transpose([1, 2, 0])
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr.astype(np.float32)




def resize(image,size):
    return  cv2.resize(image,size,cv2.INTER_AREA)



#調整明暗
def adjust_brightness(image,gamma):
    if gamma is None:
        gamma = np.random.choice(np.arange(0.5, 1.5, 0.1))
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    cv2.LUT(image.astype(np.uint8), table)
    return image

#模糊
def adjust_blur(image):
    image=cv2.blur(image, (3, 3))
    return image

def adjust_saturation(img,saturation):
    imghsv = cv2.cvtColor(img.copy().astype(np.float32), cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(imghsv)
    if saturation is None:
        saturation = np.random.choice(np.arange(0.5, 2, 0.1))
    s = s * saturation
    s = np.clip(s, 0, 255)

    hue = np.random.choice(np.arange(-20, 20, 1))
    h = h + hue
    h = np.clip(h, 0, 255)

    imghsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(imghsv.astype(np.float32), cv2.COLOR_HSV2BGR)
    return img



# # 把現有的圖片隨機擾亂
# coverage = 110
# rotation_range = 30
# zoom_range = 0.01
# shift_range = 0.02
# random_flip = 0.2
#
#
# # 對圖片進行平移(微量)、縮放(微量)、旋轉(微量)、翻轉等操作
# def random_transform(img, mask, rotation_range=rotation_range, zoom_range=zoom_range, shift_range=shift_range, random_flip=random_flip):
#     h, w = img.shape[0:2]
#     rotation = np.random.uniform(-rotation_range, rotation_range)
#     scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
#     tx = np.random.uniform(-shift_range, shift_range) * w
#     ty = np.random.uniform(-shift_range, shift_range) * h
#     mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
#     mat[:, 2] += (tx, ty)
#     new_img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_CONSTANT,
#                              borderValue=0)  # , borderMode=cv2.BORDER_REPLICATE
#     new_mask = cv2.warpAffine(mask, mat, (w, h), borderMode=cv2.BORDER_CONSTANT,
#                               borderValue=0)  # , borderMode=cv2.BORDER_REPLICATE
#     if np.random.random() < random_flip:
#         new_img = new_img[:, ::-1]
#         new_mask = new_mask[:, ::-1]
#     return new_img, new_mask
#

def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask  = mask[:,::-1]
    return image, mask

def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask  = mask[::-1,:]
    return image, mask

def random_salt_pepper_line(image, noise =0.0005, length=10):
    height,width = image.shape[:2]
    num_salt = int(noise*width*height)

    # Salt mode
    y0x0 = np.array([np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]).T
    y1x1 = y0x0 + np.random.choice(2*length, size=(num_salt,2))-length
    for (y0,x0), (y1,x1)  in zip(y0x0,y1x1):
        cv2.line(image,(x0,y0),(x1,y1), (255,255,255), 1)

    # Pepper mode
    y0x0 = np.array([np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]).T
    y1x1 = y0x0 + np.random.choice(2*length, size=(num_salt,2))-length
    for (y0,x0), (y1,x1)  in zip(y0x0,y1x1):
        cv2.line(image,(x0,y0),(x1,y1), (0,0,0), 1)

    return image

def random_salt_pepper_noise(image, noise =0.0005):
    height,width = image.shape[:2]
    num_salt = int(noise*width*height)

    # Salt mode
    yx = [np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
    image[tuple(yx)] = [255,255,255]

    # Pepper mode
    yx = [np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
    image[tuple(yx)] = [0,0,0]

    return image
