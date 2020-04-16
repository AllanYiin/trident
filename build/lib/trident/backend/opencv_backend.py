from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import cv2
import numpy as np

from .common import get_session,floatx

_session=get_session()
_backend=_session.backend
_image_backend=_session.image_backend

version=cv2.__version__
sys.stderr.write('Opencv version:{0}.\n'.format(version))

def read_image(im_path:str):
    if os.path.exists(im_path):
        img=cv2.imread(im_path)
        return img
    return None
def read_mask(im_path:str):
    if os.path.exists(im_path):
        img=cv2.imread(im_path,2)
        return img
    return None

def image2array(img):
    if isinstance(img,str):
        if os.path.exists(img):
            img=cv2.imread(img)
        else:
            return None
    if isinstance(img,np.ndarray):
        if   _backend=='tensorflow' and len(img.shape)>2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img=img.transpose([2,0,1])
        img=img.astype(floatx())
        if img.flags['C_CONTIGUOUS'] == False:
            img = np.ascontiguousarray(img)

    return img
def mask2array(img):
    if isinstance(img,str):
        if os.path.exists(img):
            img=cv2.imread(img,2)
        else:
            return None
    if isinstance(img,np.ndarray):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.squeeze(img).astype(floatx())
        if img.flags['C_CONTIGUOUS'] == False:
            img = np.ascontiguousarray(img)
    img[img > 0] = 1
    return img

def array2img(arr: np.ndarray):
    arr =np.squeeze(np.clip(arr,0,255))
    if len(arr.shape) > 2:
        if arr.shape[0]==3 or (arr.shape[0]<arr.shape[1] and arr.shape[0]<arr.shape[2]):
            arr=arr.transpose([1,2,0])
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr.astype(np.uint8)


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
