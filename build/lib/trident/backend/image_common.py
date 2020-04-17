from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import os
import random
import re
import numpy as np
import math

from skimage import  transform,exposure
from .common import *
from itertools import repeat

__all__ = ['read_image','read_mask','save_image','save_mask','image2array','array2image','mask2array','array2mask','list_pictures','normalize','unnormalize','random_crop','resize','add_noise','backend_adaptive','random_channel_shift','random_cutout']


_session=get_session()
_backend=_session.backend
_image_backend=_session.image_backend

if _image_backend=='opencv':
    from  .opencv_backend import *
else:
    from .pillow_backend import *


read_image=read_image
read_mask=read_mask
save_image=save_image
save_mask=save_mask
image2array=image2array
array2image=array2image
mask2array=mask2array
array2mask=array2mask



def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]

def check_same_size(*images):
    result=True
    height, width = images[0].shape[:2]
    #check same isze
    for img in images:
        hh, ww = images[0].shape[:2]
        if hh==height and ww==width:
            pass
        else:
            result=False
    return True


def random_augmentation(func):
    def wrapper(prob=0,*args, **kwargs):
        if random.random()<=prob:
            return func(*args, **kwargs)

    return wrapper








def add_noise(intensity=0.1):
    def img_op(image:np.ndarray):
        rr=random.randint(0,10)
        noise = np.random.standard_normal(image.shape) *(intensity*image.max())
        if rr % 2 == 0:
            noise = np.random.uniform(-1,1,image.shape) *(intensity*image.max())
        image=np.clip(image+noise,image.min(),image.max())
        return image
    return img_op

def normalize(mean,std):
    def img_op(image:np.ndarray):
        norm_mean = mean
        norm_std = std
        if isinstance(norm_mean,(float,int)) and image.ndim==3:
            norm_mean=np.array([norm_mean,norm_mean,norm_mean])
            norm_mean=np.expand_dims(norm_mean,-2)
            norm_mean = np.expand_dims(norm_mean, -2)
        if isinstance(norm_std, (float, int)) and image.ndim==3:
            norm_std = np.array([norm_std,norm_std,norm_std])
            norm_std = np.expand_dims(norm_std, -2)
            norm_std = np.expand_dims(norm_std, -2)
        if  image.ndim==3:
            return (image-norm_mean)/norm_std
        elif image.ndim==2:
            if isinstance(norm_mean, (float, int)) and isinstance(norm_std, (float, int)):
                return (image - norm_mean) / norm_std
        return image
    return img_op

def unnormalize(mean,std):
    def img_op(image:np.ndarray):
        norm_mean = mean
        norm_std = std
        if isinstance(norm_mean,tuple):
            norm_mean=list(norm_mean)

        if isinstance(norm_std,tuple):
            norm_std=list(norm_std)

        if isinstance(norm_mean,(float,int))  and isinstance(norm_std, (float, int)) and image.ndim==3:
            return image *float(norm_std)+ float(norm_mean)
        elif  isinstance(norm_mean,list) and isinstance(norm_std,list) and len(norm_mean)==1 and len(norm_std)==1:
            return image *float(norm_std[0])+float(norm_mean[0])
        elif  isinstance(norm_mean,list) and isinstance(norm_std,list) and len(norm_mean)==3 and len(norm_std)==3:
            norm_mean = np.reshape(np.array(norm_mean),(1,1,3))
            norm_std = np.reshape(np.array(norm_std),(1,1,3))
            return image *float(norm_std)+ float(norm_mean)
        return image
    return img_op



def resize(size):
    def img_op(image:np.ndarray):
        return  transform.resize(image, size,anti_aliasing=True)
    return img_op


def invert_color():
    def img_op(image:np.ndarray):
        if np.min(image)>=0 and np.max(image)<=1:
            return 1 - image
        elif np.min(image)>=-1 and np.min(image)<0 and np.max(image)<=1:
            return 1-(image*0.5+0.5)
        else:
            return 255-image
    return img_op

def gray_scale():
    def img_op(image:np.ndarray):
        if image.shape[0]==3:
            image=image.transpose([1,2,0])
        return 0.2125*image[:,:,0:] +0.7154*image[:,:,1:] +0.0721*image[:,:,2:]
    return img_op


def adjust_gamma(gamma=1.2):
    def img_op(image:np.ndarray):
        return exposure.adjust_gamma(image, 2)
    return img_op

def random_crop(w, h):
    def img_op(image:np.ndarray):
        result=np.zeros((h,w,image.shape[-1]))
        height, width = image.shape[:2]
        offset_x,offset_y=0,0
        if width>w:
            offset_x = np.random.choice(width-w)
        if height>h:
            offset_y = np.random.choice(height-h)
        crop_im=image[offset_y:offset_y+h,offset_x:offset_x+w,:]
        result[:crop_im.shape[0],:crop_im.shape[1],:]=crop_im
        return result
    return img_op


def backend_adaptive(image):
    if _session.backend=='tensorflow' and image.ndim ==3:
        image = image.astype(np.float32)
    elif _session.backend in ['pytorch','cntk'] and image.ndim ==3:
        image=np.transpose(image,[2,0,1]).astype(np.float32)
    elif isinstance(image,np.ndarray):
        return image.astype(np.float32)
    elif isinstance(image, list):
        return np.array(image).astype(np.float32)
    return image



def random_channel_shift(x, intensity=15., channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    intensity = max_x / 255 * intensity
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_cutout(img, mask ):
    h, w = img.shape[:2] if _backend=='tensorflow' or len(img.shape)==2 else img.shape[1:3]
    cutx=random.choice(range(0,w//4))
    cuty = random.choice(range(0, h//4))
    offsetx=random.choice(range(0,w))
    offsety=random.choice(range(0,h))
    block=np.zeros((min(offsety+cuty,h)-offsety,min(offsetx+cutx,w)-offsetx))
    if random.randint(0, 10) % 4 == 1:
        block = np.clip(np.random.standard_normal((min(offsety+cuty,h)-offsety,min(offsetx+cutx,w)-offsetx)) * 127.5 + 127.5, 0,255)
    elif random.randint(0, 10) % 4 == 2:
        block = np.random.uniform(0,255,(min(offsety + cuty, h) - offsety, min(offsetx + cutx, w) - offsetx))
    if _backend == 'tensorflow':
        block=np.expand_dims(block,-1)
        block = np.concatenate([block, block, block], axis=-1)
        img[offsety:min(offsety + cuty, img.shape[0]), offsetx:min(offsetx + cutx, img.shape[1]), :] = block
        mask[offsety:min(offsety + cuty, mask.shape[0]), offsetx:min(offsetx + cutx, mask.shape[1])] = 0
    else:
        block = np.expand_dims(0,block)
        block = np.concatenate([block, block, block], axis=0)
        img[:,offsety:min(offsety + cuty, img.shape[0]), offsetx:min(offsetx + cutx, img.shape[1])] = block
        mask[offsety:min(offsety + cuty, mask.shape[0]), offsetx:min(offsetx + cutx, mask.shape[1])] = 0
    return img,mask







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
# def random_contast(image):
#     beta=0
#     alpha=random.uniform(0.5, 2.0)
#     image = image.astype(np.float32) * alpha + beta
#     image = np.clip(image,0,255).astype(np.uint8)
#     return image
#
#







