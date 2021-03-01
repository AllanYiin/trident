from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import re
from itertools import repeat

import numpy as np
import six
from scipy import misc, ndimage
from skimage import color
from skimage import exposure
from skimage import morphology
from skimage import transform, exposure
from skimage.filters import *

from trident.backend.common import *
from trident.backend.tensorspec import *


if get_image_backend()=='opencv':
    from trident.backend.opencv_backend import *
else:
    from trident.backend.pillow_backend import *

from trident.backend.common import get_backend
from trident.data.label_common import check_is_onehot,get_onehot


__all__ = ['mask2trimap','color2label','label2color','mask_backend_adaptive']




def mask2trimap(kernal_range=(7,15)):
    def img_op(mask: np.ndarray):
        trimap=mask.copy().astype(np.float32)
        trimap[trimap>0]=255.0
        k_size = random.choice(range(*kernal_range))
        trimap[np.where((ndimage.grey_dilation(mask[:,:],size=(k_size,k_size)) - ndimage.grey_erosion(mask[:,:],size=(k_size,k_size)))!=0)] = 128
        return trimap
    return img_op

def alpha2triplet(kernal_range=(7,15)):
    def img_op(mask: np.ndarray):
        alpha=mask.copy() #alpha

        trimap=mask.copy().astype(np.float32)
        trimap[trimap>0]=255.0
        k_size = random.choice(range(*kernal_range))
        trimap[np.where((ndimage.grey_dilation(mask[:,:],size=(k_size,k_size)) - ndimage.grey_erosion(mask[:,:],size=(k_size,k_size)))!=0)] = 128

        binary_mask=mask.copy()


        return trimap
    return img_op


def color2label2(color_label,palette):
    num_classes=len(palette)
    if color_label.ndim==3:
        label_mask= np.zeros((color_label.shape[0], color_label.shape[1])).astype(np.int64)
        if isinstance(palette,list) and len(palette[0])==3:
            pass
        elif isinstance(palette,OrderedDict) and len(palette.value_list[0])==3:
            palette=palette.value_list
        available_pixel=OrderedDict()
        available_pixel['r']=[]
        available_pixel['g'] = []
        available_pixel['b'] = []
        for p in palette:
            available_pixel['r'].append(p[0])
            available_pixel['g'].append(p[1])
            available_pixel['b'].append(p[2])
        available_pixel['r']=np.array(list(sorted(set(available_pixel['r']))))
        available_pixel['g'] = np.array(list(sorted(set(available_pixel['g']))))
        available_pixel['b'] = np.array(list(sorted(set(available_pixel['b']))))
        def find_closest(n,available_pixel_array):
            if n in available_pixel_array:
                return n
            else:
                return available_pixel_array[np.argmin(np.abs(available_pixel_array-n))]

        for j in range(label_mask.shape[-2]):
            for k in range(label_mask.shape[-1]):
                color = tuple(color_label[j, k,:].tolist())
                if color not in palette:
                    color=tuple([find_closest(color[0],available_pixel['r']),find_closest(color[1],available_pixel['g']),find_closest(color[2],available_pixel['b'])])
                if color in palette:
                    try:
                        label_mask[ j, k] = palette.index(color)
                    except Exception as e:
                        print(e)
                        print(color)
                        label_mask[ j, k] = 0
                else:
                    print('color {0} is not in the palette '.format(color))
                    label_mask[j, k] = 0
        return label_mask
    elif color_label.ndim==4:
        results=[]
        for m in range(color_label.shape[0]):
            results.append(color2label(color_label[m],palette))

        return np.array(results)
    else:
        return color_label

def color2label(color_label,palette):
    num_classes=len(palette)
    if color_label.ndim==3:
        color_label=np.round(color_label/32).astype(np.int64)*32
        label_mask= np.zeros((color_label.shape[0], color_label.shape[1])).astype(np.int64)
        if isinstance(palette,list) and len(palette[0])==3:
            pass
        elif isinstance(palette,OrderedDict) and len(palette.value_list[0])==3:
            palette=palette.value_list
        palette=[p[0]*256*256+p[1]*256+p[2] for p in palette]


        color_label=color_label[:,:,0]*256*256+color_label[:,:,1]*256+color_label[:,:,2]
        for i in range(num_classes):
            label_mask[color_label == palette[i]] = i
        return label_mask.astype(np.int64)
    elif color_label.ndim==4:
        results=[]
        for m in range(color_label.shape[0]):
            results.append(color2label(color_label[m],palette))

        return np.array(results)
    else:
        return color_label


def label2color(label_mask,palette):
    num_classes=len(palette)
    color_label= np.zeros((*label_mask.shape,3)).astype(np.int64)
    if isinstance(palette,list) and len(palette[0])==3:
        pass
    elif isinstance(palette,OrderedDict) and len(palette.value_list[0])==3:
        palette=palette.value_list

    for i in range(num_classes):
        color_label[label_mask==i]=palette[i]

    return color_label


def mask_backend_adaptive(mask, label_mapping=None, object_type=None):
    if get_backend() == 'pytorch':
        if mask is None:
            return None
        elif isinstance(mask, np.ndarray):
            # binary mask
            if object_type == ObjectType.binary_mask:
                if mask.ndim==2 :
                    mask[mask > 0] = 1
                    return mask.astype(np.int64)
                elif mask.ndim==3 and mask.shape[-1] in [1, 2]:
                    if mask.shape[-1] ==2:
                        mask= mask[:, :, 1]
                    elif mask.shape[-1] ==1:
                        mask = mask[:, :, 0]
                    mask[mask > 0] = 1
                    return mask.astype(np.int64)
            elif object_type == ObjectType.label_mask or object_type == ObjectType.color_mask:
                if mask.ndim==2 :
                    return mask.astype(np.int64)
                if mask.ndim == 3 and mask.shape[-1] >2:
                    if check_is_onehot(mask):
                        mask=np.argmax(mask, -1).astype(np.int64)
                        return mask
                return mask.astype(np.int64)
            elif object_type == ObjectType.alpha_mask:
                if mask.ndim==2 :
                    mask=mask/255.0
                    return mask.astype(np.float32)
                if mask.ndim == 3:
                    mask=color.rgb2gray(mask.astype(np.float32))
                if mask.max()>1:
                    mask = mask /mask.max()
                return mask.astype(np.float32)
        else:
            return mask
    elif get_backend() == 'tensorflow':
        if mask is None:
            return None
        elif isinstance(mask, np.ndarray):
            # binary mask
            if object_type == ObjectType.binary_mask:
                if mask.ndim==2 :
                    mask[mask > 0] = 1
                    return mask.astype(np.int64)
                elif mask.ndim==3 and mask.shape[-1] in [1, 2]:
                    if mask.shape[-1] ==2:
                        mask= mask[:, :, 1]
                    elif mask.shape[-1] ==1:
                        mask = mask[:, :, 0]
                    mask[mask > 0] = 1
                    return mask.astype(np.int64)
            elif object_type == ObjectType.label_mask or object_type == ObjectType.color_mask:
                if mask.ndim==2 and label_mapping is not None and len(label_mapping)>0:
                    return to_onehot(mask,len(label_mapping))
                if mask.ndim == 3 and mask.shape[-1] >2:
                    return mask
                return mask
            elif object_type == ObjectType.alpha_mask:
                if mask.ndim==2 :
                    mask=mask/255.0
                    return mask.astype(np.float32)
                if mask.ndim == 3:
                    mask=color.rgb2gray(mask.astype(np.float32))
                if mask.max()>1:
                    mask = mask /mask.max()
                return mask.astype(np.float32)
        else:
            return mask


