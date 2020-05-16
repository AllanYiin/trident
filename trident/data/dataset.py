from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import itertools
import os
import pickle
import random
import string
import sys
import threading
import time
from enum import Enum, unique
from typing import List, TypeVar, Iterable, Tuple, Union

import numpy as np
from skimage import color

from trident.data.bbox_common import xywh2xyxy, xyxy2xywh
from trident.data.image_common import gray_scale, image2array, mask2array, image_backend_adaption, reverse_image_backend_adaption, \
    unnormalize, array2image, ExpectDataType, GetImageMode
from trident.data.label_common import label_backend_adaptive
from trident.data.mask_common import mask_backend_adaptive, color2label
from trident.data.samplers import *
from trident.backend.common import DataSpec, PrintException, OrderedDict,Signature
from trident.backend.load_backend import get_backend

try:
    import Queue
except ImportError:
    import queue as Queue

if get_backend() == 'pytorch':
    from trident.backend.pytorch_backend import to_numpy, to_tensor, ReplayBuffer
    import torch

__all__ = ['Dataset', 'ImageDataset', 'MaskDataset', 'LabelDataset', 'BboxDataset', 'MultipleDataset', 'Iterator',
           'NumpyDataset', 'RandomNoiseDataset']

T = TypeVar('T', int, float, str, np.ndarray)


class Dataset(List):
    def __init__(self, symbol=None, expect_data_type=None, name=''):
        super().__init__()
        self.parameter = None
        self.name = name
        self.symbol = symbol
        self.is_pair_process = False
        self.expect_data_type = expect_data_type

    def __add__(self, other):
        if other is not None and hasattr(other, '__iter__'):
            for item in other:
                if isinstance(item, (int, float, str, np.ndarray)):
                    super().append(item)

    def __len__(self):
        return super().__len__()



class MultipleDataset(List):
    def __init__(self, datasets,symbol=None, expect_data_type=None, name=''):
        super().__init__()
        self.parameter = None
        self.name = name
        self.symbol = symbol
        self.is_pair_process = False
        self.expect_data_type = expect_data_type
        for d in datasets:
            self.__add__(d)

    def __add__(self, other):
        if super().__len__()==0 and isinstance(other,Dataset):
            super().append(other)
        elif isinstance(other,Dataset):
            if len(super().__getitem__(0))==len(other):
                super().append(other)
            else:
                raise ValueError('the dataset you add does not have same length with the existing dataset')

    def add(self,other):
        self.__add__(other)

    def __getitem__(self, index: int):
        results=[]
        for i in range(super().__len__()):
            results.append(super().__getitem__(i).__getitem__(index) )
        return tuple(results)

    def __len__(self):
        return len(super().__getitem__(0))

class ImageDataset(Dataset):
    def __init__(self, images=None, expect_data_type: ExpectDataType = ExpectDataType.rgb,
                 get_image_mode: GetImageMode = GetImageMode.processed, symbol=None, name=''):
        super().__init__(symbol=symbol, expect_data_type=expect_data_type, name=name)
        self.__add__(images)
        self.dtype = np.float32
        self.get_image_mode = get_image_mode
        self.image_transform_funcs = []
        self.is_pair_process = False

    def __getitem__(self, index: int):
        img = super().__getitem__(index)  # self.pop(index)
        if isinstance(img, str) and self.get_image_mode == GetImageMode.path:
            self.parameter = DataSpec(self.name, self.symbol, "ImagePath", None)
            return img
        elif self.get_image_mode == GetImageMode.path:
            self.parameter = DataSpec(self.name, self.symbol, None, None)
            return None

        if isinstance(img, str):
            img = image2array(img)

        if self.get_image_mode == GetImageMode.raw:
            self.parameter = DataSpec(self.name, self.symbol, "ImagePath", None)
            return img
        if not isinstance(img, np.ndarray):
            raise ValueError('image data should be ndarray')
        elif isinstance(img, np.ndarray) and img.ndim not in [2, 3]:
            raise ValueError('image data dimension  should be 2 or 3, but get {0}'.format(img.ndim))
        elif self.expect_data_type == ExpectDataType.gray:
            img = color.rgb2gray(img).astype(self.dtype)
        elif self.expect_data_type == ExpectDataType.rgb and img.ndim == 2:
            img = np.repeat(np.expand_dims(img, -1), 3, -1).astype(self.dtype)
        elif self.expect_data_type == ExpectDataType.rgb and img.ndim == 3:
            img = img[:, :, :3].astype(self.dtype)
        elif self.expect_data_type == ExpectDataType.rgba:
            if img.ndim == 2:
                img = np.repeat(np.expand_dims(img, -1), 3, -1)
            if img.shape[2] == 3:
                img = np.concatenate([img, np.ones((img.shape[0], img.shape[1], 1)) * 255], axis=-1)
            img = img.astype(self.dtype)
        elif self.expect_data_type == ExpectDataType.multi_channel:
            img = img.astype(self.dtype)

        if self.get_image_mode == GetImageMode.expect and self.is_pair_process == False:
            return image_backend_adaption(img)
        elif self.get_image_mode == GetImageMode.processed and self.is_pair_process == False:
            return self.image_transform(img)
        elif self.is_pair_process == True:
            return img

        return None

    def image_transform(self, img_data):
        if len(self.image_transform_funcs) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self.image_transform_funcs:
                if not fc.__qualname__.startswith(
                        'random_') or 'crop' in fc.__qualname__ or 'rescale' in fc.__qualname__ or (
                        fc.__qualname__.startswith('random_') and random.randint(0, 10) % 2 == 0):
                    img_data = fc(img_data)
            img_data = image_backend_adaption(img_data)

            return img_data
        else:
            return img_data

    @property
    def reverse_image_transform_funcs(self):
        return_list = []
        return_list.append(reverse_image_backend_adaption)
        for i in range(len(self.image_transform_funcs)):
            fn = self.image_transform_funcs[-1 - i]
            if fn.__qualname__ == 'normalize.<locals>.img_op':
                return_list.append(unnormalize(fn.mean, fn.std))
        return_list.append(array2image)
        return return_list

    def reverse_image_transform(self, img_data):
        if len(self.reverse_image_transform_funcs) == 0:
            return reverse_image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_image_transform_funcs:
                img_data = fc(img_data)
            img_data = reverse_image_backend_adaption(img_data)

            return img_data
        else:
            return img_data


class MaskDataset(Dataset):
    def __init__(self, masks=None, class_names=None, expect_data_type: ExpectDataType = ExpectDataType.label_mask,
                 get_image_mode: GetImageMode = GetImageMode.processed, symbol=None, name=''):
        super().__init__(symbol=symbol, expect_data_type=expect_data_type, name=name)
        if expect_data_type not in [ExpectDataType.label_mask, ExpectDataType.binary_mask, ExpectDataType.alpha_mask,
                                    ExpectDataType.color_mask]:
            raise ValueError('Only mask is valid expect image type. ')

        self.__add__(masks)
        self.get_image_mode = get_image_mode
        self.palette = OrderedDict()
        self.mask_transform_funcs = []
        self.is_pair_process = False
        self.class_names = {}
        self._lab2idx = {}
        self._idx2lab = {}
        if class_names is not None:
            self.class_names = class_names

    def __getitem__(self, index: int):
        img = super().__getitem__(index)  # self.pop(index)
        if isinstance(img, str) and self.get_image_mode == GetImageMode.path:
            return img
        elif self.get_image_mode == GetImageMode.path:
            return None

        if isinstance(img, str):

            if self.expect_data_type == ExpectDataType.binary_mask:
                img = mask2array(img).astype(np.int64)
                mv = img.max()
                img[img == mv] = 255
                img[img <= 0] = 0
                img[img > 128] = 255
            elif self.expect_data_type == ExpectDataType.alpha_mask:
                img = mask2array(img).astype(np.float32)
            elif self.expect_data_type == ExpectDataType.label_mask:
                if '.png' in img:
                    img = image2array(img)
                    img = img[:, :, 1]
                else:
                    img = mask2array(img).astype(np.int64)

            elif self.expect_data_type == ExpectDataType.color_mask:
                img = image2array(img)
                if img.ndim == 2:
                    pass
                elif img.ndim == 3:
                    img = img[:, :, :3]
                    if len(self.palette) > 0:
                        img = color2label(img, self.palette).astype(np.int64)

        if self.get_image_mode == GetImageMode.raw:
            return img
        if not isinstance(img, np.ndarray):
            raise ValueError('image data should be ndarray')
        elif isinstance(img, np.ndarray) and img.ndim not in [2, 3]:
            raise ValueError('image data dimension  should be 2 or 3, but get {0}'.format(img.ndim))

        if self.get_image_mode == GetImageMode.expect and self.is_pair_process == False:
            return mask_backend_adaptive(img)
        elif self.get_image_mode == GetImageMode.processed and self.is_pair_process == False:
            return self.mask_transform(img)
        elif self.is_pair_process == True:
            return img

        return None

    def mask_transform(self, mask_data):
        if len(self.mask_transform_funcs) == 0:
            return mask_backend_adaptive(mask_data, label_mapping=self.class_names,
                                         expect_data_type=self.expect_data_type)
        if isinstance(mask_data, np.ndarray):
            for fc in self.mask_transform_funcs:
                if not fc.__qualname__.startswith(
                        'random_') or 'crop' in fc.__qualname__ or 'rescale' in fc.__qualname__ or (
                        fc.__qualname__.startswith('random_') and random.randint(0, 10) % 2 == 0):
                    mask_data = fc(mask_data)
            # mask_data = mask_backend_adaptive(mask_data)
            return mask_data
        else:
            return mask_data

    @property
    def reverse_image_transform_funcs(self):
        return_list = []
        return_list.append(reverse_image_backend_adaption)
        for i in range(len(self.image_transform_funcs)):
            fn = self.image_transform_funcs[-1 - i]
            if fn.__qualname__ == 'normalize.<locals>.img_op':
                return_list.append(unnormalize(fn.mean, fn.std))
        return_list.append(array2image)
        return return_list

    def reverse_image_transform(self, img_data):
        if len(self.reverse_image_transform_funcs) == 0:
            return reverse_image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_image_transform_funcs:
                img_data = fc(img_data)
            img_data = reverse_image_backend_adaption(img_data)

            return img_data
        else:
            return img_data


class LabelDataset(Dataset):
    def __init__(self, labels=None, expect_data_type=ExpectDataType.classification_label, class_names=None, symbol=None,
                 name=''):
        super().__init__(symbol=symbol, expect_data_type=expect_data_type, name=name)
        self.__add__(labels)
        self.dtype = np.int64
        self.class_names = {}
        self._lab2idx = {}
        self._idx2lab = {}
        if class_names is not None:
            self.class_names = class_names

        self.label_transform_funcs = []

    def binding_class_names(self, class_names=None, language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = 'en-us'
            self.class_names[language] = list(class_names)

            self.__default_language__ = language
            self._lab2idx = dict(zip(self.class_names[language], range(len(self.class_names[language]))))
            self._idx2lab = dict(zip(range(len(self.class_names[language])),self.class_names[language]))

    def __getitem__(self, index: int):
        label = super().__getitem__(index)
        return self.label_transform(label)

    def label_transform(self, label_data):
        label_data = label_backend_adaptive(label_data, self.class_names)
        if isinstance(label_data, list) and all(isinstance(elem, np.ndarray) for elem in label_data):
            label_data = np.asarray(label_data)
        if isinstance(label_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.label_transform_funcs:
                label_data = fc(label_data)
            return label_data
        else:
            return label_data


class BboxDataset(Dataset):
    def __init__(self, boxes=None, image_size=None, expect_data_type=ExpectDataType.absolute_bbox, class_names=None,
                 symbol=None, name=''):
        super().__init__(symbol=symbol, expect_data_type=expect_data_type, name=name)
        self.__add__(boxes)
        self.dtype = np.int64
        self.image_size = image_size
        self.class_names = {}
        self._lab2idx = {}
        self._idx2lab = {}
        if class_names is not None:
            self.class_names = class_names

    def binding_class_names(self, class_names=None, language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = 'en-us'
            self.class_names[language] = list(class_names)
            self.__default_language__ = language
            self._lab2idx = {v: k for k, v in enumerate(self.class_names[language])}
            self._idx2lab = {k: v for k, v in enumerate(self.class_names[language])}
            self._current_idx = -1

    def __getitem__(self, index: int):
        self._current_idx = index
        bboxes = super().__getitem__(index).astype(np.float32)
        if self.expect_data_type == ExpectDataType.relative_bbox and (self.image_size is None):
            raise RuntimeError('You need provide image size information for calculate relative_bbox. ')
        elif self.expect_data_type == ExpectDataType.relative_bbox:
            height, width = self.image_size
            bboxes[:, 0] = bboxes[:, 0] / width
            bboxes[:, 2] = bboxes[:, 2] / width
            bboxes[:, 1] = bboxes[:, 1] / height
            bboxes[:, 3] = bboxes[:, 3] / height
            return np.array(bboxes).astype(np.float32)

        elif self.expect_data_type == ExpectDataType.absolute_bbox:
            return np.array(bboxes).astype(np.float32)

    def bbox_transform(self, *bbox):

        return bbox




class NumpyDataset(Dataset):
    def __init__(self, data=None, expect_data_type=ExpectDataType.array_data, symbol=None, name=''):
        super().__init__(symbol=symbol, expect_data_type=expect_data_type, name=name)

        self.__add__(data)

        self.dtype = np.float32

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        return data


class RandomNoiseDataset(Dataset):
    def __init__(self, shape, expect_data_type=ExpectDataType.random_noise, random_mode='normal', symbol=None, name=''):
        super().__init__(symbol=symbol, expect_data_type=expect_data_type, name=name)

        self.dtype = np.float32
        self.shape = shape
        self.random_mode = random_mode

    def __getitem__(self, index: int):
        if self.random_mode == 'normal':
            return np.random.standard_normal(self.shape)
        elif self.random_mode == 'uniform':
            return np.random.uniform(-1, 1, self.shape)

    def __len__(self):
        return sys.maxsize


class Iterator(object):
    def __init__(self, data=None, label=None, mask=None, unpair=None, minibatch_size=8):
        self.is_pair_process = False
        self.signature = None
        self._data = NumpyDataset()
        self._label = LabelDataset()
        self._unpair = NumpyDataset()
        self.workers = 2
        self.itr = 0
        if data is not None and isinstance(data, (Dataset,MultipleDataset)):
            self._data = data
        if label is not None and isinstance(label, (Dataset,MultipleDataset)):
            self._label = label
            if isinstance(self._label, (MaskDataset, ImageDataset, BboxDataset,MultipleDataset)) and isinstance(self._data, ImageDataset) and len( self._label) == len(self._data):
                self._label.is_pair_process = self._data.is_pair_process = self.is_pair_process = True
            else:
                self._label.is_pair_process = self._data.is_pair_process = self.is_pair_process = False

        if unpair is not None and isinstance(unpair, Dataset):
            self._unpair = unpair

        self._minibatch_size = minibatch_size
        self.paired_transform_funcs = []
        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)
        self.buffer_size = 10
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        if self._label is not None and isinstance(self._label, (MaskDataset,BboxDataset,ImageDataset)) and isinstance(self._data, ImageDataset) and len(self._label) == len(self._data):
            self._label.is_pair_process = self._data.is_pair_process = self.is_pair_process = True
        else:
            self._label.is_pair_process = self._data.is_pair_process = self.is_pair_process = False

        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value
        if isinstance(self._label, (MaskDataset, ImageDataset, BboxDataset)) and isinstance(self._data, ImageDataset) and len(
            self._label) == len(self._data):
            self._label.is_pair_process = self._data.is_pair_process = self.is_pair_process = True
        else:
            self._label.is_pair_process = self._data.is_pair_process = self.is_pair_process = False

        self._sample_iter = iter(self.batch_sampler)

    @property
    def unpair(self):
        return self._unpair

    @unpair.setter
    def unpair(self, value):
        self._unpair = value
        self._sample_iter = iter(self.batch_sampler)

    @property
    def palette(self):
        if isinstance(self._label, MaskDataset) and self._label.expect_data_type in [ExpectDataType.label_mask,   ExpectDataType.color_mask]:
            return self._label.palette
        else:
            return None

    @property
    def minibatch_size(self):
        return self._minibatch_size

    @minibatch_size.setter
    def minibatch_size(self, value):
        self._minibatch_size = value
        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)

    def update_signature(self, arg_names):
        iterdata = self.next()
        if self.signature is None or not isinstance(self.signature,Signature):
            self.signature = Signature()
            self.signature.name = 'data_provider'
        if isinstance(arg_names, (list, tuple)) and len(iterdata)==len(arg_names):
            for i in range(len(arg_names)):
                arg = arg_names[i]
                data = iterdata[i]
                self.signature.outputs[arg] =(-1,)+ data.shape[1:] if data.ndim>1 else (-1)

        elif not isinstance(arg_names, (list, tuple)):
            raise ValueError('arg_names should be list or tuple')
        elif len(self.signature.key_list) != len(arg_names):
            raise ValueError('data feed and arg_names should be the same length')
        else:
            self.signature = None
            iterdata = self.next()



    def paired_transform(self, img_data, paired_img):

        if isinstance(img_data, list) and all(isinstance(elem, np.ndarray) for elem in img_data):
            img_data = np.asarray(img_data)
        if isinstance(img_data, str) and os.path.isfile(img_data) and os.path.exists(img_data):
            img_data = image2array(img_data)

        if len(self.paired_transform_funcs) == 0:
            return img_data, paired_img
        if isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.paired_transform_funcs:
                try:
                    img_data, paired_img = fc(img_data, paired_img)
                except:
                    PrintException()

            return img_data, paired_img
        else:
            return img_data, paired_img

    def __getitem__(self, index: int):
        # start = time.time()

        try:
            bbox = None
            mask = None
            data = self.data.__getitem__(index % len(self.data)) if len(self.data) > 0 else None
            # stop = time.time()
            # print('get data:{0}'.format(stop - start))
            # start=stop

            label = self.label.__getitem__(index % len(self.label)) if len(self.label) > 0 else None
            # stop = time.time()
            # print('get label:{0}'.format(stop - start))
            # start = stop
            if isinstance(self.label, BboxDataset):
                data, label = self.paired_transform(data, label)
                if hasattr(self.data, 'image_transform'):
                    data = self.data.image_transform(data)
                if hasattr(self.label, 'bbox_transform'):
                    new_label = self.label.bbox_transform(label)
                    if isinstance(new_label, tuple):
                        bbox, label = new_label
                    else:
                        bbox = new_label
                else:
                    bbox = label
            elif isinstance(self.label, MaskDataset):
                data, label = self.paired_transform(data, label)
                if hasattr(self.data, 'image_transform'):
                    data = self.data.image_transform(data)
                if hasattr(self.label, 'mask_transform'):
                    new_label = self.label.mask_transform(label)
                    if isinstance(new_label, tuple):
                        mask, label = new_label
                    else:
                        mask = new_label
                else:
                    mask = label.copy()
                label = None
            elif isinstance(self.label, ImageDataset):
                data, label = self.paired_transform(data, label)
                # stop = time.time()
                # print('paired_transform:{0}'.format(stop - start))
                # start = stop
                if hasattr(self.data, 'image_transform'):
                    data = self.data.image_transform( data)  # stop = time.time()  # print('data image_transform:{0}'.format(stop - start))  #
                    # start = stop
                if hasattr(self.label, 'image_transform'):
                    label = self.label.image_transform(label)  # stop = time.time()  # print('label image_transform:{0}'.format(stop - start))  #
                    # start = stop
            else:
                if hasattr(self.label, 'label_transform'):
                    label = self.label.label_transform(label)

            if hasattr(self.label, 'label_transform') and not isinstance(self.label, (BboxDataset, MaskDataset)):
                label = self.label.label_transform(label)

            unpair = self.unpair.__getitem__(index % len(self.unpair)) if len(self.unpair) > 0 else None

            return_data = []
            if self.signature is None or len(self.signature) == 0:
                self.signature=Signature()
                self.signature.name='data_provider'
                if data is not None:
                    self.signature.outputs['data' if self.data.symbol is None or len(self.data.symbol) == 0 else self.data.symbol] = (-1,)+data.shape
                if bbox is not None:
                    self.signature.outputs['bbox' if self.label.symbol is None or len(self.label.symbol) == 0 else self.label.symbol] =(-1,)+ bbox.shape
                if mask is not None:
                    self.signature.outputs['mask' if self.label.symbol is None or len(self.label.symbol) == 0 else self.label.symbol] = (-1,)+mask.shape
                if label is not None:
                    self.signature.outputs['label' if self.label.symbol is None or len(self.label.symbol) == 0 or self.label.symbol in self.signature else self.label.symbol] =(-1,)+ label.shape if isinstance(label, np.ndarray) else (-1,)
                if unpair is not None:
                    self.signature.outputs['unpair' if self.unpair.symbol is None or len(self.unpair.symbol) == 0 else self.unpair.symbol] = (-1,)+unpair.shape  # stop = time.time()  #
                    # print('signature:{0}'.format(stop - start))  # start = stop

            if data is not None:
                return_data.append(data)
            if bbox is not None:
                return_data.append(bbox)
            if mask is not None:
                return_data.append(mask)
            if label is not None:
                return_data.append(label)
            if unpair is not None:
                return_data.append(unpair)
            # stop = time.time()
            # print('prepare tuple:{0}'.format(stop - start))
            # start = stop
            return tuple(return_data)
        except:
            PrintException()

    def _next_index(self):
        return next(self._sample_iter)

    def __iter__(self):
        return self._sample_iter

    # return a batch , do minimal fetch before return
    def next(self):
        if self.out_queue.qsize() == 0:
            in_data = self._sample_iter.__next__()
            self.out_queue.put(in_data, False)

        out_data = self.out_queue.get(False)

        if self.out_queue.qsize() <= self.buffer_size // 2:
            for i in range(2):
                in_data = self._sample_iter.__next__()
                self.out_queue.put(in_data, False)

        return out_data

    # yield a batch , and trigger following fetch after yield
    def __next__(self):
        if self.out_queue.qsize() == 0:
            in_data = self._sample_iter.__next__()
            self.out_queue.put(in_data, False)

        out_data = self.out_queue.get(False)

        yield out_data
        if self.out_queue.qsize() <= self.buffer_size // 2:
            for i in range(self.buffer_size - self.out_queue.qsize()):
                in_data = self._sample_iter.__next__()
                self.out_queue.put(in_data, False)

    def __len__(self):
        return max([len(self.data) if self.data is not None else 0, len(self.unpair) if self.unpair is not None else 0])
