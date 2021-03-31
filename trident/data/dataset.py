from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod
import warnings
import collections
import copy
import hashlib
import numbers
import inspect
import itertools
import builtins
import os
import pickle
import random
import string
import sys
import threading
import time
from enum import Enum, unique
from typing import List, TypeVar, Tuple, Union, Optional, Generic, Iterable, Iterator, Sequence, Dict

import cv2
import numpy as np
from skimage import color

from trident.backend.opencv_backend import file2array
from trident.data.vision_transforms import Unnormalize

from trident.data.transform import Transform

from trident.data.bbox_common import xywh2xyxy, xyxy2xywh
from trident.data.image_common import gray_scale, image2array, mask2array, image_backend_adaption, reverse_image_backend_adaption, \
    unnormalize, array2image, GetImageMode

from trident.backend import iteration_tools
from trident.data.label_common import label_backend_adaptive
from trident.data.mask_common import mask_backend_adaptive, color2label
from trident.data.text_common import text_backend_adaption, reverse_text_backend_adaption
from trident.data.samplers import *
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec, assert_input_compatibility

try:
    import Queue
except ImportError:
    import queue as Queue

if get_backend() == 'pytorch':
    from trident.backend.pytorch_backend import to_numpy, to_tensor, ObjectType
    from trident.backend.pytorch_ops import int_shape, str2dtype, tensor_to_shape, expand_dims, cast
    import torch
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_backend import to_numpy, to_tensor, ObjectType
    from trident.backend.tensorflow_ops import int_shape, str2dtype, tensor_to_shape, expand_dims, cast

__all__ = ['Dataset', 'ZipDataset', 'ImageDataset', 'MaskDataset', 'TextSequenceDataset', 'LabelDataset', 'BboxDataset', 'LandmarkDataset', 'Iterator', 'MetricIterator',
           'NumpyDataset', 'RandomNoiseDataset']

_UID_PREFIX = collections.defaultdict(int)


def _get_global_uid(prefix=''):
    if prefix in _UID_PREFIX:
        _UID_PREFIX[prefix] += 1
        return _UID_PREFIX[prefix]
    else:
        return _UID_PREFIX[prefix]


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class DatasetBase(ABC):
    r"""
    An abstract class for all datasets.
    __getitem__ and __len__ method are aditionally needed.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class Dataset(DatasetBase):
    def __init__(self, args, symbol=None, object_type: Optional[ObjectType] = None, **kwargs):
        super().__init__()
        if args is None:
            self.items = []
        else:
            self.items = args
        self._element_spec = None

        if symbol == None:
            prefix = camel2snake(symbol) if symbol is not None else camel2snake(self.__class__.__name__.replace("Dataset", ""))
            uid = _get_global_uid(camel2snake(prefix))
            if uid == 0:
                self.symbol = prefix
            else:
                self.symbol = camel2snake(prefix) + '_' + str(uid)
        self.symbol = symbol
        self.is_spatial = False
        self.is_paired_process = False
        self.object_type = kwargs.get("expect_data_type", object_type)
        self.transform_funcs = []

    def __getitem__(self, index: int) -> Tuple:
        return self.items[index]

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name=='symbol' and isinstance(self._element_spec,TensorSpec):
            self._element_spec._name=value


    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def range(*args, **kwargs):
        """Creates a `Dataset` of a step-separated range of values.
           >>> list(Dataset.range(5))
           [0, 1, 2, 3, 4]
           >>> list(Dataset.range(2, 5))
           [2, 3, 4]
           >>> list(Dataset.range(1, 5, 2))
           [1, 3]
           >>> list(Dataset.range(1, 5, -2))
           []
           >>> list(Dataset.range(5, 1))
           []
           >>> list(Dataset.range(5, 1, -2))
           [5, 3]

           Args:
             *args: follows the same semantics as python's xrange.
               len(args) == 1 -> start = 0, stop = args[0], step = 1.
               len(args) == 2 -> start = args[0], stop = args[1], step = 1.
               len(args) == 3 -> start = args[0], stop = args[1], step = args[2].
             **kwargs:
               - dtype: Its expected dtype. (Optional, default: `tf.int64`).
           Returns:
             Dataset: A `NumpyDataset`.
           Raises:
             ValueError: if len(args) == 0.
           """
        data = None
        if len(args) == 1:
            data = np.arange(start=0, stop=args[0], step=1)
        elif len(args) == 2:
            data = np.arange(start=args[0], stop=args[1], step=1)
        elif len(args) == 3:
            data = np.arange(start=args[0], stop=args[1], step=args[2])
        else:
            raise ValueError('only maximum  3 args in arange function ')
        return NumpyDataset(data, symbol="range")

    @staticmethod
    def zip(*datasets):
        """Creates a `Dataset` by zipping together the given datasets.
            This method has similar semantics to the built-in `zip()` function
            in Python, with the main difference being that the `datasets`
            argument can be an arbitrary nested structure of `Dataset` objects.

        Args:
            datasets (Tuple[Dataset]):
        Examples:
            >>> ds1=Dataset.range(5)
            >>> ds2=Dataset.range(0,10,2)
            >>> dszip=Dataset.zip(ds1,ds2)
            >>> print(ds1.symbol)
            range
            >>> print(ds2.symbol)
            range_1
            >>> dszip.symbol
            ('range', 'range_1')
            >>> dszip[2]
            (array([2]), array([4]))


        """
        return ZipDataset(*datasets)

    @property
    def element_spec(self):
        return self._element_spec

    @element_spec.setter
    def element_spec(self, value):
        self._element_spec = value

    def len(self):
        return self.__len__()


class ZipDataset(Dataset):
    """A `Dataset` that zips its inputs together."""

    def __init__(self, *datasets, **kwargs):
        """See `Dataset.zip()` for details."""
        super().__init__()
        if not all(len(datasets[0]) == len(ds) for ds in datasets):
            raise ValueError("lengths of input datasets are inconsistent")
        self._datasets = datasets
        self.symbol = tuple([ds.symbol for ds in datasets])

    def __getitem__(self, index: int) -> Tuple:
        return tuple(ds[index] for ds in self._datasets)

    def __len__(self) -> int:
        return len(self._datasets[0])


class NumpyDataset(Dataset):
    def __init__(self, data=None, object_type=ObjectType.array_data, symbol="array", **kwargs):
        super().__init__(data, symbol=symbol, object_type=object_type, **kwargs)
        if data is not None:
            self._element_spec = TensorSpec(shape=tensor_to_shape(self.items[0], need_exclude_batch_axis=True, is_singleton=True), dtype=self.items[0].dtype, name=self.symbol,
                                            object_type=self.object_type)

    def __getitem__(self, index: int) -> Tuple:
        return None if self.items is None else self.items[index]

    def __len__(self) -> int:
        return None if self.items is None else len(self.items)


class StreamDataset(Dataset):
    r"""
    An abstract class for stream data.
    __iter__ method is aditionally needed.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def __getitem__(self):
        raise AssertionError("can not get item from StreamDataset by index")

    def __len__(self):
        raise AssertionError("StreamDataset does not have length")


class ArrayDataset(Dataset):
    def __init__(self, arrays):
        r"""
        ArrayDataset is a dataset for numpy array data, one or more numpy arrays
         are needed to initiate the dataset. And the dimensions represented sample number
         are expected to be the same.
        """
        super().__init__()
        if not all(len(arrays) == len(array) for array in arrays):
            raise ValueError("lengths of input arrays are inconsistent")
        self.arrays = arrays

    def __getitem__(self, index: int) -> Tuple:
        return self.arrays[index]

    def __len__(self) -> int:
        return len(self.arrays)


class ImageDataset(Dataset):
    def __init__(self, images, object_type: ObjectType = ObjectType.rgb, symbol="image", **kwargs):
        super().__init__(images, symbol=symbol, object_type=object_type, **kwargs)
        self.is_spatial = True
        self.is_paired_process = False

    def __getitem__(self, index: int):
        img = self.items[index]  # self.pop(index)
        if isinstance(img, str):
            img = file2array(img)

        if not isinstance(img, np.ndarray):
            raise ValueError('image data should be ndarray')
        elif isinstance(img, np.ndarray) and img.ndim not in [2, 3]:
            raise ValueError('image data dimension  should be 2 or 3, but get {0}'.format(img.ndim))
        elif self.object_type == ObjectType.gray:
            if img.ndim == 2:
                img = np.expand_dims(img, -1)
            elif (img.ndim == 3 and img.shape[-1] == 1):
                pass
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.expand_dims(img, -1)
        elif self.object_type == ObjectType.rgb and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif self.object_type == ObjectType.rgb and img.ndim == 3:
            if img.shape[-1] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
            img = img[:, :, :3]
        elif self.object_type == ObjectType.rgba:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        elif self.object_type == ObjectType.multi_channel:
            img = img.astype(np.float32)
        return img

    def data_transform(self, img_data):
        if len(self.transform_funcs) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self.transform_funcs:
                if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not image_backend_adaption:
                    img_data = fc(img_data, spec=self.element_spec)
            img_data = image_backend_adaption(img_data)

            return img_data
        else:
            return img_data

    def image_transform(self, data):
        return self.data_transform(data)

    @property
    def image_transform_funcs(self):
        return self.transform_funcs

    @image_transform_funcs.setter
    def image_transform_funcs(self, value):
        self.transform_funcs = value

    @property
    def reverse_image_transform_funcs(self):
        return_list = [reverse_image_backend_adaption]
        for i in range(len(self.transform_funcs)):
            fn = self.transform_funcs[-1 - i]
            if (inspect.isfunction(fn) and fn.__qualname__ == 'normalize.<locals>.img_op') or (isinstance(fn, Transform) and fn.name == 'normalize'):
                return_list.append(Unnormalize(fn.mean, fn.std))
        return_list.append(array2image)
        return return_list

    def reverse_image_transform(self, img_data):
        if len(self.reverse_image_transform_funcs) == 0:
            return reverse_image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_image_transform_funcs:
                if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not reverse_image_backend_adaption:
                    img_data = fc(img_data)
            img_data = reverse_image_backend_adaption(img_data)

            return img_data
        else:
            return img_data


class MaskDataset(Dataset):
    def __init__(self, masks, class_names=None, object_type: ObjectType = ObjectType.label_mask,
                 get_image_mode: GetImageMode = GetImageMode.processed, symbol="mask", **kwargs):
        super().__init__(masks, symbol=symbol, object_type=object_type, **kwargs)
        if object_type not in [ObjectType.label_mask, ObjectType.binary_mask, ObjectType.alpha_mask,
                               ObjectType.color_mask]:
            raise ValueError('Only mask is valid expect image type. ')

        self.get_image_mode = get_image_mode
        self.palette = OrderedDict()
        self.transform_funcs = []
        self.is_spatial = True
        self.is_paired_process = False
        self.class_names = {}
        self._lab2idx = {}
        self._idx2lab = {}
        if class_names is not None:
            self.class_names = class_names
        self._element_spec = TensorSpec(shape=TensorShape(self[0]), name=self.symbol, object_type=self.object_type, is_spatial=True)

    def __getitem__(self, index: int):
        mask = self.items[index]  # self.pop(index)

        if isinstance(mask, str):
            if self.object_type == ObjectType.binary_mask:
                mask = file2array(mask, flag=cv2.IMREAD_GRAYSCALE)
                mask[mask > 0] = 1
                mask[mask <= 0] = 0
                mask.astype(np.int64)
            elif self.object_type == ObjectType.alpha_mask:
                mask = file2array(mask, flag=cv2.IMREAD_GRAYSCALE)
            elif self.object_type == ObjectType.label_mask:
                if '.png' in mask:
                    mask = file2array(mask, flag=cv2.IMREAD_UNCHANGED)
                    mask = mask[:, :, 1]
                else:
                    mask = file2array(mask, flag=cv2.IMREAD_UNCHANGED)

            elif self.object_type == ObjectType.color_mask:
                mask = file2array(mask, flag=cv2.IMREAD_COLOR)
                if mask.ndim == 2:
                    pass
                elif mask.ndim == 3:
                    mask = mask[:, :, :3]
                    if len(self.palette) > 0:
                        mask = color2label(mask, self.palette).astype(np.int64)

        if not isinstance(mask, np.ndarray):
            raise ValueError('image data should be ndarray')
        elif isinstance(mask, np.ndarray) and mask.ndim not in [2, 3]:
            raise ValueError('image data dimension  should be 2 or 3, but get {0}'.format(mask.ndim))

        return mask

    def mask_transform(self, mask_data):
        if len(self.transform_funcs) == 0:
            return mask_backend_adaptive(mask_data, label_mapping=self.class_names, object_type=self.object_type)
        else:
            if isinstance(mask_data, np.ndarray):
                for fc in self.transform_funcs:
                    mask_data = fc(mask_data, spec=self.element_spec)
                mask_data = mask_backend_adaptive(mask_data, label_mapping=self.class_names, object_type=self.object_type)
                return mask_data
            else:
                return mask_data

    def data_transform(self, data):
        return self.mask_transform(data)

    @property
    def reverse_image_transform_funcs(self):
        return_list = []
        for i in range(len(self.transform_funcs)):
            fn = self.transform_funcs[-1 - i]
            if (inspect.isfunction(fn) and fn.__qualname__ == 'normalize.<locals>.img_op') or (isinstance(fn, Transform) and fn.name == 'normalize'):
                return_list.append(Unnormalize(fn.mean, fn.std))
        return_list.append(array2image)
        return return_list

    def reverse_image_transform(self, mask_data):
        if len(self.reverse_image_transform_funcs) == 0:
            return mask_data
        if isinstance(mask_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_image_transform_funcs:
                mask_data = fc(mask_data)
            # img_data = reverse_image_backend_adaption(img_data)

            return mask_data
        else:
            return mask_data


class LabelDataset(Dataset):
    def __init__(self, labels, object_type=ObjectType.classification_label, class_names=None, symbol="label",
                 **kwargs):
        super().__init__(labels, symbol=symbol, object_type=object_type, **kwargs)
        self.dtype = np.int64

        self.class_names = {}
        self._lab2idx = {}
        self._idx2lab = {}
        if class_names is not None:
            self.class_names = class_names

        self.transform_funcs = []

        shp = None
        if isinstance(self.items[0], numbers.Number):
            shp = TensorShape([0])
        else:
            shp = TensorShape(self.items)
        self._element_spec = TensorSpec(shape=shp, name=self.symbol, object_type=self.object_type, dtype=self.dtype)

    def binding_class_names(self, class_names=None, language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = 'en-us'
            self.class_names[language] = list(class_names)

            self.__default_language__ = language
            self._lab2idx = dict(zip(self.class_names[language], range(len(self.class_names[language]))))
            self._idx2lab = dict(zip(range(len(self.class_names[language])), self.class_names[language]))

    def __getitem__(self, index: int):
        label = self.items[index]
        return label

    def data_transform(self, label_data):
        label_data = label_backend_adaptive(label_data, self.class_names)
        if isinstance(label_data, list) and all(isinstance(elem, np.ndarray) for elem in label_data):
            label_data = np.asarray(label_data).astype(np.int64)
        if isinstance(label_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.transform_funcs:
                label_data = fc(label_data, spec=self.element_spec)
            return label_data
        else:
            return label_data

    def label_transform(self, data):
        return self.data_transform(data)


class BboxDataset(Dataset):
    def __init__(self, boxes, image_size=None, object_type=ObjectType.absolute_bbox, class_names=None,
                 symbol="bbox", **kwargs):
        super().__init__(boxes, symbol=symbol, object_type=object_type, **kwargs)

        self._element_spec = TensorSpec(shape=TensorShape([None, 5]), name=self.symbol, object_type=self.object_type, is_spatial=True)
        self.is_paired_process = False
        self.is_spatial = True
        self.dtype = np.int64
        self.image_size = image_size
        self.class_names = {}
        self._lab2idx = {}
        self._idx2lab = {}
        self.transform_funcs = []
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
        bboxes = self.items[index].astype(np.float32)
        if self.object_type == ObjectType.relative_bbox and (self.image_size is None):
            raise RuntimeError('You need provide image size information for calculate relative_bbox. ')
        elif self.object_type == ObjectType.relative_bbox:
            height, width = self.image_size
            bboxes[:, 0] = bboxes[:, 0] / width
            bboxes[:, 2] = bboxes[:, 2] / width
            bboxes[:, 1] = bboxes[:, 1] / height
            bboxes[:, 3] = bboxes[:, 3] / height
            return np.array(bboxes).astype(np.float32)

        elif self.object_type == ObjectType.absolute_bbox:
            return np.array(bboxes).astype(np.float32)

    def bbox_transform(self, *bbox):
        if isinstance(bbox, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.transform_funcs:
                bbox = fc(bbox, spec=self.element_spec)
            return bbox
        else:
            return bbox

    def data_transform(self, data):
        return self.bbox_transform(data)


class LandmarkDataset(Dataset):
    def __init__(self, landmarks, image_size=None, object_type=ObjectType.landmarks, symbol="landmark", **kwargs):
        super().__init__(landmarks, symbol=symbol, object_type=object_type, **kwargs)

        self.image_size = image_size
        self._element_spec = TensorSpec(shape=tensor_to_shape(self.items[0], need_exclude_batch_axis=True, is_singleton=True), dtype=self.items[0].dtype, name=self.symbol,
                                        object_type=self.object_type, is_spatial=True)
        self.is_paired_process = False
        self.is_spatial = True
        self.transform_funcs = []

    def __getitem__(self, index: int):
        # if (landmarks > 1).any() or (self.image_size is None):
        #     return np.array(landmarks).astype(np.float32)
        # # elif (landmarks > 1).any():
        # #     height, width = self.image_size
        # #     landmarks[:, 0] = landmarks[:, 0] / width
        # #     landmarks[:, 1] = landmarks[:, 1] / height
        # #     return np.array(landmarks).astype(np.float32)
        # else:
        return self.items[index]

    def landmark_transform(self, *landmarks):
        if isinstance(landmarks, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.transform_funcs:
                landmarks = fc(landmarks, spec=self.element_spec)
            return landmarks
        else:
            return landmarks

    def data_transform(self, data):
        return self.landmark_transform(data)


class RandomNoiseDataset(Dataset):
    def __init__(self, shape, object_type=ObjectType.random_noise, random_mode='normal', symbol="noise", **kwargs):
        super().__init__([], symbol=symbol, object_type=object_type, **kwargs)

        self.dtype = np.float32
        if isinstance(shape, numbers.Integral):
            self.shape = (shape,)
        else:
            self.shape = shape
        self.random_mode = random_mode
        self._element_spec = TensorSpec(shape=TensorShape(shape), name=self.symbol, object_type=self.object_type)
        self.is_paired_process = False
        self.is_spatial = False
        self.transform_funcs = []
        self.length=sys.maxsize

    def data_transform(self, noise_data):
        return noise_data

    def __getitem__(self, index: int):
        if self.random_mode == 'normal':
            return (np.random.standard_normal(self.shape)-0.5)*2
        elif self.random_mode == 'uniform':
            return np.random.uniform(-1, 1, self.shape)

    def __len__(self):
        return self.length


class TextSequenceDataset(Dataset):
    def __init__(self, corpus=None, is_onehot=False, sequence_offset=0, section_delimiter='\n\n', stopwords=None, sequence_length: int = 64, sequence_start_at='random',
                 object_type=ObjectType.corpus, symbol=None, **kwargs):
        super().__init__(None,symbol=symbol, object_type=object_type, **kwargs)
        self.sequence_start_at = sequence_start_at
        self.transform_funcs = []
        if len(section_delimiter) == 2:
            self.section_delimiter = section_delimiter
        else:
            self.section_delimiter = '\n\n'
        self.vocabs = None
        self.text2index = None
        self.index2text = None
        self.is_onehot = is_onehot
        self.is_paired_process = False
        self.is_spatial = True

        if hasattr(corpus, "__iter__"):
            new_corpus = []
            section = []
            section.append('<start/>')
            for i in range(len(corpus)):
                item = corpus[i]
                if item == '\n' and corpus[i - 1] != '\n' and len(section) > 0:
                    section.append('<end/>')
                    if i < len(corpus) - 1 and corpus[i + 1] == '\n':
                        new_corpus.append(section)
                        section = []
                        section.append('<start/>')
                    elif i < len(corpus) - 1:
                        section.append('<start/>')
                elif self.section_delimiter != '\n\n' and len(self.section_delimiter) == 2 and len(section) > 0 and item == self.section_delimiter[0] and i < len(corpus) - 1 and \
                        corpus[i - 1] == self.section_delimiter[1]:
                    section.append('<end/>')
                    if i < len(corpus) - 1 and corpus[i + 1] == '\n':
                        new_corpus.append(section)
                        section = []
                        section.append('<start/>')
                    elif i < len(corpus) - 1:
                        section.append('<start/>')
                elif item == '\r' and corpus[i - 1] == '\n':
                    pass
                elif item == '\n':
                    pass
                else:
                    section.append(item)
                    if i == len(corpus) - 1:
                        section.append('<end/>')
            if len(section) > 0:
                new_corpus.append(section)

            if self.sequence_start_at == 'random':
                for sect in new_corpus:
                    self.items.extend(sect)
                    self.items.append('\n')
                    self.items.append('\n')
            else:
                self.items.extend(new_corpus)

            chars = sorted(list(set(corpus)))

            chars.insert(0, '<start/>')
            chars.insert(1, '<end/>')
            chars.insert(2, '<unknown/>')
            chars.insert(3, '<pad/>')
            print('total distinct chars:', len(chars))
            self.vocabs = chars
            self.text2index = dict((c, i) for i, c in enumerate(chars))
            self.index2text = dict((i, c) for i, c in enumerate(chars))
        else:
            raise ValueError('corpus should be a collection.')


        self.sequence_offset = sequence_offset
        self.dtype = np.float32 if self.is_onehot else np.int64
        self.text_transform_funcs = []
        self.is_paired_process = False
        self.sequence_length = sequence_length

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = idx.__index__()
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(itertools.islice(iterator, idx, None))

    def __getitem__(self, index: int):
        sequencetext = None
        seq_base = list(self.items)
        if isinstance(self.sequence_offset, int):
            if self.sequence_start_at == 'random':
                sequencetext = seq_base[index + self.sequence_offset:builtins.min(index + self.sequence_offset + self.sequence_length, self.len())]
            elif self.sequence_start_at == 'section_start':
                sectiontext = seq_base[index]
                sequencetext = sectiontext[self.sequence_offset:builtins.min(self.sequence_offset + self.sequence_length, self.len())]
        elif isinstance(self.sequence_offset, list):
            sequencetext = []
            if self.sequence_start_at == 'random':
                for k in self.sequence_offset:
                    if 0 <= index + self.sequence_offset < len(seq_base):
                        sequencetext.append(seq_base[index + self.sequence_offset])
                    else:
                        sequencetext.append('<pad/>')

            elif self.sequence_start_at == 'section_start':
                sectiontext = seq_base[index]
                for k in self.sequence_offset:
                    if 0 <= index + self.sequence_offset < len(seq_base):
                        sequencetext.append(seq_base[index + self.sequence_offset])
                    else:
                        sequencetext.append('<pad/>')

        if len(sequencetext) != self.sequence_length:
            sequencetext.extend(['<pad/>'] * (self.sequence_length - len(sequencetext)))
        arr = None
        if self.is_onehot:
            arr = np.zeros((self.sequence_length, len(self.text2index)))
            for i in range(self.sequence_length):
                this_char = sequencetext[i]
                if i + 1 < self.sequence_length and ''.join(sequencetext[i:i + 2]) == '\n\n':
                    arr[i:, self.text2index['<pad/>']] = 1
                    break
                elif this_char in self.text2index:
                    arr[i, self.text2index[this_char]] = 1
                else:
                    arr[i, self.text2index['<unknown/>']] = 1
            arr = arr.astype(np.float32)
        else:
            arr = np.zeros((self.sequence_length))
            for i in range(self.sequence_length):
                this_char = sequencetext[i]
                if i + 1 < self.sequence_length and ''.join(sequencetext[i:i + 2]) == '\n\n':
                    arr[i:] = self.text2index['<pad/>']
                    break
                elif this_char in self.text2index:
                    arr[i] = self.text2index[this_char]
                else:
                    arr[i] = self.text2index['<unknown/>']
            arr = arr.astype(np.int64)

        if self.is_paired_process == False and len(self.text_transform_funcs) == 0:
            return text_backend_adaption(arr)
        elif self.is_paired_process == False:
            return self.text_transform(arr)
        elif self.is_paired_process == True:
            return arr

        return None

    def data_transform(self, text_data):
        if len(self.transform_funcs) == 0:
            return text_backend_adaption(text_data)
        if isinstance(text_data, np.ndarray):
            for fc in self.transform_funcs:
                text_data = fc(text_data, spec=self.element_spec)
            text_data = text_backend_adaption(text_data)

    def text_transform(self, text_data):
        return self.data_transform(text_data)

    @property
    def reverse_text_transform_funcs(self):
        return_list = []
        return_list.append(reverse_text_backend_adaption)
        for i in range(len(self.text_transform_funcs)):
            fn = self.text_transform_funcs[-1 - i]
            # if fn.__qualname__ == 'normalize.<locals>.text_op':
            #     return_list.append(unnormalize(fn.mean, fn.std))
        # return_list.append(array2image)
        return return_list

    def reverse_text_transform(self, text_data):
        if len(self.reverse_text_transform_funcs) == 0:
            return reverse_text_backend_adaption(text_data)
        if isinstance(text_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_text_transform_funcs:
                text_data = fc(text_data)
            text_data = reverse_text_backend_adaption(text_data)

            return text_data
        else:
            return text_data


class Iterator(object):
    def __init__(self, data=None, label=None, mask=None, unpair=None, sample_filter=None, minibatch_size=8, mode='tuple', is_shuffe=True, buffer_size=None, workers=2, **kwargs):
        self.is_paired_process = False
        self._data = None
        self._label = None
        self._unpair = None
        self.paired_process_symbols = []
        self.data_template = None
        self.is_shuffe = is_shuffe
        self.datasets_dict = OrderedDict()
        self.mode = mode

        self.workers = workers
        self.pass_cnt = 0
        self.pass_time_spend = 0
        if data is not None and isinstance(data, tuple):
            self._data = Dataset.zip(*data)
        elif data is not None and isinstance(data, (Dataset, ZipDataset, TextSequenceDataset)):
            self._data = data

        if label is not None and isinstance(label, tuple):
            self._label = Dataset.zip(*label)
        elif label is not None and (inspect.isgenerator(label) or isinstance(label, (Dataset, ZipDataset, TextSequenceDataset))):
            self._label = label

        if unpair is not None and isinstance(unpair, Dataset):
            self._unpair = unpair

        if self._data is None:
            self._data = NumpyDataset(symbol="")
        if self._label is None:
            self._label = NumpyDataset(symbol="")
        if self._unpair is None:
            self._unpair = NumpyDataset(symbol="")

        datasets = self.get_datasets()
        for ds in datasets:
            self.datasets_dict[ds.symbol] = ds
        data_symbols = iteration_tools.flatten([self.data.symbol], iterable_types=(list, tuple))
        label_symbols = iteration_tools.flatten([self.label.symbol], iterable_types=(list, tuple))
        # check pair_process
        data_ds = [ds for ds in datasets if len(ds) > 0 and ds.symbol in data_symbols and ds.is_spatial == True]
        label_ds = [ds for ds in datasets if len(ds) > 0 and ds.symbol in label_symbols and ds.is_spatial == True]

        if len(data_ds) > 0 and len(label_ds) > 0:
            for ds in data_ds:
                ds.is_paired_process = True
                self.paired_process_symbols.append(ds.symbol)
            for ds in label_ds:
                ds.is_paired_process = True
                self.paired_process_symbols.append(ds.symbol)

        self.data_template = OrderedDict()


        for k in range(len(datasets)):
            ds = datasets[k]
            if len(ds) > 0:
                dataitem = ds[k]
                ds.element_spec = TensorSpec.tensor_to_spec(expand_dims(dataitem, 0), object_type=ds.object_type, name=ds.symbol)
                self.data_template[ds.element_spec] = None

        self._signature=None
        self._minibatch_size = minibatch_size
        self.paired_transform_funcs = []
        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=self.is_shuffe, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)
        if buffer_size is None:
            buffer_size = 8 * minibatch_size
        self.buffer_size = buffer_size
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)
        self.sample_filter = None
        if inspect.isfunction(sample_filter) or callable(sample_filter):
            self.sample_filter = sample_filter
            self.batch_sampler.sample_filter = self.sample_filter

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


        if self._label is not None and isinstance(self._label, (MaskDataset, BboxDataset, ImageDataset)) and isinstance(self._data, ImageDataset) and len(self._label) == len(
                self._data):
            self._label.is_paired_process = self._data.is_paired_process = self.is_paired_process = True
        else:
            self._label.is_paired_process = self._data.is_paired_process = self.is_paired_process = False

        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True, drop_last=False)
        self.batch_sampler.sample_filter = self.sample_filter
        self._sample_iter = iter(self.batch_sampler)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

        if isinstance(self._label, (MaskDataset, ImageDataset, BboxDataset)) and isinstance(self._data, ImageDataset) and len(
                self._label) == len(self._data):
            self._label.is_paired_process = self._data.is_paired_process = self.is_paired_process = True
        else:
            self._label.is_paired_process = self._data.is_paired_process = self.is_paired_process = False
        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True, drop_last=False)
        self.batch_sampler.sample_filter = self.sample_filter
        self._sample_iter = iter(self.batch_sampler)

    @property
    def unpair(self):
        return self._unpair

    @unpair.setter
    def unpair(self, value):
        self._unpair = value


        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True, drop_last=False)
        self.batch_sampler.sample_filter = self.sample_filter
        self._sample_iter = iter(self.batch_sampler)

    @property
    def palette(self):
        if isinstance(self._label, MaskDataset) and self._label.object_type in [ObjectType.label_mask, ObjectType.color_mask]:
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
        self.batch_sampler.sample_filter = self.sample_filter
        self._sample_iter = iter(self.batch_sampler)
        self.buffer_size = 8 * value
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)
        self.pass_cnt = 0
        self.pass_time_spend = 0.

    @property
    def signature(self):
        datasets = self.get_datasets()
        for ds in datasets:
            if self._signature is not None and ds.symbol not in self._signature.outputs:
                self._signature =None
        if self._signature is None:
            self._signature = Signature(name='data_provider')
            self.data_template=OrderedDict()
            self.datasets_dict=OrderedDict()
            for k in range(len(datasets)):
                ds = datasets[k]
                if isinstance(ds, ZipDataset):
                    for dsitem in ds._datasets:
                        if len(dsitem) > 0:
                            dataitem = dsitem[k]
                            dsitem.element_spec = TensorSpec.tensor_to_spec(expand_dims(dataitem, 0), object_type=ds.object_type, name=ds.symbol)
                            self.datasets_dict[dsitem.symbol] = dsitem
                            self.data_template[dsitem.element_spec] = None
                            self._signature.outputs[dsitem.symbol] = ds.element_spec
                else:

                    if len(ds) > 0:
                        dataitem = ds[k]
                        ds.element_spec = TensorSpec.tensor_to_spec(expand_dims(dataitem, 0), object_type=ds.object_type, name=ds.symbol)
                        self.datasets_dict[ds.symbol] = ds
                        self.data_template[ds.element_spec] = None
                        self._signature.outputs[ds.symbol] = ds.element_spec

            self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=self.is_shuffe, drop_last=False)
            self._sample_iter = iter(self.batch_sampler)
        return self._signature

    # def update_signature(self, arg_names):
    #
    #     if len(arg_names) != len(self.datasets_dict):
    #         warnings.warn('arg_names should have same length with datasets, expect {0} got {1}'.format(len(arg_names), len(self.datasets_dict)))
    #     else:
    #         iterdata = self.next()
    #         if self.signature is None or not isinstance(self.signature, Signature):
    #             self.signature = Signature()
    #             self.signature.name = 'data_provider'
    #         if isinstance(arg_names, (list, tuple)) and len(iterdata) == len(arg_names):
    #             new_dict = OrderedDict()
    #             for i in range(len(arg_names)):
    #                 arg = arg_names[i]
    #                 data = iterdata[i]
    #                 self.signature.outputs[arg] = TensorSpec(shape=TensorShape((None,) + tuple(tensor_to_shape(data).dims)), dtype=data.dtype,
    #                                                          object_type=self.datasets_dict.value_list[i].object_type, name=arg)
    #                 ds = self.datasets_dict.value_list[i]
    #                 new_dict[arg] = ds
    #             self.datasets_dict = None
    #             self.datasets_dict = new_dict

    def paired_transform(self, datadict: Dict[TensorSpec, np.ndarray]):
        # if isinstance(img_data, list) and all(isinstance(elem, np.ndarray) for elem in img_data):
        #     img_data = np.asarray(img_data)
        # if isinstance(img_data, str) and os.path.isfile(img_data) and os.path.exists(img_data):
        #     img_data = image2array(img_data)

        if len(self.paired_transform_funcs) == 0:
            return datadict

        # if img_data.ndim>=2:
        for fc in self.paired_transform_funcs:
            try:
                datadict = fc(datadict)
            except:
                PrintException()
        return datadict

    def get_datasets(self):
        datasets = []
        if self._data and isinstance(self._data, Dataset) and not isinstance(self._data, ZipDataset) and len(self._data) > 0:
            datasets.append(self._data)
        elif self._data and isinstance(self._data, ZipDataset):
            for ds in self._data._datasets:
                datasets.append(ds)
        if self._label and isinstance(self._label, Dataset) and not isinstance(self._label, ZipDataset) and len(self._label) > 0:
            datasets.append(self._label)
        elif self._label and isinstance(self._label, ZipDataset):
            for ds in self._label._datasets:
                datasets.append(ds)
        if self._unpair and isinstance(self._unpair, Dataset) and len(self._unpair) > 0:
            datasets.append(self._unpair)
        elif isinstance(self._unpair, ZipDataset):
            for ds in self._unpair._datasets:
                datasets.append(ds)
        return tuple(datasets)

    def find_dataset(self, dataset_symbol: str):
        datasets = self.get_datasets()
        for ds in datasets:
            if ds.symbol == dataset_symbol:
                return ds

    def update_data_template(self):
        self._signature=None



    def print_statistics(self):
        print('avg. process time: {0:.5f}'.format(self.pass_time_spend / float(builtins.max(1, self.pass_cnt))))
        print('avg. batch time: {0:.5f}'.format(self._minibatch_size * self.pass_time_spend / float(builtins.max(1, self.pass_cnt))))

    def __getitem__(self, index: int):
        start_time = time.time()

        try:
            bbox = None
            mask = None

            returnData = copy.deepcopy(self.data_template)

            data = self.data.__getitem__(index % len(self.data)) if self.data is not None and len(self.data) > 0 else None
            label = self.label.__getitem__(index % len(self.label)) if self.label is not None and len(self.label) > 0 else None

            unpair = self.unpair.__getitem__(index % len(self.unpair)) if self.unpair is not None and len(self.unpair) > 0 else None

            results = iteration_tools.flatten([data, label, unpair], iterable_types=(list, tuple))
            results = tuple([item for item in results if item is not None])
            if len(returnData) != len(results):
                raise ValueError("Flattened data sh")
            for k in range(len(returnData)):
                returnData[returnData.key_list[k]] = results[k]

            if len(self.paired_process_symbols) > 0:
                returnData = self.paired_transform(returnData)

            def process_data_transform(i):
                ds = self.datasets_dict[returnData.key_list[i].name]
                returnData[returnData.key_list[i]] = unpack_singleton(ds.data_transform(returnData.value_list[i]))

            threads = []
            for i in range(len(returnData)):
                threads.append(threading.Thread(target=process_data_transform, args=(i,)))
                threads[i].start()
            for i in range(len(returnData)):
                threads[i].join()

            if self.signature is None or len(self.signature) == 0:
                self.signature = Signature(name='data_provider')
                for spec in returnData.key_list:
                    self.signature.outputs[spec.name] = spec

            self.pass_cnt += 1
            self.pass_time_spend += float(time.time() - start_time)

            return returnData
        except:
            PrintException()

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name in ['_data','_label', '_unpair']:
            self._signature = None
        if name=='_unpair':
            self._signature=None
            if isinstance(value,RandomNoiseDataset):
                value.length=len(self.data)


    def _next_index(self):
        return next(self._sample_iter)

    def __iter__(self):
        return self._sample_iter

    # return a batch , do minimal fetch before return
    def next(self):
        if self.out_queue.qsize() == 0:
            in_data = self._sample_iter.__next__()
            self.out_queue.put(in_data)

        out_data = self.out_queue.get(True)

        while self.out_queue.full():
            in_data = self._sample_iter.__next__()
            self.out_queue.put(in_data)

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


class MetricIterator(Iterator):
    """An Iterator

    """

    def __init__(self, data=None, label=None, minibatch_size=8):
        super().__init__(data, minibatch_size)
        self.is_paired_process = False
        self.signature = None
        self._data = ImageDataset()
        if data is not None and isinstance(data, (Dataset, ZipDataset)):
            self._data = data
        elif data is not None and isinstance(data, list):
            self._data = ImageDataset(images=data)
        self._label = LabelDataset()
        if label is not None and isinstance(label, (LabelDataset)):
            self._label = label
        elif label is not None and isinstance(label, list):
            self._label = LabelDataset(labels=label)
        self._unpair = ImageDataset()
        self.workers = 2
        self.pass_cnt = 0
        self.pass_time_spend = 0

        self._minibatch_size = minibatch_size
        self.paired_transform_funcs = []
        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)
        self.buffer_size = 10
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)

    def __getitem__(self, index: int):
        # start = time.time()

        try:

            anchor = self.data.__getitem__(index % len(self.data)) if self.data is not None and len(self.data) > 0 else None
            positive = self.data.__getitem__(index % len(self.data)) if self.data is not None and len(self.data) > 0 else None
            label = self.label.__getitem__(index % len(self.label)) if self.label is not None and len(self.label) > 0 else None

            available_list = list(range(len(self.data)))
            available_list.remove(index % len(self.data))

            negative = self.data.__getitem__(random.choice(available_list)) if self.data is not None and len(self.data) > 0 else None

            return_data = []
            if self.signature is None or len(self.signature) == 0:
                self.signature = Signature()
                self.signature.name = 'data_provider'
                if anchor is not None:
                    self.signature.outputs['anchor'] = (-1,) + anchor.shape
                if positive is not None:
                    self.signature.outputs['positive'] = (-1,) + positive.shape
                if negative is not None:
                    self.signature.outputs['negative'] = (-1,) + negative.shape
                if label is not None:
                    self.signature.outputs['label'] = (-1,)

            if anchor is not None:
                return_data.append(anchor)
            if positive is not None:
                return_data.append(positive)
            if negative is not None:
                return_data.append(negative)
            if label is not None:
                return_data.append(label)

            return tuple(return_data)
        except:
            PrintException()

