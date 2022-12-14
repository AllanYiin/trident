from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
import collections
import copy
import inspect
import itertools
import numbers
import random
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import TypeVar, Tuple, Optional, Iterator, Dict

import cv2
import numpy as np

from trident import context
# from trident.backend.numpy_ops import DTYPE_MAPPING
from trident.backend import dtype as Dtype
from trident.backend import iteration_tools
from trident.backend.common import *
from trident.backend.opencv_backend import file2array
from trident.backend.tensorspec import TensorSpec, ObjectType
from trident.data.image_common import image_backend_adaption, reverse_image_backend_adaption, \
    array2image, TensorShape,OrderedDict
from trident.data.label_common import label_backend_adaptive
from trident.data.mask_common import mask_backend_adaptive, color2label
from trident.data.samplers import *
from trident.data.text_common import reverse_text_backend_adaption
from trident.data.transform import Transform
from trident.data.vision_transforms import Unnormalize
from trident.data.text_transforms import ToHalfWidth
try:
    import Queue
except ImportError:
    import queue as Queue

if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import tensor_to_shape, expand_dims, cast,to_tensor
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import tensor_to_shape, expand_dims, cast,to_tensor

__all__ = ['Dataset', 'ZipDataset', 'ImageDataset', 'MaskDataset', 'TextSequenceDataset', 'LabelDataset', 'BboxDataset', 'LandmarkDataset', 'Iterator', 'MetricIterator',
           'NumpyDataset', 'RandomNoiseDataset', 'ArrayDataset','IndexDataset']

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
        self.items = []
        if args is not None and isinstance(args, np.ndarray):
            self.items=args
        elif args is not None and hasattr(args, '__iter__'):
            self.items.extend(args)
        self._element_spec = None
        self.parent = None

        if symbol is None:
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
        if index >= len(self.items):
            index = index % len(self.items)
        return self.items[index]

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name == 'symbol' and isinstance(self._element_spec, TensorSpec):
            self._element_spec._name = value

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
        super().__init__(*unpack_singleton(datasets), **kwargs)
        self.symbol = '{0}'.format(str((ds.symbol for ds in unpack_singleton(datasets))))

    def __getitem__(self, index: int) -> Tuple:
        return self.items[index]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if name.endswith('transform_funcs'):
            if 'items' in self.__dict__:
                items = self.__dict__['items']
                ds = items[0]
                if name in ds.__dict__:
                    return ds.__dict__[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name: str, value) -> None:
        if name.endswith('transform_funcs') or name=='parent':
            if 'items' in self.__dict__:
                items = self.__dict__['items']
                for ds in items:
                    if hasattr(ds, name):
                        setattr(ds, name, value)
        else:
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return 0 if len(self.items)==0 else len(self.items[0])  #len(self) % len(self.items)


class NumpyDataset(Dataset):
    def __init__(self, data=None, object_type=ObjectType.array_data, symbol="array", **kwargs):
        super().__init__(data, symbol=symbol, object_type=object_type, **kwargs)

    def __getitem__(self, index: int) -> Tuple:
        if index >= len(self.items):
            index = index % len(self.items)
        return None if self.items is None else self.items[index]

    def data_transform(self, img_data):

        if isinstance(img_data, np.ndarray):
            for fc in self.transform_funcs:
                if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not image_backend_adaption:
                    img_data = fc(img_data, spec=self.element_spec)

            return img_data
        else:
            return img_data

    def __len__(self) -> int:
        return 0 if self.items is None else len(self.items)


class IndexDataset(Dataset):
    def __init__(self, data=None,mapping_dict=None,default_mapping_value=None, object_type=ObjectType.index_data, symbol="index_data", **kwargs):
        super().__init__(data, symbol=symbol, object_type=object_type, **kwargs)
        self.mapping_dict=mapping_dict
        self.default_mapping_value=default_mapping_value
        self._element_spec = TensorSpec(shape=tensor_to_shape(self[0], need_exclude_batch_axis=True, is_singleton=True),
                                        dtype=Dtype.int64, name=self.symbol, object_type=self.object_type,
                                        is_spatial=False)

    def __getitem__(self, index: int) -> Tuple:
        if index >= len(self.items):
            index = index % len(self.items)
        if self.items is None:
            return None
        else:
            item=self.items[index]
            if self.mapping_dict is None :
                return item
            else:
                if item  in self.mapping_dict:
                    return self.mapping_dict[item]
                else:
                    return self.default_mapping_value



    def data_transform(self, data):

        if isinstance(data, np.ndarray):
            for fc in self.transform_funcs:
                if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not image_backend_adaption:
                    img_data = fc(data, spec=self.element_spec)

            return data
        else:
            return data

    def __len__(self) -> int:
        return 0 if self.items is None else len(self.items)


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

    def __getitem__(self, index):
        if index >= len(self.items):
            index = index % len(self.items)
        raise AssertionError("can not get item from StreamDataset by index")

    def __len__(self):
        raise AssertionError("StreamDataset does not have length")


class ArrayDataset(Dataset):
    def __init__(self, arrays,object_type=ObjectType.array_data,symbol='array'):
        r"""
        ArrayDataset is a dataset for numpy array data, one or more numpy arrays
         are needed to initiate the dataset. And the dimensions represented sample number
         are expected to be the same.
        """
        super().__init__(arrays,object_type=object_type,)
        self.symbol = symbol

    def __getitem__(self, index: int):
        if index >= len(self.items):
            index = index % len(self.items)
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def data_transform(self, img_data):

        if isinstance(img_data, np.ndarray):
            for fc in self.transform_funcs:
                if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not image_backend_adaption:
                    img_data = fc(img_data, spec=self.element_spec)

            return img_data
        else:
            return img_data


class ImageDataset(Dataset):
    def __init__(self, images, object_type: ObjectType = ObjectType.rgb, symbol="image", **kwargs):
        super().__init__(images, symbol=symbol, object_type=object_type, **kwargs)
        self.is_spatial = True
        self.is_paired_process = False
        if object_type==ObjectType.image_path:
            self.element_spec=TensorSpec(shape=TensorShape([None]),dtype=str,object_type=object_type,name=symbol)
        else:
            self.element_spec = TensorSpec(shape=tensor_to_shape(self.items[0] if len(self.items)>0 else None ,True,True), dtype=Dtype.float32, object_type=object_type,name=symbol)

    def __getitem__(self, index: int):
        try:
            if index >= len(self.items):
                index = index % len(self.items)
            img = self.items[index]  # self.pop(index)

            if isinstance(img, str) and self.object_type == ObjectType.image_path:
                return img
            elif isinstance(img, np.ndarray) and self.object_type != ObjectType.image_path:
                return img
            else:
                img = file2array(img)

            if not isinstance(img, np.ndarray):
                raise ValueError('image data should be ndarray')
            elif  self.object_type == ObjectType.image_path:
                return img
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
                elif img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img = img[:, :, :3]

            elif self.object_type == ObjectType.rgba:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            elif self.object_type == ObjectType.multi_channel:
                img = img.astype(np.float32)
            return img
        except Exception as e:
            print(e)
            PrintException()

    def data_transform(self, img_data):
        if self.object_type == ObjectType.image_path:
            return img_data
        if len(self.transform_funcs) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self.transform_funcs:
                if (callable(fc) or isinstance(fc, Transform)) and fc is not image_backend_adaption:
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
        if isinstance(self.parent,Iterator):
            self.parent._is_signature_update = False

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
        if self.object_type==ObjectType.image_path:
            return img_data
        elif isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_image_transform_funcs:
                if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not reverse_image_backend_adaption:
                    img_data = fc(img_data)
            img_data = reverse_image_backend_adaption(img_data)

            return img_data
        else:
            return img_data


class MaskDataset(Dataset):
    def __init__(self, masks, class_names=None, object_type: ObjectType = ObjectType.label_mask, squeeze_channel=True,symbol="mask", **kwargs):
        super().__init__(masks, symbol=symbol, object_type=object_type, **kwargs)
        if object_type not in [ObjectType.label_mask, ObjectType.binary_mask, ObjectType.alpha_mask,
                               ObjectType.color_mask]:
            raise ValueError('Only mask is valid expect image type. ')
        self.palette = OrderedDict()
        self.squeeze_channel=squeeze_channel
        self.transform_funcs = []
        self.is_spatial = True
        self.is_paired_process = False
        self.class_names = {}
        self._lab2idx = {}
        self._idx2lab = {}
        if class_names is not None:
            self.class_names = class_names
        self._element_spec = TensorSpec(shape=tensor_to_shape(self[0] if len(self)>0 else None ,need_exclude_batch_axis=True,is_singleton=True), dtype=Dtype.int64,name=self.symbol, object_type=self.object_type, is_spatial=True)

    def __getitem__(self, index: int):
        if index >= len(self.items):
            index = index % len(self.items)
        mask = self.items[index]  # self.pop(index)

        if isinstance(mask, str):
            if self.object_type == ObjectType.binary_mask:
                mask = file2array(mask, flag=cv2.IMREAD_GRAYSCALE)
                if mask.max() > 1:
                    mask = mask / mask.max()
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                mask.astype(np.int64)

            elif self.object_type == ObjectType.alpha_mask:
                mask = file2array(mask, flag=cv2.IMREAD_GRAYSCALE)
            elif self.object_type == ObjectType.label_mask:
                if '.png' in mask:
                    mask = file2array(mask, flag=cv2.IMREAD_UNCHANGED)
                    if mask.ndim==3:
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
        if ndim(mask)==2 and self.squeeze_channel==False:
            mask=np.expand_dims(mask,-1)
        return mask

    def mask_transform(self, mask_data):
        return self.data_transform(mask_data)

    def data_transform(self, data):
        if len(self.transform_funcs) == 0:
            return mask_backend_adaptive(data, label_mapping=self.class_names, object_type=self.object_type)
        else:
            if isinstance(data, np.ndarray):
                for fc in self.transform_funcs:
                    if hasattr(fc,'_apply_mask'):
                        data = fc._apply_mask(data, spec=self.element_spec)
                    else:
                        data=fc(data)
                data = mask_backend_adaptive(data, label_mapping=self.class_names, object_type=self.object_type)
                return data
            else:
                return data

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
        self.class_names = {}
        if class_names is not None:
            self.class_names = class_names
        self._lab2idx = {}
        self._idx2lab = {}
        language = context._context().locale
        self.__default_language__ = language

        if  labels is not None and len(labels)>0 and isinstance(labels[0],str):
            self.class_names[language]=list(sorted(set(self.items)))
            self.class_names['en-US'] = list(sorted(set(self.items)))
            self._lab2idx = dict(zip(self.class_names[language], range(len(self.class_names[language]))))
            self._idx2lab = dict(zip(range(len(self.class_names[language])), self.class_names[language]))


        self.dtype = np.int64
        self.transform_funcs = []

        shp = None
        if isinstance(self.items[0], numbers.Number):
            shp = TensorShape([None])
        else:
            shp = TensorShape(self.items)
        self._element_spec = TensorSpec(shape=shp, name=self.symbol, object_type=self.object_type, dtype=self.dtype)

    def binding_class_names(self, class_names=None, language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = context._context().locale
            self.class_names[language] = list(class_names)

            self.__default_language__ = language
            self._lab2idx = dict(zip(self.class_names[language], range(len(self.class_names[language]))))
            self._idx2lab = dict(zip(range(len(self.class_names[language])), self.class_names[language]))

    def __getitem__(self, index: int):
        if index >= len(self.items):
            index = index % len(self.items)
        label = self.items[index]
        if  label in self._lab2idx :
            return self._lab2idx[label]
        else:
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
        if index >= len(self.items):
            index = index % len(self.items)
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
        return self.data_transform(bbox)


    def data_transform(self, data):
        if isinstance(data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.transform_funcs:
                if hasattr(fc, '_apply_boxes'):
                    data = fc._apply_boxes(data, spec=self.element_spec)
                else:
                    data = fc(data)
            return data
        else:
            return data




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
        if index >= len(self.items):
            index = index % len(self.items)
        return self.items[index]

    def landmark_transform(self, *landmarks):
        return self.data_transform(landmarks)


    def data_transform(self, data):
        if isinstance(data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.transform_funcs:
                if hasattr(fc, '_apply_keypoints'):
                    data = fc._apply_keypoints(data, spec=self.element_spec)
                else:
                    data = fc(data)
            return data
        else:
            return data


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
        self.length = sys.maxsize

    def data_transform(self, noise_data):
        return noise_data

    def __getitem__(self, index: int):
        if self.random_mode == 'normal':
            return np.random.standard_normal(self.shape)
        elif self.random_mode == 'uniform':
            return np.random.uniform(-1, 1, self.shape)

    def __len__(self):
        return self.length


# class TextSequenceDataset(Dataset):
#     def __init__(self, corpus=None, is_onehot=False, sequence_offset=0, section_delimiter='\n\n', stopwords=None, sequence_length: int = 64, sequence_start_at='random',
#                  object_type=ObjectType.corpus, symbol=None, **kwargs):
#         super().__init__(None,symbol=symbol, object_type=object_type, **kwargs)
#         self.sequence_start_at = sequence_start_at
#         self.transform_funcs = []
#         if len(section_delimiter) == 2:
#             self.section_delimiter = section_delimiter
#         else:
#             self.section_delimiter = '\n\n'
#         self.vocabs = None
#         self.text2index = None
#         self.index2text = None
#         self.is_onehot = is_onehot
#         self.is_paired_process = False
#         self.is_spatial = True
#
#         if hasattr(corpus, "__iter__"):
#             new_corpus = []
#             section = ['<start/>']
#             for i in range(len(corpus)):
#                 item = corpus[i]
#                 if item == '\n' and corpus[i - 1] != '\n' and len(section) > 0:
#                     section.append('<end/>')
#                     if i < len(corpus) - 1 and corpus[i + 1] == '\n':
#                         new_corpus.append(section)
#                         section = ['<start/>']
#                     elif i < len(corpus) - 1:
#                         section.append('<start/>')
#                 elif self.section_delimiter != '\n\n' and len(self.section_delimiter) == 2 and len(section) > 0 and item == self.section_delimiter[0] and i < len(corpus) - 1 and \
#                         corpus[i - 1] == self.section_delimiter[1]:
#                     section.append('<end/>')
#                     if i < len(corpus) - 1 and corpus[i + 1] == '\n':
#                         new_corpus.append(section)
#                         section = ['<start/>']
#                     elif i < len(corpus) - 1:
#                         section.append('<start/>')
#                 elif item == '\r' and corpus[i - 1] == '\n':
#                     pass
#                 elif item == '\n':
#                     pass
#                 else:
#                     section.append(item)
#                     if i == len(corpus) - 1:
#                         section.append('<end/>')
#             if len(section) > 0:
#                 new_corpus.append(section)
#
#             if self.sequence_start_at == 'random':
#                 for sect in new_corpus:
#                     self.items.extend(sect)
#                     self.items.append('\n')
#                     self.items.append('\n')
#             else:
#                 self.items.extend(new_corpus)
#
#             chars = sorted(list(set(corpus)))
#
#             chars.insert(0, '<start/>')
#             chars.insert(1, '<end/>')
#             chars.insert(2, '[UNK]')
#             chars.insert(3, '[PAD]')
#             print('total distinct chars:', len(chars))
#             self.vocabs = chars
#             self.text2index = dict((c, i) for i, c in enumerate(chars))
#             self.index2text = dict((i, c) for i, c in enumerate(chars))
#         else:
#             raise ValueError('corpus should be a collection.')
#
#
#         self.sequence_offset = sequence_offset
#         self.dtype = np.float32 if self.is_onehot else np.int64
#         self.transform_funcs = []
#         self.is_paired_process = False
#         self.sequence_length = sequence_length
#
#     def _get_item_by_idx(self, iterator, idx):
#         """Get the idx-th item of the iterator"""
#         size = len(self)
#         idx = idx.__index__()
#         if not -size <= idx < size:
#             raise IndexError('index {} is out of range'.format(idx))
#         idx %= size
#         return next(itertools.islice(iterator, idx, None))
#
#     def __getitem__(self, index: int):
#         sequencetext = None
#         seq_base = list(self.items)
#         if isinstance(self.sequence_offset, int):
#             if self.sequence_start_at == 'random':
#                 sequencetext = seq_base[index + self.sequence_offset:builtins.min(index + self.sequence_offset + self.sequence_length, self.len())]
#             elif self.sequence_start_at == 'section_start':
#                 sectiontext = seq_base[index]
#                 sequencetext = sectiontext[self.sequence_offset:builtins.min(self.sequence_offset + self.sequence_length, self.len())]
#         elif isinstance(self.sequence_offset, list):
#             sequencetext = []
#             if self.sequence_start_at == 'random':
#                 for k in self.sequence_offset:
#                     if 0 <= index + self.sequence_offset < len(seq_base):
#                         sequencetext.append(seq_base[index + self.sequence_offset])
#                     else:
#                         sequencetext.append('[PAD]')
#
#             elif self.sequence_start_at == 'section_start':
#                 sectiontext = seq_base[index]
#                 for k in self.sequence_offset:
#                     if 0 <= index + self.sequence_offset < len(seq_base):
#                         sequencetext.append(seq_base[index + self.sequence_offset])
#                     else:
#                         sequencetext.append('[PAD]')
#
#         if len(sequencetext) != self.sequence_length:
#             sequencetext.extend(['[PAD]'] * (self.sequence_length - len(sequencetext)))
#         arr = None
#         if self.is_onehot:
#             arr = np.zeros((self.sequence_length, len(self.text2index)))
#             for i in range(self.sequence_length):
#                 this_char = sequencetext[i]
#                 if i + 1 < self.sequence_length and ''.join(sequencetext[i:i + 2]) == '\n\n':
#                     arr[i:, self.text2index['[PAD]']] = 1
#                     break
#                 elif this_char in self.text2index:
#                     arr[i, self.text2index[this_char]] = 1
#                 else:
#                     arr[i, self.text2index['[UNK]']] = 1
#             arr = arr.astype(np.float32)
#         else:
#             arr = np.zeros((self.sequence_length))
#             for i in range(self.sequence_length):
#                 this_char = sequencetext[i]
#                 if i + 1 < self.sequence_length and ''.join(sequencetext[i:i + 2]) == '\n\n':
#                     arr[i:] = self.text2index['[PAD]']
#                     break
#                 elif this_char in self.text2index:
#                     arr[i] = self.text2index[this_char]
#                 else:
#                     arr[i] = self.text2index['[UNK]']
#             arr = arr.astype(np.int64)
#
#         if self.is_paired_process == False and len(self.transform_funcs) == 0:
#             return text_backend_adaption(arr)
#         elif not self.is_paired_process:
#             return self.text_transform(arr)
#         elif self.is_paired_process:
#             return arr
#
#         return None
#
#     def data_transform(self, text_data):
#         if len(self.transform_funcs) == 0:
#             return text_backend_adaption(text_data)
#         if isinstance(text_data, np.ndarray):
#             for fc in self.transform_funcs:
#                 text_data = fc(text_data, spec=self.element_spec)
#             text_data = text_backend_adaption(text_data)
#
#     def text_transform(self, text_data):
#         return self.data_transform(text_data)
#
#     @property
#     def reverse_transform_funcs(self):
#         return_list = []
#         return_list.append(reverse_text_backend_adaption)
#         for i in range(len(self.transform_funcs)):
#             fn = self.transform_funcs[-1 - i]
#             # if fn.__qualname__ == 'normalize.<locals>.text_op':
#             #     return_list.append(unnormalize(fn.mean, fn.std))
#         # return_list.append(array2image)
#         return return_list
#
#     def reverse_text_transform(self, text_data):
#         if len(self.reverse_transform_funcs) == 0:
#             return reverse_text_backend_adaption(text_data)
#         if isinstance(text_data, np.ndarray):
#             # if img_data.ndim>=2:
#             for fc in self.reverse_transform_funcs:
#                 text_data = fc(text_data)
#             text_data = reverse_text_backend_adaption(text_data)
#
#             return text_data
#         else:
#             return text_data
#

class TextSequenceDataset(Dataset):
    def __init__(self, corpus=None, vocabs=None,is_onehot=False, sequence_offset=0, storage_unit='section', section_delimiter='\n', stopwords=None, sequence_length: int = 64,
                 sequence_start_at='random',min_sequence_length=3,stride=1,
                 include_segment_ids=False, include_mask_ids=False, object_type=ObjectType.corpus, symbol=None, **kwargs):
        super().__init__(None, symbol=symbol, object_type=object_type, **kwargs)
        valid_sequence_start_at = ['random', 'slide', 'follow_up', 'section_start','sentence_start',  'word_start', 'next_section_start']
        valid_storage_unit =['section','sentence']
        self.sequence_offset = sequence_offset
        self.stride=stride
        self.sequence_length = sequence_length
        if storage_unit in valid_storage_unit:
            self.storage_unit = storage_unit
        if sequence_start_at in valid_sequence_start_at:
            self.sequence_start_at = sequence_start_at
        else:
            self.sequence_start_at = 'random'
        self.length_index = OrderedDict()
        self.padding_token='[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        self.unknown_token = '[UNK]'
        self.min_sequence_length=min_sequence_length
        self.vocabs_frequency = OrderedDict()
        self.transform_funcs = []
        if len(section_delimiter) == 2:
            self.section_delimiter = section_delimiter
        else:
            self.section_delimiter = '\n'

        self.vocabs = None
        if vocabs is not None:
            self.vocabs =vocabs

        self.text2index = None
        self.index2text = None
        self.is_onehot = is_onehot
        self.is_paired_process = False
        self.is_spatial = True
        self.add_corpus(corpus)


        self.dtype = np.float32 if self.is_onehot else np.int64
        self.is_paired_process = False


    @property
    def element_spec(self):
        return self._element_spec

    @element_spec.setter
    def element_spec(self, value):
        self._element_spec = value

    def add_corpus(self, corpus):
        th=ToHalfWidth()
        if corpus is  not None:
            if isinstance(corpus, str):
                corpus=th(corpus).splitlines()
                self.items.extend(corpus)
            elif hasattr(corpus, "__iter__"):
                corpus=[th(t) for t in corpus if len(t)>self.min_sequence_length]
                self.items.extend(corpus)
            else:
                raise ValueError('corpus should be a collection.')

            if self.vocabs is None:
                if len(self.items)>0:
                    chars = sorted(list(set(''.join(self.items))))
                    chars.insert(0, '[PAD]')
                    chars.insert(1, '[CLS]')
                    chars.insert(2, '[SEP]')
                    chars.insert(3, '[UNK]')
                    chars.insert(4, '[MASK]')
                    print('total distinct chars:', len(chars))
                    self.vocabs = chars

            if self.vocabs  is not None and hasattr(self.vocabs, "__iter__"):
                self.text2index = dict((c, i) for i, c in enumerate(self.vocabs ))
                self.index2text = dict((i, c) for i, c in enumerate(self.vocabs ))
                self.vocabs_frequency = OrderedDict((c, 1) for i, c in enumerate(self.vocabs))
                self.length_index = OrderedDict()

                def start_thread(func, name=None, args=[]):
                    threading.Thread(target=func, name=name, args=args).start()

                def process_statistics():
                    total_len = 0
                    for i in range(len(self.items)):
                        _sentence=self.items[i]

                        for j in range(len(self.items[i])):
                            ch=_sentence[j]
                            if ch not in self.vocabs_frequency:
                                self.vocabs_frequency[ch] = 0
                            self.vocabs_frequency[ch] += 1

                        total_len += (len(self.items[i]) + 2)
                        self.length_index[i] = total_len

                if len(self.items)<5000000:
                    self._element_spec = TensorSpec(shape=TensorShape([None]+[self.sequence_length]),dtype=Dtype.long , name=self.symbol,
                                                    object_type=self.object_type, is_spatial=True)
                    process_statistics()
                else:
                    start_thread(process_statistics, args=[])
    def update_vocabs(self,corpus):
        cnt=len(self.vocabs)
        chars = list(sorted(set(self.vocabs[5:]+list(corpus))))
        chars.insert(0, '[PAD]')
        chars.insert(1, '[CLS]')
        chars.insert(2, '[SEP]')
        chars.insert(3, '[UNK]')
        chars.insert(4, '[MASK]')
        print('total distinct chars: {0}=>{1}', cnt,len(chars))
        self.vocabs = chars
        self.text2index = dict((c, i) for i, c in enumerate(self.vocabs))
        self.index2text = dict((i, c) for i, c in enumerate(self.vocabs))
        for i, c in enumerate(self.vocabs):
            if c not in self.vocabs_frequency:
                self.vocabs_frequency[c] = 1



    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = idx.__index__()
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(itertools.islice(iterator, idx, None))

    def __len__(self):
        if self.sequence_start_at  in [  'random']:
            return self.length_index.value_list[-1]
        elif self.sequence_start_at in [ 'section_start','next_section_start']:
            return len(self.items)

    def __getitem__(self, index: int):
        if self.sequence_start_at == 'next_section_start':
            index = index + 1
        sequencetext = None
        if isinstance(self.sequence_offset, int):
            if self.sequence_start_at in ['random', 'word_start']:
                for k, v in self.length_index.item_list:
                    if v > index:
                        last_v = self.length_index[k - 1] if k > 0 else 0
                        sectiontext = list(self.items[k])
                        sectiontext.insert(0, '[CLS]')
                        is_end = self.items[k].endswith('\n\n')
                        if is_end:
                            sectiontext = sectiontext[:-2]
                        sectiontext.append('[SEP]')
                        if k + 1 < len(self.length_index) and not is_end:
                            sectiontext.append('[CLS]')
                            sectiontext.extend(list(self.items[k + 1]))
                            is_end = self.items[k + 1].endswith('\n\n')
                            if is_end:
                                sectiontext = sectiontext[:-2]
                            sectiontext.append('[SEP]')
                        if k + 2 < len(self.length_index) and not is_end:
                            sectiontext.append('[CLS]')
                            sectiontext.extend(list(self.items[k + 2]))
                            is_end = self.items[k + 2].endswith('\n\n')
                            if is_end:
                                sectiontext = sectiontext[:-2]
                            sectiontext.append('[SEP]')
                        idx = index - last_v

                        sequencetext = sectiontext[idx + self.sequence_offset:builtins.min(idx + self.sequence_offset + self.sequence_length, len(sectiontext))]
                        break
            elif self.sequence_start_at.endswith('section_start'):
                sectiontext = list(self.items[index])
                sectiontext.insert(0, '[CLS]')
                sectiontext.append('[SEP]')
                sequencetext = sectiontext[self.sequence_offset:builtins.min(self.sequence_offset + self.sequence_length, len(sectiontext))]

        return sequencetext

    def data_transform(self, text_data):
        if len(self.transform_funcs) == 0:
            pass

        for fc in self.transform_funcs:
            text_data = fc(text_data, spec=self.element_spec)
        # text_data = text_backend_adaption(text_data)

        arr = None
        if isinstance(text_data, np.ndarray):
            return text_data
        elif self.is_onehot:
            arr = np.zeros((self.sequence_length, len(self.text2index)))
            for i in range(self.sequence_length):
                if i < len(text_data):
                    this_char = text_data[i]

                    if this_char in self.text2index:
                        arr[i, self.text2index[this_char]] = 1
                    else:
                        arr[i, self.text2index['[UNK]']] = 1
                else:
                    arr[i:, self.text2index['[PAD]']] = 1
            arr = arr.astype(np.float32)
        else:
            arr = np.zeros((self.sequence_length))
            for i in range(self.sequence_length):
                if i < len(text_data):
                    this_char = text_data[i]
                    if this_char in self.text2index:
                        arr[i] = self.text2index[this_char]
                    else:
                        arr[i] = self.text2index['[UNK]']
                else:
                    arr[i] = self.text2index['[PAD]']
            arr = arr.astype(np.int64)
        return arr

    def text_transform(self, text_data):
        return self.data_transform(text_data)

    @property
    def reverse_transform_funcs(self):
        return_list = []
        return_list.append(reverse_text_backend_adaption)
        for i in range(len(self.transform_funcs)):
            fn = self.transform_funcs[-1 - i]
            # if fn.__qualname__ == 'normalize.<locals>.text_op':
            #     return_list.append(unnormalize(fn.mean, fn.std))
        # return_list.append(array2image)
        return return_list

    def reverse_text_transform(self, text_data):
        if len(self.reverse_transform_funcs) == 0:
            return reverse_text_backend_adaption(text_data)
        if isinstance(text_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_transform_funcs:
                text_data = fc(text_data)
            text_data = reverse_text_backend_adaption(text_data)

            return text_data
        else:
            return text_data


class Iterator(object):
    def __init__(self, data=None, label=None, unpair=None, sample_filter=None, batch_size=8, mode='tuple', is_shuffle=True, buffer_size=None, workers=2, **kwargs):
        self.parent =None
        self.is_paired_process = False
        self._data = None
        self._label = None
        self._unpair = None
        self.paired_process_symbols = []
        self._is_signature_update=False

        self.is_shuffle = is_shuffle
        self.memory_cache = []

        self.data_template = OrderedDict()
        self.mode = mode

        self.workers = workers
        self.pass_cnt = 0
        self.pass_time_spend = 0
        if data is not None and isinstance(data, tuple):
            self._data = Dataset.zip(*data)
            self._data.parent = self

        elif data is not None and isinstance(data, (Dataset, ZipDataset, TextSequenceDataset)):
            self._data = data
            self._data.parent = self
            if isinstance(data,ZipDataset):
                for ds in data.items:
                    ds.parent = self

        if label is not None and isinstance(label, tuple):
            self._label = Dataset.zip(*label)
            self._label.parent = self
        elif label is not None and (inspect.isgenerator(label) or isinstance(label, (Dataset, ZipDataset, TextSequenceDataset))):
            self._label = label
            self._label.parent = self
            if isinstance(label,ZipDataset):
                for ds in label.items:
                    ds.parent = self

        if unpair is not None and isinstance(unpair, Dataset):
            self._unpair = unpair
            self._unpair.parent = self
            if isinstance(unpair,ZipDataset):
                for ds in unpair.items:
                    ds.parent = self

        if self._data is None:
            self._data = NumpyDataset(symbol="")
        if self._label is None:
            self._label = NumpyDataset(symbol="")
        if self._unpair is None:
            self._unpair = NumpyDataset(symbol="")
        self._is_signature_update = False
        datasets = self.get_datasets()

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

        for k in range(len(datasets)):
            ds = datasets[k]
            if isinstance(ds, TextSequenceDataset):
                ds.element_spec = TensorSpec(shape=TensorShape([None, ds.sequence_length]), dtype=Dtype.long, object_type=ds.object_type, name=ds.symbol)
            elif ds.object_type == ObjectType.image_path:
                ds.element_spec = TensorSpec(shape=TensorShape([None]), dtype=str, object_type=ds.object_type, name=ds.symbol)
            else:
                if len(ds) > 0:
                    dataitem = ds[k]
                    if isinstance(dataitem,np.ndarray):
                        ds.element_spec = TensorSpec.tensor_to_spec(expand_dims(dataitem, 0), object_type=ds.object_type, name=ds.symbol)
                    elif isinstance(dataitem,numbers.Number):
                        ds.element_spec =TensorSpec(shape=TensorShape([None,1]),object_type=ObjectType.regression_label,name=ds.symbol)
                    else:
                        pass

            self.data_template[ds.element_spec] = None

        self._batch_size = batch_size
        self.paired_transform_funcs = []
        self.batch_transform_funcs = []

        self.batch_sampler = BatchSampler(self, self._batch_size, is_shuffle=self.is_shuffle, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)
        if buffer_size is None:
            buffer_size = 8 * batch_size
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
        self._data.parent=self
        self._is_signature_update = False

        if self._label is not None and isinstance(self._label, (MaskDataset, BboxDataset, ImageDataset)) and isinstance(self._data, ImageDataset) and len(self._label) == len(
                self._data):
            self._label.is_paired_process = self._data.is_paired_process = self.is_paired_process = True
        else:
            self._label.is_paired_process = self._data.is_paired_process = self.is_paired_process = False

        self.batch_sampler = BatchSampler(self, self._batch_size, is_shuffle=self.is_shuffle, drop_last=False)
        self.batch_sampler.sample_filter = self.sample_filter
        self._sample_iter = iter(self.batch_sampler)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value
        self._label.parent = self
        self._is_signature_update = False
        if isinstance(self._label, (MaskDataset, ImageDataset, BboxDataset)) and isinstance(self._data, ImageDataset) and len(self._label) == len(self._data):
            self._label.is_paired_process = self._data.is_paired_process = self.is_paired_process = True

        else:
            self._label.is_paired_process = self._data.is_paired_process = self.is_paired_process = False
        self.batch_sampler = BatchSampler(self, self._batch_size, is_shuffle=self.is_shuffle, drop_last=False)
        self.batch_sampler.sample_filter = self.sample_filter
        self._sample_iter = iter(self.batch_sampler)

    @property
    def unpair(self):
        return self._unpair

    @unpair.setter
    def unpair(self, value):
        self._unpair = value
        self._unpair.parent = self
        self._is_signature_update = False
        self.batch_sampler = BatchSampler(self, self._batch_size, is_shuffle=self.is_shuffle, drop_last=False)
        self.batch_sampler.sample_filter = self.sample_filter
        self._sample_iter = iter(self.batch_sampler)

    @property
    def palette(self):
        if isinstance(self._label, MaskDataset) and self._label.object_type in [ObjectType.label_mask, ObjectType.color_mask]:
            return self._label.palette
        else:
            return None

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        self.batch_sampler = BatchSampler(self, self._batch_size, is_shuffle=self.is_shuffle, drop_last=False)
        self.batch_sampler.sample_filter = self.sample_filter
        self._sample_iter = iter(self.batch_sampler)
        self.buffer_size = 8 * value
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)
        self.pass_cnt = 0
        self.pass_time_spend = 0.

    @property
    def signature(self):
        datasets = self.get_datasets()
        if self._signature is not None and len(self._signature.outputs) == len(datasets):
            return self._signature
        else:
            self._signature = Signature(name='data_provider')
            for ds in datasets:
                spec = ds.element_spec
                self.data_template[spec] = None
                self._signature.outputs[ds.symbol] = spec
            self.batch_sampler = BatchSampler(self, self._batch_size, is_shuffle=self.is_shuffle, drop_last=False)
            self._sample_iter = iter(self.batch_sampler)
            return self._signature

    def paired_transform(self, datadict: Dict[TensorSpec, np.ndarray]):
        if len(self.paired_transform_funcs) == 0:
            return datadict

        # if img_data.ndim>=2:
        for fc in self.paired_transform_funcs:
            try:
                datadict = fc(datadict)
            except:
                PrintException()
        return datadict



    def batch_transform(self, datadict: Dict[TensorSpec, np.ndarray]):
        if len(self.batch_transform_funcs) == 0:
            return datadict
        elif len(self.batch_transform_funcs) > 0:
                # if img_data.ndim>=2:
            for fc in self.batch_transform_funcs:
                try:
                    if hasattr(fc, 'memory_cache'):
                        fc.memory_cache = self.memory_cache

                    #elif fc.__class__.__name__ in ['OneOf', 'Compose']:
                    datadict = fc(datadict)

                except:
                    PrintException()
            return datadict

    def get_datasets(self):
        datasets = []
        if self._data is not None and isinstance(self._data, Dataset) and not isinstance(self._data, ZipDataset) and len(self._data) > 0:
            datasets.append(self._data)
        elif self._data and isinstance(self._data, ZipDataset):
            for ds in self._data.items:
                datasets.append(ds)
        if self._label and isinstance(self._label, Dataset) and not isinstance(self._label, ZipDataset) and len(self._label) > 0:
            datasets.append(self._label)
        elif self._label and isinstance(self._label, ZipDataset):
            for ds in self._label.items:
                datasets.append(ds)
        if self._unpair and isinstance(self._unpair, Dataset) and len(self._unpair) > 0:
            datasets.append(self._unpair)
        elif isinstance(self._unpair, ZipDataset):
            for ds in self._unpair.items:
                datasets.append(ds)
        return datasets

    def find_dataset(self, dataset_symbol: str):
        datasets = self.get_datasets()
        for ds in datasets:
            if ds.symbol == dataset_symbol:
                return ds

    def update_signature(self):
        datasets = self.get_datasets()
        results = self[0]
        self.data_template.clear()
        self._signature = Signature(name='data_provider')

        for ds, result, value in zip(datasets, results.key_list, results.value_list):
            spec = copy.deepcopy(ds._element_spec)
            if spec.name == ds.symbol:
                if isinstance(value, np.ndarray):
                    dtype = DTYPE_MAPPING[value.dtype.type]
                elif isinstance(value, numbers.Integral):
                    dtype = numbers.Integral
                elif isinstance(value, numbers.Real):
                    dtype = numbers.Real
                spec.shape = tensor_to_shape(to_tensor(value), need_exclude_batch_axis=True, is_singleton=True)
                spec.object_type = ds.object_type
                spec.dtype = dtype

                ds._element_spec = spec
                self.data_template[spec] = None
                self._signature.outputs[ds.symbol] = spec
        self.batch_sampler = BatchSampler(self, self._batch_size, is_shuffle=self.is_shuffle, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)
        return self._signature

    def print_statistics(self):
        print('avg. process time: {0:.5f}'.format(self.pass_time_spend / float(builtins.max(1, self.pass_cnt))))
        print('avg. batch time: {0:.5f}'.format(self._batch_size * self.pass_time_spend / float(builtins.max(1, self.pass_cnt))))

    def __getitem__(self, index: int):
        start_time = time.time()

        try:

            returnData = OrderedDict()  # copy.deepcopy(self.data_template)

            data = self.data.__getitem__(index%len(self.data)) if self.data is not None and len(self.data) > 0 else None
            label = self.label.__getitem__(index%len(self.data)) if self.label is not None and len(self.label) > 0 else None
            unpair = self.unpair.__getitem__(index%len(self.unpair)) if self.unpair is not None and len(self.unpair) > 0 else None
            #((x1,x2),(x3),(x4,x5))=>(x1,x2,x3,x4,x5)
            results = iteration_tools.flatten((data, label, unpair), iterable_types=(tuple))
            #remove none
            results = tuple([item for item in results if item is not None])

            if len(returnData) > 0 and len(returnData) != len(self.get_datasets()):
                raise ValueError("Flattened data should have same length as datasets")

            for n in range(len(self.get_datasets())):
                ds = self.get_datasets()[n]
                spec = ds.element_spec
                returnData[spec] = results[n]

            if len(self.paired_process_symbols) > 0:

                returnData = self.paired_transform(returnData)

            # for batch transform cache data in memory
            if len(self.batch_transform_funcs) > 0:
                for trans in self.batch_transform_funcs:
                    if is_instance(trans,"VisionTransform"):
                        trans.memory_cache=self.memory_cache
                self.memory_cache.append(copy.deepcopy(returnData))
                if len(self.memory_cache) > self.batch_size*2:
                    self.memory_cache.pop(0)
                if len(self.memory_cache) >=4:
                    returnData = self.batch_transform(returnData)



            # if len(self.parent.image_transform_funcs) > 0:
                # for fc in self.parent.image_transform_funcs:
                #     try:
                #         if
                #         datadict = fc(datadict)
                #     except:
                #         PrintException()
                # return datadict

            # for i in range(len(returnData)):
            #     ds = self.get_datasets()[i]
            #     if returnData.value_list[i] is None:
            #         pass
            #     else:
            #         returnData[ds.element_spec] = unpack_singleton(ds.data_transform(returnData.value_list[i]))
            def process_data_transform(i):
                ds = self.get_datasets()[i]
                returnData[ds.element_spec] = unpack_singleton(ds.data_transform(returnData.value_list[i]))

            threads = []
            for i in range(len(returnData)):
                threads.append(threading.Thread(target=process_data_transform, args=(i,)))
                threads[i].start()
            for i in range(len(returnData)):
                threads[i].join()


            if self.signature is None or len(self.signature.outputs) == 0 or len(self._signature.outputs)!=len(self.get_datasets())  or self._is_signature_update == False:
                self._signature = Signature(name='data_provider')
                for spec,ds in zip(returnData.key_list,self.get_datasets()):
                    data_value=returnData[spec]
                    if data_value is not None:
                        new_spec = copy.deepcopy(spec)
                        new_spec.shape=tensor_to_shape(data_value,need_exclude_batch_axis=True,is_singleton=True)
                        new_spec.object_type=ds.object_type
                        self._signature.outputs[ds.symbol] =new_spec
                        ds.element_spec=new_spec
                        self.data_template[ds.element_spec] = None


                self._is_signature_update=True

            if self._is_signature_update:
                self.data_template=OrderedDict()
                for ds in self.get_datasets():
                    self.data_template[ds.element_spec]=None

            self.pass_cnt += 1
            self.pass_time_spend += float(time.time() - start_time)

            return returnData
        except Exception as e:
            print(e)
            PrintException()

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name in ['_data', '_label', '_unpair']:
            self._signature = None
        if name == '_unpair':
            self._signature = None
            if isinstance(value, RandomNoiseDataset):
                value.length = len(self.data)

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

    def __init__(self, data=None, label=None, batch_size=8):
        super().__init__(data, batch_size)
        self.is_paired_process = False
        self._signature = None
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

        self._batch_size = batch_size
        self.paired_transform_funcs = []
        self.batch_sampler = BatchSampler(self, self._batch_size, is_shuffle=self.is_shuffle, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)
        self.buffer_size = 10
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)

    def __getitem__(self, index: int):
        # start = time.time()

        try:

            anchor = self.data.__getitem__(index % len(self.data)) if self.data is not None and len(self.data) > 0 else None
            positive = self.data.__getitem__(index% len(self.data)) if self.data is not None and len(self.data) > 0 else None
            label = self.label.__getitem__(index % len(self.label)) if self.label is not None and len(self.label) > 0 else None

            available_list = list(range(len(self.data)))
            available_list.remove(index % len(self.data))

            negative = self.data.__getitem__(random.choice(available_list)) if self.data is not None and len(self.data) > 0 else None

            return_data = []
            if self.signature is None or len(self.signature) == 0:
                self._signature = Signature()
                self._signature.name = 'data_provider'
                if anchor is not None:
                    self._signature.outputs['anchor'] = (-1,) + anchor.shape
                if positive is not None:
                    self._signature.outputs['positive'] = (-1,) + positive.shape
                if negative is not None:
                    self._signature.outputs['negative'] = (-1,) + negative.shape
                if label is not None:
                    self._signature.outputs['label'] = (-1,)

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

