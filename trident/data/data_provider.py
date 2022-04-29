from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import builtins
import copy
import inspect
import itertools
import locale
import numbers
import os
import random
import string
import uuid
import warnings
from typing import List, Iterable

import numpy as np
from trident import context

from trident.misc.ipython_utils import is_in_ipython

from trident.data.vision_transforms import Unnormalize, Normalize
from trident.data.image_common import image_backend_adaption,reverse_image_backend_adaption
from trident.data.transform import Transform

try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve
from trident.backend.common import *
from trident.backend.tensorspec import *
from trident.backend.opencv_backend import *
from trident.data.text_common import *
from trident.data.image_common import *
from trident.data.label_common import *
from trident.data.samplers import *
from trident.data.dataset import *
from trident import context

ctx = context._context()
_trident_dir = get_trident_dir()

__all__ = ['ImageDataProvider', 'DataProvider', 'TextSequenceDataProvider']


class ImageDataProvider(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset_name='', traindata=None, testdata=None, batch_size=8, mode='tuple', **kwargs):
        self.__name__ = dataset_name
        self.dynamic_padding=False
        self.uuid = uuid.uuid4().node
        self.traindata = traindata
        if isinstance(self.traindata, Iterator):
            self.traindata.parent = self

        self.testdata = testdata
        if isinstance(self.testdata, Iterator):
            self.testdata.parent = self

        self.annotations = {}

        self.scenario = 'train'
        self._class_names = {}
        if mode in ['tuple', 'dict']:
            self.mode = mode
        else:
            raise ValueError("Valid mode should be tuple or dict ")

        self._batch_size = batch_size
        self.__default_language__ = ctx.locale
        if len(self._class_names) > 0:
            if ctx.locale in self._class_names:
                self.__default_language__ = ctx.locale
            for k in self._class_names.keys():
                if ctx.locale.split('-')[0] in k:
                    self.__default_language__ = k
                    break

        self._idx2lab = {}
        self._lab2idx = {}

        self.tot_minibatch = 0
        self.tot_records = 0
        self.tot_epochs = 0
        self._image_transform_funcs = []
        self._label_transform_funcs = []
        self._paired_transform_funcs = []
        self._batch_transform_funcs = []
        cxt = context._context()
        cxt.regist_data_provider(self)

    @property
    def signature(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.signature
        elif self.traindata is not None:
            return self.traindata.signature
        else:
            return None

    def update_signature(self, arg_names):
        if self.scenario == 'test' and self.testdata is not None:
            self.testdata.update_signature(arg_names)
            print(self.testdata.signature)
        elif self.traindata is not None:
            self.traindata.update_signature(arg_names)
            print(self.traindata.signature)
        else:
            return None

    @property
    def batch_sampler(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.batch_sampler
        elif self.traindata is not None:
            return self.traindata.batch_sampler
        else:
            return []

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        if self.traindata is not None and hasattr(self.traindata, '_batch_size'):
            self.traindata.batch_size = self._batch_size
        if self.testdata is not None and hasattr(self.testdata, '_batch_size'):
            self.testdata.batch_size = self._batch_size

    @property
    def palette(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.palette
        elif self.traindata is not None:
            return self.traindata.palette
        else:
            return None

    @property
    def image_transform_funcs(self):
        return self._image_transform_funcs

    @image_transform_funcs.setter
    def image_transform_funcs(self, value):
        value = [] if value is None else value
        self._image_transform_funcs = value
        if self.traindata is not None and hasattr(self.traindata.data, 'transform_funcs'):
            self.traindata._is_signature_update = False
            dss = self.traindata.get_datasets()
            for ds in dss:
                if isinstance(ds, ImageDataset):
                    ds.transform_funcs = value
                if value is not None and hasattr(value, '__getitem__') and isinstance(ds, (
                BboxDataset, MaskDataset, LandmarkDataset)):
                    ds.transform_funcs = [t for t in ds.transform_funcs if
                                          not hasattr(t, 'is_spatial') or t.is_spatial == False]
                    for t in list(range(len(value)))[::-1]:
                        if isinstance(value[t], Transform) and hasattr(value[t], 'is_spatial') and value[t].is_spatial:
                            ds.transform_funcs.insert(0, value[t])

        if self.testdata is not None and len(self.testdata.data) > 0 and hasattr(self.testdata.data, 'transform_funcs'):
            dss_t = self.testdata.get_datasets()
            for ds in dss_t:
                if isinstance(ds, ImageDataset):
                    ds.transform_funcs = value
                if value is not None and hasattr(value, '__getitem__') and isinstance(ds, (
                BboxDataset, MaskDataset, LandmarkDataset)):
                    ds.transform_funcs = [t for t in ds.transform_funcs if
                                          not hasattr(t, 'is_spatial') or t.is_spatial == False]
                    for t in list(range(len(value)))[::-1]:
                        if isinstance(value[t], Transform) and hasattr(value[t], 'is_spatial') and value[t].is_spatial:
                            ds.transform_funcs.insert(0, value[t])

    def image_transform(self, img_data):
        if img_data.ndim == 4:
            return np.asarray([self.image_transform(im) for im in img_data])
        if len(self.image_transform_funcs) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self.image_transform_funcs:
                if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not image_backend_adaption:
                    img_data = fc(img_data)
            img_data = image_backend_adaption(img_data)

            return img_data
        else:
            return img_data

    @property
    def reverse_image_transform_funcs(self):
        return_list = [reverse_image_backend_adaption]
        if self.image_transform_funcs is None or self.image_transform_funcs == []:
            dss = self.traindata.get_datasets()
            for ds in dss:
                if isinstance(ds, ImageDataset) and len(ds.transform_funcs) > 0:
                    self.image_transform_funcs = ds.transform_funcs
                    break
        for i in range(len(self.image_transform_funcs)):
            fn = self.image_transform_funcs[-1 - i]
            if (inspect.isfunction(fn) and fn.__qualname__ == 'normalize.<locals>.img_op') or isinstance(fn, Normalize):
                return_list.append(Unnormalize(fn.mean, fn.std))
        # return_list.append(array2image)
        return return_list

    def reverse_image_transform(self, img_data: np.ndarray):
        if img_data.ndim == 4:
            return np.array([self.reverse_image_transform(im) for im in img_data])
        if len(self.reverse_image_transform_funcs) == 0:
            return reverse_image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            img_data = reverse_image_backend_adaption(img_data)
            # if img_data.ndim>=2:
            for fc in self.reverse_image_transform_funcs:
                img_data = fc(img_data)

        return img_data

    @property
    def label_transform_funcs(self):
        return self._label_transform_funcs

    @label_transform_funcs.setter
    def label_transform_funcs(self, value):
        self._label_transform_funcs = value
        if self.traindata is not None and hasattr(self.traindata.label, 'label_transform_funcs'):
            self.traindata.label.label_transform_funcs = self._label_transform_funcs
            self.traindata.update_signature()
        if self.testdata is not None and hasattr(self.testdata.label, 'label_transform_funcs'):
            self.testdata.label.label_transform_funcs = self._label_transform_funcs
            self.testdata.update_signature()

    @property
    def paired_transform_funcs(self):
        return self._paired_transform_funcs

    @paired_transform_funcs.setter
    def paired_transform_funcs(self, value):
        self._paired_transform_funcs = value

        if self.traindata is not None and hasattr(self.traindata, 'paired_transform_funcs'):
            self.traindata.paired_transform_funcs = value
            self.traindata._is_signature_update = False

        if self.testdata is not None and hasattr(self.testdata, 'paired_transform_funcs'):
            self.testdata.paired_transform_funcs = value
            self.testdata._is_signature_update = False

    @property
    def batch_transform_funcs(self):
        return self._batch_transform_funcs

    @batch_transform_funcs.setter
    def batch_transform_funcs(self, value):
        self._batch_transform_funcs = value
        self.traindata.batch_transform_funcs = value
        if self.testdata is not None:
            self.traindata.batch_transform_funcs = value

    def batch_transform(self, batchdata):
        if hasattr(self, '_batch_transform_funcs') and len(self._batch_transform_funcs) > 0:

            if isinstance(batchdata, tuple):
                new_batchdata = OrderedDict()
                for ds in self.traindata.get_datasets():
                    new_batchdata[ds.element_spec] = None

            if isinstance(batchdata, OrderedDict):
                if not all([isinstance(k, TensorSpec) for k in batchdata.key_list]):
                    new_batchdata = copy.deepcopy(self.traindata.data_template)
                    for i in range(len(batchdata)):
                        new_batchdata[new_batchdata.key_list[i]] = batchdata[batchdata.key_list[i]]
                    batchdata = new_batchdata

                for trans in self._batch_transform_funcs:
                    batchdata = trans(batchdata)
                return batchdata
            else:
                return batchdata

        else:
            return batchdata

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, value):
        self._class_names = value
        if self.traindata is not None and hasattr(self.traindata.label, 'class_names'):
            self.traindata.label.class_names = self._class_names
        if self.testdata is not None and hasattr(self.testdata.label, 'class_names'):
            self.testdata.label.class_names = self._class_names

    def preview_images(self, key=None, is_concate=True):
        image_ds = [ds.symbol for ds in self.traindata.get_datasets() if
                    isinstance(ds, ImageDataset) and ds.object_type != ObjectType.image_path]
        if len(image_ds) == 0:
            print(red_color('This data_provider not have any ImageDataset in it.'))
            return None
        else:
            if key is None:
                orig_mode = self.mode
                self.mode = 'dict'
                data = self.next()
                self.mode = orig_mode
                return_images = OrderedDict()
                for k, v in data.items():
                    if k.name in image_ds:
                        batch_imgs = v
                        batch_imgs = self.reverse_image_transform(batch_imgs)

                        batch_imgs = np.concatenate([img for img in batch_imgs], axis=-1 if batch_imgs[0].ndim == 2 or (
                                    batch_imgs[0].ndim == 3 and batch_imgs[0].shape[0] in [1, 3, 4]) else -2)
                        return_images[k.name] = batch_imgs
                if is_in_ipython():
                    for k, v in return_images.items():
                        print(blue_color(k), flush=True)
                        from IPython import display
                        display.display(array2image(v))

                else:
                    return return_images


            elif isinstance(key, slice):
                start = key.start if key.start is not None else 0
                stop = key.stop
                results = []
                for k in range(start, stop, 1):
                    img = self.traindata.data.__getitem__(k)

                    if isinstance(img, np.ndarray):
                        for fc in self.image_transform_funcs:
                            if (inspect.isfunction(fc) or isinstance(fc,
                                                                     Transform)) and fc is not image_backend_adaption and fc is not Normalize and fc is not normalize:
                                img = fc(img)
                    results.append(img)
                if is_concate:
                    results = np.concatenate(results, axis=-1 if results[0].ndim == 2 or (
                                results[0].ndim == 3 and results[0].shape[0] in [1, 3, 4]) else -2)
                    return array2image(results)
                else:
                    return [array2image(img) for img in results]
            elif isinstance(key, int):
                img = self.traindata.data.__getitem__(key)
                if isinstance(img, np.ndarray):
                    for fc in self.image_transform_funcs:
                        if (inspect.isfunction(fc) or isinstance(fc,
                                                                 Transform)) and fc is not image_backend_adaption and fc is not Normalize and fc is not normalize:
                            img = fc(img)
                return array2image(img)

    def label_statistics(self):
        if isinstance(self.traindata.label, LabelDataset):
            unique, counts = np.unique(np.array(self.traindata.label.items), return_counts=True)
            class_names_mapping = self.class_names[list(self.class_names.keys())[0]]

            unique = [class_names_mapping[s] if s in class_names_mapping else class_names_mapping[int(s)] if isinstance(
                class_names_mapping, list) else str(s) for s in unique]
            max_len = get_string_actual_length(unique) + 5
            for i in range(len(unique)):
                s = unique[i]
                current_name = class_names_mapping[str(unique[i])] if not isinstance(s,
                                                                                     numbers.Integral) and isinstance(
                    class_names_mapping, dict) and unique[i] in class_names_mapping else class_names_mapping[
                    int(s)] if isinstance(s, numbers.Integral) and isinstance(class_names_mapping, list) and isinstance(
                    s, numbers.Integral) < len(class_names_mapping) else str(unique[i])
                # class_name = ''.join([' '] * max_len) if class_names_mapping is None or len(class_names_mapping) == 0 else self.index2label(unique[i]) + ''.join([' '] * (max_len - get_string_actual_length(self.index2label(unique[i]))))
                bar = ['█'] * int(builtins.round(50 * counts[i] / float(len(self.traindata.label.items))))
                print('{0:<10} {1} {2:<50} {3:,} ({4:.3%})'.format(s, current_name, ''.join(bar), counts[i],
                                                                   counts[i] / float(len(self.traindata.label.items))))

    def _next_index(self):
        return self.__next__()

    def __iter__(self):

        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata._sample_iter
        else:
            return self.traindata._sample_iter

    def __len__(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.__len__()
        elif self.traindata is not None:
            return self.traindata.__len__()
        else:
            return 0

    def next(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.next()

        else:
            return self.traindata.next()

    def __next__(self):
        if self.scenario == 'test' and self.testdata is not None:
            return next(self.testdata)
        else:
            return next(self.traindata)

    def next_train(self):
        return self.batch_transform(self.traindata.next())

    def next_test(self):
        if self.testdata is not None:
            return self.batch_transform(self.testdata.next())

        else:
            return None

    def get_all_data(self, is_shuffle=False, topk=-1):
        idxes = np.arange(len(self.traindata.data))
        if is_shuffle:
            np.random.shuffle(idxes)
        data = []
        if topk == -1:
            topk = len(self.traindata.data)

        for i in range(topk):
            data.append(self.traindata.data.__getitem__(idxes[i]))

        return data

    def binding_class_names(self, class_names=None, language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = ctx.locale
            self.class_names[language] = list(class_names)
            print('Mapping class_names  in {0}   success, total {1} class names added.'.format(language,
                                                                                               len(class_names)))
            self.__default_language__ = language
            self._lab2idx = {v: k for k, v in enumerate(self.class_names[language])}
            self._idx2lab = {k: v for k, v in enumerate(self.class_names[language])}

            if self.traindata is not None and hasattr(self.traindata.label, 'class_names'):
                self.traindata.label.binding_class_names(class_names, language)
            if self.testdata is not None and hasattr(self.testdata.label, 'class_names'):
                self.testdata.label.binding_class_names(class_names, language)

    def change_language(self, lang):
        self.__default_language__ = lang
        if self.class_names is None or len(self.class_names.items()) == 0 or lang not in self.class_names:
            warnings.warn('You dont have {0} language version class names', category='mapping', stacklevel=1,
                          source=self.__class__)
        else:
            self._lab2idx = {v: k for k, v in enumerate(self.class_names[lang])}
            self._idx2lab = {k: v for k, v in enumerate(self.class_names[lang])}

    def index2label(self, idx: (int, str)):
        if isinstance(idx, str) and idx.isnumeric():
            idx = int(idx)
        if self._idx2lab is None or len(self._idx2lab.items()) == 0:
            raise ValueError('You dont have proper mapping class names')
        elif isinstance(self._idx2lab, dict) and idx in self._idx2lab:
            return self._idx2lab[idx]
        elif isinstance(self._idx2lab, list) and idx < len(self._idx2lab):
            return self._idx2lab[idx]
        elif idx not in self._idx2lab:
            raise ValueError('Index :{0} is not exist in class names'.format(idx))
        else:
            return self._idx2lab[idx]

    def label2index(self, label):
        if self._lab2idx is None or len(self._lab2idx.items()) == 0:
            raise ValueError('You dont have proper mapping class names')
        elif label not in self._lab2idx:
            raise ValueError('label :{0} is not exist in class names'.format(label))
        else:
            return self._lab2idx[label]

    def get_language(self):
        return self.__default_language__

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name == 'traindata' or name == 'testdata':
            if isinstance(value, Iterator):
                object.__getattribute__(self, name).parent = self
                # self.traindata.parent=self

        if name == 'mode':
            if isinstance(self.traindata, Iterator):
                self.traindata.mode = value
                self.traindata.batch_sampler.mode = value
            if isinstance(self.testdata, Iterator):
                self.testdata.mode = value
                self.testdata.batch_sampler.mode = value


DataProvider = ImageDataProvider


class TextSequenceDataProvider(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset_name='', traindata=None, testdata=None, batch_size=8, sequence_length=64,
                 dynamic_padding=False, mode='tuple', **kwargs):
        self.__name__ = dataset_name
        self._sequence_length = sequence_length
        self.uuid = uuid.uuid4().node
        if mode in ['tuple', 'dict']:
            self.mode = mode
        else:
            raise ValueError("Valid mode should be tuple or dict ")
        self._text_transform_funcs = []
        self._label_transform_funcs = []
        self._paired_transform_funcs = []
        self._batch_transform_funcs = []
        cxt = context._context()
        cxt.regist_data_provider(self)

        self.traindata = traindata
        if isinstance(self.traindata, Iterator):
            self.traindata.parent = self
        self.testdata = testdata
        if isinstance(self.testdata, Iterator):
            self.testdata.parent = self
        self.dynamic_padding = dynamic_padding
        if isinstance(self.traindata, Iterator):
            self.traindata.batch_sampler.dynamic_padding = self.dynamic_padding
            for ds in self.traindata.get_datasets():
                if isinstance(ds, TextSequenceDataset):
                    ds.sequence_length = self.sequence_length
                    ds.dynamic_padding = dynamic_padding
        if isinstance(self.testdata, Iterator):
            self.testdata.batch_sampler.dynamic_padding = self.dynamic_padding
            for ds in self.testdata.get_datasets():
                if isinstance(ds, TextSequenceDataset):
                    ds.sequence_length = self.sequence_length
                    ds.dynamic_padding = dynamic_padding

        self.annotations = {}

        self.scenario = 'train'
        self._class_names = {}

        self._batch_size = batch_size
        self.__default_language__ = 'en-us'
        if len(self._class_names) > 0:
            if ctx.locale in self._class_names:
                self.__default_language__ = ctx.locale
            for k in self._class_names.keys():
                if ctx.locale.split('-')[0] in k:
                    self.__default_language__ = k
                    break

        self._idx2lab = {}
        self._lab2idx = {}

        self.tot_minibatch = 0
        self.tot_records = 0
        self.tot_epochs = 0

    @property
    def signature(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.signature
        elif self.traindata is not None:
            return self.traindata.signature
        else:
            return None

    def update_signature(self, arg_names):
        if self.scenario == 'test' and self.testdata is not None:
            self.testdata.update_signature(arg_names)
            print(self.testdata.signature)
        elif self.traindata is not None:
            self.traindata.update_signature(arg_names)
            print(self.traindata.signature)
        else:
            return None

    @property
    def batch_sampler(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.batch_sampler
        elif self.traindata is not None:
            return self.traindata.batch_sampler
        else:
            return []

    @property
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value
        datasets = self.traindata.get_datasets()
        for ds in datasets:
            if isinstance(ds, TextSequenceDataset):
                ds.sequence_length = value
        if hasattr(self, '_text_transform_funcs') and len(self._text_transform_funcs) > 0:
            for tm in self._text_transform_funcs:
                if isinstance(tm, VocabsMapping):
                    tm.sequence_length = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        if self.traindata is not None and hasattr(self.traindata, '_batch_size'):
            self.traindata.batch_size = self._batch_size
        if self.testdata is not None and hasattr(self.testdata, '_batch_size'):
            self.testdata.batch_size = self._batch_size

    @property
    def palette(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.palette
        elif self.traindata is not None:
            return self.traindata.palette
        else:
            return None

    @property
    def text_transform_funcs(self):
        return self._text_transform_funcs

    @text_transform_funcs.setter
    def text_transform_funcs(self, value):
        self._text_transform_funcs = value
        if self.traindata is not None and hasattr(self.traindata.data, 'text_transform_funcs'):
            self.traindata.data.text_transform_funcs = self._text_transform_funcs
            if len(self.traindata.unpair) > 0:
                self.traindata.unpair.text_transform_funcs = self._text_transform_funcs
        if self.testdata is not None and len(self.testdata.data) > 0 and hasattr(self.testdata.data,
                                                                                 'text_transform_funcs'):
            self.testdata.data.text_transform_funcs = self._text_transform_funcs
        if self.testdata is not None and len(self.testdata.data) > 0 and len(self.testdata.unpair) > 0:
            self.testdata.unpair.text_transform_funcs = self._text_transform_funcs

    def data_transform(self, text_data):
        if text_data.ndim == 4:
            return [self.data_transform(im) for im in text_data]
        if len(self._text_transform_funcs) == 0:
            return text_backend_adaption(text_data)
        if isinstance(text_data, np.ndarray):
            for fc in self._text_transform_funcs:
                if not fc.__qualname__.startswith(
                        'random_') or 'crop' in fc.__qualname__ or 'rescale' in fc.__qualname__ or (
                        fc.__qualname__.startswith('random_') and random.randint(0, 10) % 2 == 0):
                    text_data = fc(text_data)

            text_data = text_backend_adaption(text_data)

            return text_data
        else:
            return text_data

    def text_transform(self, text_data):
        return self.data_transform(text_data)

    @property
    def reverse_text_transform_funcs(self):
        return_list = []
        return_list.append(reverse_text_backend_adaption)
        for i in range(len(self.text_transform_funcs)):
            fn = self.text_transform_funcs[-1 - i]
            if fn.__qualname__ == 'normalize.<locals>.text_op':
                return_list.append(Unnormalize(fn.mean, fn.std))
        # return_list.append(array2text)
        return return_list

    def reverse_text_transform(self, text_data: np.ndarray):
        if len(self.reverse_text_transform_funcs) == 0:
            return reverse_text_backend_adaption(text_data)
        if isinstance(text_data, np.ndarray):
            # if text_data.ndim>=2:
            for fc in self.reverse_text_transform_funcs:
                text_data = fc(text_data)
            text_data = reverse_text_backend_adaption(text_data)

        return text_data

    @property
    def label_transform_funcs(self):
        return self._label_transform_funcs

    @label_transform_funcs.setter
    def label_transform_funcs(self, value):
        self._label_transform_funcs = value
        if self.traindata is not None and hasattr(self.traindata.label, 'label_transform_funcs'):
            self.traindata.label.label_transform_funcs = self._label_transform_funcs
        if self.testdata is not None and hasattr(self.testdata.label, 'label_transform_funcs'):
            self.testdata.label.label_transform_funcs = self._label_transform_funcs

    @property
    def paired_transform_funcs(self):
        return self._paired_transform_funcs

    @paired_transform_funcs.setter
    def paired_transform_funcs(self, value):
        self._paired_transform_funcs = value

        if self.traindata is not None and hasattr(self.traindata, 'paired_transform_funcs'):
            self.traindata.paired_transform_funcs = value

        if self.testdata is not None and hasattr(self.testdata, 'paired_transform_funcs'):
            self.testdata.paired_transform_funcs = value

    @property
    def batch_transform_funcs(self):
        return self._batch_transform_funcs

    @batch_transform_funcs.setter
    def batch_transform_funcs(self, value):
        self._batch_transform_funcs = value
        self.traindata.update_signature()
        self.traindata.batch_sampler._batch_transform_funcs = value
        if self.testdata is not None:
            self.testdata.update_signature()

    def batch_transform(self, batchdata):
        if hasattr(self, '_batch_transform_funcs') and len(self._batch_transform_funcs) > 0:

            if isinstance(batchdata, tuple):
                new_batchdata = copy.deepcopy(self.traindata.data_template)
                for i in range(len(batchdata)):
                    new_batchdata[new_batchdata.key_list[i]] = batchdata[i]
                batchdata = new_batchdata
            if isinstance(batchdata, OrderedDict):
                if not all([isinstance(k, TensorSpec) for k in batchdata.key_list]):
                    new_batchdata = copy.deepcopy(self.traindata.data_template)
                    for i in range(len(batchdata)):
                        new_batchdata[new_batchdata.key_list[i]] = batchdata[batchdata.key_list[i]]
                    batchdata = new_batchdata

                for trans in self._batch_transform_funcs:
                    batchdata = trans(batchdata)
                return batchdata
            else:
                return batchdata

        else:
            return batchdata

    def _next_index(self):
        return self.__next__()

    def __iter__(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata._sample_iter
        else:
            return self.traindata._sample_iter

    def __len__(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.__len__()
        elif self.traindata is not None:
            return self.traindata.__len__()
        else:
            return 0

    def next(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.next()
        else:
            return self.traindata.next()

    def __next__(self):
        if self.scenario == 'test' and self.testdata is not None:
            return next(self.testdata)
        else:
            return next(self.traindata)

    def next_train(self):
        results = self.traindata.next()
        # if self.dynamic_padding:
        #     for i in range(int_shape(results)[1]):
        #         if np.all(np.equal(results[:,int_shape(results)[1]-i-1],0)):
        #             pass
        #         else:
        #             results=results[:,:int_shape(results)[1]-i]

        return results

    def next_test(self):
        if self.testdata is not None:
            results = self.testdata.next()
            return results

        else:
            return None

    def index2text(self, idx: int):
        index2textdict = None

        if isinstance(self.traindata.data,TextSequenceDataset):
            index2textdict = self.traindata.data.index2text
        elif isinstance(self.traindata.data, ZipDataset) and any([isinstance(ds,TextSequenceDataset) for ds in self.traindata.data.items]):
            index2textdict = [ds for ds in self.traindata.data.items if isinstance(ds,TextSequenceDataset) ][0].index2text
        elif isinstance(self.traindata.label,TextSequenceDataset):
            index2textdict = self.traindata.label.index2text
        else:
            index2textdict = self.traindata.data.index2text
        if idx in index2textdict:
            return  index2textdict[idx]
        else:
            return '[UNK]'


    def text2index(self, text_data: str):
        text2indexdict = None
        if isinstance(self.traindata.data,TextSequenceDataset):
            text2indexdict = self.traindata.data.text2index
        elif isinstance(self.traindata.data, ZipDataset) and any([isinstance(ds,TextSequenceDataset) for ds in self.traindata.data.items]):
            text2indexdict = [ds for ds in self.traindata.data.items if isinstance(ds,TextSequenceDataset) ][0].text2index
        elif isinstance(self.traindata.label,TextSequenceDataset):
            text2indexdict = self.traindata.label.text2index
        else:
            text2indexdict = self.traindata.data.text2index
        if text_data in text2indexdict:
            return text2indexdict[text_data]
        else:
            return text2indexdict['[UNK]']

    def index2label(self, idx: int):
        index2textdict = None
        if self.scenario == 'test' and self.testdata is not None and self.testdata.label is not None:
            if isinstance(self.testdata.label, ZipDataset):
                for dataset in self.testdata.label.items:
                    if isinstance(dataset, TextSequenceDataset):
                        index2textdict = dataset.index2text
                        break
            else:
                index2textdict = self.traindata.label.index2text
        else:
            if isinstance(self.traindata.label, ZipDataset):
                for dataset in self.traindata.label.items:
                    if isinstance(dataset, TextSequenceDataset):
                        index2textdict = dataset.index2text
                        break
            else:
                index2textdict = self.traindata.label.index2text
        if idx in index2textdict:
            return index2textdict[idx]
        else:
            return '[UNK]'

    def label2index(self, text_data: str):
        text2indexdict = None
        if self.scenario == 'test' and self.testdata is not None and self.testdata.label is not None:
            if isinstance(self.testdata.label, ZipDataset):
                for dataset in self.testdata.label.items:
                    if isinstance(dataset, TextSequenceDataset):
                        text2indexdict = dataset.text2index
                        break
            else:
                text2indexdict = self.traindata.label.text2index
        else:
            if isinstance(self.traindata.label, ZipDataset):
                for dataset in self.traindata.label.items:
                    if isinstance(dataset, TextSequenceDataset):
                        text2indexdict = dataset.text2index
                        break
            else:
                text2indexdict = self.traindata.label.text2index

        if text_data in text2indexdict:
            return text2indexdict[text_data]
        else:
            return text2indexdict['[UNK]']

    @property
    def vocabs(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.data.vocabs
        else:
            return self.traindata.data.vocabs

    @property
    def label_vocabs(self):
        if self.scenario == 'test' and self.testdata is not None:
            return self.testdata.label.vocabs
        else:
            return self.traindata.label.vocabs

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name == 'traindata' or name == 'testdata':
            if isinstance(value, Iterator):
                object.__getattribute__(self, name).parent = self

        if name in ['traindata', 'testdata', 'sequence_length']:
            if hasattr(self, 'sequence_length') and self.sequence_length is not None:
                if hasattr(self, 'traindata') and isinstance(self.traindata, Iterator):

                    for ds in self.traindata.get_datasets():
                        if isinstance(ds, TextSequenceDataset):
                            ds.sequence_length = self.sequence_length
                if hasattr(self, 'testdata') and isinstance(self.testdata, Iterator):
                    for ds in self.testdata.get_datasets():
                        if isinstance(ds, TextSequenceDataset):
                            ds.sequence_length = self.sequence_length










