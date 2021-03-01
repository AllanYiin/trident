from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import builtins
import copy
import inspect
import itertools
import locale
import os
import random
import string
import uuid
import warnings
from typing import List

import numpy as np
from trident import context

from trident.misc.ipython_utils import is_in_ipython

from trident.data.vision_transforms import Unnormalize, Normalize

from trident.data.transform import Transform

try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve
from trident.backend.common import *
from trident.backend.opencv_backend import *
from trident.data.text_common import *
from trident.data.image_common import *
from trident.data.label_common import *
from trident.data.samplers import *
from trident.data.dataset import *

_session = get_session()
_trident_dir = get_trident_dir()
_locale = locale.getdefaultlocale()[0].lower()

__all__ = ['ImageDataProvider', 'DataProvider', 'TextSequenceDataProvider']


class ImageDataProvider(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset_name='', traindata=None, testdata=None, minibatch_size=8, mode='tuple', **kwargs):
        self.__name__ = dataset_name
        self.uuid = uuid.uuid4().node
        self.traindata = traindata
        self.testdata = testdata
        self.annotations = {}

        self.scenario = 'train'
        self._class_names = {}
        if mode in ['tuple', 'dict']:
            self.mode = mode
        else:
            raise ValueError("Valid mode should be tuple or dict ")

        self._minibatch_size = minibatch_size
        self.is_flatten = bool(kwargs['is_flatten']) if 'is_flatten' in kwargs else False
        self.__default_language__ = 'en-us'
        if len(self._class_names) > 0:
            if _locale in self._class_names:
                self.__default_language__ = _locale
            for k in self._class_names.keys():
                if _locale.split('-')[0] in k:
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
        self.spatial_transform_funcs = []
        cxt=context._context()
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
    def minibatch_size(self):
        return self._minibatch_size

    @minibatch_size.setter
    def minibatch_size(self, value):
        self._minibatch_size = value
        if self.traindata is not None and hasattr(self.traindata, '_minibatch_size'):
            self.traindata.minibatch_size = self._minibatch_size
        if self.testdata is not None and hasattr(self.testdata, '_minibatch_size'):
            self.testdata.minibatch_size = self._minibatch_size

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
        self._image_transform_funcs = value
        if self.traindata is not None and hasattr(self.traindata.data, 'transform_funcs'):
            self.traindata.data.transform_funcs =copy.deepcopy(value)

            self.traindata.update_data_template()
        if self.testdata is not None and len(self.testdata.data) > 0 and hasattr(self.testdata.data, 'transform_funcs'):
            self.testdata.data.transform_funcs =copy.deepcopy(value)

            self.testdata.update_data_template()

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
        for i in range(len(self.image_transform_funcs)):
            fn = self.image_transform_funcs[-1 - i]
            if (inspect.isfunction(fn) and fn.__qualname__ == 'normalize.<locals>.img_op') or (isinstance(fn, Transform) and fn.name == 'normalize'):
                return_list.append(Unnormalize(fn.mean, fn.std))
        #return_list.append(array2image)
        return return_list

    def reverse_image_transform(self, img_data: np.ndarray):
        if img_data.ndim == 4:
            return np.asarray([self.reverse_image_transform(im) for im in img_data])
        if len(self.reverse_image_transform_funcs) == 0:
            return reverse_image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_image_transform_funcs:
                img_data = fc(img_data)
            #img_data = reverse_image_backend_adaption(img_data)

        return img_data

    @property
    def label_transform_funcs(self):
        return self._label_transform_funcs

    @label_transform_funcs.setter
    def label_transform_funcs(self, value):
        self._label_transform_funcs = value
        if self.traindata is not None and hasattr(self.traindata.label, 'label_transform_funcs'):
            self.traindata.label.label_transform_funcs = self._label_transform_funcs
            self.traindata.update_data_template()
        if self.testdata is not None and hasattr(self.testdata.label, 'label_transform_funcs'):
            self.testdata.label.label_transform_funcs = self._label_transform_funcs
            self.testdata.update_data_template()

    @property
    def paired_transform_funcs(self):
        return self._paired_transform_funcs

    @paired_transform_funcs.setter
    def paired_transform_funcs(self, value):
        self._paired_transform_funcs = value

        if self.traindata is not None and hasattr(self.traindata, 'paired_transform_funcs'):
            self.traindata.paired_transform_funcs =copy.deepcopy(value)


            self.traindata.update_data_template()

        if self.testdata is not None and hasattr(self.testdata, 'paired_transform_funcs'):
            self.testdata.paired_transform_funcs =copy.deepcopy(value)
            self.testdata.update_data_template()

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

    def preview_images(self, key=None,is_concate=True):
        if not isinstance(self.traindata.data,ImageDataset):
            return None
        else:
            if key is None and isinstance(self.traindata.data,ImageDataset):
                data= self.next()
                data=enforce_singleton(data)
                data = self.reverse_image_transform(data)
                if is_concate:
                    data = np.concatenate([img for img in data], axis=-1 if data[0].ndim==2 or (data[0].ndim==3 and data[0].shape[0] in [1,3,4]) else -2)
                    return array2image(data)
                else:
                    return [array2image(img) for img in data]
            elif isinstance(key, slice):
                start = key.start if key.start is not None else 0
                stop = key.stop
                results=[]
                for k in range(start,stop,1):
                    img=self.traindata.data.__getitem__(k)

                    if isinstance(img, np.ndarray):
                        for fc in self.image_transform_funcs:
                            if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not image_backend_adaption and  fc is not Normalize and fc is not normalize:
                                img = fc(img)
                    results.append(img)
                if is_concate:
                    results = np.concatenate(results, axis=-1 if results[0].ndim==2 or (results[0].ndim==3 and results[0].shape[0] in [1,3,4]) else -2)
                    return array2image(results)
                else:
                    return [array2image(img) for img in results]
            elif isinstance(key, int):
                img=self.traindata.data.__getitem__(key)
                if isinstance(img, np.ndarray):
                    for fc in self.image_transform_funcs:
                        if (inspect.isfunction(fc) or isinstance(fc, Transform)) and fc is not image_backend_adaption and fc is not Normalize and fc is not normalize:
                            img = fc(img)
                return array2image(img)

    def label_statistics(self):
        if isinstance(self.traindata.label,LabelDataset):
            unique, counts = np.unique(np.array(self.traindata.label.items), return_counts=True)
            max_len=builtins.max([get_string_actual_length(item) for item in self.class_names[self.__default_language__]])+5
            for i in range(len(unique)):
                class_names=''.join([' ']*max_len) if self.class_names is None or len(self.class_names) == 0 else self.index2label(unique[i])+''.join([' ']*(max_len-get_string_actual_length(self.index2label(unique[i]))))
                bar=['█']*int(builtins.round(50*counts[i] / float(len(self.traindata.label.items))))
                print('{0:<10} {1} {2:<50} {3:,} ({4:.3%})'.format(unique[i],class_names,''.join(bar),counts[i],counts[i]/float(len(self.traindata.label.items))))




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
        return self.traindata.next()


    def next_test(self):
        if self.testdata is not None:
            return self.testdata.next()

        else:
            return None

    def get_all_data(self, is_shuffle=False,topk=-1):
        idxes = np.arange(len(self.traindata.data))
        if is_shuffle :
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
                language = 'en-us'
            self.class_names[language] = list(class_names)
            print('Mapping class_names  in {0}   success, total {1} class names added.'.format(language, len(class_names)))
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
            warnings.warn('You dont have {0} language version class names', category='mapping', stacklevel=1, source=self.__class__)
        else:
            self._lab2idx = {v: k for k, v in enumerate(self.class_names[lang])}
            self._idx2lab = {k: v for k, v in enumerate(self.class_names[lang])}

    def index2label(self, idx: int):
        if self._idx2lab is None or len(self._idx2lab.items()) == 0:
            raise ValueError('You dont have proper mapping class names')
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
        if name=='mode':
            if isinstance(self.traindata,Iterator):
                self.traindata.mode=value
                self.traindata.batch_sampler.mode=value
            if isinstance(self.testdata,Iterator):
                self.testdata.mode=value
                self.testdata.batch_sampler.mode = value





DataProvider = ImageDataProvider


class TextSequenceDataProvider(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset_name='', traindata=None, testdata=None, minibatch_size=8, mode='tuple', **kwargs):
        self.__name__ = dataset_name
        if mode in ['tuple', 'dict']:
            self.mode = mode
        else:
            raise ValueError("Valid mode should be tuple or dict ")

        self.traindata = traindata
        self.testdata = testdata
        self.annotations = {}

        self.scenario = 'train'
        self._class_names = {}

        self._minibatch_size = minibatch_size
        self.is_flatten = bool(kwargs['is_flatten']) if 'is_flatten' in kwargs else False
        self.__default_language__ = 'en-us'
        if len(self._class_names) > 0:
            if _locale in self._class_names:
                self.__default_language__ = _locale
            for k in self._class_names.keys():
                if _locale.split('-')[0] in k:
                    self.__default_language__ = k
                    break

        self._idx2lab = {}
        self._lab2idx = {}

        self.tot_minibatch = 0
        self.tot_records = 0
        self.tot_epochs = 0
        self._text_transform_funcs = []
        self._label_transform_funcs = []
        self._paired_transform_funcs = []
        self.spatial_transform_funcs = []

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
    def minibatch_size(self):
        return self._minibatch_size

    @minibatch_size.setter
    def minibatch_size(self, value):
        self._minibatch_size = value
        if self.traindata is not None and hasattr(self.traindata, '_minibatch_size'):
            self.traindata.minibatch_size = self._minibatch_size
        if self.testdata is not None and hasattr(self.testdata, '_minibatch_size'):
            self.testdata.minibatch_size = self._minibatch_size

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
        if self.testdata is not None and len(self.testdata.data) > 0 and hasattr(self.testdata.data, 'text_transform_funcs'):
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
                if not fc.__qualname__.startswith('random_') or 'crop' in fc.__qualname__ or 'rescale' in fc.__qualname__ or (
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
        return self.traindata.next()


    def next_test(self):
        if self.testdata is not None:
            return self.testdata.next()

        else:
            return None

    def index2text(self, idx: int):
        index2textdict = None
        if self.scenario == 'test' and self.testdata is not None:
            index2textdict = self.testdata.data.index2text
        else:
            index2textdict = self.traindata.data.index2text
        return index2textdict[idx]

    def text2index(self, text_data: str):
        text2indexdict = None
        if self.scenario == 'test' and self.testdata is not None:
            text2indexdict = self.testdata.data.text2index
        else:
            text2indexdict = self.traindata.data.text2index
        if text_data in text2indexdict:
            return text2indexdict[text_data]
        else:
            return text2indexdict['<unknown/>']

    def index2label(self, idx: int):
        index2textdict = None
        if self.scenario == 'test' and self.testdata is not None and self.testdata.label is not None:
            index2textdict = self.testdata.label.index2text
        else:
            index2textdict = self.traindata.label.index2text
        return index2textdict[idx]

    def label2index(self, text_data: str):
        text2indexdict = None
        if self.scenario == 'test' and self.testdata is not None and self.testdata.label is not None:
            text2indexdict = self.testdata.label.text2index
        else:
            text2indexdict = self.traindata.label.text2index
        if text_data in text2indexdict:
            return text2indexdict[text_data]
        else:
            return text2indexdict['<unknown/>']

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







