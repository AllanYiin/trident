"""Fashion-MNIST dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from backend.common import image_data_format,floatx
from backend.load_backend import  get_trident_dir
from backend.image_common import *
import os
import gzip
import numpy as np
from six.moves.urllib.request import urlretrieve
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen
import urllib
from tqdm import tqdm
import warnings
import time


import hashlib
import multiprocessing as mp
import os
import random
import shutil
import sys
import tarfile
import threading
import time
import warnings
import zipfile
from abc import abstractmethod
from contextlib import closing
from multiprocessing.pool import ThreadPool
import itertools

import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlopen
#from six.moves.urllib.request import urlretrieve
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
from backend.image_common import *


_trident_dir=get_trident_dir()



def load_mnist( dataset_name='mnist',kind='train'):
    dirname = os.path.join(_trident_dir,'datasets',dataset_name)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            pass

    #base = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    base = 'http://yann.lecun.com/exdb/mnist/'
    if dataset_name=='fashion-mnist':
        base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'


    """Load MNIST data from `path`"""
    labels_file ='{0}-labels-idx1-ubyte.gz'.format(kind)
    images_file= '{0}-images-idx3-ubyte.gz'.format(kind)
    download_file(base+labels_file, dirname, labels_file,dataset_name+'_labels_{0}'.format(kind))
    download_file(base+images_file, dirname, images_file,dataset_name+'_images_{0}'.format(kind))
    labels_path=os.path.join(dirname,labels_file)
    images_path = os.path.join(dirname, images_file)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


class TqdmProgress(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_file(src,dirname,filename,desc=''):
    if os.path.exists(os.path.join(dirname,filename)):
        print('archive file is already existing, donnot need download again.')
    else:
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                with TqdmProgress(unit='B', unit_scale=True, miniters=1,desc=desc) as t:  # all optional kwargs
                    urlretrieve(src, filename=os.path.join(dirname,filename), reporthook=t.update_to, data=None)
            except HTTPError as e:
                raise Exception(error_msg.format(src,os.path.join(dirname,filename), e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(src,os.path.join(dirname,filename), e.errno, e.reason))
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(os.path.join(dirname, filename)):
                    os.remove(os.path.join(dirname, filename))
        except:
            raise



class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset_name=''):
        self.__name__=dataset_name
        self.data ={}
        self.labels ={}
        self.masks ={}
        self.class_names={}
        self.palettes=None
        self.current_scenario='raw'
        self.__default_language__='en-us'
        self.__current_idx2lab__={}
        self.__current_lab2idx__ = {}
        self.__initialized = False
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self,scenario=None):
        if not isinstance(self.data,dict) or  len(self.data.items())==0 :
            return 0
        if self.current_scenario not in self.data:
            raise ValueError('Current Scenario {0} dont have data.'.format(self.current_scenario))
        else:
            return len(self.data[self.current_scenario])

    def mapping(self,data,labels=None,masks=None,scenario=None):
        if scenario is None:
            scenario= 'raw'
        elif scenario not in ['training','testing','validation','train','val','test','raw']:
            raise ValueError('Only training,testing,validation,val,test,raw is valid senario')
        self.current_scenario=scenario
        if data is not None and hasattr(data, '__len__'):
            self.data[scenario]=list(data)
            print('Mapping data  in {0} scenario  success, total {1} record addeds.'.format(scenario,len(data)))
            self.__initialized = True
        if labels is not None and hasattr(labels,'__len__'):
            if len(labels)!=len(data):
                raise ValueError('labels and data count are not match!.')
            else:
                self.labels[scenario]=list(labels)
                print('Mapping label  in {0} scenario  success, total {1} records added.'.format(scenario, len(labels)))
        if masks is not None and hasattr(masks, '__len__'):
            if len(masks)!=len(data):
                raise ValueError('masks and data count are not match!.')
            else:
                self.masks[scenario]=list(masks)
                print('Mapping mask  in {0} scenario  success, total {1} records added.'.format(scenario, len(masks)))

    def binding_class_names(self,class_names=None,language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = 'en-us'
            self.class_names[language] = list(class_names)
            print('Mapping class_names  in {0}   success, total {1} class names added.'.format(language, len(class_names)))
            self.__default_language__=language
            self.__current_lab2idx__= {v: k for k, v in enumerate(self.class_names[language] )}
            self.__current_idx2lab__={k: v for k, v in enumerate(self.class_names[language] )}

            # if len(list(set(self.labels[scenario])))!=len(self.class_names):
            #     warnings.warn('Distinct labels count is not match with class_names', category='mapping', stacklevel=1, source=self.__class__)
            #

    def change_language(self, lang):
        self.__default_language__ = lang
        if self.class_names is None or len(self.class_names.items())==0 or lang not in self.class_names :
            warnings.warn('You dont have {0} language version class names', category='mapping', stacklevel=1, source=self.__class__)
        else:
            self.__current_lab2idx__ = {v: k for k, v in enumerate(self.class_names[lang])}
            self.__current_idx2lab__ = {k: v for k, v in enumerate(self.class_names[lang])}

    def index2label(self, idx:int):
        if self.__current_idx2lab__  is None or len(self.__current_idx2lab__ .items())==0:
            raise ValueError('You dont have proper mapping class names')
        elif  idx not in self.__current_idx2lab__ :
            raise ValueError('Index :{0} is not exist in class names'.format(idx))
        else:
            return self.__current_idx2lab__[idx]

    def label2index(self ,label):
        if self.__current_lab2idx__  is None or len(self.__current_lab2idx__ .items())==0:
            raise ValueError('You dont have proper mapping class names')
        elif  label not in self.__current_lab2idx__ :
            raise ValueError('label :{0} is not exist in class names'.format(label))
        else:
            return self.__current_lab2idx__[label]
    def get_language(self):
        return self.__default_language__






class DataProvider(object):
    def __init__(self, dataset, batch_size=1, shuffle=True,num_workers=0,  pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle=shuffle
        self.split_idxs=range(self.__len__())
        self.batch_idxs=range(self.__len__())
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')


        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True
    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'sampler', 'drop_last'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)





