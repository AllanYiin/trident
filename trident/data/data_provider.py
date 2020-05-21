from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import locale
import os
import random
import string
import warnings

import numpy as np

try:
    from urllib.request import urlretrieve
except ImportError:
    from six.moves.urllib.request import urlretrieve
from trident.backend.common import *
from trident.data.image_common import *
from trident.data.label_common import *
from trident.data.samplers import *
from trident.data.dataset import *
_session =get_session()
_trident_dir=get_trident_dir()
_locale = locale.getdefaultlocale()[0].lower()



class DataProvider(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset_name='',data=None,labels=None,masks=None,scenario=None,minibatch_size=8,**kwargs):
        self.__name__=dataset_name
        self.__initialized = False
        self._signature=None
        self.data = {}
        self.labels = {}
        self.annotations = {}
        self.masks = {}

        if scenario is None:
            scenario= 'train'
        elif scenario not in ['training','testing','validation','train','val','test','raw']:
            raise ValueError('Only training,testing,validation,val,test,raw is valid senario')
        self._current_scenario=scenario
        if data is not None and hasattr(data, '__len__'):
            self.data[self._current_scenario]=np.array(data)

            print('Mapping data  in {0} scenario  success, total {1} record addeds.'.format(scenario,len(data)))
            self.__initialized = True
        if labels is not None and hasattr(labels,'__len__'):
            if len(labels)!=len(data):
                raise ValueError('labels and data count are not match!.')
            else:
                self.labels[self._current_scenario]=np.array(labels)
                print('Mapping label  in {0} scenario  success, total {1} records added.'.format(scenario, len(labels)))
        if masks is not None and hasattr(masks, '__len__'):
            if len(masks)!=len(data):
                raise ValueError('masks and data count are not match!.')
            else:
                self.masks[self._current_scenario]=np.array(masks)
                print('Mapping mask  in {0} scenario  success, total {1} records added.'.format(scenario, len(masks)))


        self.class_names={}
        self.palettes=None
        self._minibatch_size = minibatch_size
        self.is_flatten=bool(kwargs['is_flatten']) if 'is_flatten' in kwargs else False
        self.__default_language__='en-us'
        if len(self.class_names)>0:
            if _locale in self.class_names:
                self.__default_language__ =_locale
            for k in self.class_names.keys():
                if _locale.split('-')[0] in k:
                    self.__default_language__ = k
                    break

        self._idx2lab={}
        self._lab2idx = {}

        self.batch_sampler=BatchSampler(self ,self._minibatch_size,is_shuffle=True,drop_last=False)
        self._sample_iter =iter(self.batch_sampler)

        self.tot_minibatch=0
        self.tot_records=0
        self.tot_epochs=0
        self.image_transform_funcs=[]
        self.label_transform_funcs = []
        self.paired_transform_funcs = []
        self.spatial_transform_funcs = []

    @property
    def minibatch_size(self):
        return self._minibatch_size

    @minibatch_size.setter
    def minibatch_size(self, value):
        self._minibatch_size = value
        self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True, drop_last=False)
        self._sample_iter = iter(self.batch_sampler)

    @property
    def current_scenario(self):
        return self._current_scenario


    @current_scenario.setter
    def current_scenario(self,value):
        if self._current_scenario!=value:
            self._current_scenario=value
            self.batch_sampler = BatchSampler(self, self._minibatch_size, is_shuffle=True,drop_last=False)
            self._sample_iter = iter(self.batch_sampler)


    def _check_data_available(self):
        if len(self.data[self._current_scenario])>0:
            pass
        elif 'train' in self.data and len(self.data['train'])>0:
            self._current_scenario= 'train'
        elif 'raw' in self.data and len(self.data['raw'])>0:
            self._current_scenario= 'raw'
        elif 'test' in self.data and len(self.data['test'])>0:
            self._current_scenario= 'test'

    def __getitem__(self, index:int):
        if self.tot_records == 0:
            self._check_data_available()
        # if len(self.data[self.current_scenario])>index and self.current_scenario in self.masks and len(self.masks[self.current_scenario])>index and len(self.labels[self.current_scenario])==0:
        #     return self.data[self.current_scenario][index], self.masks[self.current_scenario][index]
        if len(self.data[self._current_scenario])>index and self._current_scenario in self.labels and len(self.labels[self._current_scenario])>index :
            return self.image_transform(self.data[self._current_scenario][index]), self.label_transform(self.labels[self._current_scenario][index])
        return self.image_transform(self.data[self._current_scenario][index]),

    def _next_index(self):
        return next(self._sample_iter)

    def __iter__(self):
        return self._sample_iter

    def __len__(self):
        if not isinstance(self.data,dict) or  len(self.data.items())==0 :
            return 0
        if self._current_scenario not in self.data:
            raise ValueError('Current Scenario {0} dont have data.'.format(self._current_scenario))
        elif len(self.data[self._current_scenario])==0:
            self._check_data_available()
            return len(self.data[self._current_scenario])
        else:
            return len(self.data[self._current_scenario])

    def next(self):
        return next(self._sample_iter)

    def __next__(self):
        return next(self._sample_iter)
        # # if minibach_size is not None and minibach_size != self.minibatch_size:
        # #     self.minibatch_size = minibach_size
        # #     self.batch_sampler = BatchSampler(range(len(self.data[self.current_scenario])), self.minibatch_size,
        # #                                       is_shuffle=True, drop_last=False)
        # #     self._sample_iter = iter(self.batch_sampler)
        #
        # if self.batch_sampler is None or len(self.batch_sampler) == 0:
        #     self.batch_sampler = BatchSampler(range(len(self.data[self.current_scenario])), self.minibatch_size,
        #                                       is_shuffle=True, drop_last=False)
        #     self._sample_iter = iter(self.batch_sampler)

        # index = self._next_index()  # may raise StopIteration
        # batch = self.__getitem__(index)  # may raise StopIteration
        # # batch= zip(*batch)
        # self.tot_minibatch += 1
        # self.tot_records += len(batch[0])
        # self.tot_epochs = self.tot_records // self.__len__()
        # yield  batch

    def get_all_data(self,is_shuffle=False,topk=100):
        if is_shuffle==False:
            data= self.data[self._current_scenario][:topk]
            if isinstance(data[0],str):
                data = [image2array(d) for d in data]
            return data
        else:
            idxes=np.random.shuffle(np.arange(len(self)))
            data = self.data[self._current_scenario][idxes][:topk]
            if isinstance(data[0], str):
                data = [image2array(d) for d in data]
            return data




    def reset_statistics(self):
        self.tot_minibatch = 0
        self.tot_records = 0
        self.tot_epochs = 0
        self._check_data_available()

    def image_transform(self, img_data):

        if isinstance(img_data, list) and all(isinstance(elem, np.ndarray) for elem in img_data):
            img_data=np.asarray(img_data)
        if isinstance(img_data,str) and os.path.isfile(img_data) and os.path.exists(img_data):
            img_data=image2array(img_data)

        if len(self.image_transform_funcs)==0:
            return image_backend_adaption(img_data)
        if isinstance(img_data,np.ndarray):
            #if img_data.ndim>=2:
            for fc in self.image_transform_funcs:
                img_data=fc(img_data)
            img_data=image_backend_adaption(img_data)
            if img_data.dtype!=np.float32:
                raise ValueError('')
            return img_data
        else:
            return img_data


    def label_transform(self, label_data):
        label_data=label_backend_adaptive(label_data,self.class_names)
        if isinstance(label_data, list) and all(isinstance(elem, np.ndarray) for elem in label_data):
            label_data = np.asarray(label_data)
        if isinstance(label_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.label_transform_funcs:
                label_data = fc(label_data)
            return label_data
        else:
            return label_data


    def mapping(self,data,labels=None,masks=None,scenario=None):
        if scenario is None:
            scenario= 'train'
        elif scenario not in ['training','testing','validation','train','val','test','raw']:
            raise ValueError('Only training,testing,validation,val,test,raw is valid senario')
        self._current_scenario=scenario
        if data is not None and hasattr(data, '__len__'):
            self.data[scenario]=data
            print('Mapping data  in {0} scenario  success, total {1} record addeds.'.format(scenario,len(data)))
            self.__initialized = True
        if labels is not None and hasattr(labels,'__len__'):
            if len(labels)!=len(data):
                raise ValueError('labels and data count are not match!.')
            else:
                self.labels[scenario]=np.array(labels)
                print('Mapping label  in {0} scenario  success, total {1} records added.'.format(scenario, len(labels)))
        if masks is not None and hasattr(masks, '__len__'):
            if len(masks)!=len(data):
                raise ValueError('masks and data count are not match!.')
            else:
                self.masks[scenario]=np.array(masks)
                print('Mapping mask  in {0} scenario  success, total {1} records added.'.format(scenario, len(masks)))

    def binding_class_names(self,class_names=None,language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = 'en-us'
            self.class_names[language] = list(class_names)
            print('Mapping class_names  in {0}   success, total {1} class names added.'.format(language, len(class_names)))
            self.__default_language__=language
            self._lab2idx= {v: k for k, v in enumerate(self.class_names[language])}
            self._idx2lab={k: v for k, v in enumerate(self.class_names[language])}

            # if len(list(set(self.labels[scenario])))!=len(self.class_names):
            #     warnings.warn('Distinct labels count is not match with class_names', category='mapping', stacklevel=1, source=self.__class__)
            #

    def change_language(self, lang):
        self.__default_language__ = lang
        if self.class_names is None or len(self.class_names.items())==0 or lang not in self.class_names :
            warnings.warn('You dont have {0} language version class names', category='mapping', stacklevel=1, source=self.__class__)
        else:
            self._lab2idx = {v: k for k, v in enumerate(self.class_names[lang])}
            self._idx2lab = {k: v for k, v in enumerate(self.class_names[lang])}

    def index2label(self, idx:int):
        if self._idx2lab  is None or len(self._idx2lab .items())==0:
            raise ValueError('You dont have proper mapping class names')
        elif  idx not in self._idx2lab :
            raise ValueError('Index :{0} is not exist in class names'.format(idx))
        else:
            return self._idx2lab[idx]

    def label2index(self ,label):
        if self._lab2idx  is None or len(self._lab2idx .items())==0:
            raise ValueError('You dont have proper mapping class names')
        elif  label not in self._lab2idx :
            raise ValueError('label :{0} is not exist in class names'.format(label))
        else:
            return self._lab2idx[label]

    def get_language(self):
        return self.__default_language__






class DataProviderV2(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset_name='',traindata=None,testdata=None,minibatch_size=8,**kwargs):
        self.__name__=dataset_name

        self.traindata = traindata
        self.testdata = testdata
        self.annotations = {}

        self.scenario='train'
        self._class_names={}

        self._minibatch_size = minibatch_size
        self.is_flatten=bool(kwargs['is_flatten']) if 'is_flatten' in kwargs else False
        self.__default_language__='en-us'
        if len(self._class_names)>0:
            if _locale in self._class_names:
                self.__default_language__ =_locale
            for k in self._class_names.keys():
                if _locale.split('-')[0] in k:
                    self.__default_language__ = k
                    break

        self._idx2lab={}
        self._lab2idx = {}



        self.tot_minibatch=0
        self.tot_records=0
        self.tot_epochs=0
        self._image_transform_funcs=[]
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

    def update_signature(self,arg_names):
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
        if self.scenario=='test' and self.testdata is not None:
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
            self.testdata.minibatch_size= self._minibatch_size

    @property
    def palette(self):
        if self.scenario=='test' and self.testdata is not None:
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
        if self.traindata is not None and hasattr(self.traindata.data, 'image_transform_funcs'):
            self.traindata.data.image_transform_funcs = self._image_transform_funcs
            if len(self.traindata.unpair) > 0:
                self.traindata.unpair.image_transform_funcs = self._image_transform_funcs
        if self.testdata is not None and len(self.testdata.data)>0 and hasattr(self.testdata.data, 'image_transform_funcs'):
            self.testdata.data.image_transform_funcs = self._image_transform_funcs
        if self.testdata is not None and len(self.testdata.data)>0 and len(self.testdata.unpair) > 0:
            self.testdata.unpair.image_transform_funcs = self._image_transform_funcs

    def image_transform(self, img_data):
        if img_data.ndim==4:
            return [self.image_transform(im) for im in img_data]
        if len(self.image_transform_funcs) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self.image_transform_funcs:
                if not fc.__qualname__.startswith('random_') or  'crop' in fc.__qualname__  or  'rescale' in fc.__qualname__  or  (fc.__qualname__.startswith('random_') and random.randint(0,10)%2==0):
                    img_data = fc(img_data)

            img_data = image_backend_adaption(img_data)

            return img_data
        else:
            return img_data


    @property
    def reverse_image_transform_funcs(self):
        return_list=[]
        return_list.append(reverse_image_backend_adaption)
        for i in range(len(self.image_transform_funcs)):
            fn=self.image_transform_funcs[-1-i]
            if fn.__qualname__=='normalize.<locals>.img_op':
                return_list.append(unnormalize(fn.mean,fn.std))
        #return_list.append(array2image)
        return return_list

    def reverse_image_transform(self, img_data:np.ndarray):
        if len(self.reverse_image_transform_funcs) == 0:
            return reverse_image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            # if img_data.ndim>=2:
            for fc in self.reverse_image_transform_funcs:
                img_data = fc(img_data)
            img_data = reverse_image_backend_adaption(img_data)

        return img_data


    @property
    def label_transform_funcs(self):
        return self._label_transform_funcs

    @label_transform_funcs.setter
    def label_transform_funcs(self, value):
        self._label_transform_funcs = value
        if self.traindata is not None and hasattr(self.traindata.label, 'label_transform_funcs'):
            self.traindata.label.label_transform_funcs=self._label_transform_funcs
        if self.testdata is not None and hasattr(self.testdata.label, 'label_transform_funcs'):
            self.testdata.label.label_transform_funcs=self._label_transform_funcs

    @property
    def paired_transform_funcs(self):
        return self._paired_transform_funcs

    @paired_transform_funcs.setter
    def paired_transform_funcs(self, value):
        self._paired_transform_funcs = value

        if self.traindata is not None and hasattr(self.traindata, 'paired_transform_funcs'):
            self.traindata.paired_transform_funcs = self._paired_transform_funcs

        if self.testdata is not None  and hasattr(self.testdata, 'paired_transform_funcs'):
            self.testdata.paired_transform_funcs = self._paired_transform_funcs


    def image_transform(self, img_data):
        if img_data.ndim==4:
            return [self.image_transform(im) for im in img_data]
        if len(self.image_transform_funcs) == 0:
            return image_backend_adaption(img_data)
        if isinstance(img_data, np.ndarray):
            for fc in self.image_transform_funcs:
                if fc.__qualname__.startswith('random_') and random.randint(10)%2==0:
                    img_data = fc(img_data)
            img_data = image_backend_adaption(img_data)

            return img_data
        else:
            return img_data



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

    def _next_index(self):
        return self.__next__()

    def __iter__(self):
        if self.scenario=='test' and self.testdata is not None:
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
            result = self.testdata.next()
            return result
        else:
            result= self.traindata.next()
            return result

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


    def get_all_data(self,is_shuffle=False,get_image_mode = GetImageMode.expect, topk=-1):
        orig_get_image_mode = None
        if hasattr(self.traindata.data, 'get_image_mode'):
            orig_get_image_mode = self.traindata.data.get_image_mode
            self.traindata.data.get_image_mode =get_image_mode

        idxes = np.arange(len(self.traindata.data))
        if is_shuffle == True:
            np.random.shuffle(idxes)
        data = []
        if topk==-1:
            topk=len(self.traindata.data)
        for i in range(topk):
            data.append(self.traindata.data[idxes[i]])
        if hasattr(self.traindata.data, 'get_image_mode'):
            self.traindata.data.get_image_mode = orig_get_image_mode
        return data



    def binding_class_names(self,class_names=None,language=None):
        if class_names is not None and hasattr(class_names, '__len__'):
            if language is None:
                language = 'en-us'
            self.class_names[language] = list(class_names)
            print('Mapping class_names  in {0}   success, total {1} class names added.'.format(language, len(class_names)))
            self.__default_language__=language
            self._lab2idx= {v: k for k, v in enumerate(self.class_names[language])}
            self._idx2lab={k: v for k, v in enumerate(self.class_names[language])}

            if self.traindata is not None and hasattr(self.traindata.label,'class_names'):
                self.traindata.label.binding_class_names(class_names,language)
            if self.testdata is not None and hasattr(self.testdata.label,'class_names'):
                self.testdata.label.binding_class_names(class_names,language)



    def change_language(self, lang):
        self.__default_language__ = lang
        if self.class_names is None or len(self.class_names.items())==0 or lang not in self.class_names :
            warnings.warn('You dont have {0} language version class names', category='mapping', stacklevel=1, source=self.__class__)
        else:
            self._lab2idx = {v: k for k, v in enumerate(self.class_names[lang])}
            self._idx2lab = {k: v for k, v in enumerate(self.class_names[lang])}

    def index2label(self, idx:int):
        if self._idx2lab  is None or len(self._idx2lab .items())==0:
            raise ValueError('You dont have proper mapping class names')
        elif  idx not in self._idx2lab :
            raise ValueError('Index :{0} is not exist in class names'.format(idx))
        else:
            return self._idx2lab[idx]

    def label2index(self ,label):
        if self._lab2idx  is None or len(self._lab2idx .items())==0:
            raise ValueError('You dont have proper mapping class names')
        elif  label not in self._lab2idx :
            raise ValueError('label :{0} is not exist in class names'.format(label))
        else:
            return self._lab2idx[label]

    def get_language(self):
        return self.__default_language__







