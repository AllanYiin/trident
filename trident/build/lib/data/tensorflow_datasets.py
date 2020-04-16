from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

#from .tensorflow_blocks import *


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





