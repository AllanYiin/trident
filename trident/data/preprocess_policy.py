from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import random
import time
import builtins
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from trident.backend.common import *
from trident.backend.tensorspec import TensorSpec, assert_input_compatibility, ObjectType,object_type_inference
from trident.data.image_common import image_backend_adaption

if get_backend() == 'pytorch':
    from trident.backend.pytorch_ops import *
elif get_backend() == 'tensorflow':
    from trident.backend.tensorflow_ops import *

__all__ = ['PreprocessPolicy', 'PreprocessPolicyItem']


class PreprocessPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.




    """

    def __init__(self,*args):
        self.policies = []
        for arg in args:
            self.add(arg)
        self.pass_cnt=0
        self.pass_time_spend=0.

    def add(self, item):
        if isinstance(item, PreprocessPolicyItem):
            if len(item.name) == 0:
                item.name = 'item_{0}'.format(len(self.policies))
        self.policies.append(item)
    def reset_statistics(self):
        self.pass_cnt=0
        self.pass_time_spend=0
        for item in self.policies:
            if isinstance(item, PreprocessPolicyItem):
                item.reset_statistics()

    def print_statistics(self):
        print('avg. process time: {0:.5f}'.format(self.pass_time_spend/float(builtins.max(1,self.pass_cnt))))

        for item in self.policies:
            if isinstance(item,PreprocessPolicyItem):
                print(' policy {0}   hit-rate={1:.3%}'.format(item.name, item.hit_rate))
                print(' avg. time spend (true): {0}'.format(item.time_spend_true/float(builtins.max(1,item.count_true))))
                print(' avg. time spend (false):{0}'.format(item.time_spend_false/float(builtins.max(1,item.count_false))))


    def __call__(self, img,spec:TensorSpec=None,**kwargs):
        if isinstance(img, np.ndarray):
            start_time = time.time()
            if spec is None:
                spec = TensorSpec(shape=to_tensor(img.shape), object_type=object_type_inference(img))
            if spec.object_type==ObjectType.rgb or spec.object_type==ObjectType.rgb or spec.object_type==ObjectType.gray:
                if (img.ndim==3 and img.shape[0] in [1,3,4]):
                    img=img.transpose(1,2,0)

            for i in range(len(self.policies)):
                try:
                    item = self.policies[i]
                    img = item(img,spec=spec)
                except Exception as e:
                    print(e)
            img=image_backend_adaption(img)
            self.pass_cnt+=1
            self.pass_time_spend+=float(time.time()-start_time)
            return img

    def __repr__(self):
        return "PreprocessPolicy"


class PreprocessPolicyItem(object):
    def __init__(self, condition_if, then_process, else_process=None, name=''):
        # lambda or function
        if inspect.isfunction(condition_if) or callable(condition_if) or isinstance(condition_if, bool):
            self.condition_if = condition_if
        else:
            print('PreprocessPolicyItem {0} condition_if is not callable'.format(name))
        if callable(then_process) or (isinstance(then_process, list) and callable(then_process[0])):
            self.then_process = then_process
        else:
            print('PreprocessPolicyItem {0} then_process is not callable'.format(name))

        if else_process is None or callable(else_process) or (
                isinstance(else_process, list) and callable(else_process[0])):
            self.else_process = else_process
        else:
            print('PreprocessPolicyItem {0} else_process is not callable'.format(name))
        self.name = name
        self.count_true = 0
        self.count_false = 0
        self.time_spend_true = 0.
        self.time_spend_false = 0.

    @property
    def hit_rate(self):
        return self.count_true / float(max((self.count_true + self.count_false), 1))

    def reset_statistics(self):
        self.count_true = 0
        self.count_false = 0
        self.time_spend_true = 0
        self.time_spend_false = 0

    def __call__(self, img,spec:TensorSpec=None,**kwargs):
        start_time=time.time()
        bool_if=None
        if isinstance(self.condition_if,bool):
            bool_if=self.condition_if
        elif inspect.isfunction(self.condition_if) or callable(self.condition_if) :
            argspec=inspect.getfullargspec(self.condition_if)
            if  "spec" in argspec.args:
                bool_if =self.condition_if(img,spec=spec)
            else:
                bool_if = self.condition_if(img)

        if bool_if==True:
            if isinstance(self.then_process, list):
                for proc in self.then_process:
                    img = proc(img,spec=spec)
            elif callable(self.then_process) or inspect.isfunction(self.then_process):
                img = self.then_process(img,spec=spec)
            self.count_true += 1
            self.time_spend_true+=float(time.time()-start_time)
        elif bool_if==False:
            if self.else_process is not None:
                if isinstance(self.else_process, list):
                    for proc in self.else_process:
                        img = proc(img,spec=spec)
                elif callable(self.else_process) or inspect.isfunction(self.else_process):
                    img = self.else_process(img,spec=spec)
            self.count_false += 1
            self.time_spend_false+= float(time.time() - start_time)
        else:
            self.count_false += 1
            self.time_spend_false += float(time.time() - start_time)
            pass
        return img