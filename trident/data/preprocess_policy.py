from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .image_common import image_backend_adaptive

__all__ = ['PreprocessPolicy', 'PreprocessPolicyItem']


class PreprocessPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.




    """

    def __init__(self):
        self.policies = []

    def add(self, item):
        if isinstance(item, PreprocessPolicyItem):
            if len(item.name) == 0:
                item.name = 'item_{0}'.format(len(self.policies))
        self.policies.append(item)

    def print_statistics(self):
        for item in self.policies:
            if isinstance(item,PreprocessPolicyItem):
                print('policy {0}   hit-rate={1:.3%}'.format(item.name, item.hit_rate))

    def __call__(self, img):
        for i in range(len(self.policies)):
            try:
                item = self.policies[i]
                img = item(img)
            except Exception as e:
                print(e)
        img=image_backend_adaptive(img)
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

    @property
    def hit_rate(self):
        return self.count_true / float(max((self.count_true + self.count_false), 1))

    def reset_statistics(self):
        self.count_true = 0
        self.count_false = 0

    def __call__(self, img):
        bool_if =self.condition_if(img)

        if bool_if==True:
            if isinstance(self.then_process, list):
                for proc in self.then_process:
                    img = proc(img)
            elif callable(self.then_process) or inspect.isfunction(self.then_process):
                img = self.then_process(img)
            self.count_true += 1
        elif bool_if==False:
            if self.else_process is not None:
                if isinstance(self.else_process, list):
                    for proc in self.else_process:
                        img = proc(img)
                elif callable(self.else_process) or inspect.isfunction(self.else_process):
                    img = self.else_process(img)
            self.count_false += 1
        else:
            self.count_false += 1
            pass
        return img