from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import warnings

import numpy as np

from trident.backend.common import *
from trident.backend.common import get_backend
from trident.callbacks.callback_base import *
from trident.context import split_path, make_dir_if_need, sanitize_path
if get_backend()=='pytorch':
    from trident.backend.pytorch_ops import to_numpy,to_tensor
elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import  to_numpy,to_tensor

__all__ = ['SavingStrategyCallback','CyclicSavingStrategyCallback']

class SavingStrategyCallback(CallbackBase):
    def __init__(self, **kwargs):
        super(SavingStrategyCallback, self).__init__()

    def on_model_saving_start(self, training_context):
        pass

    def on_model_saving_end(self, training_context):
        pass



class CyclicSavingStrategyCallback(SavingStrategyCallback):
    def __init__(self, repeat_period=5,**kwargs):
        super(CyclicSavingStrategyCallback, self).__init__()
        self.repeat_period=repeat_period
        self.counter=0
        self.origin_save_path =None


    def on_model_saving_start(self, training_context):
        if 'save_path' in training_context and self.origin_save_path is None:
            self.origin_save_path = training_context['save_path']
            folder,filename,ext=split_path(self.origin_save_path)
            training_context['save_path']=os.path.join(folder,filename+'_{0}'.format(self.counter)+ext)

    def on_model_saving_end(self, training_context):
        training_context['save_path']= self.origin_save_path
        self.origin_save_path =None

