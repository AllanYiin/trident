import os
import builtins
import numbers
import time

import numpy as np
from trident.backend.common import get_backend,to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session,get_backend, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path,make_dir_if_need,Signature
_session = get_session()
_backend =get_backend()
working_directory=_session.working_directory


if _backend == 'pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_backend import Tensor
    from trident.backend.pytorch_ops import *

elif _backend == 'tensorflow':
    import tensorflow as tf
    from trident.backend.tensorflow_backend import Tensor
    from trident.backend.tensorflow_ops import *



class HistoryBase(OrderedDict):
    def __init__(self, name='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name=name
        self.training_name=None
        self._enable_tensorboard=False
        self.summary_writer=None

    @property
    def enable_tensorboard(self):
        return self._enable_tensorboard

    @enable_tensorboard.setter
    def enable_tensorboard(self,value):
        self._enable_tensorboard=value
        if value==True:
            if get_backend() == 'pytorch':
                try:
                    from  trident.loggers.pytorch_tensorboard import SummaryWriter
                    self.summary_writer = SummaryWriter(os.path.join(working_directory, 'Logs'))

                except Exception as e:
                    print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                    print(e)
                    PrintException()
            elif get_backend() == 'tensorflow':
                try:
                    from trident.loggers.tensorflow_tensorboard import SummaryWriter
                    self.summary_writer = SummaryWriter(os.path.join(working_directory, 'Logs'))
                except Exception as e:
                    print('Tensorboard initialize failed, please check the installation status about Tensorboard.')
                    print(e)
                    PrintException()



    def regist(self,data_name:str):
        if data_name not in self:
            self[data_name]=[]

    def collect(self,data_name:str,step:int,value:(float,Tensor)):
        if data_name not in self:
            self.regist(data_name)
        if is_tensor(value):
            value=to_numpy(value.copy().cpu().detach()).mean()
            self[data_name].append((step, value))
        else:
            self[data_name].append((step, value))
        if self.enable_tensorboard:
            if self.training_name is None:
                self.summary_writer.add_scalar( self.name+"/"+data_name, value, global_step=step, walltime=time.time())
            else:
                self.summary_writer.add_scalar(self.training_name+ "/"+self.name + "/" + data_name, value, global_step=step, walltime=time.time())

    def reset(self):
        for i in range(len(self)):
            self.value_list[i]=[]
    def get_keys(self):
        return self.key_list

    def get_series(self,data_name):
        if data_name in self:
            steps,values=zip(*self[data_name].copy())
            return list(steps),list(values)
        else:
            raise ValueError('{0} is not in this History.'.format(data_name))

    def get_last(self,data_name):
        if data_name in self:
            return self[data_name][-1]
        else:
            raise ValueError('{0} is not in this History.'.format(data_name))

    def get_best(self,data_name,is_larger_better=True):
            if data_name in self:
                steps,values=zip(*self[data_name].copy())
                if is_larger_better:
                    return builtins.max(values)
                else:
                    return builtins.min(values)
            else:
                raise ValueError('{0} is not in this History.'.format(data_name))