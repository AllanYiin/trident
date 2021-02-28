import os
import sys
import builtins
import numbers
import time

import numpy as np
from trident import context

from trident.backend.common import get_backend,to_list, addindent, get_time_suffix, format_time, get_terminal_size, get_session,get_backend, \
    snake2camel, PrintException, unpack_singleton, enforce_singleton, OrderedDict, split_path, sanitize_path,make_dir_if_need,Signature


ctx = context._context()
_backend =get_backend()
working_directory=ctx.working_directory


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

        self.summary_writer=ctx.summary_writer

    @property
    def enable_tensorboard(self):
        return ctx.enable_tensorboard


    def regist(self,data_name:str):
        if data_name not in self:
            self[data_name]=[]

    def collect(self, data_name: str, step: int, value: (float,np.ndarray, Tensor)):
        if data_name not in self:
            self.regist(data_name)

        if value is not None:
            if isinstance(value,(list,tuple)):
                value=to_tensor(value)
            elif isinstance(value,np.ndarray):
                value = to_tensor(value)


            if get_backend() == 'pytorch':
                if isinstance(value,numbers.Number):
                    self[data_name].append((step, value))
                elif is_tensor(value):
                    value=value.copy().cpu().detach().mean().item()
                    self[data_name].append((step, value))
                elif isinstance(value, np.ndarray):
                    value = value.mean()[0]
                    self[data_name].append((step, value))
            elif get_backend() == 'tensorflow':
                with tf.device('/cpu:0'):
                    value = tf.identity(value).numpy().mean()
                    self[data_name].append((step, value))

            # if is_tensor(value):
            #     if get_backend()=='pytorch':
            #         value=to_numpy(value.copy().cpu().detach()).mean()
            #         self[data_name].append((step, value))
            #     elif get_backend()=='tensorflow':
            #         with tf.device('/cpu:0'):
            #             value = to_numpy(tf.identity(value)).mean()
            #             self[data_name].append((step, value))
            #
            #
            # else:
            #     self[data_name].append((step, value))
            if ctx.enable_tensorboard:
                if self.training_name is None:
                    ctx.summary_writer.add_scalar( self.name+"/"+data_name, value, global_step=step, walltime=time.time())
                else:
                    ctx.summary_writer.add_scalar(self.training_name+ "/"+self.name + "/" + data_name, value, global_step=step, walltime=time.time())

    def reset(self):
        for i in range(len(self)):
            self.value_list[i]=[]
    def get_keys(self):
        return self.key_list

    def get_series(self,data_name):

        if data_name in self and self[data_name] is not None and len(self[data_name])>=1:
            steps,values=zip(*self[data_name].copy())
            return list(steps),list(values)
        else:
            sys.stderr.write('{0} is not in this history.'.format(data_name))
            return [], []

    def get_last(self,data_name):
        if data_name in self:
            return self[data_name][-1]
        else:
            return []
            #raise ValueError('{0} is not in this History.'.format(data_name))

    def get_best(self,data_name,is_larger_better=True):
            if data_name in self:
                steps,values=zip(*self[data_name].copy())
                if is_larger_better:
                    return builtins.max(values)
                else:
                    return builtins.min(values)
            else:
                raise ValueError('{0} is not in this History.'.format(data_name))