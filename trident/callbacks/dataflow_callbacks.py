from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import warnings

import numpy as np

from trident.backend.common import *
from trident.backend.common import get_backend
from trident.callbacks.callback_base import CallbackBase,_valid_when
from trident.data.image_common import *
from trident.context import split_path, make_dir_if_need, sanitize_path
if get_backend()=='pytorch':
    import torch
    import torch.nn as nn
    from trident.backend.pytorch_ops import to_numpy,to_tensor

elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import  to_numpy,to_tensor
elif get_backend()=='jax':
    from trident.backend.jax_ops import  to_numpy,to_tensor


__all__ = ['DataProcessCallback']

class DataProcessCallback(CallbackBase):
    def __init__(self, when='on_data_received',policy=None, **kwargs):
        super(DataProcessCallback, self).__init__()
        if when in _valid_when:
            self.when = when
        else:
            raise ValueError("{0} is not valid event trigger.".format(when))
        self.policy = policy

    def on_batch_start(self, training_context):
        try:
            train_data = training_context['train_data']
            test_data = training_context['test_data']
            data_provider

            input = train_data[train_data.key_list[0]]
            new_input = []
            for i in range(input.shape[0]):
                try:
                    new_input.append(self.policy(input[i]))
                except:
                    new_input.append(input[i])

            new_input = np.array(new_input).astype(np.float32)

            train_data[train_data.key_list[0]] = new_input

            if test_data is not None and len(test_data) > 0:
                input = test_data[test_data.key_list[0]]
                new_input = []
                for i in range(input.shape[0]):
                    try:
                        new_input.append(self.policy(input[i]))
                    except:
                        new_input.append(input[i])

                new_input = np.array(new_input).astype(np.float32)

                test_data[test_data.key_list[0]] = new_input
        except Exception as e:
            print(e)
