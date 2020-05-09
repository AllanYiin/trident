from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import warnings

import numpy as np

from ..backend.common import *
from ..backend.load_backend import get_backend
from ..callbacks import CallbackBase
from ..data.image_common import *

if get_backend()=='pytorch':
    import torch
    import torch.nn as nn
    from ..backend.pytorch_ops import to_numpy,to_tensor
    from ..optims.pytorch_losses import CrossEntropyLoss
elif get_backend()=='tensorflow':
    from ..backend.tensorflow_ops import  to_numpy,to_tensor
    from ..optims.tensorflow_losses import CrossEntropyLoss


__all__ = ['DataProcessCallback']

class DataProcessCallback(CallbackBase):
    def __init__(self, policy=None, **kwargs):
        super(DataProcessCallback, self).__init__()
        self.policy = policy

    def on_batch_start(self, training_context):
        try:
            train_data = training_context['train_data']
            test_data = training_context['test_data']

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
