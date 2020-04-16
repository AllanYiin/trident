import os
import sys
import time
from shutil import copyfile
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import  Optimizer
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.hooks as hooks
from collections import OrderedDict,defaultdict
from functools import partial
import numpy as np
from ..backend.common import get_session,addindent,get_time_prefix,get_class,format_time,get_terminal_size,snake2camel,camel2snake
from ..backend.pytorch_backend import *

Adam=optim.adam.Adam




def get_optimizer(optimizer_name,**kwargs):
    if optimizer_name is None:
        return None
    optimizer_modules = ['trident.optimizers.pytorch_optimizers','torch.optim']
    optimizer_class = get_class(camel2snake(optimizer_name), optimizer_modules)
    return optimizer_class

