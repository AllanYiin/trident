import warnings
import math
import numpy as np
from  ..callbacks import *
from ..backend.common import *
from ..backend.load_backend import get_backend

if get_backend()=='pytorch':
    from ..backend.pytorch_backend import to_numpy,to_tensor
elif get_backend()=='tensorflow':
    from ..backend.tensorflow_backend import  to_numpy,to_tensor
elif get_backend()=='cntk':
    from ..backend.cntk_backend import  to_numpy,to_tensor
