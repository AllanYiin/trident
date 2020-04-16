from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from layers.pytorch_activations import *
from layers.pytorch_normalizations import *
from data.pytorch_datasets import *

version=torch.__version__
stderr.write('Pytorch version:{0}.\n'.format(version))
if version<'1.0.0':
    raise ValueError('Not support Pytorch below 1.0' )
