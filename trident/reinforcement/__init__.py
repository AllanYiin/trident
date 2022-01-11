from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from trident.backend.common import get_backend
if get_backend()=='pytorch':
    from . import pytorch_policies as policies

elif get_backend()=='tensorflow':
    from . import tensorflow_policies as policies

from . import utils

