"""trident backend"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from trident.context import *
from trident.backend.common import *
from trident.backend import dtype
import  trident.backend.numpy_ops

import trident.backend.load_backend
if get_backend()=='pytorch':
    from trident.backend.pytorch_ops import *


elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import *


from trident.backend.tensorspec import *

if get_backend()=='pytorch':
    from trident.backend.pytorch_backend import *

elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_backend import *


from trident.loggers.history import *
from trident.backend.model import *

from trident.data.dataset import *
from trident.data.data_provider import *
from trident.callbacks import *
from trident.misc import *

if get_backend()=='pytorch':
    from trident.optims.pytorch_optimizers import *
    from trident.layers.pytorch_activations import *
    from trident.layers.pytorch_initializers import *
    from trident.layers.pytorch_layers import *
    from trident.layers.pytorch_pooling import *
    from trident.layers.pytorch_blocks import *
    from trident.layers.pytorch_normalizations import *
    from trident.layers.pytorch_rnn import *
    from trident.layers.pytorch_transformers import *

    from trident.optims.pytorch_constraints import *
    from trident.optims.pytorch_regularizers import *
    from trident.optims.pytorch_losses import *
    from trident.optims.pytorch_metrics import *
    from trident.optims.pytorch_trainer import *

elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_serialization import *
    from trident.optims.tensorflow_optimizers import *

    from trident.layers.tensorflow_activations import *
    from trident.layers.tensorflow_initializers import *
    from trident.layers.tensorflow_layers import *
    from trident.layers.tensorflow_pooling import *
    from trident.layers.tensorflow_blocks import *
    from trident.layers.tensorflow_normalizations import *
    from trident.layers.tensorflow_rnn import *

    from trident.optims.tensorflow_constraints import *
    from trident.optims.tensorflow_regularizers import *
    from trident.optims.tensorflow_losses import *
    from trident.optims.tensorflow_metrics import *
    from trident.optims.tensorflow_trainer import *




elif get_backend()=='onnx':
    import_or_install('onnx_runtime')
    pass

from trident.data.image_common import *
from trident.data.bbox_common import *
from trident.data.label_common import *
from trident.data.mask_common import *
from trident.data.transform import *
from trident.data.vision_transforms import *
from trident.data.text_transforms import *
from trident.data.data_loaders import *

from trident.optims.trainers import TrainingPlan
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import *
from trident.backend.iteration_tools import *
import trident.models


