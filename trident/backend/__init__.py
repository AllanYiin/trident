"""trident backend"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

from trident.backend.common import *
from trident.backend.tensorspec import *
import  trident.backend.numpy_ops
import trident.backend.load_backend
from trident.backend.model import *



from trident.data import *
from trident.callbacks import *
from trident.misc import *

if get_backend()=='pytorch':
    from trident.backend.pytorch_ops import *
    from trident.backend.pytorch_backend import *
    from trident.optims.pytorch_optimizers import *
    from trident.layers.pytorch_activations import *
    from trident.layers.pytorch_layers import *
    from trident.layers.pytorch_pooling import *
    from trident.layers.pytorch_blocks import *
    from trident.layers.pytorch_normalizations import *
    from trident.layers.pytorch_rnn import *

    from trident.optims.pytorch_constraints import *
    from trident.optims.pytorch_regularizers import *
    from trident.optims.pytorch_losses import *
    from trident.optims.pytorch_metrics import *

    from trident.optims.pytorch_trainer import *

elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_ops import *
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_serialization import *
    from trident.optims.tensorflow_optimizers import *

    from trident.layers.tensorflow_activations import *
    from trident.layers.tensorflow_layers import *
    from trident.layers.tensorflow_pooling import *
    from trident.layers.tensorflow_blocks import *
    from trident.layers.tensorflow_normalizations import *


    from trident.optims.tensorflow_constraints import *
    from trident.optims.tensorflow_regularizers import *
    from trident.optims.tensorflow_losses import *
    from trident.optims.tensorflow_metrics import *

    from trident.optims.tensorflow_trainer import *


elif get_backend()=='onnx':
    import_or_install()
    pass

from trident.optims.trainers import TrainingPlan
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import *
from trident.backend.iteration_tools import *
from trident.models import *
#

