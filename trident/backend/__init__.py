from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
from trident.backend.common import get_session,set_session,get_plateform, get_trident_dir, epsilon, floatx, camel2snake, \
    snake2camel, addindent, format_time, get_time_suffix, get_function, get_class, get_terminal_size, gcd, \
    get_divisors, isprime, next_prime, prev_prime, nearest_prime,PrintException, if_else

#from trident.backend.load_backend import get_backend, get_image_backend
from trident.backend.model import *


from trident.data import *
from trident.data.data_loaders import *
from trident.data.image_common import *
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

from trident.callbacks import *
from trident.optims.trainers import TrainingPlan
from trident.misc.ipython_utils import *
from trident.misc.visualization_utils import *
from trident.backend.iteration_tools import *
from trident.models import *


