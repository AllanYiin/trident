from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
from trident.backend.load_backend import get_backend, get_image_backend, PrintException, if_else
from trident.backend.load_backend import get_session, get_trident_dir, epsilon, set_epsilon, floatx, set_floatx, camel2snake, \
    snake2camel, addindent, format_time, get_time_suffix, get_function, get_class, get_terminal_size, gcd, \
    get_divisors, \
    isprime, next_prime, prev_prime, nearest_prime
from trident.backend.load_backend import to_tensor, to_numpy, to_list, Sequential, Layer
from trident.callbacks import *
from trident.data import *
from trident.data.data_loaders import *
from trident.data.image_common import *
from trident.misc import *

if get_backend()=='pytorch':
    from trident.backend.pytorch_backend import *
    from trident.backend.pytorch_ops import *
    from ..layers.pytorch_activations import *
    from ..layers.pytorch_layers import *
    from ..layers.pytorch_pooling import *
    from ..layers.pytorch_blocks import *
    from ..layers.pytorch_normalizations import *
    from ..optims.pytorch_regularizers import *

    from ..optims.pytorch_constraints import *
    from ..optims.pytorch_losses import *
    from ..optims.pytorch_metrics import *
    from ..optims.pytorch_optimizers import *
    from ..optims.pytorch_trainer import *
    from ..data.pytorch_datasets import *


elif get_backend()=='tensorflow':
    from trident.backend.tensorflow_backend import *
    from trident.backend.tensorflow_ops import *
    from trident.backend.tensorflow_serialization import *
    from ..layers.tensorflow_activations import *

    from ..layers.tensorflow_layers import *
    from ..layers.tensorflow_pooling import *
    from ..layers.tensorflow_blocks import *
    from ..layers.tensorflow_normalizations import *

    from ..optims.tensorflow_regularizers import*
    from ..optims.tensorflow_constraints import *
    from ..optims.tensorflow_losses import *
    from ..optims.tensorflow_metrics import *
    from ..optims.tensorflow_trainer import *
    from ..optims.tensorflow_optimizers import *




from  ..optims.trainers import *
from ..optims.trainers import *

from ..misc.ipython_utils import *
from ..misc.visualization_utils import *
from .iteration_tools import *
from ..callbacks  import *


