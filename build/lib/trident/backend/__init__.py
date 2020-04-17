from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from .load_backend import get_backend, get_image_backend, PrintException, if_else
from .load_backend import get_session, get_trident_dir, epsilon, set_epsilon, floatx, set_floatx, camel2snake, \
    snake2camel, addindent, format_time, get_time_suffix, get_function, get_class, get_terminal_size, gcd, \
    get_divisors, \
    isprime, next_prime, prev_prime, nearest_prime
from .load_backend import to_tensor, to_numpy, to_list, Sequential, Layer
from ..callbacks import *
from ..data import *
from ..data.data_loaders import *
from ..data.image_common import *
from ..misc import *

if get_backend()=='pytorch':
    from .pytorch_backend import *
    from .pytorch_ops import *
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


elif get_backend()=='cntk':
    from .cntk_backend import *
    from ..layers.cntk_activations import *
    from ..layers.cntk_normalizations import *
    from ..layers.cntk_layers import *
    from ..layers.cntk_blocks import *
    from ..optims.cntk_optimizers import  *
    from ..optims.cntk_trainer import *
    from ..optims.cntk_losses import *
    from ..optims.cntk_metrics import *
    from ..optims.cntk_constraints import *
    from ..optims.cntk_regularizers import *

elif get_backend()=='tensorflow':
    from .tensorflow_backend import *
    from .tensorflow_ops import *
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


