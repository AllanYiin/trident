from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf


from ..layers.tensorflow_activations import *
from ..layers.tensorflow_normalizations import *
from ..layers.tensorflow_activations import *
from ..layers.tensorflow_normalizations import *
from ..data.tensorflow_datasets import *


version=tf.version
sys.stderr.write('Tensorflow version:{0}.\n'.format(version.VERSION))

if version.VERSION<'2.0.0':
    raise ValueError('Not support Tensorflow below 2.0' )



def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

