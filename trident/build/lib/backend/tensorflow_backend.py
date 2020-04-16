from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf

from layers.tensorflow_activations import *
from layers.tensorflow_normalizations import *

from layers.tensorflow_activations import *
from layers.tensorflow_normalizations import *
from data.tensorflow_datasets import *



version=tf.version
sys.stderr.write('Tensorflow version:{0}.\n'.format(version.VERSION))

if version.VERSION<'2.0.0':
    raise ValueError('Not support Tensorflow below 2.0' )
