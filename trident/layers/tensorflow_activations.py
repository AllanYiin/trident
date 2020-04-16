from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .load_backend import *

import numpy as np
import six
import tensorflow as tf
from tensorflow.keras import backend as K

def identity(x,name='identity'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.linear(x)

def sigmoid(x,name='sigmoid'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.sigmoid(x)

def tanh(x,name='tanh'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.tanh(x)

def relu(x,upper_limit=None,name='relu'):
    if upper_limit<=0:
        raise ValueError('Upper limit should greater than 0!')
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.keras.activations.relu(x),0,upper_limit)
        return tf.keras.activations.relu(x)

def leaky_relu(x,alpha=0.01,upper_limit=None,name='leaky_relu'):
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.keras.activations.relu(x,alpha), -np.inf, upper_limit)
        return tf.keras.activations.relu(x,alpha)

def elu(x,alpha=0.01,upper_limit=None,name='elu'):
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.keras.activations.elu(x,alpha),-np.inf,upper_limit)
        return tf.keras.activations.elu(x,alpha)

def lrelu(x,leak=0.2,upper_limit=None,name='lrelu'):
    with tf.keras.backend.name_scope(name)as scope:
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        if upper_limit is not None:
            return K.clip(f1 * x + f2 * tf.math.abs(x),-np.inf,upper_limit)
        return f1 * x + f2 * tf.math.abs(x)


def smooth_relu(x,upper_limit=None,name='smooth_relu'):
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.math.log(1 + tf.math.exp(x)),-np.inf,upper_limit)
        return tf.math.log(1 + tf.math.exp(x))

def prelu(x,upper_limit=None,name='prelu'):
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.keras.layers.PReLU()(x),-np.inf,upper_limit)
        return tf.keras.layers.PReLU()(x)

def swish(x,name='swish'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.sigmoid(x) * x


def selu(x,name='selu'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.selu(x)


def lecun_tanh(x,name='lecun_tanh'):
    with tf.keras.backend.name_scope(name)as scope:
        return 1.7159 * tf.keras.activations.tanh(2/3 * x)

def softsign(x,name='softsign'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.softsign(x)

def softplus(x,name='softplus'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.softplus(x)

def hard_sigmoid(x,name='hard_sigmoid'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.hard_sigmoid(x)

def hard_tanh(x,name='hard_tanh'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.backend.clip(x,-1,1)


def logit(x,name='logit'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.math.log.log(x / (1 - x))

def loglog(x,name='loglog'):
    with tf.keras.backend.name_scope(name)as scope:
        return  1-tf.math.exp(-tf.math.exp(x))

def softmax(x,name='softmax'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.softmax(x)



def get_activation(identifier):
    if identifier is None:
        return identity
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        fn=identity
        mod=fn.__module__
        obj_dict = fn.__globals__
        for k, v in obj_dict.items():
            if k == identifier and mod=='trident.backend.tensorflow_activations':
                fn = v
                return fn
        raise ValueError('Not valid activation functions name : ' + str(identifier))

