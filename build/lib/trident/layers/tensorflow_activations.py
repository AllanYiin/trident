from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..backend.common import *

import numpy as np
import six
import tensorflow as tf
from tensorflow.keras import backend as K


__all__ = ['Identity','Sigmoid','Tanh','Relu','Relu6','LeakyRelu','LeakyRelu6','SmoothRelu','PRelu','Swish','Elu','HardSigmoid','HardSwish','Selu','LecunTanh','SoftSign','SoftPlus','HardTanh','Logit','LogLog','Mish','Softmax','identity','sigmoid','tanh','relu','relu6','leaky_relu','leaky_relu6','smooth_relu','prelu','swish','elu','hard_sigmoid','hard_swish','selu','lecun_tanh','softsign','softplus','hard_tanh','logit','loglog','mish','softmax','get_activation']


def identity(x,name='identity'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.linear(x)

Identity=tf.keras.layers.Lambda(identity)

def sigmoid(x,name='sigmoid'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.sigmoid(x)

Sigmoid=tf.keras.layers.Lambda(sigmoid)

def tanh(x,name='tanh'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.tanh(x)

Tanh=tf.keras.layers.Lambda(tanh)
def relu(x,upper_limit=None,name='relu'):
    if upper_limit<=0:
        raise ValueError('Upper limit should greater than 0!')
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.keras.activations.relu(x),0,upper_limit)
        return tf.keras.activations.relu(x)

def relu6(x,name='relu6'):
    with tf.keras.backend.name_scope(name)as scope:
        return K.clip(tf.keras.activations.relu(x),0,6)


Relu=tf.keras.layers.ReLU
Relu6=tf.keras.layers.Lambda(relu6)

def leaky_relu(x,alpha=0.01,upper_limit=None,name='leaky_relu'):
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.keras.activations.relu(x,alpha), -np.inf, upper_limit)
        return tf.keras.activations.relu(x,alpha)

def leaky_relu6(x,alpha=0.01,name='leaky_relu'):
    with tf.keras.backend.name_scope(name)as scope:
        return K.clip(tf.keras.activations.relu(x,alpha), -6, 6)

LeakyRelu=tf.keras.layers.LeakyReLU
LeakyRelu6=tf.keras.layers.Lambda(leaky_relu6)

def elu(x,alpha=0.01,upper_limit=None,name='elu'):
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.keras.activations.elu(x,alpha),-np.inf,upper_limit)
        return tf.keras.activations.elu(x,alpha)

Elu=tf.keras.layers.ELU
lrelu=leaky_relu


def smooth_relu(x,upper_limit=None,name='smooth_relu'):
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.math.log(1 + tf.math.exp(x)),-np.inf,upper_limit)
        return tf.math.log(1 + tf.math.exp(x))
SmoothRelu=tf.keras.layers.Lambda(smooth_relu)

def prelu(x,upper_limit=None,name='prelu'):
    with tf.keras.backend.name_scope(name)as scope:
        if upper_limit is not None:
            return K.clip(tf.keras.layers.PReLU()(x),-np.inf,upper_limit)
        return tf.keras.layers.PReLU()(x)
PRelu=tf.keras.layers.PReLU

def swish(x,name='swish'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.sigmoid(x) * x

Swish=tf.keras.layers.Lambda(swish)


def selu(x,name='selu'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.selu(x)

Selu=tf.keras.layers.Lambda(selu)



def lecun_tanh(x,name='lecun_tanh'):
    with tf.keras.backend.name_scope(name)as scope:
        return 1.7159 * tf.keras.activations.tanh(2/3 * x)

LecunTanh=tf.keras.layers.Lambda(lecun_tanh)

def softsign(x,name='softsign'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.softsign(x)
SoftSign=tf.keras.layers.Lambda(softsign)

def softplus(x,name='softplus'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.softplus(x)
SoftPlus=tf.keras.layers.Lambda(softplus)

def hard_sigmoid(x,name='hard_sigmoid'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.hard_sigmoid(x)
HardSigmoid=tf.keras.layers.Lambda(hard_sigmoid)


def hard_tanh(x,name='hard_tanh'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.backend.clip(x,-1,1)
HardTanh=tf.keras.layers.Lambda(hard_tanh)


def hard_swish(x,name='hard_tanh'):
    with tf.keras.backend.name_scope(name)as scope:
        return  x * hard_sigmoid(x)
HardSwish=tf.keras.layers.Lambda(hard_swish)


def logit(x,name='logit'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.math.log.log(x / (1 - x))

Logit=tf.keras.layers.Lambda(logit)

def loglog(x,name='loglog'):
    with tf.keras.backend.name_scope(name)as scope:
        return  1-tf.math.exp(-tf.math.exp(x))

LogLog=tf.keras.layers.Lambda(loglog)


def softmax(x,name='softmax'):
    with tf.keras.backend.name_scope(name)as scope:
        return tf.keras.activations.softmax(x)

Softmax=tf.keras.layers.Softmax

def mish(x,name='mish'):
    with tf.keras.backend.name_scope(name)as scope:
        return x*tf.keras.activations.tanh(tf.keras.activations.softplus(x))

Mish=tf.keras.layers.Lambda(mish)

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

