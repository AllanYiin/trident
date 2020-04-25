from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import numpy as np
import six
import six
import tensorflow as tf

from ..backend.common import *
from ..backend.tensorflow_backend import Layer, to_numpy, to_tensor, is_tensor, Sequential
from ..backend.tensorflow_ops import *

__all__ = ['Identity','Sigmoid','Tanh','Relu','Relu6','LeakyRelu','LeakyRelu6','SmoothRelu','PRelu','Swish','Elu','HardSigmoid','HardSwish','Selu','LecunTanh','SoftSign','SoftPlus','HardTanh','Logit','LogLog','Mish','Softmax','BertGELU','GPTGELU','identity','sigmoid','tanh','relu','relu6','leaky_relu','leaky_relu6','smooth_relu','p_relu','swish','elu','hard_sigmoid','hard_swish','selu','lecun_tanh','soft_sign','soft_plus','hard_tanh','logit','log_log','mish','softmax','log_sum_exp','LogSoftmax','bert_gelu','gpt_gelu','get_activation']


def identity(x):
    return x


class Identity(Layer):
    def __init__(self,name=None):
        super(Identity, self).__init__(name=name)
    def forward(self, x, mask=None):
        return x


def sigmoid(x):
    return tf.nn.sigmoid(x)



class Sigmoid(Layer):
    def __init__(self,name=None):
        super(Sigmoid, self).__init__(name=name)
    def forward(self, x, mask=None):
        return sigmoid(x)

def tanh(x):
    return tf.nn.tanh(x)

class Tanh(Layer):
    def __init__(self,name=None):
        super(Tanh, self).__init__(name=name)
    def forward(self, x, mask=None):
        return tanh(x)

def relu(x,upper_limit=None):
    if upper_limit is not None and upper_limit<=0:
        raise ValueError('Upper limit should greater than 0!')
    elif upper_limit is not None:
        return clip(tf.nn.relu(x),0,upper_limit)
    return tf.nn.relu(x)



class Relu(Layer):
    def __init__(self,name=None):
        super(Relu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return relu(x)

def relu6(x):
    return clip(tf.nn.relu(x),0,6)


class Relu6(Layer):
    def __init__(self,name=None):
        super(Relu6, self).__init__(name=name)
    def forward(self, x, mask=None):
        return relu6(x)



def leaky_relu(x,alpha=0.01,upper_limit=None):
    if upper_limit is not None:
        return clip(tf.nn.leaky_relu(x,alpha), -np.inf, upper_limit)
    return tf.nn.leaky_relu(x,alpha)



class LeakyRelu(Layer):
    def __init__(self,name=None):
        super(LeakyRelu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return leaky_relu(x)

def leaky_relu6(x,alpha=0.01):
    return clip(tf.nn.leaky_relu(x,alpha), -6, 6)

class LeakyRelu6(Layer):
    def __init__(self,name=None):
        super(LeakyRelu6, self).__init__(name=name)
    def forward(self, x, mask=None):
        return leaky_relu6(x)


def elu(x,alpha=0.01,upper_limit=None):
    if upper_limit is not None:
        return clip(tf.nn.elu(x,alpha),-np.inf,upper_limit)
    return tf.nn.elu(x,alpha)

class Elu(Layer):
    def __init__(self,name=None):
        super(Elu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return elu(x)

lrelu=leaky_relu


def smooth_relu(x,upper_limit=None):
    if upper_limit is not None:
        return clip(tf.math.log(1 + tf.math.exp(x)),-np.inf,upper_limit)
    return tf.math.log(1 + tf.math.exp(x))

class SmoothRelu(Layer):
    def __init__(self,name=None):
        super(SmoothRelu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return smooth_relu(x)

def p_relu(x,upper_limit=None):
    if upper_limit is not None:
        return clip(tf.keras.layers.PReLU()(x),-np.inf,upper_limit)
    return tf.keras.layers.PReLU()(x)




class PRelu(Layer):
    def __init__(self, num_parameters=1, init=0.25):
        super(PRelu, self).__init__()
        self.num_parameters=num_parameters
        self.init = init
        self.weight =None

    def build(self, input_shape):
        if self._built == False:
            self.weight=tf.Variable(tf.random.normal(shape=[self.num_parameters],mean=0, stddev=1)*self.init, name='weight')
            self._built = True
    def forward(self, *x):
        x = enforce_singleton(x)
        pos = tf.nn.relu(x)
        neg = self.weight * (x - tf.abs(x)) * 0.5
        return pos+neg



def swish(x):
    return tf.nn.sigmoid(x) * x


class Swish(Layer):
    def __init__(self,name=None):
        super(Swish, self).__init__(name=name)
    def forward(self, x, mask=None):
        return swish(x)


def selu(x):
    return tf.nn.selu(x)


class Selu(Layer):
    def __init__(self,name=None):
        super(Selu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return selu(x)



def lecun_tanh(x):
    return 1.7159 * tf.nn.tanh(2/3 * x)

class LecunTanh(Layer):
    def __init__(self,name=None):
        super(LecunTanh, self).__init__(name=name)
    def forward(self, x, mask=None):
        return lecun_tanh(x)



def soft_sign(x):
    return tf.nn.softsign(x)

class SoftSign(Layer):
    def __init__(self,name=None):
        super(SoftSign, self).__init__(name=name)
    def forward(self, x, mask=None):
        return soft_sign(x)


def soft_plus(x):
    return tf.nn.softplus(x)

class SoftPlus(Layer):
    def __init__(self,name=None):
        super(SoftPlus, self).__init__(name=name)
    def forward(self, x, mask=None):
        return soft_plus(x)

def hard_sigmoid(x):
    return relu6(x+3)/6

class HardSigmoid(Layer):
    def __init__(self,name=None):
        super(HardSigmoid, self).__init__(name=name)
    def forward(self, x, mask=None):
        return hard_sigmoid(x)

def hard_tanh(x):
    return tf.keras.backend.clip(x,-1,1)

class HardTanh(Layer):
    def __init__(self,name=None):
        super(HardTanh, self).__init__(name=name)
    def forward(self, x, mask=None):
        return hard_tanh(x)

def hard_swish(x):
    return  x * hard_sigmoid(x)

class HardSwish(Layer):
    def __init__(self,name=None):
        super(HardSwish, self).__init__(name=name)
    def forward(self, x, mask=None):
        return hard_swish(x)


def logit(x):
        return tf.math.log(x / (1 - x))


class Logit(Layer):
    def __init__(self,name=None):
        super(Logit, self).__init__(name=name)
    def forward(self, x, mask=None):
        return logit(x)


def log_log(x):
    return  1-tf.math.exp(-tf.math.exp(x))

class LogLog(Layer):
    def __init__(self,name=None):
        super(LogLog, self).__init__(name=name)
    def forward(self, x, mask=None):
        return log_log(x)



def softmax(x,axis=-1):
    return tf.nn.softmax(x,axis=axis)

class Softmax(Layer):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return tf.nn.softmax(x)

def log_sum_exp(x):
    """Activation function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x : input tensor
    """
    x_max = x.data.max()
    return log(reduce_sum(exp(x-x_max), 1, keepdims=True)) + x_max


class LogSoftmax(Layer):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return log_sum_exp(x)




def mish(x):
    return x*tf.nn.tanh(tf.nn.softplus(x))



class Mish(Layer):
    def __init__(self,name=None):
        super(Mish, self).__init__(name=name)
    def forward(self, x, mask=None):
        return mish(x)



def bert_gelu(x):

  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  return x *  0.5 * (1.0 + tf.nn.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))


class BertGELU(Layer):
    def __init__(self,name=None):
        super(BertGELU, self).__init__(name=name)
    def forward(self, x, mask=None):
        return bert_gelu(x)



def gpt_gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 /np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))

class GPTGELU(Layer):
    def __init__(self,name=None):
        super(GPTGELU, self).__init__(name=name)
    def forward(self, x, mask=None):
        return gpt_gelu(x)



def get_activation(fn_name):
    if fn_name is None:
        return None
    fn_modules = ['trident.layers.tensorflow_activations']
    try:
        if isinstance(fn_name,str):
            if fn_name.lower()==fn_name:
                 activation_fn = get_function(fn_name, ['trident.layers.tensorflow_activations'] if fn_name in __all__ else  fn_modules)
                 return activation_fn
            else:
                try:
                    activation_fn =  get_function(camel2snake(fn_name), fn_modules)
                    return activation_fn()
                except Exception:
                    activation_fn = tf.keras.activations.get(fn_name)
                    return activation_fn

        elif getattr(fn_name, '__module__', None) == 'trident.layers.tensorflow_activations':
            if inspect.isfunction(fn_name):
                return fn_name
            elif isinstance(fn_name, Layer):
                return fn_name()
        else:
            if callable(fn_name) :
                result=inspect.getfullargspec(fn_name)
                if 1<=len(result.args)<=2:
                    return fn_name if inspect.isfunction(fn_name) else fn_name()
                else:
                    raise ValueError('Unknown activation function/ class')
    except Exception:
        return None


