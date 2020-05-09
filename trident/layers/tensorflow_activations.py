from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import numpy as np
import six
import six
import tensorflow as tf

from trident.backend.common import *
from trident.backend.tensorflow_backend import Layer, Sequential
from trident.backend.tensorflow_ops import *

__all__ = ['Identity','Sigmoid','Tanh','Relu','Relu6','LeakyRelu','LeakyRelu6','SmoothRelu','PRelu','Swish','Elu','HardSigmoid','HardSwish','Selu','LecunTanh','SoftSign','SoftPlus','HardTanh','Logit','LogLog','Mish','Softmax','BertGELU','GPTGELU','get_activation']



class Identity(Layer):
    def __init__(self,name=None):
        super(Identity, self).__init__(name=name)
    def forward(self, x, mask=None):
        return x






class Sigmoid(Layer):
    def __init__(self,name=None):
        super(Sigmoid, self).__init__(name=name)
    def forward(self, x, mask=None):
        return sigmoid(x)


class Tanh(Layer):
    def __init__(self,name=None):
        super(Tanh, self).__init__(name=name)
    def forward(self, x, mask=None):
        return tanh(x)



class Relu(Layer):
    def __init__(self,name=None):
        super(Relu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return relu(x)



class Relu6(Layer):
    def __init__(self,name=None):
        super(Relu6, self).__init__(name=name)
    def forward(self, x, mask=None):
        return relu6(x)





class LeakyRelu(Layer):
    def __init__(self,alpha=0.02,name=None):
        super(LeakyRelu, self).__init__(name=name)
        self.alpha=alpha

    def forward(self, x, mask=None):
        return leaky_relu(x,alpha=self.alpha)


class LeakyRelu6(Layer):
    def __init__(self,alpha=0.02,name=None):
        super(LeakyRelu6, self).__init__(name=name)
        self.alpha = alpha
    def forward(self, x, mask=None):
        return leaky_relu6(x,alpha=self.alpha)



class Elu(Layer):
    def __init__(self,name=None):
        super(Elu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return elu(x)

class SmoothRelu(Layer):
    def __init__(self,name=None):
        super(SmoothRelu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return smooth_relu(x)




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




class Swish(Layer):
    def __init__(self,name=None):
        super(Swish, self).__init__(name=name)
    def forward(self, x, mask=None):
        return swish(x)




class Selu(Layer):
    def __init__(self,name=None):
        super(Selu, self).__init__(name=name)
    def forward(self, x, mask=None):
        return selu(x)



class LecunTanh(Layer):
    def __init__(self,name=None):
        super(LecunTanh, self).__init__(name=name)
    def forward(self, x, mask=None):
        return lecun_tanh(x)




class SoftSign(Layer):
    def __init__(self,name=None):
        super(SoftSign, self).__init__(name=name)
    def forward(self, x, mask=None):
        return soft_sign(x)



class SoftPlus(Layer):
    def __init__(self,name=None):
        super(SoftPlus, self).__init__(name=name)
    def forward(self, x, mask=None):
        return soft_plus(x)


class HardSigmoid(Layer):
    def __init__(self,name=None):
        super(HardSigmoid, self).__init__(name=name)
    def forward(self, x, mask=None):
        return hard_sigmoid(x)


class HardTanh(Layer):
    def __init__(self,name=None):
        super(HardTanh, self).__init__(name=name)
    def forward(self, x, mask=None):
        return hard_tanh(x)

class HardSwish(Layer):
    def __init__(self,name=None):
        super(HardSwish, self).__init__(name=name)
    def forward(self, x, mask=None):
        return hard_swish(x)



class Logit(Layer):
    def __init__(self,name=None):
        super(Logit, self).__init__(name=name)
    def forward(self, x, mask=None):
        return logit(x)


class LogLog(Layer):
    def __init__(self,name=None):
        super(LogLog, self).__init__(name=name)
    def forward(self, x, mask=None):
        return log_log(x)




class Softmax(Layer):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return tf.nn.softmax(x)




class LogSoftmax(Layer):
    def __init__(self):
        super(LogSoftmax, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return reduce_logsumexp(x)






class Mish(Layer):
    def __init__(self,name=None):
        super(Mish, self).__init__(name=name)
    def forward(self, x, mask=None):
        return mish(x)




class BertGELU(Layer):
    def __init__(self,name=None):
        super(BertGELU, self).__init__(name=name)
    def forward(self, x, mask=None):
        return bert_gelu(x)



class GPTGELU(Layer):
    def __init__(self,name=None):
        super(GPTGELU, self).__init__(name=name)
    def forward(self, x, mask=None):
        return gpt_gelu(x)



def get_activation(fn_name):
    '''
    get the proper activation function

    Args:
        fn_name (string ,function,Type):

    Returns: function or Layer class

    Examples:
    >>> get_activation('relu')
    <function relu at 0x0000021143F49D90>
    >>> get_activation('Relu')
    Relu()
    >>> get_activation(Mish)
    Mish()
    >>> get_activation(LeakyRelu(alpha=0.1))
    LeakyRelu(alpha=0.1)


    '''
    if fn_name is None:
        return None
    elif isinstance(fn_name,Layer):
        return fn_name

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


