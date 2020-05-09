from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import string
from functools import partial
from pydoc import locate

import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as af
from torch.nn.parameter import Parameter

from trident.backend.common import get_function, get_class, camel2snake, enforce_singleton
from trident.backend.pytorch_backend import Layer
from trident.backend.pytorch_ops import *

__all__ = ['Identity', 'Sigmoid', 'Tanh', 'Relu', 'Relu6', 'LeakyRelu', 'LeakyRelu6', 'SmoothRelu', 'PRelu', 'Swish',
           'Elu', 'HardSigmoid', 'HardSwish', 'Selu', 'LecunTanh', 'SoftSign', 'SoftPlus', 'HardTanh', 'Logit',
           'LogLog', 'Mish', 'Softmax', 'BertGelu', 'GptGelu','LogSoftmax',  'get_activation']



class Identity(Layer):
    '''Identity activation Layer
    '''
    def __init__(self, name=''):
        super(Identity, self).__init__(name=name)

    def forward(self, x):
        return x


class Relu(Layer):
    '''Relu activation Layer
    '''
    def __init__(self, name=''):
        super(Relu, self).__init__(name=name)

    def forward(self, *x):
        x = enforce_singleton(x)
        return relu(x)

class Relu6(Layer):
    '''Relu6 activation Layer
    '''
    def __init__(self, ):
        super(Relu6, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return relu6(x)


class LeakyRelu(Layer):
    '''leaky_relu activation Layer
    '''
    def __init__(self, alpha=0.2):
        super(LeakyRelu, self).__init__()
        self.alpha=alpha

    def forward(self, *x):
        x = enforce_singleton(x)
        return leaky_relu(x,self.alpha)

    def extra_repr(self):
        s = 'alpha={alpha}'
        return s.format(**self.__dict__)


class LeakyRelu6(Layer):
    '''leaky_relu6 activation Layer
    '''
    def __init__(self):
        super(LeakyRelu6, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return leaky_relu6(x)

    def extra_repr(self):
        s = 'alpha={alpha}'
        return s.format(**self.__dict__)

class SmoothRelu(Layer):
    '''smooth_relu activation Layer
    '''
    def __init__(self):
        super(SmoothRelu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return smooth_relu(x)

class PRelu(Layer):
    '''PRelu activation Layer
    '''
    def __init__(self, num_parameters=1, init=0.25):
        super(PRelu, self).__init__()
        self.num_parameters=num_parameters
        self.init = init
        self.weight = Parameter(torch.Tensor(self.num_parameters).fill_(self.init))

    def build(self, input_shape):
        if self._built == False:
            self.weight.to(self.device)
            self._built = True
    def forward(self, *x):
        x = enforce_singleton(x)
        return F.prelu(x, self.weight)




class Sigmoid(Layer):
    """Sigmoid activation layer.
       # Arguments
           x: Input tensor.

       # Returns
           Tensor, output of Sigmoid transformation.

       """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return sigmoid(x)




class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return tanh(x)


class Swish(Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return swish(x)



class HardSigmoid(Layer):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()

        self.inplace = inplace

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_sigmoid(x, inplace=self.inplace)


class HardSwish(Layer):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()

        self.inplace = inplace

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_swish(x, inplace=self.inplace)


class HardTanh(Layer):
    def __init__(self, ):
        super(HardTanh, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_tanh(x)

class Selu(Layer):
    def __init__(self, inplace=False):
        super(Selu, self).__init__()

        self.inplace = inplace

    def forward(self, *x):
        x = enforce_singleton(x)
        return selu(x)

class Elu(Layer):
    def __init__(self):
        super(Elu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return elu(x)


class LecunTanh(Layer):
    def __init__(self):
        super(LecunTanh, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_swish(x)


class SoftSign(Layer):
    def __init__(self):
        super(SoftSign, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return soft_sign(x)

class SoftPlus(Layer):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return soft_plus(x)


class Logit(Layer):
    def __init__(self, ):
        super(Logit, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return logit(x)

class LogLog(Layer):
    def __init__(self, ):
        super(LogLog, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return log_log(x)



class Mish(Layer):
    '''

    '''

    def __init__(self):
        super().__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return mish(x)



class Softmax(Layer):
    """Softmax activation layer.
       # Arguments
           x: Input tensor.
           axis: Integer, axis along which the softmax normalization is applied.

       # Returns
           Tensor, output of softmax transformation.

       # Raises
           ValueError: In case `dim(x) == 1`.
       """
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return softmax(x)



class LogSoftmax(Layer):
    def __init__(self):
        super(LogSoftmax, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return reduce_logsumexp(x)




class BertGelu(Layer):
    r"""Bert uses GELU as the activation function for the position-wise network.
    """

    def __init__(self):
        super(BertGelu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return bert_gelu(x)





class GptGelu(Layer):
    r"""For information: OpenAI GPT's GELU is slightly different (and gives
    slightly different results).
    """

    def __init__(self):
        super(GptGelu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return gpt_gelu(x)




def get_activation(fn_name):
    if fn_name is None:
        return None
    fn_modules = ['trident.layers.pytorch_activations', 'trident.backend.pytorch_ops','torch.nn.functional']
    try:
        if isinstance(fn_name, str):
            if fn_name.lower() == fn_name:
                if fn_name=='p_relu' or fn_name=='prelu':
                    return PRelu()
                activation_fn = get_function(fn_name, [ 'trident.backend.pytorch_ops','trident.layers.pytorch_activations'] if fn_name in __all__ else fn_modules)
                return activation_fn
            else:
                try:
                    activation_fn = get_function(camel2snake(fn_name), fn_modules)
                    return activation_fn()
                except Exception:
                    activation_fn = get_class(fn_name, [
                       'trident.backend.pytorch_ops', 'trident.layers.pytorch_activations'] if fn_name in __all__ else fn_modules)
                    return activation_fn
        elif getattr(fn_name, '__module__', None) == 'trident.layers.pytorch_activations':
            if inspect.isfunction(fn_name):
                return fn_name
            elif inspect.isclass(fn_name) and inspect._is_type(fn_name):
                return fn_name()
            elif isinstance(fn_name, Layer):
                return fn_name
        else:
            if callable(fn_name):
                result = inspect.getfullargspec(fn_name)
                if 1 <= len(result.args) <= 2:
                    return fn_name if inspect.isfunction(fn_name) else fn_name()
                else:
                    raise ValueError('Unknown activation function/ class')
    except Exception:
        return None

