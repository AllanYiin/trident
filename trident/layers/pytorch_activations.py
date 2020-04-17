from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
import string
from functools import partial
from pydoc import locate

import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as af
from torch.nn.parameter import Parameter

from ..backend.common import get_function, get_class, camel2snake, enforce_singleton
from ..backend.pytorch_backend import Layer

__all__ = ['Identity', 'Sigmoid', 'Tanh', 'Relu', 'Relu6', 'LeakyRelu', 'LeakyRelu6', 'SmoothRelu', 'PRelu', 'Swish',
           'Elu', 'HardSigmoid', 'HardSwish', 'Selu', 'LecunTanh', 'SoftSign', 'SoftPlus', 'HardTanh', 'Logit',
           'LogLog', 'Mish', 'Softmax', 'BertGelu', 'GptGelu', 'identity', 'sigmoid', 'tanh', 'relu', 'relu6',
           'leaky_relu', 'leaky_relu6', 'smooth_relu', 'p_relu', 'swish', 'elu', 'hard_sigmoid', 'hard_swish', 'selu',
           'lecun_tanh', 'soft_sign', 'soft_plus', 'hard_tanh', 'logit', 'log_log', 'mish', 'softmax', 'bert_gelu',
           'gpt_gelu', 'get_activation']

'''
'''


class Identity(Layer):
    def __init__(self, name=''):
        super(Identity, self).__init__(name=name)

    def forward(self, x):
        return x


'''identity activation function 
'''


def identity(x):
    return x


class Relu(Layer):
    def __init__(self, name=''):
        super(Relu, self).__init__(name=name)

    def forward(self, *x):
        x = enforce_singleton(x)
        return relu(x)


'''relu activation function 
'''


def relu(x):
    return torch.relu(x)


class Relu6(Layer):
    def __init__(self, ):
        super(Relu6, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return relu6(x)


'''relu6 activation function 
'''


def relu6(x):
    return F.relu6(x)


class LeakyRelu(Layer):
    def __init__(self, ):
        super(LeakyRelu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return leaky_relu(x)


'''leaky_relu activation function 
'''


def leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.2)


class LeakyRelu6(Layer):
    def __init__(self):
        super(LeakyRelu6, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return leaky_relu6(x)


'''leaky_relu6 activation function 
'''


def leaky_relu6(x):
    return torch.clamp(F.leaky_relu(x, negative_slope=0.2), -6, 6)


class SmoothRelu(Layer):
    def __init__(self):
        super(SmoothRelu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return smooth_relu(x)


'''smooth_relu activation function 
'''


def smooth_relu(x):
    return torch.log(1 + torch.exp(x))


'''PRelu activation function Layer
'''



class PRelu(Layer):
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



'''p_relu activation function 
'''
p_relu = torch.prelu

'''Softmax activation function layer
'''


class Sigmoid(Layer):
    """Softmax activation function.
       # Arguments
           x: Input tensor.
           axis: Integer, axis along which the softmax normalization is applied.

       # Returns
           Tensor, output of softmax transformation.

       # Raises
           ValueError: In case `dim(x) == 1`.
       """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return sigmoid(x)


'''softmax activation function 
'''


def sigmoid(x):
    return torch.sigmoid(x)


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return tanh(x)


'''tanh activation function 
'''


def tanh(x):
    return torch.tanh(x)


class Swish(Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return swish(x)


'''swish activation function 
'''


def swish(x):
    return x * sigmoid(x)


class HardSigmoid(Layer):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()

        self.inplace = inplace

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_sigmoid(x, inplace=self.inplace)


def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6


class HardSwish(Layer):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()

        self.inplace = inplace

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_swish(x, inplace=self.inplace)


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardTanh(Layer):
    def __init__(self, ):
        super(HardTanh, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_tanh(x)


def hard_tanh(x):
    return torch.clamp(x, -1, 1)


class Selu(Layer):
    def __init__(self, inplace=False):
        super(Selu, self).__init__()

        self.inplace = inplace

    def forward(self, *x):
        x = enforce_singleton(x)
        return selu(x)


'''selu activation function 
'''


def selu(x):
    return torch.selu(x)


class Elu(Layer):
    def __init__(self):
        super(Elu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return elu(x)


def elu(x):
    return F.elu(x)


class LecunTanh(Layer):
    def __init__(self):
        super(LecunTanh, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return hard_swish(x)


def lecun_tanh(x):
    return 1.7159 * torch.tanh(2 / 3 * x)


class SoftSign(Layer):
    def __init__(self):
        super(SoftSign, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return soft_sign(x)


def soft_sign(x):
    return x.exp().add(1).log()


class SoftPlus(Layer):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return soft_plus(x)


def soft_plus(x):
    return F.softplus(x)


class Logit(Layer):
    def __init__(self, ):
        super(Logit, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return logit(x)


def logit(x):
    return (x / (1 - x)).log()


class LogLog(Layer):
    def __init__(self, ):
        super(LogLog, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return log_log(x)


def log_log(x):
    return 1 - torch.exp(-torch.exp(x))


class Mish(Layer):
    '''
        #Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        #https://arxiv.org/abs/1908.08681v1
    '''

    def __init__(self):
        super().__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return mish(x)


'''mish activation function 
'''


def mish(x):
    return x * (torch.tanh(F.softplus(x)))


class Softmax(Layer):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return softmax(x)


def softmax(x):
    return torch.softmax(x, dim=-1)


class BertGelu(Layer):
    r"""Bert uses GELU as the activation function for the position-wise network.
    """

    def __init__(self):
        super(BertGelu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return bert_gelu(x)


def bert_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GptGelu(Layer):
    r"""For information: OpenAI GPT's GELU is slightly different (and gives
    slightly different results).
    """

    def __init__(self):
        super(GptGelu, self).__init__()

    def forward(self, *x):
        x = enforce_singleton(x)
        return gpt_gelu(x)


def gpt_gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def get_activation(fn_name):
    if fn_name is None:
        return None
    fn_modules = ['trident.layers.pytorch_activations', 'torch.nn.functional']
    try:
        if isinstance(fn_name, str):
            if fn_name.lower() == fn_name:
                activation_fn = get_function(fn_name, [
                    'trident.layers.pytorch_activations'] if fn_name in __all__ else fn_modules)
                return activation_fn
            else:
                try:
                    activation_fn = get_function(camel2snake(fn_name), fn_modules)
                    return activation_fn()
                except Exception:
                    activation_fn = get_class(fn_name, [
                        'trident.layers.pytorch_activations'] if fn_name in __all__ else fn_modules)
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

