from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pydoc import locate
import six
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.activation as af
from ..backend.common import get_function,camel2snake

__all__ = ['Identity','Sigmoid','Tanh','Relu','Relu6','LeakyRelu','LeakyRelu6','SmoothRelu','PRelu','Swish','Elu','HardSigmoid','HardSwish','Selu','LecunTanh','SoftSign','SoftPlus','HardTanh','Logit','LogLog','Mish','Softmax','identity','sigmoid','tanh','relu','relu6','leaky_relu','leaky_relu6','smooth_relu','prelu','swish','elu','hard_sigmoid','hard_swish','selu','lecun_tanh','softsign','softplus','hard_tanh','logit','loglog','mish','softmax','get_activation']


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def identity(x):
    return x


Relu= af.ReLU
relu =F.relu
Relu6 =af.ReLU6
relu6 =F.relu6
leaky_relu = F.leaky_relu
LeakyRelu = af.LeakyReLU

class LeakyRelu6(nn.Module):
    def __init__(self):
        super(LeakyRelu6, self).__init__()
    def forward(self, x):
        return leaky_relu6(x)

def leaky_relu6(x):
    return torch.clamp(F.leaky_relu(x),-6,6)


class SmoothRelu(nn.Module):
    def __init__(self):
        super(SmoothRelu, self).__init__()
    def forward(self, x):
        return smooth_relu(x)

def smooth_relu(x):
    return  torch.log(1 + torch.exp(x))


PRelu=nn.PReLU
prelu=F.prelu


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
    def forward(self, x):
        return sigmoid(x)

sigmoid=torch.sigmoid


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()
    def forward(self, x):
        return tanh(x)

def tanh(x):
    return  x.tanh()


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return swish(x)

def swish(x):
    return x * sigmoid(x)

class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)

def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)

def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardTanh(nn.Module):
    def __init__(self,):
        super(HardTanh, self).__init__()
    def forward(self, x):
        return hard_tanh(x)

def hard_tanh(x):
    return  x.clamp(-1,1)


class Selu(nn.Module):
    def __init__(self, inplace=False):
        super(Selu, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return selu(x, inplace=self.inplace)
selu=F.selu


class Elu(nn.Module):
    def __init__(self, inplace=False):
        super(Elu, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return elu(x, inplace=self.inplace)
elu=F.elu


class LecunTanh(nn.Module):
    def __init__(self):
        super(LecunTanh, self).__init__()
    def forward(self, x):
        return hard_swish(x)

def lecun_tanh(x):
    return 1.7159 * F.tanh(2 / 3 * x)


class SoftSign(nn.Module):
    def __init__(self):
        super(SoftSign, self).__init__()
    def forward(self, x):
        return softsign(x)

def softsign(x):
    return x.exp().add(1).log()



class SoftPlus(nn.Module):
    def __init__(self):
        super(SoftSign, self).__init__()
    def forward(self, x):
        return softsign(x)

def softplus(x):
    return x.exp().log().add(1)



class Logit(nn.Module):
    def __init__(self,):
        super(Logit, self).__init__()
    def forward(self, x):
        return logit(x)

def logit(x):
    return  (x / (1 - x)).log()


class LogLog(nn.Module):
    def __init__(self,):
        super(LogLog, self).__init__()
    def forward(self, x):
        return loglog(x)

def loglog(x):
    return   1-torch.exp(-torch.exp(x))




class Mish(nn.Module):
    '''
        #Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        #https://arxiv.org/abs/1908.08681v1
    '''
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return mish(x)

def mish(x):
    return x *( torch.tanh(F.softplus(x)))


class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
    def forward(self, x):
        return softmax(x)

def softmax(x):
    return x.softmax()




class BertGELU(nn.Module):
    r"""Bert uses GELU as the activation function for the position-wise network.
    """
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(2.0)))


class GPTGELU(nn.Module):
    r"""For information: OpenAI GPT's GELU is slightly different (and gives
    slightly different results).
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))



# def get_activation(identifier):
#     if identifier is None:
#         return Identity
#     elif isinstance(identifier, six.string_types):
#         identifier = str(identifier)
#         fn=get_activation
#         mod=fn.__module__
#         obj_dict = fn.__globals__
#         for k, v in obj_dict.items():
#             if k.lower() == identifier.lower() and k.lower() !=k and mod=='layers.pytorch_activations':
#                 fn = v
#                 return fn
#         raise ValueError('Not valid activation functions name : ' + str(identifier))
#     elif callable(identifier):
#         return identifier
#     else:
#         return Identity
#

def get_activation(fn_name):
    if fn_name is None:
        return None
    fn_modules = ['trident.layers.pytorch_activations','torch', 'torch.nn']
    activation_fn = get_function(camel2snake(fn_name), fn_modules)
    return activation_fn
