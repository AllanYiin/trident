from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from backend.common import  epsilon,get_function,camel2snake
import numpy as np
import cntk as C
from cntk.ops.functions import ModelFormat, CloneMethod, Function, BlockFunction, load_model, register_native_user_function, native_user_function
from cntk.internal import sanitize_input, sanitize_shape, sanitize_axis, sanitize_dynamic_axes, sanitize_axis_list, sanitize_multi_axis_reduction_list, typemap, sanitize_pooling_args, sanitize_convolution_args, sanitize_permutation, sanitize_dtype_cntk


from cntk.ops.functions import Function, BlockFunction

import six


__all__ = ['Identity','Sigmoid','Tanh','Relu','Relu6','LeakyRelu','LeakyRelu6','SmoothRelu','PRelu','Swish','Elu','HardSigmoid','HardSwish','Selu','LecunTanh','SoftSign','SoftPlus','HardTanh','Logit','LogLog','Mish','Softmax','identity','sigmoid','tanh','relu','relu6','leaky_relu','leaky_relu6','smooth_relu','prelu','swish','elu','hard_sigmoid','hard_swish','selu','lecun_tanh','softsign','softplus','hard_tanh','logit','loglog','mish','softmax','get_activation']

@C.typemap
def Identity(name=''):
    @C.BlockFunction('Identity', name=name)
    def inner(x):
        return  identity(x)
    return inner

def identity(x,name=''):
    return x



@typemap
def Sigmoid(name=''):
    @C.BlockFunction('Sigmoid', name=name)
    def inner(x):
        return  C.sigmoid(x)
    return inner

def sigmoid(x, name=''):
    return C.sigmoid(x)


@typemap
def Tanh(name=''):
    @C.BlockFunction('Tanh', name=name)
    def inner(x):
        return tanh(x)
    return inner
tanh=C.tanh




@typemap
def Relu(upper_limit=np.inf,name=''):
    @C.BlockFunction('Relu', name=name)
    def inner(x):
        return relu(x,upper_limit)
    return inner

def relu(x,upper_limit=np.inf,name=''):
    if upper_limit<=0:
        raise ValueError('Upper limit should greater than 0!')
    if upper_limit is not None:
        return C.clip(C.relu(x),0.,float(upper_limit),name=name)
    return C.relu(x,name=name)

@typemap
def Relu6(upper_limit=np.inf,name=''):
    @C.BlockFunction('Relu6', name=name)
    def inner(x):
        return relu6(x)
    return inner

def relu6(x,name=''):
    return relu(x,6)

@typemap
def LeakyRelu(alpha=0.2,upper_limit=np.inf,name=''):
    @C.BlockFunction('LeakyRelu', name=name)
    def inner(x):
        return leaky_relu(x,alpha,upper_limit)
    return inner

def leaky_relu(x,alpha=0.2,upper_limit=np.inf,name=''):
    if upper_limit<=0:
        raise ValueError('Upper limit should greater than 0!')
    if upper_limit is not None:
        return C.clip(C.leaky_relu(x,alpha),-1*upper_limit,upper_limit,name=name)
    return C.leaky_relu(x,alpha,name=name)


@typemap
def LeakyRelu6(alpha=0.2,name=''):
    @C.BlockFunction('LeakyRelu6', name=name)
    def inner(x):
        return leaky_relu6(x,alpha)
    return inner

def leaky_relu6(x,alpha=0.2, upper_limit=6,name=''):
    return leaky_relu(x,upper_limit=6)



@typemap
def SmoothRelu(name=''):
    @C.BlockFunction('SmoothRelu', name=name)
    def inner(x):
        return smooth_relu(x)
    return inner

def smooth_relu(x,name=''):
     return C.log(1 + C.exp(x))


def PRelu(input_shape,name=''):
    alpha = C.Parameter(input_shape[0], init=C.he_normal())
    @C.BlockFunction('PRelu', name=name)
    def inner(x):
        return prelu(x,alpha)
    return inner


def prelu(x,alpha,name=''):
    return C.param_relu(alpha,x)


@typemap
def Swish(name=''):
    @C.BlockFunction('Swish', name=name)
    def inner(x):
        return swish(x)
    return inner

def swish(x, name=''):
    return  x * C.sigmoid(x)





@typemap
def Elu(name=''):
    @C.BlockFunction('Elu', name=name)
    def inner(x):
        return elu(x)
    return inner
elu=C.elu

@typemap
def HardSigmoid(name=''):
    @C.BlockFunction('HardSigmoid', name=name)
    def inner(x):
        return hard_sigmoid(x)
    return inner

def hard_sigmoid(x, name=''):
    return  relu6(x+3)/6

@typemap
def HardSwish(name=''):
    @C.BlockFunction('HardSwish', name=name)
    def inner(x):
        return hard_swish(x)
    return inner

def hard_swish(x, name=''):
    return  x * hard_sigmoid(x)


@typemap
def Selu(name=''):
    @C.BlockFunction('Selu', name=name)
    def inner(x):
        return selu(x)
    return inner

selu=C.selu

@typemap
def LecunTanh(name=''):
    @C.BlockFunction('LecunTanh', name=name)
    def inner(x):
        return lecun_tanh(x)
    return inner

def lecun_tanh(x, name=''):
    return 1.7159 * C.tanh(2 / 3 * x)


@typemap
def SoftSign(name=''):
    @C.BlockFunction('SoftSign', name=name)
    def inner(x):
        return softsign(x)
    return inner

def softsign(x, name=''):
    return C.log(1 + C.exp(x))


@typemap
def SoftPlus(name=''):
    @C.BlockFunction('SoftPlus', name=name)
    def inner(x):
        return softplus(x)
    return inner

def softplus(x, name=''):
    return C.log(C.exp(x) + 1)




@typemap
def HardTanh(name=''):
    @C.BlockFunction('HardTanh', name=name)
    def inner(x):
        return hard_tanh(x)
    return inner

def hard_tanh(x,name=''):
    return C.clip(x,-1,1)


@typemap
def Logit(name=''):
    @C.BlockFunction('Logit', name=name)
    def inner(x):
        return logit(x)
    return inner

def logit(x,name=''):
    return C.log(x / (1 - x))

@typemap
def LogLog(name=''):
    @C.BlockFunction('LogLog', name=name)
    def inner(x):
        return loglog(x)
    return inner

def loglog(x,name=''):
    return 1-C.exp(-C.exp(x))


@typemap
def Mish(name=''):
    '''
        #Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        #https://arxiv.org/abs/1908.08681v1

    '''
    @C.BlockFunction('Mish', name=name)
    def inner(x):
        return mish(x)
    return inner

def mish(x,name=''):
    return x *( C.tanh(softplus(x)))


@typemap
def Softmax(name=''):
    @C.BlockFunction('Softmax', name=name)
    def inner(x):
        return softmax(x)
    return inner

def softmax(x,name=''):
    return C.softmax(x,0)


#
# def bert_gelu(x)
#     return x * 0.5 * (1.0 + torch.erf(x / C..sqrt(2.0)))
#
#
# class BertGELU(nn.Module):
#     r"""Bert uses GELU as the activation function for the position-wise network.
#     """
#     def forward(self, x):
#
#
#
# class GPTGELU(nn.Module):
#     r"""For information: OpenAI GPT's GELU is slightly different (and gives
#     slightly different results).
#     """
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(
#             math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#
#




def get_activation(fn_name):
    if fn_name is None:
        return None
    fn_modules = ['trident.layers.pytorch_activations','torch', 'torch.nn']
    activation_fn = get_function(camel2snake(fn_name), fn_modules)
    return activation_fn


