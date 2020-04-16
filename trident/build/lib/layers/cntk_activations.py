from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from backend.common import  epsilon,floatx

import cntk as C
import six
from cntk.internal import sanitize_input


@C.typemap
def identity(x,name=''):
    @C.BlockFunction('identity', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return x
    return apply_x(x)

@C.typemap
def sigmoid(x,name=''):
    @C.BlockFunction('sigmoid', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.sigmoid(x)
    return apply_x(x)

@C.typemap
def tanh(x,name=''):
    @C.BlockFunction('tanh', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.tanh(x)
    return apply_x(x)

@C.typemap
def relu(x,upper_limit=None,name=''):
    if upper_limit<=0:
        raise ValueError('Upper limit should greater than 0!')
    @C.BlockFunction('relu', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        if upper_limit is not None:
            return C.clip(C.relu(x),0,upper_limit)
        return C.relu(x)
    return apply_x(x)

@C.typemap
def leaky_relu(x,alpha=0.01,name=''):
    @C.BlockFunction('leaky_relu', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.leaky_relu(x,alpha)
    return apply_x(x)

@C.typemap
def elu(x,alpha=0.01,name=''):
    @C.BlockFunction('elu', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.elu(x,alpha)
    return apply_x(x)

@C.typemap
def lrelu(x,leak=0.2,name=''):
    @C.BlockFunction('lrelu', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * C.abs(x)
    return apply_x(x)

@C.typemap
def smooth_relu(x,name=''):
    @C.BlockFunction('smooth_relu', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.log(1 + C.exp(x))
    return apply_x(x)

@C.typemap
def prelu(x,name=''):
    @C.BlockFunction('prelu', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        alpha=C.Parameter(x.shape, init=C.he_normal())
        return C.param_relu(alpha,x)
    return apply_x(x)

@C.typemap
def swish(x,name=''):
    @C.BlockFunction('swish', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        r = C.element_times(x, C.sigmoid(x))
        return r
    return apply_x(x)


@C.typemap
def selu(x,name=''):
    @C.BlockFunction('selu', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.selu(x)
    return apply_x(x)

@C.typemap
def lecun_tanh(x,name=''):
    @C.BlockFunction('lecun_tanh', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return 1.7159 * C.tanh(2/3 * x)
    return apply_x(x)

@C.typemap
def softsign(x,name=''):
    @C.BlockFunction('softsign', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.log(1 + C.exp(x))
    return apply_x(x)

@C.typemap
def softplus(x,name=''):
    @C.BlockFunction('softplus', name=name)
    def apply_x(x):
        return C.log(C.exp(x) + 1)
    return apply_x(x)


@C.typemap
def hard_sigmoid(x,name=''):
    @C.BlockFunction('hard_sigmoid', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        x = (0.2 * x) + 0.5
        x = C.clip(x, 0.0, 1.0)
        return x
    return apply_x(x)

@C.typemap
def hard_tanh(x,name=''):
    @C.BlockFunction('hard_tanh', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.clip(x,-1,1)
    return apply_x(x)


@C.typemap
def logit(x,name=''):
    @C.BlockFunction('logit', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.log(x / (1 - x))
    return apply_x(x)



@C.typemap
def loglog(x,name=''):
    @C.BlockFunction('loglog', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return 1-C.exp(-C.exp(x))
    return apply_x(x)


@C.typemap
def softmax(x,name=''):
    @C.BlockFunction('softmax', name=name)
    def apply_x(x):
        x = sanitize_input(x)
        return C.softmax(x)
    return apply_x(x)



def get_activation(identifier):
    if identifier is None:
        return identity()
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        fn=relu
        mod=fn.__module__
        obj_dict = fn.__globals__
        for k, v in obj_dict.items():
            if k == identifier and mod=='trident.backend.cntk_activations':
                fn = v
                return fn
        raise ValueError('Not valid activation functions name : ' + str(identifier))



# def get(identifier):
#
#     """Get the `identifier` activation function.
#
#
#
#     # Arguments
#
#         identifier: None or str, name of the function.
#
#
#
#     # Returns
#
#         The activation function, `linear` if `identifier` is None.
#
#
#
#     # Raises
#
#         ValueError if unknown identifier
#
#     """
#     if identifier is None:
#         return C.ops.lea
#     if isinstance(identifier, six.string_types):
#         identifier = str(identifier)
#         return deserialize(identifier)
#     elif callable(identifier):
#         if isinstance(identifier, Layer):
#             warnings.warn(
#                 'Do not pass a layer instance (such as {identifier}) as the '
#                 'activation argument of another layer. Instead, advanced '
#                 'activation layers should be used just like any other '
#                 'layer in a model.'.format(
#                     identifier=identifier.__class__.__name__))
#         return identifier
#     else:
#         raise ValueError('Could not interpret '
#                          'activation function identifier:', identifier)