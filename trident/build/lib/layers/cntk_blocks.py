from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from backend.common import  epsilon,floatx

from backend.load_backend import *
from layers.cntk_normalizations import  *




def conv2d_block(filter_shape, num_filters, strides=(1, 1), init=C.xavier(0.1),activation='relu', normalization="batch",pad=True, bias=False, dilation=1, drop_rate=0.0, name=''):
    normalization_fn = get_normalization(normalization)
    activation_fn=get_activation(activation)
    def apply_x(x):
        x =C.layers.Convolution2D(filter_shape, num_filters, activation=None, strides=strides, init=init, pad=pad, bias=bias,dilation=dilation,name='')(x)
        if normalization_fn != None:
            x = normalization_fn(x)
        x = activation_fn(x)
        if drop_rate > 0:
            x = Dropout(drop_rate)(x)
        return _inject_name(x, name)

    return apply_x

