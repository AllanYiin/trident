from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from backend.load_backend import *
import torch
import torch.nn as nn
import torch.nn.functional as F # import torch functions


def _gcd(x, y):
    gcds=[]
    gcd = 1
    if x % y == 0:
        gcds.append(int(y))
    for k in range(int(y //2), 0, -1):
        if x % k == 0 and y % k == 0:
            gcd = k
            gcds.append(int(k))
    return gcds

class GCD_Conv2D(nn.Module):
    def __init__(self,
            filter_shape,
            input_filters=None,
            num_filters =None,
            strides=(1, 1),
            padding='valid',
            activation = None,
            init = 'glorot_uniform',
            use_bias = True,
            init_bias = 'zeros',
            divisor_rank = 0,
            dilation = (1, 1),
            weights_contraint = None ,**kwargs):
        super(GCD_Conv2D, self).__init__()
        self.num_filters = num_filters
        self.input_filters=input_filters
        self.filter_shape = conv_utils.normalize_tuple(filter_shape, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation= conv_utils.normalize_tuple(dilation, 2, 'dilation')
        #self.activation = get_activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(init)
        self.bias_initializer = tf.keras.initializers.get(init_bias)
        self.kernel_regularizer = tf.keras.regularizers.get(None)
        self.bias_regularizer = tf.keras.regularizers.get(None)
        self.activity_regularizer = tf.keras.regularizers.get(None)
        self.kernel_constraint = tf.keras.constraints.get(weights_contraint)
        self.divisor_rank = divisor_rank

        gcd_list = []
        gcd_list.extend(_gcd(self.input_filters, self.num_filters))
        # divisor_rank=min(divisor_rank,len(gcd_list))

        self.gcd = gcd_list[0]
        num_filters_1 = gcd_list[0]
        num_filters_2 = self.num_filters
        if self.input_filters == self.gcd  or self.num_filters == self.gcd :
            self.groups = 1
        else:
            self.groups = gcd_list[min(int(divisor_rank), len(gcd_list))]
        num_filters_per_group = (int(self.input_filters // self.groups),)
        print('input:{0}   output:{1}  gcd:{2}'.format(self.input_filters, self.num_filters , self.groups))

        self.group_conv=nn.Conv2d(num_filters_per_group, num_filters_1, kernel_size=filter_shape, stride=strides, padding=padding, dilation=dilation, groups=self.groups, bias=use_bias)
        self.pointwise_conv=nn.Conv2d(num_filters_1, self.num_filters, kernel_size=(1,1) ,stride=(1,1), padding=padding, dilation=1, groups=1, bias=use_bias)
    def forward(self, x):
        if self.input_filters != self.gcd and self.input_filters != self.gcd:
            x=self.group_conv(x)
        x=self.pointwise_convx()
        return x