import os
os.environ['TRIDENT_BACKEND']='cntk'
from trident import get_backend as T
from trident import layers
from trident.layers import *
import cntk as C

import cntk as C
import numpy as np



def gcd_conv_in(x, filter_shape, num_filters, dilation=1,strides=(1, 1),padding='same', init=C.he_normal(0.002), bias=False,divisor_rank=0,name=''):
    x=T.gcd_conv2d(x,filter_shape=filter_shape, num_filters=num_filters, strides=1, padding=padding, activation=None,divisor_rank=divisor_rank)
    x = T.InstanceNormalization()(x)
    return x

def gcd_conv_in_relu(x, filter_shape, num_filters, dilation=1,strides=(1, 1),padding='same', init=C.he_normal(0.002), divisor_rank=0,bias=False,name=''):
    x=T.gcd_conv2d(x,filter_shape=filter_shape, num_filters=num_filters, strides=1, padding=padding, activation=None,divisor_rank=divisor_rank,name=name)
    x = T.InstanceNormalization()(x)
    x=T.leaky_relu(x)
    return x

def conv_in(x, filter_shape=(3,3), num_filters=None, dilation=1,strides=(1, 1),padding='same',init=C.he_normal(0.002), bias=False,name=''):
    x=T.conv2d(x,filter_shape=filter_shape,num_filters=num_filters, activation=None, strides=strides, padding=padding,init=init, bias=bias,dilation=dilation,name=name)(x)
    x =T.InstanceNormalization()(x)
    return x


def conv_in_relu(x, filter_shape=(3,3), num_filters=None, dilation=1,strides=(1, 1),padding='same',init=C.he_normal(0.002), bias=False,name=''):
    x = T.conv2d(x, filter_shape=filter_shape, num_filters=num_filters, activation=None, strides=strides,
                 padding=padding, init=init, bias=bias, dilation=dilation, name=name)(x)
    x = T.InstanceNormalization()(x)
    return T.leaky_relu(x)


def squeeze_excite(x,reduction_channel):
    in_channel=x.shape[0]
    s=C.reduce_mean(x,[1,2],keepdims=True)
    #squeeze
    s = T.gcd_conv2d(s,filter_shape=(3,3), num_filters=reduction_channel, strides=1, padding='same', activation=None)
    s=T.leaky_relu(s,upper_limit=6)
    #excite
    s = T.gcd_conv2d(s,filter_shape=(1,1), num_filters=in_channel, strides=1, padding='same', activation=None)
    s=C.sigmoid(s)
    return x+s*x

#[3,5,7]
#7,11,13
#19,31,37
def efficirent_block_fixation( x,base=16,ratios=None,filter_shape=(3,3), stride=(1,1), is_shortcut=False, drop_connectt_rate=1,name=''):
    in_filters=x.shape[0]
    if ratios is None:
        ratios=[3,5,7]

    x=gcd_conv_in_relu(x, filter_shape, num_filters=base*ratios[0], dilation=1,strides=(1, 1),padding='same', init=C.he_normal(0.002), bias=False,name='')
    x1= gcd_conv_in_relu(x, filter_shape, num_filters=base * ratios[1], dilation=1, strides=(1, 1), padding='same', init=C.he_normal(0.002), bias=False, name='')
    x1 = gcd_conv_in_relu(x1, filter_shape, num_filters=base * ratios[0], dilation=1, strides=(1, 1), padding='same',init=C.he_normal(0.002), bias=False, name='')
    if is_shortcut:
        x1=x1+(1-drop_connectt_rate)*x
    x1 = gcd_conv_in_relu(x1, filter_shape, num_filters=base * ratios[0], dilation=1, strides=(1, 1), padding='same', init=C.he_normal(0.002), bias=False, name='')
    return x1

def efficirent_block_expasion( x,base=16,ratios=None,internal_filters=None, out_filters=None,filter_shape=(3,3), stride=(1,1), is_shortcut=False, drop_connectt_rate=1,name=''):
    in_filters=x.shape[0]
    if ratios is None:
        ratios=[3,5,7]

    x=gcd_conv_in_relu(x, filter_shape, num_filters=base*ratios[0], dilation=1,strides=(1, 1),padding='same', init=C.he_normal(0.002), bias=False,name='')
    x1= gcd_conv_in_relu(x, filter_shape, num_filters=base * ratios[1], dilation=1, strides=(1, 1), padding='same', init=C.he_normal(0.002), bias=False, name='')
    x1 = gcd_conv_in_relu(x1, filter_shape, num_filters=base * ratios[0], dilation=1, strides=(1, 1), padding='same',init=C.he_normal(0.002), bias=False, name='')
    x1 = squeeze_excite(x1, base * ratios[0] // 4)
    if is_shortcut:
        x1=x1+(1-drop_connectt_rate)*x
    x = gcd_conv_in_relu(x, filter_shape, num_filters=base * ratios[0], dilation=1, strides=(2, 2), padding='same', init=C.he_normal(0.002), bias=False, name='')
    x1 = gcd_conv_in_relu(x1, filter_shape, num_filters=base * ratios[2], dilation=1, strides=(1, 1), padding='same', init=C.he_normal(0.002), bias=False, name='')
    return x1

def efficirent_block_deflation( x,base=16,ratios=None,internal_filters=None, out_filters=None,filter_shape=(3,3), stride=(1,1), is_shortcut=False, drop_connectt_rate=1,name=''):
    in_filters=x.shape[0]
    if ratios is None:
        ratios=[3,5,7]

    x=gcd_conv_in_relu(x, filter_shape, num_filters=base*ratios[2], dilation=1,strides=(1, 1),padding='same', init=C.he_normal(0.002), bias=False,name='')
    x1= gcd_conv_in_relu(x, filter_shape, num_filters=base * ratios[1], dilation=1, strides=(1, 1), padding='same', init=C.he_normal(0.002), bias=False, name='')
    x1 = gcd_conv_in_relu(x1, filter_shape, num_filters=base * ratios[2], dilation=1, strides=(1, 1), padding='same',init=C.he_normal(0.002), bias=False, name='')
    x1 = squeeze_excite(x1, base * ratios[2] // 4)
    if is_shortcut:
        x1=x1+(1-drop_connectt_rate)*x
    x1 = gcd_conv_in_relu(x1, filter_shape, num_filters=base * ratios[0], dilation=1, strides=(1, 1), padding='same', init=C.he_normal(0.002), bias=False, name='')
    return x1

def backbond(x,d = 0.4,name=''):  #1,128,800
    x=conv_in_relu(x,(3,3),32, strides=1,  name='first_layer')
    x=efficirent_block_fixation(x,16,ratios=[3,5,7],filter_shape=(3,3),stride=(1,1),is_shortcut=True,drop_connectt_rate=0.4*1/7)#112
    x=efficirent_block_fixation(x,16,ratios=[3,5,7],filter_shape=(3,3),stride=(1,1),is_shortcut=True,drop_connectt_rate=0.4*1/7)#112
    x = efficirent_block_fixation(x, 24, ratios=[3, 5, 7], filter_shape=(3, 3), stride=(1, 1), is_shortcut=True,drop_connectt_rate=0.4 * 1 / 7)  # 112
    x = efficirent_block_expasion(x, 24, ratios=[3, 5, 7], filter_shape=(3, 3), stride=(1, 1), is_shortcut=False,drop_connectt_rate=0.4 * 1 / 7)  # 112
    x =efficirent_block_fixation(x, 32,  ratios=[3,5,7],filter_shape=(3,3),stride=(1,1),is_shortcut=True,drop_connectt_rate=0.4*2/7)
    x = efficirent_block_fixation(x, 32, ratios=[3,5,7], filter_shape=(3, 3), stride=(1, 1), is_shortcut=True,drop_connectt_rate=0.4 * 2 / 7)
    x = efficirent_block_fixation(x, 40, ratios=[3,5,7], filter_shape=(3, 3), stride=(1, 1), is_shortcut=True, drop_connectt_rate=0.4 * 2 / 7)
    x = efficirent_block_expasion(x,40,ratios=[3,5,7],filter_shape=(3,3),stride=(1,1),is_shortcut=False,drop_connectt_rate=0.4*2/7)
    x = efficirent_block_fixation(x, 48, ratios=[3,5,7], filter_shape=(3, 3), stride=(1, 1), is_shortcut=True,drop_connectt_rate=0.4 * 2 / 7)
    x = efficirent_block_expasion(x,48, ratios=[3,5,7], filter_shape=(3, 3), stride=(1, 1), is_shortcut=False,drop_connectt_rate=0.4 * 2 / 7)
    x = gcd_conv_in_relu(x, (3, 3), x.shape[0//19], strides=1, name='backend_end')
    return x


input_var=C.input_variable((3,128,200))
z=backbond(input_var)



def upsampling_resize(x):
    xr = C.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2], 1))
    xx = C.splice(xr, xr, axis=-1)  # axis=-1 refers to the last axis
    xy = C.splice(xx, xx, axis=-3)  # axis=-3 refers to the middle axis
    r = C.reshape(xy, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2))
    return r

# def aspp(input, name=''):
#     img_features =C.layers.AveragePooling((2, 2),(2, 2), pad=True)(input)
#     img_features = gcd_conv_in_relu(img_features, (3, 3), 32, strides=1, name='img_features')
#     img_features = upsampling_resize(img_features)
#     # print(img_features.shape)
#
#     atrous_block1 =gcd_conv_in_relu(input, (3, 3), 32, strides=1, name='img_features')
#     # print(atrous_block1.shape)
#
#     atrous_block6 = depthwise_conv2d(input,(3,3),1,1,name='atrous_block6')
#     atrous_block6 = InstanceNormalization()(atrous_block6)
#     atrous_block6 = Convolution((3, 3), 128, activation=leaky_leaky_relu, pad=True, strides=1, bias=False, dilation=6,  init=C.he_normal(0.002),name='atrous_block6_dilation')(atrous_block6)
#     # print(atrous_block6.shape)
#     atrous_block12 = depthwise_conv2d(input,(3,3),1,1,name='atrous_block12')
#     atrous_block12 = InstanceNormalization()(atrous_block12)
#     atrous_block12 = Convolution((3, 3), 128, activation=leaky_leaky_relu, pad=True, strides=1, bias=False, dilation=12,  init=C.he_normal(0.002),name='atrous_block12_dilation')(atrous_block12)
#     # print(atrous_block12.shape)
#     atrous_block18 = depthwise_conv2d(input,(3,3),1,1,name='atrous_block18')
#     atrous_block18 = InstanceNormalization()(atrous_block18)
#     atrous_block18 = Convolution((3, 3), 128, activation=leaky_leaky_relu, pad=True, strides=1, bias=False, dilation=18,  init=C.he_normal(0.002),name='atrous_block18_dilation')(atrous_block18)
#     # print(atrous_block18.shape)
#     net = splice(img_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18, axis=0)
#     net = C.layers.Convolution2D((3,3), 192, activation=leaky_leaky_relu, strides=1,pad=True,bias=False, init=C.he_normal(0.002))(net)
#     net = InstanceNormalization()(net)
#     return _inject_name(net,name)





