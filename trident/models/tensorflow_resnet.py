from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import itertools
import math
import os
from functools import reduce
from functools import wraps
from itertools import repeat

import tensorflow as tf

from trident.backend.common import *
from trident.backend.tensorflow_backend import *
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive,download_file_from_google_drive
from trident.layers.tensorflow_activations import get_activation, Identity, Relu
from trident.layers.tensorflow_blocks import *
from trident.layers.tensorflow_layers import *
from trident.layers.tensorflow_normalizations import get_normalization, BatchNorm
from trident.layers.tensorflow_pooling import get_pooling, GlobalAvgPool2d, MaxPool2d
from trident.optims.tensorflow_trainer import *

__all__ = ['basic_block','bottleneck', 'ResNet','ResNet50','ResNet101','ResNet152']

_session = get_session()
_epsilon=_session.epsilon
_trident_dir=_session.trident_dir


dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass



def basic_block(num_filters=64,base_width=64,strides=1,expansion = 4,conv_shortcut=False,use_bias=True,name=None):
    shortcut = Identity()
    if strides>1 or conv_shortcut is True:
        shortcut =Conv2d_Block((1,1),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias,name=name + '_downsample')

    return ShortCut2d(Sequential(Conv2d_Block((3,3),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation='relu',use_bias=use_bias,name=name + '_0_conv'),
                                 Conv2d_Block((3,3),num_filters=num_filters,strides=1,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias,name=name + '_1_conv')),
                      shortcut,activation='relu',name=name)

def bottleneck(num_filters=64,strides=1,expansion = 4,conv_shortcut=True,use_bias=True,name=None):
    #width = int(num_filters * (base_width / 64.)) * 1#groups'
    shortcut = Identity()
    shortcut_name='0'
    if conv_shortcut is True:
        shortcut =Conv2d_Block((1,1),num_filters=num_filters*expansion,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias)
        shortcut_name = '0'

    return ShortCut2d({shortcut_name:shortcut,
        '1':Sequential(Conv2d_Block((1,1),num_filters=num_filters ,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation='relu',use_bias=use_bias),
                                 Conv2d_Block((3, 3), num_filters=num_filters , strides=1, auto_pad=True,padding_mode='zero',normalization='batch', activation='relu',use_bias=use_bias,name=name),
                                 Conv2d_Block((1,1),num_filters=num_filters*expansion,strides=1,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias,name=name)),
                     },activation='relu',name=name)



def ResNet(block, layers, input_shape=(224, 224,3), num_classes=1000, use_bias=True, zero_init_residual=False,
           width_per_group=64, replace_stride_with_dilation=None, include_top=True, model_name='',
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Args
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns
        A Keras model instance.
    Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """



    def _make_layer(block, num_filters, blocklayers, strides=1, dilate=False,use_bias=use_bias,layer_name=''):
        layers = OrderedDict()
        layers['0']=block(num_filters=num_filters, strides=strides, expansion = 4, conv_shortcut=True,use_bias=use_bias, name=layer_name+'1')

        for k in range(1, blocklayers):
            layers['{0}'.format(k)]=block(num_filters=num_filters,  strides=1, expansion = 4, conv_shortcut=False, use_bias=use_bias,name=layer_name+'{0}'.format(k+1))

        laters_block=Sequential(layers)
        laters_block._name=layer_name
        return laters_block

    flow_list=[]
    resnet = Sequential()
    resnet.add_module('conv1',Conv2d_Block((7,7),64,strides=2,use_bias=use_bias,auto_pad=True,padding_mode='zero',normalization='batch',activation='relu',name='first_block'))
    resnet.add_module('maxpool',(MaxPool2d((3,3),strides=2,auto_pad=True,padding_mode='zero')))
    resnet.add_module('conv2_block',(_make_layer(block, 64, layers[0],strides=1, dilate=None,use_bias=use_bias,layer_name='conv2_block' )))
    resnet.add_module('conv3_block',(_make_layer(block, 128, layers[1], strides=2, dilate=None,use_bias=use_bias,layer_name='conv3_block' )))
    resnet.add_module('conv4_block',(_make_layer(block, 256, layers[2], strides=2, dilate=None,use_bias=use_bias,layer_name='conv4_block' )))
    resnet.add_module('conv5_block' ,(_make_layer(block, 512, layers[3], strides=2, dilate=None,use_bias=use_bias,layer_name='conv5_block' )))
    resnet.add_module('avg_pool',GlobalAvgPool2d(name='avg_pool'))
    if include_top:
        resnet.add_module('fc',Dense(num_classes,activation=None,name='fc'))
        resnet.add_module('softmax', SoftMax(name='softmax'))
    resnet._name=model_name
    model=ImageClassificationModel(input_shape=input_shape,output=resnet)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'imagenet_labels1.txt'), 'r', encoding='utf-8-sig') as f:
        labels = [l.rstrip() for l in f]
        model.class_names=labels
    model.preprocess_flow=[resize((input_shape[0],input_shape[1]),keep_aspect=True), to_bgr(), normalize([103.939, 116.779, 123.68], [1, 1, 1])]
    #model.summary()
    return model

#
# def ResNet18(include_top=True,
#              weights='imagenet',
#              input_shape=None,
#              classes=1000,
#              **kwargs):
#     if input_shape is not None and len(input_shape)==3:
#         input_shape=tuple(input_shape)
#     else:
#         input_shape=(3, 224, 224)
#     resnet18 = ResNet(basic_block, [2, 2, 2, 2], input_shape, model_name='resnet18')

def ResNet50(include_top=True,
             pretrained=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(224, 224,3)
    resnet50 =ResNet(bottleneck, [3, 4, 6, 3], input_shape,num_classes=classes,include_top=include_top, model_name='resnet50')
    if pretrained==True:
        download_model_from_google_drive('1vReSW_l8fldyYQ6ay5HCYFGoMaGbdW2T',dirname,'resnet50_tf.pth')
        recovery_model=load(os.path.join(dirname,'resnet50_tf.pth'))
        recovery_model.eval()

        if include_top==False:
            recovery_model.__delitem__(-1)
        else:
            if classes!=1000:

                new_fc = Dense(classes, activation=None, name='fc')
                new_fc.input_shape=recovery_model.fc.input_shape
                recovery_model.fc=new_fc


        resnet50.model=recovery_model
        resnet50.rebinding_input_output(input_shape)
        resnet50.signature = get_signature(resnet50.model.forward)
    return resnet50

def ResNet101(include_top=True,
             pretrained=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(224, 224,3)
    resnet101 =ResNet(bottleneck, [3, 4, 23, 3], input_shape,num_classes=classes,include_top=include_top, model_name='resnet101')
    if pretrained==True:
        download_model_from_google_drive('13QYdFX3CvsNiegi-iUX1PUC0KKKgPNwr',dirname,'resnet101_tf.pth')
        recovery_model=load(os.path.join(dirname,'resnet101_tf.pth'))
        recovery_model.eval()
        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                recovery_model.fc = Dense(classes, activation=None, name='fc')

        resnet101.model=recovery_model
        resnet101.rebinding_input_output(input_shape)
        resnet101.signature = get_signature(resnet101.model.forward)
    return resnet101


def ResNet152(include_top=True,
             pretrained=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(224, 224,3)
    resnet152 =ResNet(bottleneck, [3, 8, 36, 3], input_shape,num_classes=classes,include_top=include_top, model_name='resnet152')
    if pretrained==True:
        download_model_from_google_drive('1TeVBB5ynW9E4_EgxIdjugLT8oaXnQH_c',dirname,'resnet152.pth')
        recovery_model=load(os.path.join(dirname,'resnet152.pth'))
        recovery_model.eval()

        if include_top == False:
            recovery_model.__delitem__(-1)
        else:
            if classes != 1000:
                recovery_model.fc = Dense(classes, activation=None, name='fc')

        resnet152.model=recovery_model
        resnet152.rebinding_input_output(input_shape)
        resnet152.signature = get_signature(resnet152.model.forward)
    return resnet152


#
#
# resnet34=ResNet(basic_block, [3, 4, 6, 3], (3, 224, 224))
# resnet50=ResNet(bottleneck, [3, 4, 6, 3], (3, 224, 224))
# resnet101=ResNet(bottleneck, [3, 4, 23, 3], (3, 224, 224))
# resnet152=ResNet(bottleneck, [3, 8, 36, 3], (3, 224, 224))

