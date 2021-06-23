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
from trident.models.pretrained_utils import _make_recovery_model_include_top

from trident.backend.common import *
from trident.backend.tensorflow_backend import *
from trident.backend.tensorflow_ops import *
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive,download_file_from_google_drive
from trident.layers.tensorflow_activations import get_activation, Identity, Relu
from trident.layers.tensorflow_blocks import *
from trident.layers.tensorflow_layers import *
from trident.layers.tensorflow_normalizations import get_normalization, BatchNorm
from trident.layers.tensorflow_pooling import GlobalAvgPool2d, MaxPool2d
from trident.optims.tensorflow_trainer import *
from trident.data.vision_transforms import Resize,Normalize
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



def basic_block(num_filters=64,base_width=64,strides=1,expansion = 4,conv_shortcut=False,use_bias=False,name=None):
    shortcut = Identity()
    if strides>1 or conv_shortcut is True:
        shortcut =Conv2d_Block((1,1),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias,name=name + '_downsample')

    return ShortCut2d(Sequential(Conv2d_Block((3,3),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation='relu',use_bias=use_bias,name=name + '_0_conv'),
                                 Conv2d_Block((3,3),num_filters=num_filters,strides=1,auto_pad=True,padding_mode='zero',normalization='batch',activation=None,use_bias=use_bias,name=name + '_1_conv')),
                      shortcut,activation='relu',name=name)

def bottleneck(num_filters=64,strides=1,expansion = 4,conv_shortcut=True,use_bias=False,name=None):
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



def ResNet(block, layers, input_shape=(224, 224,3), num_classes=1000, use_bias=False,  include_top=True, model_name='',
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Args
        block: a function that returns output tensor for the stacked residual blocks.
        layers:  list of integer, the number of  repeat units in each blocks.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            It should have exactly 3 inputs channels.
        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        include_top: whether to include the fully-connected layer at the top of the network.
        model_name: string, model name.

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
    resnet.add_module('layer1',(_make_layer(block, 64, layers[0],strides=1, dilate=None,use_bias=use_bias,layer_name='layer1' )))
    resnet.add_module('layer2',(_make_layer(block, 128, layers[1], strides=2, dilate=None,use_bias=use_bias,layer_name='layer2' )))
    resnet.add_module('layer3',(_make_layer(block, 256, layers[2], strides=2, dilate=None,use_bias=use_bias,layer_name='layer3' )))
    resnet.add_module('layer4' ,(_make_layer(block, 512, layers[3], strides=2, dilate=None,use_bias=use_bias,layer_name='layer4' )))

    if include_top:
        resnet.add_module('avg_pool', GlobalAvgPool2d(name='avg_pool'))
        resnet.add_module('fc',Dense(num_classes,activation=None,name='fc'))
        resnet.add_module('softmax', SoftMax(name='softmax'))
    resnet._name=model_name
    model=ImageClassificationModel(input_shape=input_shape,output=resnet)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'imagenet_labels1.txt'), 'r', encoding='utf-8-sig') as f:
        labels = [l.rstrip() for l in f]
        model.class_names=labels
        input_np_shape=to_numpy(input_shape)
    model.preprocess_flow=[Resize((input_np_shape[0],input_np_shape[1]),keep_aspect=True), to_bgr(), Normalize([103.939, 116.779, 123.68], [1, 1, 1])]
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

def ResNet18(include_top=True,
             pretrained=True,
             freeze_features=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(3, 224, 224)
    resnet18 = ResNet(basic_block, [2, 2, 2, 2], input_shape, use_bias=False,include_top=include_top,model_name='resnet18')

    with tf.device(get_device()):
        if pretrained:
            download_model_from_google_drive('1vReSW_l8fldyYQ6ay5HCYFGoMaGbdW2T', dirname, 'resnet50_tf.pth')
            recovery_model = load(os.path.join(dirname, 'resnet50_tf.pth'))
            recovery_model = fix_layer(recovery_model)
            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            resnet18.model = recovery_model
        else:
            resnet18.model = _make_recovery_model_include_top(resnet18.model, include_top=include_top, classes=classes, freeze_features=True)

        resnet18.model.input_shape = input_shape
    return resnet18


def ResNet50(include_top=True,
             pretrained=True,
             freeze_features=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(224, 224,3)
    resnet50 =ResNet(bottleneck, [3, 4, 6, 3], input_shape,num_classes=classes,include_top=include_top, model_name='resnet50')
    with tf.device(get_device()):
        if pretrained:
            download_model_from_google_drive('1vReSW_l8fldyYQ6ay5HCYFGoMaGbdW2T',dirname,'resnet50_tf.pth')
            recovery_model=load(os.path.join(dirname,'resnet50_tf.pth'))
            recovery_model = fix_layer(recovery_model)
            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            resnet50.model = recovery_model
        else:
            resnet50.model = _make_recovery_model_include_top(resnet50.model, include_top=include_top, classes=classes, freeze_features=True)

        resnet50.model.input_shape = input_shape
        return resnet50

def ResNet101(include_top=True,
             pretrained=True,
            freeze_features=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(224, 224,3)
    resnet101 =ResNet(bottleneck, [3, 4, 23, 3], input_shape,num_classes=classes,include_top=include_top, use_bias=True,model_name='resnet101')
    with tf.device(get_device()):
        if pretrained==True:
            download_model_from_google_drive('13QYdFX3CvsNiegi-iUX1PUC0KKKgPNwr',dirname,'resnet101_tf.pth')
            recovery_model=load(os.path.join(dirname,'resnet101_tf.pth'))
            recovery_model = fix_layer(recovery_model)
            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            resnet101.model = recovery_model
        else:
            resnet101.model = _make_recovery_model_include_top(resnet101.model, include_top=include_top, classes=classes, freeze_features=True)

        resnet101.model.input_shape = input_shape
        return resnet101


def ResNet152(include_top=True,
             pretrained=True,
            freeze_features=True,
             input_shape=None,
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(224, 224,3)
    resnet152 =ResNet(bottleneck, [3, 8, 36, 3], input_shape,num_classes=classes,include_top=include_top, model_name='resnet152')
    with tf.device(get_device()):
        if pretrained==True:
            download_model_from_google_drive('1TeVBB5ynW9E4_EgxIdjugLT8oaXnQH_c',dirname,'resnet152_tf.pth')
            recovery_model=load(os.path.join(dirname,'resnet152_tf.pth'))
            recovery_model = fix_layer(recovery_model)
            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            resnet152.model = recovery_model
        else:
            resnet152.model = _make_recovery_model_include_top(resnet152.model, include_top=include_top, classes=classes, freeze_features=True)

        resnet152.model.input_shape = input_shape
        return resnet152


#
#
# resnet34=ResNet(basic_block, [3, 4, 6, 3], (3, 224, 224))
# resnet50=ResNet(bottleneck, [3, 4, 6, 3], (3, 224, 224))
# resnet101=ResNet(bottleneck, [3, 4, 23, 3], (3, 224, 224))
# resnet152=ResNet(bottleneck, [3, 8, 36, 3], (3, 224, 224))

