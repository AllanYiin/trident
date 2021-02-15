from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from copy import deepcopy
import builtins
import tensorflow as tf
from trident.models.pretrained_utils import _make_recovery_model_include_top

from trident.backend.common import *
from trident.backend.tensorflow_backend import *
from trident.backend.tensorflow_ops import *
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive, unpickle
from trident.layers.tensorflow_activations import Identity
from trident.layers.tensorflow_blocks import *
from trident.layers.tensorflow_layers import *
from trident.layers.tensorflow_pooling import GlobalAvgPool2d
from trident.optims.tensorflow_trainer import *
from trident.data.vision_transforms import Resize,Normalize

__all__ = ['efficient_block','EfficientNet','EfficientNetB0','EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4','EfficientNetB5','EfficientNetB6','EfficientNetB7']


_session = get_session()
_epsilon=_session.epsilon
_trident_dir=_session.trident_dir

DEFAULT_BLOCKS_ARGS = [
        {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16, 'expand_ratio': 1, 'id_skip': True,
         'strides': 1, 'se_ratio': 0.25},
        {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24, 'expand_ratio': 6, 'id_skip': True,
         'strides': 2, 'se_ratio': 0.25},
        {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40, 'expand_ratio': 6, 'id_skip': True,
         'strides': 2, 'se_ratio': 0.25},
        {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80, 'expand_ratio': 6, 'id_skip': True,
         'strides': 2, 'se_ratio': 0.25},
        {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112, 'expand_ratio': 6, 'id_skip': True,
         'strides': 1, 'se_ratio': 0.25},
        {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192, 'expand_ratio': 6, 'id_skip': True,
         'strides': 2, 'se_ratio': 0.25},
        {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320, 'expand_ratio': 6, 'id_skip': True,
         'strides': 1, 'se_ratio': 0.25}]


def efficient_block( expand_ratio=1 , filters_in=32, filters_out=16, kernel_size=3, strides=1, zero_pad=0, se_ratio=0,  drop_rate=0.2,is_shortcut=True,name='',**kwargs):
    expand_ratio=kwargs.get('expand_ratio',expand_ratio)
    is_shortcut=kwargs.get('id_skip',is_shortcut)
    filters_in = kwargs.get('filters_in', filters_in)
    filters_out = kwargs.get('filters_out', filters_out)
    kernel_size = kwargs.get('kernel_size', kernel_size)
    is_shortcut=filters_in==filters_out and strides==1 and kwargs.get('id_skip',is_shortcut)
    filters = filters_in * expand_ratio
    if expand_ratio ==1 and strides==1:

        bottleneck=Sequential(
            DepthwiseConv2d_Block((kernel_size,kernel_size),depth_multiplier=1,strides=strides,auto_pad=True,padding_mode='zero',normalization='batch',activation='swish',name=name + 'dwconv'),
            SqueezeExcite( se_filters= builtins.max(1, int(filters_in * se_ratio)),num_filters=filters_in,use_bias=True) if 0 < se_ratio <= 1 else Identity(),
            Conv2d_Block((1,1),num_filters=filters_out,strides=1,auto_pad=True,normalization='batch', activation=None,name=name + 'se'),
            Dropout(dropout_rate=drop_rate) if is_shortcut and drop_rate > 0 else Identity()
        )


        if is_shortcut:
            return ShortCut2d(Identity(),bottleneck)
        else:
            return bottleneck

    else:
        bottleneck=Sequential(Conv2d_Block((1, 1), num_filters=filters, strides=1, auto_pad=True, normalization='batch', activation='swish' ,name=name + 'expand_bn'),
            DepthwiseConv2d_Block((kernel_size, kernel_size), depth_multiplier=1, strides=strides, auto_pad=True,padding_mode='zero', normalization='batch', activation='swish',name=name + 'dwconv'),
            SqueezeExcite(se_filters= builtins.max(1, int(filters_in * se_ratio)),num_filters=filters,use_bias=True) if 0 < se_ratio <= 1 else Identity(),
            Conv2d_Block((1, 1), num_filters=filters_out, strides=1, auto_pad=True,normalization='batch', activation=None,name=name + 'se'),
            Dropout(dropout_rate=drop_rate) if is_shortcut and drop_rate > 0 else Identity()
        )
        if is_shortcut:
            return ShortCut2d(Identity(),bottleneck)
        else:
            return bottleneck


dirname = os.path.join(_trident_dir, 'models')
if not os.path.exists(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients.

    'Coefficients:
        width,depth,res,dropout
    """
    params_dict = {
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b5': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 model_name='efficientnet',
                 include_top=True,
                 num_classes=1000,**kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in your Keras config at `~/.keras/keras.json`.

    Args
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.

        model_name: string, model name.
        include_top: whether to include the fully-connected layer at the top of the network.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.

        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    Returns
        A Efficientnet model instance.


    """
    default_block_args=deepcopy(DEFAULT_BLOCKS_ARGS)
    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = builtins.max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    flow_list=[]
    efficientnet = Sequential(name=model_name)
    efficientnet.add_module('stem',Conv2d_Block((3,3),round_filters(32),strides=2,use_bias=False,auto_pad=True,padding_mode='zero',normalization='batch',activation='swish',name='stem'))
    b = 0
    blocks = float(builtins.sum(args['repeats'] for args in default_block_args))
    for (i, args) in enumerate(default_block_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        # args['filters_in'] = round_filters(args['filters_in'])
        # args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            efficientnet.add_module('block{}{}'.format(i + 1, chr(j + 97)),efficient_block(expand_ratio=args['expand_ratio'] , filters_in=round_filters(args['filters_in']), filters_out=round_filters(args['filters_out']), kernel_size=args['kernel_size'] , strides=args['strides'] , zero_pad=0, se_ratio=args['se_ratio'] ,drop_connect_rate=drop_connect_rate * b / blocks, name='block{}{}_'.format(i + 1, chr(j + 97)))),
            b += 1
    efficientnet.add_module('top_conv', Conv2d_Block((1, 1), round_filters(1280), strides=1, use_bias=False, auto_pad=True,  padding_mode='zero', normalization='batch', activation='swish', name='top_conv'))
    efficientnet.add_module('avg_pool',GlobalAvgPool2d(name='avg_pool'))
    if include_top:
        if dropout_rate > 0:
            efficientnet.add_module('top_dropout',Dropout(dropout_rate,name='top_dropout'))
        efficientnet.add_module('fc',Dense(num_classes,activation=None,name='fc'))
        efficientnet.add_module('softmax', SoftMax(name='softmax'))
    if isinstance(default_size,int):
        default_size=(default_size,default_size,3)
    elif len(default_size)==1:
        default_size=(default_size[0],default_size[0],3)
    model=ImageClassificationModel(input_shape=default_size,output=efficientnet)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)) ,'imagenet_labels1.txt'), 'r', encoding='utf-8-sig') as f:
        labels = [l.rstrip() for l in f]
        model.class_names=labels
    model.preprocess_flow=[Resize((default_size[0],default_size[1]),keep_aspect=True),Normalize(0,255),Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    return model


def EfficientNetB0(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=(224,224,3),
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(224, 224,3)
    effb0 = EfficientNet(1.0, 1.0, default_size=input_shape, dropout_rate= 0.2, model_name='efficientnet-b0',include_top=include_top, num_classes=classes)
    with tf.device(get_device()):
        if pretrained:
            download_model_from_google_drive('1pO4wRWY6N4e7U_7E2H-NhBPEF4MlR4ru',dirname,'efficientnet-b0_tf.pth')
            recovery_model=load(os.path.join(dirname,'efficientnet-b0_tf.pth'))
            recovery_model = fix_layer(recovery_model)

            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            effb0.model = recovery_model
        else:
            effb0.model = _make_recovery_model_include_top(effb0.model, include_top=include_top, classes=classes, freeze_features=False)

        effb0.model.input_shape = input_shape
        effb0.model.name='efficientnet-b0'
    return effb0


def EfficientNetB1(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=(240,240,3),
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(240, 240,3)
    effb1 =EfficientNet(1.0, 1.1, default_size=input_shape, dropout_rate= 0.2, model_name='efficientnet-b1',include_top=include_top,num_classes=classes)
    with tf.device(get_device()):
        if pretrained==True:
            download_model_from_google_drive('1zCWDn4lwHCn4exAnGfBSPh9YHYTGdIYt', dirname, 'efficientnet-b1_tf.pth')
            recovery_model = load(os.path.join(dirname, 'efficientnet-b1_tf.pth'))
            recovery_model = fix_layer(recovery_model)

            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            effb1.model = recovery_model
        else:
            effb1.model = _make_recovery_model_include_top(effb1.model, include_top=include_top, classes=classes, freeze_features=False)

        effb1.model.input_shape = input_shape
        effb1.model.name = 'efficientnet-b1'
    return effb1


def EfficientNetB2(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=(260,260,3),
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(260, 260,3)
    effb2 =EfficientNet(1.1, 1.2, default_size=input_shape, dropout_rate= 0.3, model_name='efficientnet-b2',include_top=include_top,num_classes=classes)
    with tf.device(get_device()):
        if pretrained==True:
            download_model_from_google_drive('1YQgy7PTgj8VereQfaxKJCZshIZK_uqtI', dirname, 'efficientnet-b2_tf.pth')
            recovery_model = load(os.path.join(dirname, 'efficientnet-b2_tf.pth'))
            recovery_model = fix_layer(recovery_model)

            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            effb2.model = recovery_model
        else:
            effb2.model = _make_recovery_model_include_top(effb2.model, include_top=include_top, classes=classes, freeze_features=False)

        effb2.model.input_shape = input_shape
        effb2.model.name = 'efficientnet-b2'
    return effb2


def EfficientNetB3(include_top=True,
             pretrained=True,
             freeze_features=False,
             input_shape=(300,300,3),
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(300, 300,3)
    effb3 =EfficientNet(1.2, 1.4, default_size=input_shape, dropout_rate=0.3, model_name='efficientnet-b3',include_top=include_top,num_classes=classes)
    with tf.device(get_device()):
        if pretrained==True:
            download_model_from_google_drive('1LgG4bsYnkY-uj6sLqbebRP0va3gDEkgS', dirname, 'efficientnet-b3_tf.pth')
            recovery_model = load(os.path.join(dirname, 'efficientnet-b3_tf.pth'))
            recovery_model = fix_layer(recovery_model)

            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            effb3.model = recovery_model
        else:
            effb3.model = _make_recovery_model_include_top(effb3.model, include_top=include_top, classes=classes, freeze_features=False)

        effb3.model.input_shape = input_shape
        effb3.model.name = 'efficientnet-b3'
    return effb3


def EfficientNetB4(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=(380,380,3),
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(380, 380,3)
    effb4 =EfficientNet(1.4, 1.8, 380, 0.4, model_name='efficientnet-b4', include_top=include_top, num_classes=classes)
    with tf.device(get_device()):
        if pretrained==True:
            download_model_from_google_drive('1eOUvMemIysmAa0ePdGKz01NO1EYWI2Dp', dirname, 'efficientnet-b4_tf.pth')
            recovery_model = load(sanitize_path(os.path.join(dirname, 'efficientnet-b4_tf.pth')))
            recovery_model = fix_layer(recovery_model)

            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            effb4.model = recovery_model
        else:
            effb4.model = _make_recovery_model_include_top(effb4.model, include_top=include_top, classes=classes, freeze_features=False)

        effb4.model.input_shape = input_shape
        effb4.model.name = 'efficientnet-b4'
    return effb4


def EfficientNetB5(include_top=True,
             pretrained=True,
             freeze_features=False,
             input_shape=(456,456,3),
             classes=1000,
             **kwargs):
    """

    Args:
        freeze_features ():
        include_top ():
        pretrained ():
        input_shape ():
        classes ():
        **kwargs ():

    Returns:

    """
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(456,456,3)
    effb5 =EfficientNet(1.6, 2.2, default_size=input_shape, dropout_rate= 0.4, model_name='efficientnet-b5',include_top=include_top,num_classes=classes)
    with tf.device(get_device()):
        if pretrained == True:
            download_model_from_google_drive('1o_JQkIFUP1_-9AkiTs-x8q7gyDMyqiJz', dirname, 'efficientnet-b5_tf.pth')
            recovery_model = load(os.path.join(dirname, 'efficientnet-b5_tf.pth'))
            recovery_model = fix_layer(recovery_model)

            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            effb5.model = recovery_model
        else:
            effb5.model = _make_recovery_model_include_top(effb5.model, include_top=include_top, classes=classes, freeze_features=False)

        effb5.model.input_shape = input_shape
        effb5.model.name = 'efficientnet-b5'
    return effb5



def EfficientNetB6(include_top=True,
             pretrained=True,
            freeze_features=False,
             input_shape=(528,528,3),
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(528, 528,3)
    effb6 =EfficientNet(1.8, 2.6, default_size=input_shape, dropout_rate= 0.5, model_name='efficientnet-b6',include_top=include_top,num_classes=classes)
    with tf.device(get_device()):
        if pretrained==True:
            download_model_from_google_drive('1-dUqwaLzv2V7m8w4jkvFBlHTCsZq54JY', dirname, 'efficientnet-b6_tf.pth')
            recovery_model = load(os.path.join(dirname, 'efficientnet-b6_tf.pth'))
            recovery_model = fix_layer(recovery_model)

            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            effb6.model = recovery_model
        else:
            effb6.model = _make_recovery_model_include_top(effb6.model, include_top=include_top, classes=classes, freeze_features=False)

        effb6.model.input_shape = input_shape
        effb6.model.name = 'efficientnet-b6'
    return effb6



def EfficientNetB7(include_top=True,
             pretrained=True,
             freeze_features=False,
             input_shape=(600,600,3),
             classes=1000,
             **kwargs):
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)
    else:
        input_shape=(600, 600,3)
    effb7 =EfficientNet(2.0, 3.1, default_size=input_shape, dropout_rate=0.5, model_name='efficientnet-b7',include_top=include_top,num_classes=classes)
    with tf.device(get_device()):
        if pretrained == True:
            download_model_from_google_drive('1NcsqfYpnIXme8nk8qrrvNAVQOGK-EOZz', dirname, 'efficientnet-b7_tf.pth')
            recovery_model = load(os.path.join(dirname, 'efficientnet-b7_tf.pth'))
            recovery_model = fix_layer(recovery_model)

            recovery_model = _make_recovery_model_include_top(recovery_model, include_top=include_top, classes=classes, freeze_features=freeze_features)
            effb7.model = recovery_model
        else:
            effb7.model = _make_recovery_model_include_top(effb7.model, include_top=include_top, classes=classes, freeze_features=False)

        effb7.model.input_shape = input_shape
        effb7.model.name = 'efficientnet-b7'
    return effb7