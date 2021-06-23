import os
from collections import namedtuple
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Callable, Any, Optional, Tuple, List

from trident.models.pretrained_utils import _make_recovery_model_include_top

from trident.backend.common import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential, summary, get_device, ModuleDict, fix_layer, load
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity, Relu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization, BatchNorm2d
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *
from trident.data.vision_transforms import Resize,Normalize
_session = get_session()
_epsilon = _session.epsilon
_trident_dir = _session.trident_dir
dirname = os.path.join(_trident_dir, 'models')

__all__ = ['InceptionV3', 'inception_v3', 'InceptionOutputs', '_InceptionOutputs']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs



def inception_v3(input_shape=(3,224,224),num_classes=1000,include_top=True,model_name='inception_v3'):
    model= Sequential(
    Conv2d_Block((3, 3), num_filters=32, strides=2, use_bias=False, normalization='batch', activation='relu',name='Conv2d_1a_3x3'),
    Conv2d_Block((3, 3), num_filters=32, use_bias=False, normalization='batch', activation='relu',name='Conv2d_2a_3x3'),
    Conv2d_Block((3, 3), num_filters=64, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='Conv2d_2b_3x3'),
    MaxPool2d((3, 3), strides=2,name='maxpool1'),
    Conv2d_Block((1, 1), num_filters=80, use_bias=False, normalization='batch', activation='relu',name='Conv2d_3b_1x1'),
    Conv2d_Block((3, 3), num_filters=192, use_bias=False, normalization='batch', activation='relu',name='Conv2d_4a_3x3'),
    MaxPool2d((3, 3), strides=2,name='maxpool2'),
    InceptionA(pool_features=32,name='Mixed_5b'),
    InceptionA(pool_features=64,name='Mixed_5c'),
    InceptionA(pool_features=64,name='Mixed_5d'),
    InceptionB(name='Mixed_6a'),
    InceptionC(channels_7x7=128,name='Mixed_6b'),
    InceptionC(channels_7x7=160,name='Mixed_6c'),
    InceptionC( channels_7x7=160,name='Mixed_6d'),
    InceptionC( channels_7x7=192,name='Mixed_6e'),
    ModuleDict({'InceptionAux':InceptionAux(1000),
                'Classifier':Sequential(
                    InceptionD(name='Mixed_7a'),
                    InceptionE(name='Mixed_7b'),
                    InceptionE(name='Mixed_7c'),
                    AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    Dropout(),
                    Dense(num_classes,name='fc')
                )}, is_multicasting=True)
    ,name=model_name)
    if not include_top:
        model.remove_at(-1)
    return model



def InceptionA(pool_features, name=None):
    return ShortCut2d(
        Conv2d_Block((1, 1), num_filters=64, use_bias=False, normalization='batch', activation='relu', name='branch1x1'),
        Sequential(
            Conv2d_Block((1, 1), num_filters=48, use_bias=False, normalization='batch', activation='relu', name='branch5x5_1'),
            Conv2d_Block((5, 5), num_filters=64, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch5x5_2')
        ),
        Sequential(
            Conv2d_Block((1, 1), num_filters=64, use_bias=False, normalization='batch', activation='relu', name='branch3x3dbl_1'),
            Conv2d_Block((3, 3), num_filters=96, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch3x3dbl_2'),
            Conv2d_Block((3, 3), num_filters=96, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch3x3dbl_3')
        ),
        Sequential(
            AvgPool2d((3,3), strides=1, auto_pad=True),
            Conv2d_Block((1, 1), num_filters=pool_features, use_bias=False, normalization='batch', activation='relu', name='branch_pool')
        ),

        mode='concate',name=name
    )

def InceptionB(name=None):
    return ShortCut2d(
        Conv2d_Block((3, 3), num_filters=384, strides=2, use_bias=False, normalization='batch', activation='relu'),
        Sequential(
           Conv2d_Block((1, 1), num_filters=64, use_bias=False, normalization='batch', activation='relu'),
            Conv2d_Block((3, 3), num_filters=96, auto_pad=True, use_bias=False, normalization='batch', activation='relu'),
            Conv2d_Block((3, 3), num_filters=96, strides=2, use_bias=False, normalization='batch', activation='relu')
        ),
        MaxPool2d((3,3), strides=2),
        mode='concate',name=name
    )


def InceptionC(channels_7x7, name=None):
    return ShortCut2d(
        Conv2d_Block((1, 1), num_filters=192, use_bias=False, normalization='batch', activation='relu',name='branch1x1'),
        Sequential(
            Conv2d_Block((1, 1), num_filters=channels_7x7, auto_pad=True, use_bias=False, normalization='batch', activation='relu',name='branch3x3dbl_1'),
            Conv2d_Block((7, 1), num_filters=channels_7x7, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch7x7dbl_2'),
            Conv2d_Block((1, 7), num_filters=192, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch7x7dbl_3'),
        ),
        Sequential(
            Conv2d_Block((1, 1), num_filters=channels_7x7, auto_pad=True, use_bias=False, normalization='batch', activation='relu',name='branch7x7dbl_1'),
            Conv2d_Block((7, 1), num_filters=channels_7x7, auto_pad=True, use_bias=False, normalization='batch', activation='relu',name='branch7x7dbl_2'),
            Conv2d_Block((1, 7), num_filters=channels_7x7, auto_pad=True, use_bias=False, normalization='batch', activation='relu',name='branch7x7dbl_3'),
            Conv2d_Block((7, 1), num_filters=channels_7x7, auto_pad=True, use_bias=False, normalization='batch', activation='relu',name='branch7x7dbl_4'),
            Conv2d_Block((1, 7), num_filters=192, auto_pad=True, use_bias=False, normalization='batch', activation='relu',name='branch7x7dbl_5')
        ),
        Sequential(
            AvgPool2d((3,3), strides=1, auto_pad=True),
            Conv2d_Block((1, 1), num_filters=192, use_bias=False, normalization='batch', activation='relu', name='branch_pool')

        ),
        mode='concate',name=name
    )


def InceptionD(name=None):
    return ShortCut2d(
        Sequential(
        Conv2d_Block((1, 1), num_filters=192, use_bias=False, normalization='batch', activation='relu', name='branch3x3_1'),
        Conv2d_Block((3, 3), num_filters=320, strides=2,use_bias=False, normalization='batch', activation='relu', name='branch3x3_2'),
        ),

        Sequential(
        Conv2d_Block((1, 1), num_filters=192, use_bias=False, normalization='batch', activation='relu', name='branch7x7x3_1'),
        Conv2d_Block((1, 7), num_filters=192, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch7x7x3_2'),
        Conv2d_Block((7, 1), num_filters=192, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch7x7x3_3'),
        Conv2d_Block((3, 3), num_filters=192, strides=2, use_bias=False, normalization='batch', activation='relu', name='branch7x7x3_4'),
        ),
        MaxPool2d((3,3), strides=2),
        mode='concate',name=name
    )

def InceptionE(name=None):
    return ShortCut2d(
        Conv2d_Block((1, 1), num_filters=320, use_bias=False, normalization='batch', activation='relu', name='branch1x1'),
        Sequential(
            Conv2d_Block((1, 1), num_filters=384, use_bias=False, normalization='batch', activation='relu', name='branch3x3_1'),
            ShortCut2d(
                Conv2d_Block((1, 3), num_filters=384, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch3x3_2a'),
                Conv2d_Block((3, 1), num_filters=384, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch3x3_2b')
            ,mode='concate')
        ),

        Sequential(
            Conv2d_Block((1, 1), num_filters=448, use_bias=False, normalization='batch', activation='relu', name='branch3x3dbl_1'),
            Conv2d_Block((3, 3), num_filters=384, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch3x3dbl_2'),
            ShortCut2d(
                Conv2d_Block((1, 3), num_filters=384, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch3x3dbl_3a'),
                Conv2d_Block((3, 1), num_filters=384, auto_pad=True, use_bias=False, normalization='batch', activation='relu', name='branch3x3dbl_3b')
             , mode='concate')
        ),
        Sequential(
            AvgPool2d((3, 3), strides=1, auto_pad=True),
            Conv2d_Block((1,1),num_filters=192,use_bias=False,normalization='batch',activation='relu', name='branch_pool')
        ),
        mode='concate',name=name)

def InceptionAux(num_classes=1000,name=None):
    return Sequential(
        AvgPool2d((5,5), strides=3),
       Conv2d_Block((1, 1), num_filters=128, use_bias=False, normalization='batch', activation='relu'),
        Conv2d_Block((5, 5), num_filters=768, use_bias=False, normalization='batch', activation='relu'),
        AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        Dense(num_classes)

    )


def InceptionV3(include_top=True, pretrained=True,freeze_features=True, input_shape=(3, 224, 224), classes=1000, **kwargs):
    if input_shape is not None and len(input_shape) == 3:
        input_shape = tuple(input_shape)
    else:
        input_shape = (3, 224, 224)
    model = ImageClassificationModel(input_shape=input_shape,output=inception_v3(input_shape, model_name='inception_v3', include_top=include_top,num_classes=classes))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels1.txt'), 'r',
              encoding='utf-8-sig') as f:
        labels = [l.rstrip() for l in f]
        model.class_names = labels
    model.preprocess_flow = [Resize((input_shape[2], input_shape[1]), keep_aspect=True), Normalize(0, 255),
                             Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # if pretrained:
    #     download_model_from_google_drive('1bxnoDerzoNfiZZLft4ocD3DAgx4v6aTN', dirname, 'efficientnet-b0.pth')
    #     recovery_model = fix_layer(load(os.path.join(dirname, 'efficientnet-b0.pth')))
    #     recovery_model = _make_recovery_model_include_top(recovery_model,input_shape=input_shape, include_top=include_top, classes=classes, freeze_features=freeze_features)
    #     effb0.model = recovery_model
    # else:
    #     effb0.model = _make_recovery_model_include_top( effb0.model, include_top=include_top, classes=classes, freeze_features=True)
    #
    # effb0.model .input_shape = input_shape
    # effb0.model .to(get_device())
    return model



