from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import math
import os
import uuid
from collections import *
from collections import deque
from copy import copy, deepcopy
from functools import partial
from itertools import repeat

import numpy as np

import tensorflow as tf
from trident.backend.common import *
from trident.backend.tensorspec import *
from trident.backend.tensorflow_backend import to_numpy, to_tensor, Layer, Sequential, summary, fix_layer,get_device
from trident.backend.tensorflow_ops import *
from trident.data.image_common import *
from trident.data.utils import download_model_from_google_drive,download_file,get_image_from_google_drive
from trident.layers.tensorflow_activations import get_activation, Identity, Relu
from trident.layers.tensorflow_blocks import *
from trident.layers.tensorflow_layers import *
from trident.layers.tensorflow_normalizations import get_normalization, BatchNorm2d
from trident.layers.tensorflow_pooling import *
from trident.optims.tensorflow_trainer import *
from trident.backend.tensorflow_serialization import load,save
__all__ = ['DenseNet','DenseNet121','DenseNet161','DenseNet169','DenseNet201','DenseNetFcn']

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



def DenseLayer(growth_rate,name=''):
    """
    The basic normalization, convolution and activation combination for dense connection

    Args:
        growth_rate (int):The growth rate regulates how much new information each layer contributes to the global state
        name (str): None of this dense layer

    Returns:
        An instrance of dense layer.

    """
    items = OrderedDict()
    items['norm']=BatchNorm2d()
    items['relu']=Relu()
    items['conv1']=Conv2d_Block((1,1),num_filters=4 * growth_rate,strides=1,activation='relu',auto_pad=True,padding_mode='zero',use_bias=False,normalization='batch')
    items['conv2']=Conv2d((3,3),num_filters=growth_rate,strides=1,auto_pad=True,padding_mode='zero',use_bias=False)
    return  Sequential(items)


class DenseBlock(Layer):
    def __init__(self, num_layers,  growth_rate=32, drop_rate=0,keep_output=False,name=''):
        """
        The dense connected block.
        Feature-maps of eachconvolution layer are used as inputs into all subsequent layers

        Args:
            num_layers (int):  number of dense layers in this block
            growth_rate (int):The growth rate regulates how much new information each layer contributes to the global state
            drop_rate (decimal):  the drop out rate of this dense block
            keep_output (bool):  If True, the output tensor will kept
            name (str):Name of this dense block .

        Returns:
            An instrance of dense block.

        """
        super(DenseBlock, self).__init__()
        if len(name)>0:
            self.name=name
        self.keep_output=keep_output
        for i in range(num_layers):
            layer = DenseLayer(growth_rate,name='denselayer%d' % (i + 1))
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.named_children():
            new_features = layer(x)
            x=concate([x,new_features], -1)
        return x


def Transition(reduction,name=''):
    """
     The block for transition-down, down-sampling by average pooling
    Args:
        reduction (float): The depth_multiplier to transition-down the dense features
        name (str): Name of the transition-down process

    Returns:
        An instrance of transition-down .


    """
    items=OrderedDict()
    items['norm']=BatchNorm2d()
    items['relu']=Relu()
    items['conv1']=Conv2d((1, 1),num_filters=None, depth_multiplier=reduction, strides=1, auto_pad=True,padding_mode='zero',use_bias=False)
    items['pool']=AvgPool2d(2,2,auto_pad=True)
    return Sequential(items,name=name)


def TransitionDown(reduction,name=''):
    """
     The block for transition-down, down-sampling by using depthwise convolution with strides==2

    Args:
        reduction (float): The depth_multiplier to transition-down the dense features
        name (str): Name of the transition-down process

    Returns:
        An instrance of transition-down .

    """
    return DepthwiseConv2d_Block((3,3),depth_multiplier=reduction,strides=2,activation='leaky_relu',normalization='batch', dropout_rate=0.2)

def TransitionUp(output_idx=None,num_filters=None,name=''):
    return ShortCut2d(TransConv2d((3,3),num_filters=num_filters,strides=2,auto_pad=True),output_idx=output_idx,mode= 'concate',name=name)

def DenseNet(blocks,
             growth_rate=32,
             initial_filters=64,
             include_top=True,
             pretrained=True,
             input_shape=(224,224,3),
             num_classes=1000,
             name='',
             **kwargs):
    """'
    Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained on ImageNet.

    Args
        blocks (tuple/ list of int ): numbers of building blocks for the dense layers.

        growth_rate (int):The growth rate regulates how much new information each layer contributes to the global state

        initial_filters (int): the channel of the first convolution layer

        pretrained (bool): If True, returns a model pre-trained on ImageNet.

        input_shape (tuple or list): the default input image size in CHW order (C, H, W)

        num_classes (int): number of classes

        name (string): anme of the model

    Returns
        A trident image classification model instance.

    """
    densenet=Sequential()
    densenet.add_module('conv1/conv',Conv2d_Block((7,7),num_filters=initial_filters,strides=2,use_bias=False,auto_pad=True,padding_mode='zero',activation='relu',normalization='batch', name='conv1/conv'))
    densenet.add_module('maxpool', (MaxPool2d((3, 3), strides=2, auto_pad=True, padding_mode='zero')))
    densenet.add_module('denseblock1', DenseBlock(blocks[0],growth_rate=growth_rate))
    densenet.add_module('transitiondown1', Transition(0.5))
    densenet.add_module('denseblock2', DenseBlock(blocks[1], growth_rate=growth_rate))
    densenet.add_module('transitiondown2', Transition(0.5))
    densenet.add_module('denseblock3', DenseBlock(blocks[2], growth_rate=growth_rate))
    densenet.add_module('transitiondown3', Transition(0.5))
    densenet.add_module('denseblock4', DenseBlock(blocks[3], growth_rate=growth_rate))
    densenet.add_module('classifier_norm',BatchNorm2d(name='classifier_norm'))
    densenet.add_module('classifier_relu', Relu(name='classifier_relu'))
    densenet.add_module('avg_pool', GlobalAvgPool2d(name='avg_pool'))
    if include_top:
        densenet.add_module('classifier', Dense(num_classes, activation=None, name='classifier'))
        densenet.add_module('softmax', SoftMax( name='softmax'))
    densenet.name = name

    model=ImageClassificationModel(input_shape=input_shape,output=densenet)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_labels1.txt'), 'r',encoding='utf-8-sig') as f:
        labels = [l.rstrip() for l in f]
        model.class_names = labels
    model.preprocess_flow = [resize((input_shape[0], input_shape[1]), keep_aspect=True), normalize(0, 255),  normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # model.summary()
    return model


def DenseNetFcn(blocks=(4, 5, 7, 10, 12),
             growth_rate=16,
             initial_filters=64,
             pretrained=False,
             input_shape=(224,224,3),
             num_classes=10,
             name='',
             **kwargs):
    """
    Instantiates the DenseNet FCN architecture.
    Optionally loads weights pre-trained on ImageNet.

    Args
        blocks (tuple/ list of int ): numbers of building blocks for the dense layers.

        growth_rate (int):The growth rate regulates how much new information each layer contributes to the global state

        initial_filters (int): the channel of the first convolution layer

        pretrained (bool): only False is valid for DenseNet FCN

        input_shape (tuple or list): the default input image size in CHW order (C, H, W)

        num_classes (int): number of classes

        name (string): anme of the model

    Returns
        A trident image segmentation model instance.

    """

    model = ImageSegmentationModel(input_shape=input_shape, output=_DenseNetFcn2(blocks=blocks,
             growth_rate=growth_rate,
             initial_filters=initial_filters,
             num_classes=num_classes,
             name=name,
             **kwargs))
    model.signature = Signature(name='DenseNetFcn')
    if is_tensor(model._input_shape):
        model.signature.inputs['input']=TensorSpec(shape=model._input_shape,name='input')
    if is_tensor(model._output_shape):
        model.signature.outputs['output']=TensorSpec(shape=model._output_shape,name='output')

    model.preprocess_flow = [resize((input_shape[2], input_shape[1]), keep_aspect=True), normalize(0, 255),
                             normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # model.summary()
    return model

class _DenseNetFcn2(Layer):
    def __init__(self, blocks=(4, 5, 7, 10, 12),
             growth_rate=16,
             initial_filters=64,
             num_classes=10,
             name='',
             **kwargs):
        super(_DenseNetFcn2, self).__init__()
        self.blocks=blocks
        self.num_classes=num_classes
        self.growth_rate=growth_rate
        self.name=name
        self.initial_filters=initial_filters
        self.first_layer=Conv2d_Block((3, 3), num_filters=self.initial_filters, strides=2, use_bias=False, auto_pad=True,
                                                    padding_mode='zero', activation='relu', normalization='batch',
                                                    name='first_layer')
        for i in range(len(self.blocks)-1):
            num_filters=self.initial_filters+self.blocks[i+1]*self.growth_rate
            self.add_module('denseblock_down{0}'.format(i+1),DenseBlock(self.blocks[i], growth_rate=self.growth_rate, name='denseblock_down{0}'.format(i+1)))
            self.add_module('transition_down{0}'.format(i+1),TransitionDown(0.5,name='transition_down{0}'.format(i+1)))
            self.add_module('transition_up{0}'.format(i + 1), TransConv2d_Block((3,3),num_filters=num_filters,strides=2,auto_pad=True,activation='relu',normalization='batch',name='transition_up{0}'.format(i + 1)))
            self.add_module('denseblock_up{0}'.format(i + 1),DenseBlock(self.blocks[i], growth_rate=self.growth_rate, name='denseblock_up{0}'.format(i + 1)))

        self.bottleneck=DenseBlock(self.blocks[4], growth_rate=self.growth_rate, name='bottleneck')
        self.upsample= Upsampling2d(scale_factor=2,mode='bilinear')
        self.last_layer=Conv2d((1, 1), num_filters=self.num_classes, strides=1, activation=None)
        self.softmax=SoftMax()

    def forward(self, *x):
        x=enforce_singleton(x)
        skips=[]
        x=self.first_layer(x)
        for i in range(len(self.blocks) - 1):
            x=getattr(self,'denseblock_down{0}'.format(i+1))(x)
            skips.append(x)
            x=getattr(self,'transition_down{0}'.format(i+1))(x)

        x=self.bottleneck(x)
        for i in range(len(self.blocks) - 1):
            x = getattr(self, 'transition_up{0}'.format(len(self.blocks)-1- i))(x)
            output = skips.pop()
            x = concate([x, output], axis=-1)
            x=getattr(self,'denseblock_up{0}'.format(len(self.blocks)-1-i))(x)
        x=self.upsample(x)
        x=self.last_layer(x)
        x=self.softmax(x)
        return x



def DenseNet121(include_top=True,
             pretrained=True,
             input_shape=(224,224,3),
             classes=1000,
             **kwargs):
    """
    Constructor the image classicication model with DenseNet121 as backbond

    Args:
        include_top ():
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        input_shape (tuple or list): the default input image size in CHW order (C, H, W)
        classes (int): number of classes

    References
        Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf

    Returns:
        the image classicication model with DenseNet121

    Examples:
        >>> dense121 = DenseNet121(include_top=True,pretrained=True,input_shape=(224,224,3),classes=1000)
        >>> 'n02124075' in dense121.infer_single_image(get_image_from_google_drive('1SwablQsZO8mBuB84xnr1IoOisE3pm03l'),1).key_list[0]
        True

    """

    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)

    densenet121 =DenseNet([6, 12, 24, 16],32,64, include_top=include_top, pretrained=True,input_shape=input_shape, num_classes=classes,name='densenet121')
    if pretrained==True:
        download_model_from_google_drive('1XZvwockMwLJAZjUKbDt687-pssz58GP9',dirname,'densenet121_tf.pth')
        recovery_model=load(os.path.join(dirname,'densenet121_tf.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model.name = 'densenet121'
        recovery_model.eval()
        with tf.device(get_device()):
            if include_top==False:
                recovery_model.remove_at(-1)
                recovery_model.remove_at(-1)
            else:
                if classes!=1000:
                    recovery_model.remove_at(-1)
                    recovery_model.remove_at(-1)
                    recovery_model.add_module('classifier', Dense(classes, activation=None, name='classifier'))
                    recovery_model.add_module('softmax', SoftMax(name='softmax'))
                    densenet121.class_names = []


        recovery_model.signature = Signature(name='DenseNetFcn')
        if is_tensor(recovery_model._input_shape):
            recovery_model.signature.inputs['input'] = TensorSpec(shape=recovery_model._input_shape, name='input')
        if is_tensor(recovery_model._output_shape):
            recovery_model.signature.outputs['output'] = TensorSpec(shape=recovery_model._output_shape, name='output')
        densenet121.model = recovery_model
    return densenet121


def DenseNet161(include_top=True,
             pretrained=True,
             input_shape=(224,224,3),
             classes=1000,
             **kwargs):
    """
    Constructor the image classicication model with DenseNet161 as backbond

    Args:
        include_top ():
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        input_shape (tuple or list): the default input image size in CHW order (C, H, W)
        classes (int): number of classes

    References
        Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf

    Returns:
        the image classicication model with DenseNet161

    Examples:
        >>> dense161 = DenseNet161(include_top=True,pretrained=True,input_shape=(224,224,3),classes=1000)
        >>> 'n02124075' in dense161.infer_single_image(get_image_from_google_drive('1SwablQsZO8mBuB84xnr1IoOisE3pm03l'),1).key_list[0]
        True

    """

    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)

    densenet161 =DenseNet([6, 12, 36, 24],48,96, include_top=include_top, pretrained=True,input_shape=input_shape, num_classes=classes,name='densenet161')
    if pretrained==True:
        print('There is no DenseNet161 pretrained model avaiable ')
        # download_model_from_google_drive('1n3HRkdPbxKrLVua9gOCY6iJnzM8JnBau',dirname,'densenet161_tf.pth')
        # recovery_model=load(os.path.join(dirname,'densenet161_tf.pth'))
        # recovery_model = fix_layer(recovery_model)
        # recovery_model.name = 'densenet161'
        # recovery_model.eval()
        #
        # if include_top==False:
        #     recovery_model.remove_at(-1)
        # else:
        #     if classes!=1000:
        #         recovery_model.remove_at(-1)
        #         recovery_model.remove_at(-1)
        #         recovery_model.add_module('classifier', Dense(classes, activation=None, name='classifier'))
        #         recovery_model.add_module('softmax', SoftMax(name='softmax'))
        # recovery_model.signature = Signature(name='DenseNetFcn')
        # if is_tensor(recovery_model._input_shape):
        #     recovery_model.signature.inputs['input'] = TensorSpec(shape=recovery_model._input_shape, name='input')
        # if is_tensor(recovery_model._output_shape):
        #     recovery_model.signature.outputs['output'] = TensorSpec(shape=recovery_model._output_shape, name='output')
        # densenet161.model=recovery_model

    return densenet161




def DenseNet169(include_top=True,
             pretrained=True,
             input_shape=(224,224,3),
             classes=1000,
             **kwargs):
    """
    Constructor the image classicication model with DenseNet169 as backbond

    Args:
        include_top ():
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        input_shape (tuple or list): the default input image size in CHW order (C, H, W)
        classes (int): number of classes

    References
        Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf

    Returns:
        the image classicication model with DenseNet169

    Examples:
        >>> dense169 = DenseNet169(include_top=True,pretrained=True,input_shape=(224,224,3),classes=1000)
        >>> 'n02124075' in dense169.infer_single_image(get_image_from_google_drive('1SwablQsZO8mBuB84xnr1IoOisE3pm03l'),1).key_list[0]
        True

    """
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)

    densenet169 =DenseNet([6, 12, 32, 32],32,64, include_top=include_top, pretrained=True,input_shape=input_shape, num_classes=classes,name='densenet169')
    if pretrained==True:
        download_model_from_google_drive('1tfLRxX4EipkIc23MF5xZR23rYdxk3AZ9',dirname,'densenet169_tf.pth')
        recovery_model=load(os.path.join(dirname,'densenet169_tf.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model.name = 'densenet169'
        recovery_model.eval()
        with tf.device(get_device()):
            if include_top==False:
                recovery_model.remove_at(-1)
                recovery_model.remove_at(-1)
            else:
                if classes!=1000:
                    recovery_model.remove_at(-1)
                    recovery_model.remove_at(-1)
                    recovery_model.add_module('classifier', Dense(classes, activation=None, name='classifier'))
                    recovery_model.add_module('softmax', SoftMax(name='softmax'))
        recovery_model.signature = Signature(name='DenseNetFcn')
        if is_tensor(recovery_model._input_shape):
            recovery_model.signature.inputs['input'] = TensorSpec(shape=recovery_model._input_shape, name='input')
        if is_tensor(recovery_model._output_shape):
            recovery_model.signature.outputs['output'] = TensorSpec(shape=recovery_model._output_shape, name='output')

        densenet169.model=recovery_model

    return densenet169



def DenseNet201(include_top=True,
             pretrained=True,
             input_shape=(224,224,3),
             classes=1000,
             **kwargs):
    """
    Constructor the image classicication model with DenseNet201 as backbond

    Args:
        include_top ():
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        input_shape (tuple or list): the default input image size in CHW order (C, H, W)
        classes (int): number of classes

    References
        Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf

    Returns:
        the image classicication model with DenseNet201

    Examples:
        >>> dense201 = DenseNet201(include_top=True,pretrained=True,input_shape=(224,224,3),classes=1000)
        >>> 'n02124075' in dense201.infer_single_image(get_image_from_google_drive('1SwablQsZO8mBuB84xnr1IoOisE3pm03l'),1).key_list[0]
        True

    """
    if input_shape is not None and len(input_shape)==3:
        input_shape=tuple(input_shape)

    densenet201 =DenseNet([6, 12, 48, 32],32,64, include_top=include_top, pretrained=True,input_shape=input_shape, num_classes=classes,name='densenet201')
    if pretrained==True:
        download_model_from_google_drive('1dJfgus11jXVoCLWfZqTgZ6jKKtrB70om',dirname,'densenet201_tf.pth')
        recovery_model=load(os.path.join(dirname,'densenet201_tf.pth'))
        recovery_model = fix_layer(recovery_model)
        recovery_model._name = 'densenet201'
        recovery_model.eval()
        with tf.device(get_device()):
            if include_top==False:
                recovery_model.remove_at(-1)
                recovery_model.remove_at(-1)
            else:
                if classes!=1000:
                    recovery_model.remove_at(-1)
                    recovery_model.remove_at(-1)
                    recovery_model.add_module('classifier', Dense(classes, activation=None, name='classifier'))
                    recovery_model.add_module('softmax', SoftMax(name='softmax'))
        recovery_model.signature = Signature(name='DenseNetFcn')
        if is_tensor(recovery_model._input_shape):
            recovery_model.signature.inputs['input'] = TensorSpec(shape=recovery_model._input_shape, name='input')
        if is_tensor(recovery_model._output_shape):
            recovery_model.signature.outputs['output'] = TensorSpec(shape=recovery_model._output_shape, name='output')

        densenet201.model=recovery_model
    return densenet201