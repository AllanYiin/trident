import os
import sys
import codecs

os.environ['TRIDENT_BACKEND'] = 'pytorch'
import  trident  as T
from trident import *
import math
import numpy as np
import linecache
import PIL
import PIL.Image as Image




dataset=load_cifar('cifar100')
dataset.image_transform_funcs=[
                               random_center_crop(64,64,(0.9,1.1)),
                               random_adjust_gamma((0.8,1.2)),
                               random_adjust_contast((0.8,1.2)),
                               add_noise(0.05),
                               normalize(0,255),
                               normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]






#= torch.load('resnet50_model_pytorch_cifar100_224.pth')

resnet50=ResNet50(pretrained=False,classes=100,input_shape=(3,64,64))
resnet50.model.__delitem__(1)
#resnet50.load_model('Models/resnet50.pth.tar')
resnet50.summary()
resnet50.with_optimizer(optimizer='Ranger',lr=1e-3,betas=(0.9, 0.999))\
    .with_loss(CrossEntropyLoss)\
    .with_metric(accuracy,name='accuracy')\
    .with_metric(accuracy,topk=5,name='top5_accuracy')\
    .with_regularizer('l2')\
    .with_constraint('max_min_norm')\
    .with_learning_rate_scheduler(reduce_lr_on_plateau,monitor='accuracy',mode='max',factor=0.75,patience=5,cooldown=2,threshold=5e-5,warmup=0)\
    .with_model_save_path('Models/resnet50.pth')



def gcd_bottleneck(base_num,growth,strides=1,conv_shortcut=True,use_bias=False,name=''):
    #width = int(num_filters * (base_width / 64.)) * 1#groups'

    num_filters=growth*base_num

    shortcut =GcdConv2d((3,3),num_filters=num_filters,strides=strides,auto_pad=True,padding_mode='zero',self_norm=True,divisor_rank=0,activation=None,use_bias=use_bias,name=name + '_downsample')
    shortcut_name = 'downsample'
    return ShortCut2d({'branch1':Sequential(GcdConv2d((3,3),num_filters=base_num*prev_prime(growth) ,strides=strides,auto_pad=True,padding_mode='zero',self_norm=True,divisor_rank=0,activation='selu',use_bias=use_bias,name=name + '_0_conv'),
                                 GcdConv2d((3, 3), num_filters=base_num*next_prime(growth) , strides=1, auto_pad=True,padding_mode='zero', self_norm=True,divisor_rank=0,activation='selu',use_bias=use_bias,name=name + '_1_conv'),
                                 GcdConv2d((3,3),num_filters=num_filters,strides=1,auto_pad=True,padding_mode='zero',activation=None,self_norm=True,divisor_rank=0,use_bias=use_bias,name=name + '_2_conv')),
                      shortcut_name:shortcut},activation='selu')


def GcdResNet(block, layers, input_shape=(3, 224, 224), num_classes=1000, use_bias=False, zero_init_residual=False,
           width_per_group=64, replace_stride_with_dilation=None, include_top=True, model_name='',
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
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
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """


    # if not (weights in {'imagenet', None} or os.path.exists(weights)):
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization), `imagenet` '
    #                      '(pre-training on ImageNet), '
    #                      'or the path to the weights file to be loaded.')
    #
    # if weights == 'imagenet' and include_top and classes != 1000:
    #     raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
    #                      ' as true, `classes` should be 1000')

    def _make_layer(block, base_num, growths,blocklayers, strides=1, dilate=False,use_bias=use_bias,layer_name=''):
        conv_shortcut=False
        if strides!=1 or block is bottleneck:
            conv_shortcut=True
        #growths = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        #32,48,64
        #40,60,80,100
        #
        layers = []
        layers.append(block(base_num,growths[0], strides=strides, conv_shortcut=conv_shortcut,use_bias=use_bias, name=layer_name+'.0'))

        for k in range(1, blocklayers):
            layers.append(block(base_num,growths[k],  strides=1,  conv_shortcut=False, use_bias=use_bias,name=layer_name+'.{0}'.format(k)))

        laters_block=Sequential(*layers)
        laters_block.name=layer_name
        return laters_block

    flow_list=[]
    resnet = Sequential()
    resnet.add_module('first_block',Conv2d_Block((3,3),18,strides=1,use_bias=use_bias,auto_pad=True,padding_mode='zero',normalization='batch',activation='leaky_relu',name='first_block'))

    resnet.add_module('layer1',(_make_layer(block, 8, [4, 6, 8, 10, 12, 14, 16, 18, 20], layers[0],strides=1, dilate=1,use_bias=use_bias,layer_name='layer1' )))
    resnet.add_module('layer2',(_make_layer(block, 10, [6, 8, 10, 12, 14, 16, 18, 20], layers[1], strides=2, dilate=1,use_bias=use_bias,layer_name='layer2' )))
    resnet.add_module('layer3',(_make_layer(block, 14, [ 8, 10, 12, 14, 16, 18, 20], layers[2], strides=2, dilate=1,use_bias=use_bias,layer_name='layer3' )))
    resnet.add_module('layer4' ,(_make_layer(block, 18, [10, 12, 14, 16, 18, 20], layers[3], strides=2, dilate=1,use_bias=use_bias,layer_name='layer4' )))
    resnet.add_module('last_block',Conv2d((1, 1), 100,strides=1, use_bias=use_bias, auto_pad=True, padding_mode='zero', activation='leaky_relu', name='last_block'))
    resnet.add_module('avg_pool',GlobalAvgPool2d(name='avg_pool'))
    if include_top:
        resnet.add_module('fc',Dense(num_classes,activation='softmax',name='fc'))
    resnet.name=model_name
    model=ImageClassificationModel(input_shape=input_shape,output=resnet)
    model.preprocess_flow=[resize((input_shape[2],input_shape[1]),keep_aspect=True),normalize(0,255),normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]

    return model



gcdresnet50 =GcdResNet(gcd_bottleneck, [2,3,4,2], (3,64,64),num_classes=100,include_top=True, model_name='gcd_resnet50')
#gcdresnet50.load_model('Models/gcdresnet50.pth.tar')
gcdresnet50.summary()
gcdresnet50.with_optimizer(optimizer='Ranger',lr=1e-3,betas=(0.9, 0.999))\
    .with_loss(CrossEntropyLoss)\
    .with_metric(accuracy,name='accuracy')\
    .with_metric(accuracy,topk=5,name='top5_accuracy')\
    .with_regularizer('l2')\
    .with_constraint('max_min_norm')\
    .with_learning_rate_scheduler(reduce_lr_on_plateau,monitor='accuracy',mode='max',factor=0.75,patience=5,cooldown=2,threshold=5e-5,warmup=0)\
    .with_model_save_path('Models/gcdresnet50.pth')


plan=TrainingPlan()\
    .add_training_item(resnet50)\
    .add_training_item(gcdresnet50)\
    .with_data_loader(dataset)\
    .repeat_epochs(300)\
    .within_minibatch_size(16)\
    .print_progress_scheduling(10,unit='batch')\
    .save_model_scheduling(500,unit='batch')

plan.start_now()