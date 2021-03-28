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
from enum import Enum, unique
from functools import partial
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch.nn import init
from torch.nn.parameter import Parameter

from trident.backend.common import *
from trident.backend.tensorspec import *
from trident.backend.pytorch_backend import to_numpy, to_tensor, Layer, Sequential,get_device
from trident.backend.pytorch_ops import *
from trident.data.image_common import *
from trident.data.utils import download_file_from_google_drive
from trident.layers.pytorch_activations import get_activation, Identity, LeakyRelu,Sigmoid
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_normalizations import get_normalization, BatchNorm
from trident.layers.pytorch_pooling import *
from trident.optims.pytorch_trainer import *

__all__ = ['gan_builder', 'UpsampleMode', 'BuildBlockMode','MinibatchDiscrimination','MinibatchDiscriminationLayer']

_session = get_session()
_device =get_device()
_epsilon = _session.epsilon
_trident_dir = _session.trident_dir


def resnet_block(num_filters=64, strides=1, activation='leaky_relu', normalization='batch', use_spectral=False,
                 dilation=1, name=''):
    kernal = 1 if strides == 1 else 3
    return [ShortCut2d(Sequential(
        Conv2d_Block((3, 3), depth_multiplier=0.5, strides=1, auto_pad=True, padding_mode='replicate',
                     use_spectral=use_spectral, normalization=normalization, activation=activation, use_bias=False,
                     dilation=dilation, name=name + '_0_conv'),
        Conv2d_Block((1, 1), depth_multiplier=2, strides=1, auto_pad=True, padding_mode='replicate',
                     use_spectral=use_spectral, normalization=normalization, activation=activation, use_bias=False,
                     name=name + '_1_conv')), Identity(), activation=activation, name=name),
        Conv2d_Block((kernal, kernal), num_filters=num_filters, strides=strides, auto_pad=True,
                         padding_mode='replicate', use_spectral=use_spectral, normalization=normalization,
                         activation=activation, use_bias=False, name=name + '_conv')]


def separable_resnet_block(num_filters=64, strides=1, activation='leaky_relu', normalization='batch',
                           use_spectral=False, dilation=1, name=''):
    kernal = 1 if strides == 1 else 3
    return [ShortCut2d(Sequential(
        SeparableConv2d_Block((3, 3), depth_multiplier=1, strides=1, auto_pad=True, padding_mode='replicate',
                              use_spectral=use_spectral, normalization=normalization, activation=activation,
                              use_bias=False, dilation=dilation, name=name + '_0_conv'),
        Conv2d_Block((1, 1), depth_multiplier=2, strides=1, auto_pad=True, padding_mode='replicate',
                     use_spectral=use_spectral, normalization=normalization, activation=activation, use_bias=False,
                     name=name + '_1_conv')), Identity(), activation=activation, name=name),
            Conv2d_Block((kernal, kernal), num_filters=num_filters, strides=strides, auto_pad=True,
                         padding_mode='replicate', use_spectral=use_spectral, normalization=normalization,
                         activation=activation, use_bias=False, name=name + '_conv')]


def bottleneck_block(num_filters=64, strides=1, reduce=4, activation='leaky_relu', normalization='batch',
                     use_spectral=False, dilation=1, name=''):
    shortcut = Conv2d_Block((3, 3), num_filters=num_filters, strides=strides, auto_pad=True, padding_mode='zero',
                            normalization=normalization, activation=None, name=name + '_downsample')
    return ShortCut2d(Sequential(Conv2d_Block((1, 1), depth_multiplier=1, strides=1, auto_pad=True, padding_mode='replicate',
                                              use_spectral=use_spectral, normalization=normalization,
                                              activation=activation, use_bias=False, name=name + '_0_conv'),
                                 Conv2d_Block((3, 3), depth_multiplier=1 / reduce, strides=strides, auto_pad=True,
                                              padding_mode='replicate', use_spectral=use_spectral,
                                              normalization=normalization, activation=activation, use_bias=False,
                                              dilation=dilation, name=name + '_1_conv'),
                                 Conv2d_Block((1, 1), num_filters=num_filters, strides=1, auto_pad=True,
                                              padding_mode='replicate', use_spectral=use_spectral,
                                              normalization=normalization, activation=None, use_bias=False,
                                              name=name + '_2_conv')), shortcut, activation=activation, name=name)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname.find("Conv2d_Block") == -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)



class MinibatchDiscrimination(Layer):
    def __init__(self, num_filters=None, hidden_filters=None, use_mean=True,name=None):
        super().__init__(name=name)
        self.num_filters = num_filters
        self.hidden_filters = hidden_filters
        self.use_mean = use_mean
    def build(self, *input_shape: TensorShape):
        if not self._built:
            if  self.num_filters is None:
                self.num_filters=minimum(self.input_filters//2,64)
            if  self.hidden_filters is None:
                self.hidden_filters=self.num_filters//2
            self.register_parameter('weight',Parameter(random_normal((self.input_filters, self.num_filters, self.hidden_filters)).to(get_device())))

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.weight.view(self.input_filters, -1))
        matrices = matrices.view(-1, self.num_filters, self.hidden_filters)
        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.use_mean:
            o_b /= x.size(0) - 1
        x = torch.cat([x, o_b], 1)
        return x


class MinibatchDiscriminationLayer(Layer):
    def __init__(self, averaging= 'spatial', name=None):
        super(MinibatchDiscriminationLayer, self).__init__(name=name)
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging

    def adjusted_std(self,t, dim=0, keepdim=True):
        return torch.sqrt(torch.mean((t - torch.mean(t, dim=dim, keepdim=keepdim)) ** 2, dim=dim, keepdim=keepdim) + 1e-8)

    def forward(self, x):
        shape = list(x.size())

        target_shape = deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2,3], keepdim=True)             # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0,2,3], keepdim=True)                   # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:                                                           # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)



class UpsampleMode(Enum):
    pixel_shuffle = 'pixel_shuffle'
    transpose = 'transpose'
    nearest = 'nearest'
    bilinear = 'bilinear'


class BuildBlockMode(Enum):
    base = 'base'
    resnet = 'resnet'
    bottleneck = 'bottleneck'
    separable_resnet = 'separable_resnet'


class NetworkType(Enum):
    encoder = 'encoder'
    decoder = 'decoder'
    autoencoder = 'autoencoder'


def gan_builder(noise_shape=100, image_width=256, upsample_mode='nearest', generator_build_block='resnet',
        discriminator_build_block='resnet', generator_network_type='decoder', discriminator_network_type='encoder',
        use_spectral=False, activation='leaky_relu', generator_norm='batch', discriminator_norm='batch',
        use_dilation=False, use_dropout=False, use_self_attention=False,use_minibatch_discrimination=False):
    noise_input = torch.tensor(data=np.random.normal(0, 1, size=(2, noise_shape)))

    def build_generator():
        layers = []
        initial_size = 8

        if image_width in [192, 96, 48]:
            initial_size = 6
        elif image_width in [144, 72, 36]:
            initial_size = 9
        elif image_width in [160, 80]:
            initial_size = 10

        layers.append(Dense(256 * initial_size * initial_size, activation=None, name='fc'))
        layers.append(Reshape((256, initial_size, initial_size), name='reshape'))

        if upsample_mode == UpsampleMode.pixel_shuffle.value:
            layers.append(Conv2d_Block((3, 3), 256, strides=1, auto_pad=True, use_spectral=use_spectral, use_bias=False,activation=activation, normalization=generator_norm))
        else:
            layers.append(Conv2d_Block((3, 3), 256, strides=1, auto_pad=True, use_spectral=use_spectral, use_bias=False,activation=activation, normalization=generator_norm, padding_mode='replicate'))

        current_width = initial_size
        i = 0
        while current_width < image_width:
            filter = 256
            scale = 2 if (image_width // current_width) % 2 == 0 else (image_width // current_width)

            dilation = 1
            if use_dilation:
                dilation = 2 if current_width >= 64 else 1

            if upsample_mode == UpsampleMode.transpose.value:
                layers.append(
                    TransConv2d_Block((3, 3), depth_multiplier=1, strides=scale, auto_pad=True, padding_mode='replicate',
                                      use_spectral=use_spectral, use_bias=False, activation=activation,
                                      normalization=generator_norm, dilation=dilation,
                                      name='transconv_block{0}'.format(i)))
            elif upsample_mode == UpsampleMode.pixel_shuffle.value:
                if i > 0:
                    layers.append(Conv2d_Block((3, 3), depth_multiplier=4, strides=1, auto_pad=True, use_spectral=use_spectral,
                                        use_bias=False, activation=activation, normalization=generator_norm,
                                        padding_mode='replicate'))
                    #filter = filter * 4
                layers.append(Upsampling2d(scale_factor=scale, mode=upsample_mode, name='{0}{1}'.format(upsample_mode, i)))
                #filter = filter // (scale * scale)
            else:
                if i > 0:
                    layers.append(
                        Conv2d_Block((3, 3), depth_multiplier=2, strides=1, auto_pad=True, use_spectral=use_spectral,
                                        use_bias=False, activation=activation, normalization=generator_norm,
                                        padding_mode='replicate'))

                layers.append(Upsampling2d(scale_factor=scale, mode=upsample_mode, name='{0}{1}'.format(upsample_mode, i)))


            if generator_build_block == BuildBlockMode.base.value:
                layers.append(Conv2d_Block((3, 3), num_filters=filter, strides=1, auto_pad=True, padding_mode='replicate',
                                           use_spectral=use_spectral, use_bias=False, activation=activation,
                                           normalization=generator_norm, dilation=dilation,
                                           name='base_block{0}'.format(i)))
            elif generator_build_block == BuildBlockMode.resnet.value:
                layers.extend(resnet_block(num_filters=filter, strides=1, activation=activation, use_spectral=use_spectral,
                                           normalization=generator_norm, dilation=dilation,
                                           name='resnet_block{0}'.format(i)))
            elif generator_build_block == BuildBlockMode.bottleneck.value:
                layers.append(bottleneck_block(num_filters=filter, strides=1, activation=activation, use_spectral=use_spectral,
                                               normalization=generator_norm, dilation=dilation,
                                               name='resnet_block{0}'.format(i)))
            if use_self_attention and current_width == initial_size * 2:
                layers.append(SelfAttention(8, name='self_attention'))
            if use_dropout and current_width == initial_size * 4:
                layers.append(Dropout(0.2))

            current_width = current_width * scale
            i = i + 1
        layers.append(Conv2d((5, 5), 3, strides=1, auto_pad=True, use_bias=False, activation='tanh', name='last_layer'))
        return Sequential(layers, name='generator')

    def build_discriminator():
        layers = []
        layers.append(
            Conv2d((5, 5), 32, strides=1, auto_pad=True, use_bias=False, activation=activation, name='first_layer'))
        layers.append(Conv2d_Block((3, 3), 64, strides=2, auto_pad=True, use_spectral=use_spectral, use_bias=False,
                                   activation=activation, normalization=discriminator_norm, name='second_layer'))
        filter = 64
        current_width = image_width // 2
        i = 0
        while current_width > 8:
            filter = filter * 2 if i % 2 == 1 else filter
            if discriminator_build_block == BuildBlockMode.base.value:
                layers.append(
                    Conv2d_Block((3, 3), num_filters=filter, strides=2, auto_pad=True, use_spectral=use_spectral, use_bias=False,
                                 activation=activation, normalization=discriminator_norm,
                                 name='base_block{0}'.format(i)))
            elif discriminator_build_block == BuildBlockMode.resnet.value:
                layers.extend(resnet_block(num_filters=filter, strides=2, activation=activation, use_spectral=use_spectral,
                                           normalization=discriminator_norm, name='resnet_block{0}'.format(i)))

            elif discriminator_build_block == BuildBlockMode.bottleneck.value:
                layers.append(
                    bottleneck_block(num_filters=filter, strides=2, reduce=2, activation=activation, use_spectral=use_spectral,
                                     normalization=discriminator_norm, name='bottleneck_block{0}'.format(i)))

            current_width = current_width // 2
            i = i + 1
        if use_self_attention:
            layers.insert(-2, SelfAttention(8, name='self_attention'))
        if use_dropout :
            layers.insert(-1, Dropout(0.5))
        layers.append(Conv2d_Block((3, 3), 128, strides=2, auto_pad=True, use_bias=False, activation='leaky_relu', use_spectral=use_spectral,normalization=discriminator_norm,name='last_conv'))
        layers.append(Flatten()),
        if use_minibatch_discrimination:
            layers.append(MinibatchDiscriminationLayer(name='minibatch_dis'))
        layers.append(Dense(1, use_bias=False, name='fc'))
        layers.append(Sigmoid())
        dis=Sequential(layers, name='discriminator')
        out=dis(to_tensor(TensorShape([None,3,image_width,image_width]).get_dummy_tensor()).to(get_device()))
        if use_spectral:
            new_layers=[]
            for layer in dis:
                if isinstance(layer,Dense):
                    new_layers.append(torch.nn.utils.spectral_norm(layer))
                else:
                    new_layers.append(layer)
            return Sequential(new_layers, name='discriminator')


    def build_autoencoder(gan_role='discriminator'):
        ae=Sequential()
        normalization = generator_norm if gan_role == 'generator' else discriminator_norm
        build_block = generator_build_block if gan_role == 'generator' else discriminator_build_block
        layers = []
        initial_size = image_width // 8
        layers.append(
            Conv2d((3, 3), 32, strides=2, auto_pad=True, use_bias=False, activation=activation, name='first_layer'))
        filter = 32
        dilation = 1

        for i in range(3):
            filter = filter + 32
            if use_dilation:
                dilation = 2 if i < 2 else 1
            if build_block == BuildBlockMode.base.value:
                layers.append(
                    Conv2d_Block((3, 3), filter, strides=2, auto_pad=True, use_spectral=use_spectral, use_bias=False,
                                 activation=activation, normalization=normalization, name='base_block{0}'.format(i)))
            elif build_block == BuildBlockMode.resnet.value:
                layers.extend(resnet_block(filter, strides=2, activation=activation, use_spectral=use_spectral,
                                           normalization=normalization, name='resnet_block{0}'.format(i)))
            elif build_block == BuildBlockMode.bottleneck.value:
                layers.append(
                    bottleneck_block(num_filters=filter, strides=2, reduce=4, activation=activation, use_spectral=use_spectral,
                                     normalization=normalization, name='bottleneck_block{0}'.format(i)))
            if use_self_attention and i == 1:
                layers.append(SelfAttention(16, name='self_attention'))
        #filter = filter // 2
        encoder=Sequential()
        for layer in layers:
            if layer.name is not None and len(layer.name)>2:
                encoder.add_module(layer.name,layer)
            else:
                encoder.add(layer)
        ae.add_module('encoder',encoder)
        layers=[]
        ae.add_module('embeddings',Conv2d((1, 1), filter, strides=1, auto_pad=True, use_bias=False, activation=activation, keep_output=True, name='embeddings'))

        for i in range(4):
            if upsample_mode == UpsampleMode.transpose.value:
                layers.append(
                    TransConv2d_Block((3, 3), num_filters=filter if i == 0 else filter // 2, strides=2, auto_pad=True,
                                      use_spectral=use_spectral, use_bias=False, activation=activation,
                                      normalization=normalization, dilation=dilation,
                                      name='transconv_block{0}'.format(i)))
            elif upsample_mode == UpsampleMode.pixel_shuffle.value:
                layers.append(
                    Conv2d_Block((3, 3), num_filters=4 * filter if i == 0 else 2 * filter, strides=1, auto_pad=True,
                                 use_spectral=use_spectral, use_bias=False, activation=activation,
                                 normalization=normalization))
                layers.append(Upsampling2d(scale_factor=2, mode=upsample_mode, name='{0}{1}'.format(upsample_mode, i)))
            else:
                layers.append(Upsampling2d(scale_factor=2, mode=upsample_mode, name='{0}{1}'.format(upsample_mode, i)))

            filter = filter if i == 0 else filter + 32
            if use_dropout and i == 0:
                layers.append(Dropout(0.2))

            if build_block == BuildBlockMode.base.value:
                layers.append(
                    Conv2d_Block((3, 3), filter, strides=1, auto_pad=True, use_spectral=use_spectral, use_bias=False,
                                 activation=activation, normalization=normalization, dilation=dilation,
                                 name='base_block{0}'.format(i)))
            elif build_block == BuildBlockMode.resnet.value:
                layers.extend(resnet_block(filter, strides=1, activation=activation, use_spectral=use_spectral,
                                           normalization=normalization, dilation=dilation,
                                           name='resnet_block{0}'.format(i)))
            elif build_block == BuildBlockMode.bottleneck.value:
                layers.append(bottleneck_block(filter, strides=1, activation=activation, use_spectral=use_spectral,
                                               normalization=normalization, dilation=dilation,
                                               name='resnet_block{0}'.format(i)))
            elif build_block == BuildBlockMode.separable_resnet.value:
                layers.extend(
                    separable_resnet_block(filter, strides=2, activation=activation, use_spectral=use_spectral,
                                           normalization=discriminator_norm, name='resnet_block{0}'.format(i)))

            filter = filter if i == 0 else filter + 32
        layers.append(Conv2d((3, 3), 3, strides=1, auto_pad=True, use_bias=False, activation='tanh', name='last_layer'))
        decoder=Sequential()
        for layer in layers:
            if layer.name is not None and len(layer.name)>2:
                decoder.add_module(layer.name, layer)
            else:
                decoder.add(layer)
        ae.add_module('decoder', decoder)
        ae.input_shape=TensorShape([None,3,image_width,image_width])
        return ae

    gen = Model(
        input_shape=(3, image_width, image_width) if generator_network_type == 'autoencoder' else (noise_shape),
        output=build_generator())
    gen.model.name = 'generator'

    dis = ImageClassificationModel(input_shape=(3, image_width, image_width), output=build_autoencoder(
        'discriminator') if discriminator_network_type == 'autoencoder' else build_discriminator())
    dis.model.name = 'discriminator'

    gen.model.apply(weights_init_normal)
    dis.model.apply(weights_init_normal)
    return gen, dis
