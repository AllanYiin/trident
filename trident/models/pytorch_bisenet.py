from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import *

import torch

from trident.backend.common import *
from trident.backend.pytorch_backend import Layer, Sequential, get_device
from trident.layers.pytorch_activations import Identity, Relu
from trident.layers.pytorch_blocks import *
from trident.layers.pytorch_layers import *
from trident.layers.pytorch_pooling import *
from trident.layers.pytorch_normalizations import *

__all__ = [ 'BiSeNetV2']

_session = get_session()
_device = get_device()
_epsilon = _session.epsilon
_trident_dir = _session.trident_dir

dirname = os.path.join(_trident_dir, 'models')


def DetailBranch(base_filter=64):
    return Sequential(
        OrderedDict({
            's1': Sequential(
                Conv2d_Block((3, 3), num_filters=base_filter, strides=2, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn'),
                Conv2d_Block((3, 3), num_filters=base_filter, strides=1, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn')
            ),
            's2': Sequential(
                Conv2d_Block((3, 3), num_filters=base_filter, strides=2, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn'),
                Conv2d_Block((3, 3), num_filters=base_filter, strides=1, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn'),
                Conv2d_Block((3, 3), num_filters=base_filter, strides=1, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn')
            ),
            's3': Sequential(
                Conv2d_Block((3, 3), num_filters=base_filter * 2, strides=2, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn'),
                Conv2d_Block((3, 3), num_filters=base_filter * 2, strides=1, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn'),
                Conv2d_Block((3, 3), num_filters=base_filter * 2, strides=1, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn')
            ),
        })
    )


def StemBlock():
    return Sequential(
        Conv2d_Block((3, 3), num_filters=16, strides=2, auto_pad=True, dilation=1, use_bias=False,
                     activation=Relu(inplace=True), normalization='bn'),
        ShortCut2d(
            OrderedDict({
                'left': Sequential(
                    Conv2d_Block((1, 1), num_filters=8, strides=1, auto_pad=False, dilation=1, use_bias=False,
                                 activation=Relu(inplace=True), normalization='bn'),
                    Conv2d_Block((3, 3), num_filters=16, strides=2, auto_pad=True, dilation=1, use_bias=False,
                                 activation=Relu(inplace=True), normalization='bn')
                ),
                'right': MaxPool2d((3, 3), strides=2, auto_pad=True)
            }), mode='concate', axis=1),
        Conv2d_Block((3, 3), num_filters=16, strides=1, auto_pad=True, dilation=1, use_bias=False,
                     activation=Relu(inplace=True), normalization='bn')
    )


def ContextEmbeddingBlock(base_filter=64):
    return Sequential(
        ShortCut2d(
            Sequential(
                GlobalAvgPool2d(keepdims=True),
                BatchNorm2d(),
                Conv2d_Block((1, 1), num_filters=base_filter * 2, strides=1, auto_pad=True, dilation=1, use_bias=False,
                             activation=Relu(inplace=True), normalization='bn'),
            ),
            Identity()
            , mode='add'),
        Conv2d_Block((3, 3), num_filters=base_filter * 2, strides=1, auto_pad=True, dilation=1, use_bias=False,
                     activation=None, normalization='bn')
    )


def GatherExpansionLayer1(num_filters, exp_ratio=6):
    return ShortCut2d(
        Sequential(
            Conv2d_Block((3, 3), depth_multiplier=1, strides=1, auto_pad=True, dilation=1, use_bias=False,
                         activation=Relu(inplace=True), normalization='bn'),
            DepthwiseConv2d_Block((3, 3), depth_multiplier=exp_ratio, strides=1, auto_pad=True, dilation=1,
                                  use_bias=False,
                                  activation=None, normalization='bn'),
            Conv2d_Block((1, 1), num_filters=num_filters, strides=1, auto_pad=True, dilation=1, use_bias=False,
                         activation=None, normalization='bn'),
        ),
        Identity(),
        mode='add', activation=Relu(inplace=True)
    )


def GatherExpansionLayer2(num_filters, exp_ratio=6):
    return ShortCut2d(
        Sequential(
            Conv2d_Block((3, 3), depth_multiplier=1, strides=1, auto_pad=True, dilation=1, use_bias=False,
                         activation=Relu(inplace=True), normalization='bn'),
            DepthwiseConv2d_Block((3, 3), depth_multiplier=exp_ratio, strides=2, auto_pad=True, dilation=1,
                                  use_bias=False,
                                  activation=None, normalization='bn'),
            DepthwiseConv2d_Block((3, 3), depth_multiplier=1, strides=1, auto_pad=True, dilation=1,
                                  use_bias=False,
                                  activation=None, normalization='bn'),
            Conv2d_Block((1, 1), num_filters=num_filters, strides=1, auto_pad=True, dilation=1, use_bias=False,
                         activation=None, normalization='bn'),
        ),
        Sequential(
            DepthwiseConv2d_Block((3, 3), depth_multiplier=1, strides=2, auto_pad=True, dilation=1,
                                  use_bias=False,
                                  activation=None, normalization='bn'),
            Conv2d_Block((1, 1), num_filters=num_filters, strides=1, auto_pad=True, dilation=1, use_bias=False,
                         activation=None, normalization='bn'),
        ),
        mode='add', activation=Relu(inplace=True)
    )


class SemanticBranch(Layer):

    def __init__(self, base_filter=64):
        super(SemanticBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = Sequential(
            GatherExpansionLayer2(base_filter // 2),
            GatherExpansionLayer1(base_filter // 2)
        )
        self.S4 = Sequential(
            GatherExpansionLayer2(base_filter),
            GatherExpansionLayer1(base_filter)
        )
        self.S5_4 = Sequential(
            GatherExpansionLayer2(base_filter * 2),
            GatherExpansionLayer1(base_filter * 2),
            GatherExpansionLayer1(base_filter * 2),
            GatherExpansionLayer1(base_filter * 2)
        )
        self.S5_5 = ContextEmbeddingBlock(base_filter)

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BilateralGuidedAggLayer(Layer):

    def __init__(self, base_filter=64):
        super(BilateralGuidedAggLayer, self).__init__()
        self.left1 = Sequential(
            DepthwiseConv2d_Block((3, 3), depth_multiplier=1, strides=1, auto_pad=True, dilation=1,
                                  use_bias=False,
                                  activation=None, normalization='bn'),
            Conv2d((1, 1), num_filters=base_filter * 2, strides=1, auto_pad=True, dilation=1,
                   use_bias=False,
                   activation=None))

        self.left2 = Sequential(
            Conv2d_Block((3, 3), num_filters=base_filter * 2, strides=2, auto_pad=True, dilation=1,
                         use_bias=False,
                         activation=None, normalization='bn'),
            AvgPool2d((3, 3), strides=2, auto_pad=True))

        self.right1 = Sequential(Conv2d_Block((3, 3), num_filters=base_filter * 2, strides=1, auto_pad=True, dilation=1,
                                              use_bias=False,
                                              activation=None, normalization='bn'),
                                 Upsampling2d(scale_factor=4, mode='bilinear', align_corners=False)
                                 )

        self.right2 = Sequential(
            DepthwiseConv2d_Block((3, 3), depth_multiplier=1, strides=1, auto_pad=True, dilation=1,
                                  use_bias=False,
                                  activation=None, normalization='bn'),

            Conv2d_Block((1, 1), num_filters=base_filter * 2, strides=1, auto_pad=True, dilation=1,
                         use_bias=False,
                         activation=None, normalization='bn')
        )

        self.up2 = Upsampling2d(scale_factor=4, mode='bilinear', align_corners=False)
        ##TODO: does this really has no relu?
        self.conv = Conv2d_Block((3, 3), num_filters=base_filter * 2, strides=1, auto_pad=True, dilation=1,
                                 use_bias=False,
                                 activation=Relu(inplace=True), normalization='bn')

    # x_d=>from detail branch
    # x_s=>from
    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)

        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out


def SegmentHead(mid_chan, up_factor=8, aux=True, num_classes=11):
    return Sequential(
        Dropout(0.1),
        Sequential(
            Upsampling2d(scale_factor=up_factor // 2, mode='bilinear', align_corners=False),
            Conv2d_Block((3, 3), num_filters=mid_chan, strides=1, auto_pad=True, dilation=1, use_bias=False,
                         activation=Relu(inplace=True), normalization='bn', keep_output=True),
            Conv2d((1, 1), num_filters=num_classes, strides=1, auto_pad=True, dilation=1, use_bias=False,
                   activation=SoftMax(axis=1, add_noise=True, noise_intensity=0.08), keep_output=True),
            Upsampling2d(scale_factor=2, mode='bilinear', align_corners=False)
        )

    )


def SegmentHeadV2(mid_chan, up_factor=8, aux=True, num_classes=11):
    return Sequential(
        Dropout(0.1),
        Sequential(
            Upsampling2d(scale_factor=up_factor // 2, mode='bilinear', align_corners=False),
            Conv2d_Block((3, 3), num_filters=mid_chan, strides=1, auto_pad=True, dilation=3, use_bias=False,
                         activation=Sigmoid(), normalization='bn', keep_output=True),
            Conv2d((1, 1), num_filters=num_classes, strides=1, auto_pad=True, dilation=1, use_bias=False,
                   activation=None, keep_output=True),
            Upsampling2d(scale_factor=2, mode='bilinear', align_corners=False),
            SoftMax(axis=1, add_noise=True, noise_intensity=0.08)
        )

    )




class BiSeNetV2(Layer):

    def __init__(self, n_classes=80, aux_mode='train'):
        super(BiSeNetV2, self).__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segmentic = SemanticBranch()
        self.bga = BilateralGuidedAggLayer()
        ## TODO: what is the number of mid chan ?
        self.head = SegmentHead(128, up_factor=8, aux=False,num_classes=n_classes)
        self.aux2 = SegmentHead(128, up_factor=4,num_classes=n_classes)
        self.aux3 = SegmentHead(128, up_factor=8,num_classes=n_classes)
        self.aux4 = SegmentHead(128, up_factor=16,num_classes=n_classes)
        self.aux5_4 = SegmentHead(128, up_factor=32,num_classes=n_classes)


    def forward(self, x):
        size = x.size()[2:]
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segmentic(x)


        feat_head = self.bga(feat_d, feat_s)
        segment_result = self.head(feat_head)
        results=OrderedDict()
        results['output']=segment_result
        results['segment_aux2'] = self.aux2(feat2)
        results['segment_aux3'] = self.aux3(feat3)
        results['segment_aux4'] = self.aux4(feat4)
        results['segment_aux5_4'] = self.aux5_4(feat5_4)
        if self.training:
            return results
        else:
            return segment_result
