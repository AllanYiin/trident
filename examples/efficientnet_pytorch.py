import os

os.environ['TRIDENT_BACKEND'] = 'pytorch'
import trident

#C.debugging.set_computation_network_trace_level(10)
#$C.debugging.set_checked_mode(True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import trident.backend as T

spec_b0 = ((16, 1, 3, 1, 1, 0.25, 0.2),  # size, expand, kernel, stride, repeat, se_ratio, dc_ratio
           (24, 6, 3, 2, 2, 0.25, 0.2), (40, 6, 5, 2, 2, 0.25, 0.2), (80, 6, 3, 2, 3, 0.25, 0.2),
           (112, 6, 5, 1, 3, 0.25, 0.2), (192, 6, 5, 2, 4, 0.25, 0.2), (320, 6, 3, 1, 1, 0.25, 0.2),)


def mb_block(x, size_out, expand=1, kernel=1, strides=1, se_ratio=0.25, dc_ratio=0.2, **kw):
    """ MobileNet Bottleneck Block. """
    input_shape = x.size(1)
    expand_shape = input_shape * expand
    se_shape = input_shape * se_ratio
    x1 = T.Conv2d(kernel_size=(1,1), input_filters=input_shape, num_filters=expand_shape, strides=1)(x)
    x1 = T.Conv2d(kernel_size=(kernel, kernel), input_filters=expand_shape, num_filters=expand_shape, strides=strides,groups=expand_shape)(x1)
    se = F.adaptive_avg_pool2d(x1, 1)
    se = T.Conv2d(kernel_size=(1, 1), input_filters=x.size(1), num_filters=se_shape, strides=1, activation=T.swish)(x1)
    se = T.Conv2d(kernel_size=(1, 1), input_filters=se_shape, num_filters=input_shape, activation=T.swish)(x1)

    x1 = T.Conv2d(kernel_size=(1, 1), input_filters=input_shape, num_filters=input_shape, activation=None)(x1)
    if strides == 1 and input_shape == size_out:
        if dc_ratio == 0:
            x1 += x
        else:
            dc_ratio = 1.0 - dc_ratio
            drop_mask = torch.rand([x.shape[0], 1, 1, 1], device=x.device, requires_grad=False) + dc_ratio
            x1 += x / dc_ratio * drop_mask.floor()
    return x1


class EfficientNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.first_layer = T.Conv2d(kernel_size=(1, 1), input_filters=3, num_filters=32, strides=2, activation=T.swish)

    def forward(self, x):
        x = self.first_layer(x)
        for size, expand, kernel, strides, repeat, se_ratio, dc_ratio in spec_b0:
            for i in range(repeat):
                strides = strides if i == 0 else 1
                x = mb_block(x, size, expand, kernel, strides, se_ratio, dc_ratio)
        x = T.GcdConv2d(kernel_size=(1, 1), input_filters=x.size(1), num_filters=1280, strides=1, activation=None)(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = nn.Dropout2d()(x)
        x = x.view(x.size(1), -1)
        x = nn.Linear(x.size(1), 100)(x)
        return x


m = EfficientNet()

T.summary(m)
