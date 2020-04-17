import os
import sys
import codecs

os.environ['TRIDENT_BACKEND'] = 'cntk'
import collections

import math
import cntk as C
from cntk.layers import *
from cntk.ops.functions import *
from cntk.ops import *
from cntk.learners import *
import numpy as np
import linecache

import trident as T


# C.debugging.set_computation_network_trace_level(1000)
# C.debugging.set_checked_mode(True)
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def calculate_flops(x):
    flops = 0
    for p in x.parameters:
        flops += p.value.size
    return flops


data = T.load_mnist('fashion-mnist', 'train', is_flatten=False)
dataset = T.DataProvider('fashion-mnist')
dataset.mapping(data=data[0], labels=data[1], scenario='train')

input_var = C.input_variable((1,28, 28), dtype=np.float32)
label_var = C.input_variable((10), dtype=np.float32)


def bottleneck_block(inp, out_filters, normalization='instance', before_shortcut_relu=False, after_shortcut_relu=False,
                     shortcut_connect_rate=0.5, divisor_rank=0):
    normalization_fn = T.get_normalization(normalization)
    in_filters = None
    if hasattr(inp, 'shape'):
        in_filters = getattr(inp, 'shape')[0]
    elif isinstance(inp, C.Function):
        in_filters = inp.arguments[0].shape[0]

    sc = T.conv2d(inp, (1, 1), num_filters=in_filters // 4, strides=1, padding='same', activation=None,
                  name='bottleneck_1X1_1')
    sc = normalization_fn()(sc)
    if before_shortcut_relu:
        sc = T.leaky_relu6(sc)
    sc = T.conv2d(sc, (3, 3), num_filters=in_filters // 4, strides=1, padding='same', activation=None,
                  name='bottleneck_3X3_1')
    sc = normalization_fn()(sc)
    if before_shortcut_relu:
        sc = T.leaky_relu6(sc)
    sc = T.conv2d(sc, (1, 1), num_filters=in_filters, strides=1, padding='same', activation=None,
                  name='bottleneck_1X1_2')
    inp = inp + shortcut_connect_rate * sc
    inp = T.conv2d(inp, (3, 3), num_filters=out_filters, strides=1, padding='same', activation=None,
                   name='bottleneck_final')
    inp = normalization_fn()(inp)
    if after_shortcut_relu:
        inp = T.leaky_relu6(inp)
    return inp


#
#           [BaseX3]
#         \          [BaseX7]
#         \          [BaseX5]
#          \         [BaseX3]
#          ___(+)____\
#            [BaseX5]

#   Reversed
#           [BaseX7]
#         \          [BaseX3]
#         \          [BaseX5]
#          \         [BaseX7]
#          ___(+)____\
#            [BaseX5]


def gcd_branch3_block(inp, base=16, factors=None, normalization='instance', before_shortcut_relu=False,
                      after_shortcut_relu=False, shortcut_connect_rate=0.8, divisor_rank=0):
    if factors is None:
        factors = [3, 5, 7]
    normalization_fn = T.get_normalization(normalization)
    in_filters = None
    if hasattr(inp, 'shape'):
        in_filters = getattr(inp, 'shape')[0]
    elif isinstance(inp, C.Function):
        in_filters = inp.arguments[0].shape[0]

    inp = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[0]), strides=1, padding='same', activation=None,
                       divisor_rank=divisor_rank, name='gcd_block1_{0}_{1}'.format(base, factors[0]))
    inp = normalization_fn()(inp)
    inp = T.leaky_relu6(inp)

    sc = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[2]), strides=1, padding='same', activation=None,
                      divisor_rank=divisor_rank, name='gcd_block2_{0}_{1}'.format(base, factors[2]))
    sc = normalization_fn()(sc)
    if before_shortcut_relu:
        sc = T.leaky_relu6(sc)
    sc = T.gcd_conv2d(sc, (3, 3), num_filters=int(base * factors[1]), strides=1, padding='same', activation=None,
                      divisor_rank=divisor_rank, name='gcd_block3_{0}_{1}'.format(base, factors[1]))
    sc = normalization_fn()(sc)
    if before_shortcut_relu:
        sc = T.leaky_relu6(sc)
    sc = T.gcd_conv2d(sc, (3, 3), num_filters=int(base * factors[0]), strides=1, padding='same', activation=None,
                      divisor_rank=divisor_rank, name='gcd_block4_{0}_{1}'.format(base, factors[0]))
    inp = inp + shortcut_connect_rate * sc
    inp = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[1]), strides=1, padding='same', activation=None,
                       divisor_rank=divisor_rank, name='gcd_block5_{0}_{1}'.format(base, factors[2]))
    inp = normalization_fn()(inp)
    if after_shortcut_relu:
        inp = T.leaky_relu6(inp)
    return inp


#   [3,5,7,11]
#                       [BaseX7]---\--------------\
#         \          [BaseX3]      [BaseX11]
#         \          [BaseX5]             \
#          \         [BaseX7]      [BaseX7]
#                     ___(+)____\____(+)______\
#                                      |

def gcd_double_branch_block(inp, base=16, factors=None, normalization='instance', before_shortcut_relu=False,
                            after_shortcut_relu=False, divisor_rank=0):
    if factors is None:
        factors = [3, 5, 7, 11]
    in_filters = None
    if hasattr(inp, 'shape'):
        in_filters = int(getattr(inp, 'shape'))[0]
    elif isinstance(inp, C.Function):
        in_filters = inp.arguments[0].shape[0]

    normalization_fn = T.get_normalization(normalization)
    inp = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[2]), strides=1, padding='same', activation=None,
                       divisor_rank=divisor_rank)
    inp = normalization_fn()(inp)
    inp = T.leaky_relu6(inp)

    sc1 = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[0]), strides=1, padding='same', activation=None,
                       divisor_rank=divisor_rank)
    sc1 = normalization_fn()(sc1)
    if before_shortcut_relu:
        sc1 = T.leaky_relu6(sc1)
    sc1 = T.gcd_conv2d(sc1, (3, 3), num_filters=int(base * factors[1]), strides=1, padding='same', activation=None,
                       divisor_rank=divisor_rank)
    sc1 = normalization_fn()(sc1)
    if before_shortcut_relu:
        sc1 = T.leaky_relu6(sc1)
    sc1 = T.gcd_conv2d(sc1, (3, 3), num_filters=int(base * factors[2]), strides=1, padding='same', activation=None,
                       divisor_rank=divisor_rank)

    sc2 = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[3]), strides=1, padding='same', activation=None,
                       divisor_rank=divisor_rank)
    sc2 = normalization_fn()(sc2)
    if before_shortcut_relu:
        sc2 = T.leaky_relu6(sc2)
    sc2 = T.gcd_conv2d(sc2, (3, 3), num_filters=int(base * factors[2]), strides=1, padding='same', activation=None,
                       divisor_rank=divisor_rank)
    inp = inp + 0.5 * sc1 + 0.5 * sc2
    inp = normalization_fn()(inp)
    if after_shortcut_relu:
        inp = T.leaky_relu6(inp)
    return inp


# def gcd_block(inp, num_filters=16, factors=None, divisor_rank=0):
#     if factors is None:
#         factors = [3, 5, 7]
#     input_shape = inp.shape[0]
#     sc = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[0]), strides=1, padding='same', activation=None,
#                       divisor_rank=divisor_rank, name='gcd_block2_{0}_{1}'.format(base, factors[0]))
#     sc = T.leaky_relu6(sc)
#     sc = T.gcd_conv2d(sc, (3, 3), num_filters=int(base * factors[1]), strides=1, padding='same', activation=None,
#                       divisor_rank=divisor_rank, name='gcd_block3_{0}_{1}'.format(base, factors[1]))
#     sc = T.leaky_relu6(sc)
#     sc = T.gcd_conv2d(sc, (3, 3), num_filters=input_shape, strides=1, padding='same', activation=None,
#                       divisor_rank=divisor_rank, name='gcd_block4_{0}'.format(input_shape))
#     inp = inp + sc
#     inp = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[2]), strides=1, padding='same', activation=None,
#                        divisor_rank=0, name='gcd_block5_{0}_{1}'.format(base, factors[2]))
#     inp = T.InstanceNormalization()(inp)
#     return inp
#
# #
# x4 = C.layers.Convolution2D((5, 5), 24, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
# x4 = gcd_block(x4, 8, [3, 5, 7])
# x4 = gcd_block(x4, 12, [3, 5, 7])
#
# x4 = T.gcd_conv2d(x4, (3, 3), num_filters=16*7, strides=2, padding='same', activation=None, divisor_rank=0)  # 8*13
# x4 = T.InstanceNormalization()(x4)
# x4 = T.leaky_relu6(x4)
# x4 = gcd_block(x4, 8, [7, 11, 13])
# x4 = gcd_block(x4, 12, [7, 11, 13])
#
# x4 = T.gcd_conv2d(x4, (3, 3), num_filters=16*13, strides=2, padding='same', activation=None, divisor_rank=0)  # 8*13
# x4 = T.InstanceNormalization()(x4)
# x4 = T.leaky_relu6(x4)
# x4 = gcd_block(x4, 8, [13,17, 19])
# x4 = gcd_block(x4, 12, [13,17, 19])
#
# x4 = T.gcd_conv2d(x4, (3, 3), num_filters=16*19, strides=2, padding='same', activation=None, divisor_rank=0)  # 8*13
# x4 = T.InstanceNormalization()(x4)
# x4 = T.leaky_relu6(x4)
# x4 = gcd_block(x4, 8, [19,23,29])
# x4 = gcd_block(x4, 12, [19,23,29])
# x4 = C.layers.Convolution2D((1, 1), 100, T.leaky_relu6, strides=1, pad=True)(x4)
# x4 = C.squeeze(C.layers.GlobalAveragePooling()(x4))
# z4 = C.layers.Dense((100), activation=T.sigmoid)(x4)


#
# x5 = C.layers.Convolution2D((5, 5), 32, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
# x5 = gcd_double_block(x5, 8, [3, 5, 7, 11], divisor_rank=1)
# x5 = T.gcd_conv2d(x5, (3, 3), num_filters=64, strides=2, padding='same', activation=None, divisor_rank=0)  # 7*11*2
# x5 = T.InstanceNormalization()(x5)
# x5 = T.leaky_relu6(x5)
# x5 = gcd_double_block(x5, 8, [11, 13, 17, 19], divisor_rank=1)
# x5 = T.gcd_conv2d(x5, (3, 3), num_filters=144, strides=2, padding='same', activation=None, divisor_rank=0)  # 71*11*2
# x5 = T.InstanceNormalization()(x5)
# x5 = T.leaky_relu6(x5)
# x5 = gcd_double_block(x5, 8, [19, 23, 29, 31])
# x5 = T.gcd_conv2d(x5, (3, 3), num_filters=256, strides=2, padding='same', activation=None, divisor_rank=0)  # 8*13
# x5 = T.InstanceNormalization()(x5)
# x5 = T.leaky_relu6(x5)
# x5 = gcd_block(x5, 72, [3, 5, 7])
# x5 = C.layers.Convolution2D((1, 1), num_filters=100, activation=T.leaky_relu6, strides=1, pad=True)(x5)
# x5 = C.squeeze(C.layers.GlobalAveragePooling()(x5))
# z5 = T.sigmoid(x5)


classifiertype='dense'

def baselineNet(input_var):
    global  classifiertype
    return T.Sequential2(input_var,[
        T.Conv2d((5, 5), num_filters=32, strides=1, activation='leaky_relu6', padding='same'),
        T.Conv2d_Block((3, 3),num_filters= 64,strides= 1, auto_pad=True, activation='leaky_relu6', normalization='instance'),
        T.Conv2d_Block((3, 3), num_filters=64, strides=2, auto_pad=True, activation='leaky_relu6', normalization='instance'),
        T.Conv2d_Block((3, 3), num_filters=128,strides= 1, auto_pad=True, activation='leaky_relu6', normalization='instance', dropout_rate=0.2),
        T.ShortCut({'left': [T.Conv2d((1, 1), num_filters=64, strides=1, activation=None, padding='same'),
            T.Conv2d((3, 3), num_filters=256, strides=1, activation=None, padding='same')],
            'right': [T.Conv2d((1, 1), num_filters=256, strides=1, activation=None, padding='same')]}),
        T.Conv2d_Block((3, 3), num_filters=512, strides=2, auto_pad=True, activation='leaky_relu6',
                       normalization='instance'),
        T.Conv2d_Block((3, 3), num_filters=128,strides= 2, auto_pad=True, activation='leaky_relu6', normalization='instance'),
        T.Conv2d_Block((3, 3), num_filters=256, strides=1, auto_pad=True, activation='leaky_relu6', normalization='instance'),
    T.Conv2d((1, 1), num_filters=550, strides=1, activation=None,  padding='same'),
    T.Classifier(num_classes=10,is_multiselect=False,classifier_type=classifiertype)
    ],name='')






def challengerNet(input_var):
    global classifiertype
    return T.Sequential2(input_var,[
            T.Conv2d((5, 5), num_filters=32, strides=1, activation='leaky_relu6', padding='same'),
            T.GcdConv2d_Block((3, 3),num_filters= 48,strides= 1, auto_pad=True, activation='leaky_relu6', normalization=None,divisor_rank=0),
            T.GcdConv2d_Block((3, 3), num_filters=80, strides=2, auto_pad=True, activation='leaky_relu6', normalization=None,divisor_rank=0),
            T.GcdConv2d_Block((3, 3), num_filters=112,strides= 1, auto_pad=True, activation='leaky_relu6', normalization=None,divisor_rank=0, dropout_rate=0.2),
            T.ShortCut({
                'left': [T.GcdConv2d((1, 1), num_filters=96, strides=1, activation=None, padding='same', divisor_rank=0),
                    T.GcdConv2d((3, 3), num_filters=256, strides=1, activation=None, padding='same', divisor_rank=0)],
                'right': [
                    T.GcdConv2d((1, 1), num_filters=256, strides=1, activation=None, padding='same', divisor_rank=0)]}),
            T.GcdConv2d_Block((3, 3), num_filters=176,strides= 2, auto_pad=True, activation='leaky_relu6', normalization=None,divisor_rank=0),
            T.GcdConv2d_Block((3, 3), num_filters=208, strides=1, auto_pad=True, activation='leaky_relu6', normalization=None,divisor_rank=0),
            T.GcdConv2d_Block((3, 3), num_filters=352, strides=1,auto_pad=True, activation='leaky_relu6', normalization=None,divisor_rank=0),
            T.GcdConv2d((3, 3), num_filters=550, strides=1, padding='same', activation=None,divisor_rank=0),#這個活化函數必須拿掉，讓梯度好順利傳遞
        T.Classifier(num_classes=10,is_multiselect=False,classifier_type=classifiertype)
        ],name='')


def challengerNet0(input_var):
    global classifiertype
    return T.Sequential2(input_var,[
            T.Conv2d((5, 5), num_filters=32, strides=1, activation='leaky_relu6', padding='same'),
            T.GcdConv2d_Block((3, 3),num_filters= 48,strides= 1, auto_pad=True, activation='leaky_relu6', normalization='instance',divisor_rank=0,self_norm=False),
            T.GcdConv2d_Block((3, 3), num_filters=80, strides=2, auto_pad=True, activation='leaky_relu6', normalization='instance',divisor_rank=0,self_norm=False),
            T.GcdConv2d_Block((3, 3), num_filters=112,strides= 1, auto_pad=True, activation='leaky_relu6', normalization='instance',divisor_rank=0,self_norm=False, dropout_rate=0.2),
            T.ShortCut({
                'left': [T.GcdConv2d((1, 1), num_filters=96, strides=1, activation=None, padding='same', divisor_rank=0,self_norm=False),
                    T.GcdConv2d((3, 3), num_filters=256, strides=1, activation=None, padding='same', divisor_rank=0,self_norm=False)],
                'right': [
                    T.GcdConv2d((1, 1), num_filters=256, strides=1, activation=None, padding='same', divisor_rank=0,self_norm=False)]}),
            T.GcdConv2d_Block((3, 3), num_filters=176,strides= 2, auto_pad=True, activation='leaky_relu6', normalization='instance',divisor_rank=0,self_norm=False),
            T.GcdConv2d_Block((3, 3), num_filters=208, strides=1, auto_pad=True, activation='leaky_relu6', normalization='instance',divisor_rank=0,self_norm=False),
            T.GcdConv2d_Block((3, 3), num_filters=352, strides=1, auto_pad=True, activation='leaky_relu6', normalization='instance',divisor_rank=0,self_norm=False),
            T.GcdConv2d((3, 3), num_filters=550, strides=1, padding='same', activation=None,divisor_rank=0,self_norm=False),#這個活化函數必須拿掉，讓梯度好順利傳遞
        T.Classifier(num_classes=10,is_multiselect=False,classifier_type=classifiertype)
        ],name='')

def challenger2Net(input_var):
    global  classifiertype
    return T.Sequential2(input_var,[
        T.Conv2d((5, 5), num_filters=32, strides=1, activation='leaky_relu6', padding='same'),
        T.SepatableConv2d_Block((3, 3),depth_multiplier=2,strides= 1,auto_pad=True, activation='leaky_relu6', normalization='instance'),
        T.SepatableConv2d_Block((3, 3), depth_multiplier=1, strides=2, auto_pad=True, activation='leaky_relu6', normalization='instance'),
        T.SepatableConv2d_Block((3, 3), depth_multiplier=2,strides= 1, auto_pad=True, activation='leaky_relu6', normalization='instance', dropout_rate=0.2),
        T.ShortCut({'left': [T.SeparableConv2d((1, 1), depth_multiplier=2, strides=1, activation=None, padding='same'),
            T.Conv2d((1, 1), num_filters=128, strides=1,activation=None, padding='same')],
            'right': [T.SeparableConv2d((1, 1), depth_multiplier=1, strides=1, activation=None, padding='same')]}),
        T.SepatableConv2d_Block((3, 3), depth_multiplier=2,strides= 2, auto_pad=True, activation='leaky_relu6', normalization='instance'),
    T.SeparableConv2d((1, 1), depth_multiplier=2,strides=1, activation=None,  padding='same'),
    T.Classifier(num_classes=10,is_multiselect=False,classifier_type=classifiertype)
    ],name='')


# z_challenger1 = challengerNet(input_var, divisor_rank=0, before_shortcut_relu=False, after_shortcut_relu=False)
# z_challenger2 = challengerNet(input_var, divisor_rank=1, before_shortcut_relu=False, after_shortcut_relu=False)
# z_challenger3 = challengerNet(input_var, divisor_rank=0, before_shortcut_relu=True, after_shortcut_relu=False)
# z_challenger4 = challengerNet(input_var, divisor_rank=0, before_shortcut_relu=False, after_shortcut_relu=True)

#os.remove('model_log_fmnist_baseline.txt')
f = codecs.open('model_log_fmnist_baseline.txt', 'a', encoding='utf-8-sig')
# model = Function.load('Models/model5_cifar100.model')
#, 'baseline': z_baseline

# dict=collections.OrderedDict(sorted( {'challenger1': z_challenger1, 'challenger2': z_challenger2, 'challenger3': z_challenger3, 'challenger4': z_challenger4}.items(), key=lambda t: t[0]))

for cls in ['dense']:
    classifiertype=cls
    for learning_rate in [1e-2]:

        k1='baseline'
        z1=baselineNet(input_var)

        k2 = ' gcd_challenger'
        z2 = challengerNet(input_var)
        k3 = 'depthwise_challenger'
        z3 =challenger2Net(input_var)


        prefix1='fmnist_model_{0}_{1}_{2}'.format(k1,cls,learning_rate).replace('.','_')
        prefix2 = 'fmnist_model_{0}_{1}_{2}'.format(k2, cls, learning_rate).replace('.', '_')
        prefix3 = 'fmnist_model_{0}_{1}_{2}'.format(k3, cls, learning_rate).replace('.', '_')
        flops_baseline = calculate_flops(z1)
        print('flops_{0}:{1}'.format(k1, flops_baseline))

        flops_challenger = calculate_flops(z2)
        print('flops_{0}:{1}'.format(k2, flops_challenger))

        flops_challenger0 = calculate_flops(z3)
        print('flops_{0}:{1}'.format(k3, flops_challenger0))

        loss1 = C.cross_entropy_with_softmax(z1, label_var)
        err1 = 1 - C.classification_error(z1, label_var)
        loss2 = C.cross_entropy_with_softmax(z2, label_var)
        err2 = 1 - C.classification_error(z2, label_var)
        loss3 = C.cross_entropy_with_softmax(z3, label_var)
        err3 = 1 - C.classification_error(z3, label_var)

        lr = learning_rate
        C.logging.log_number_of_parameters(z1)
        C.logging.log_number_of_parameters(z2)
        C.logging.log_number_of_parameters(z3)


        progress_printer1 = C.logging.ProgressPrinter(freq=50, first=5, tag='Training', num_epochs=10)
        progress_printer2 = C.logging.ProgressPrinter(freq=50, first=5, tag='Training', num_epochs=10)
        progress_printer3 = C.logging.ProgressPrinter(freq=50, first=5, tag='Training', num_epochs=10)
        learner1 = C.adam(z1.parameters, lr=C.learning_rate_schedule([lr], C.UnitType.minibatch),
                         l1_regularization_weight=5e-6,
                         momentum=C.momentum_schedule(0.75))


        trainer1 = C.Trainer(z1, (loss1, err1), learner1, progress_printer1)

        learner2 = C.adam(z2.parameters, lr=C.learning_rate_schedule([lr], C.UnitType.minibatch),
                         l1_regularization_weight=5e-6, momentum=C.momentum_schedule(0.75))

        trainer2 = C.Trainer(z2, (loss2, err2), learner2, progress_printer2)

        learner3 = C.adam(z3.parameters, lr=C.learning_rate_schedule([lr], C.UnitType.minibatch),
                          l1_regularization_weight=5e-6, momentum=C.momentum_schedule(0.75))
        # learner =cntkx.learners.RAdam(z_class.parameters, learning_rate, 0.912, beta2=0.999,l1_regularization_weight=1e-3,
        # l2_regularization_weight=5e-4, epoch_size=300)
        trainer3 = C.Trainer(z3, (loss3, err3), learner3, progress_printer3)
        tot_loss = 0
        tot_metrics = 0
        mbs=0
        index_dict={}
        index_dict['loss1']=0
        index_dict['loss2'] = 0
        index_dict['loss3'] = 0
        index_dict['err1'] = 0
        index_dict['err2'] = 0
        index_dict['err3'] = 0

        for epoch in range(200):
            print('epoch {0}'.format(epoch))
            mbs=0
            #raw_imgs, raw_labels = dataset.next_bach(64)
            while  (mbs+1)%3000>0 :
                try:
                    # 定義數據如何對應變數
                    #if epoch>0 or  mbs + 1 > 500:
                    raw_imgs, raw_labels = dataset.next_bach(64)
                    trainer1.train_minibatch({input_var: raw_imgs, label_var: raw_labels})
                    trainer2.train_minibatch({input_var: raw_imgs, label_var: raw_labels})
                    trainer3.train_minibatch({input_var: raw_imgs, label_var: raw_labels})
                    index_dict['loss1'] +=trainer1.previous_minibatch_loss_average
                    index_dict['loss2'] +=trainer2.previous_minibatch_loss_average
                    index_dict['loss3'] +=trainer3.previous_minibatch_loss_average
                    index_dict['err1'] +=trainer1.previous_minibatch_evaluation_average
                    index_dict['err2'] +=trainer2.previous_minibatch_evaluation_average
                    index_dict['err3'] +=trainer3.previous_minibatch_evaluation_average

                    if mbs==500 or (mbs+1)%1000==0:
                        print(prefix1)
                        for p in z1.parameters:
                            #print('{0}   {1}'.format(p.owner.root_function.name, node.owner.op_name))
                            print('{0}   {1}'.format(p.uid, p.value.shape))
                            print('max: {0:.4f} min: {1:.4f} mean:{2:.4f}'.format(p.value.max(), p.value.min(), p.value.mean()))
                        print('')
                        print(prefix2)
                        for p in z2.parameters:
                            # print('{0}   {1}'.format(p.owner.root_function.name, node.owner.op_name))
                            print('{0}   {1}'.format(p.uid, p.value.shape))
                            print('max: {0:.4f} min: {1:.4f} mean:{2:.4f}'.format(p.value.max(), p.value.min(),
                                                                                  p.value.mean()))
                    if (mbs + 1) % 50 == 0:
                        f.writelines(['model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format(prefix1, learning_rate, epoch, mbs + 1,index_dict['loss1']/50,index_dict['err1']/50)])

                        f.writelines(['model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format( prefix2, learning_rate, epoch, mbs + 1, index_dict['loss2']/50, index_dict['err2']/50)])
                        f.writelines(['model: {0}  learningrate {1}  epoch {2}  {3}/ 1000 loss: {4} metrics: {5} \n'.format(prefix3, learning_rate, epoch, mbs + 1,index_dict['loss3']/50,index_dict['err3']/50)])
                        index_dict['loss1'] = 0
                        index_dict['loss2'] = 0
                        index_dict['loss3'] = 0
                        index_dict['err1'] = 0
                        index_dict['err2'] = 0
                        index_dict['err3'] = 0
                    if(mbs + 1) % 500 == 0 or mbs ==0 or mbs==1 or  (mbs + 1) == 10 or (mbs + 1) == 50  or (mbs + 1) == 100 or  (mbs + 1) == 200 or  (mbs + 1) == 300 or  (mbs + 1) == 400:
                            tot_p=0
                            tot_zero=0
                            max_p=0
                            min_p=0
                            for p in z1.parameters:
                                tot_p+=p.value.size
                                tot_zero+=np.equal(p.value,0).astype(np.float32).sum()
                                max_p=max(max_p,p.value.max())
                                min_p = min(min_p, p.value.min())
                            print('baseline: zero ratio {0:.3%} max: {1:.4f}  min: {2:.4f}'.format(tot_zero/tot_p,max_p,min_p))

                            tot_p = 0
                            tot_zero = 0
                            max_p = 0
                            min_p = 0
                            for p in z2.parameters:
                                tot_p += p.value.size
                                tot_zero += np.equal(p.value, 0).astype(np.float32).sum()
                                max_p = max(max_p, p.value.max())
                                min_p = min(min_p, p.value.min())
                            print('challenger zero ratio {0:.3%} max: {1:.4f}  min: {2:.4f}'.format(tot_zero / tot_p, max_p, min_p))

                    if (mbs + 1) % 250 == 0:
                        lr = learning_rate / (1 + 0.05 * 4 * (epoch + (mbs / 250.)))
                        print('learning rate:{0}'.format(lr))
                        z1.save('Models/{0}.model'.format(prefix1))
                        z2.save('Models/{0}.model'.format(prefix2))
                        z3.save('Models/{0}.model'.format(prefix3))
                except Exception as e:
                    PrintException()
                    print(e)
                mbs += 1
            trainer1.summarize_training_progress()
            trainer2.summarize_training_progress()
            trainer3.summarize_training_progress()
