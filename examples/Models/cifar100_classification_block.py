import os
import sys
import codecs

os.environ['TRIDENT_BACKEND'] = 'cntk'
from trident import backend as T
import math
import cntk as C
from cntk.layers import *
from cntk.ops.functions import *
from cntk.ops import *
from cntk.learners import *
import numpy as np
import linecache


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


data = T.load_cifar('cifar100', 'train', is_flatten=False)
dataset = T.Dataset('cifar100')
dataset.mapping(data=data[0], labels=data[1], scenario='train')

input_var = C.input_variable((3, 32, 32), dtype=np.float32)
label_var = C.input_variable((100), dtype=np.float32)


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
                      after_shortcut_relu=False, shortcut_connect_rate=0.5, divisor_rank=0):
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

def baselineNet(input_var, normalization='instance', before_shortcut_relu=False, after_shortcut_relu=False):
    x7 = T.conv2d(input_var, (5, 5), num_filters=32, activation=T.leaky_relu6, strides=1, padding='same')
    x7 = bottleneck_block(x7, 64, normalization='instance', before_shortcut_relu=before_shortcut_relu,
                          after_shortcut_relu=after_shortcut_relu)
    x7 = bottleneck_block(x7, 64, normalization='instance', before_shortcut_relu=before_shortcut_relu,
                          after_shortcut_relu=after_shortcut_relu)
    x7 = bottleneck_block(x7, 128, normalization='instance', before_shortcut_relu=before_shortcut_relu,
                          after_shortcut_relu=after_shortcut_relu)
    x7 = T.conv2d(x7, (3, 3), num_filters=128, strides=2, padding='same', activation=None)
    x7 = T.InstanceNormalization()(x7)
    x7 = T.leaky_relu6(x7)

    x7 = bottleneck_block(x7, 256, normalization='instance', before_shortcut_relu=before_shortcut_relu,
                          after_shortcut_relu=after_shortcut_relu)
    x7 = bottleneck_block(x7, 512, normalization='instance', before_shortcut_relu=before_shortcut_relu,
                          after_shortcut_relu=after_shortcut_relu)
    x7 = bottleneck_block(x7, 512, normalization='instance', before_shortcut_relu=before_shortcut_relu,
                          after_shortcut_relu=after_shortcut_relu)
    x7 = T.conv2d(x7, (3, 3), num_filters=512, strides=2, padding='same', activation=None)
    x7 = T.InstanceNormalization()(x7)
    x7 = T.leaky_relu6(x7)
    x7 = T.conv2d(x7, (1, 1), num_filters=100, activation=None, strides=1, padding='same')
    x7 = C.squeeze(C.layers.GlobalAveragePooling()(x7))
    z7 = T.sigmoid(x7)
    return z7


def challengerNet(input_var, normalization='instance', before_shortcut_relu=False, after_shortcut_relu=False,
                  divisor_rank=0):
    x7 = T.conv2d(input_var, (5, 5), num_filters=32, activation=T.leaky_relu6, strides=1, padding='same')
    x7 = gcd_branch3_block(x7, base=12, factors=[3, 5, 7], normalization='instance',
                           before_shortcut_relu=before_shortcut_relu, after_shortcut_relu=after_shortcut_relu,
                           divisor_rank=divisor_rank)
    x7 = gcd_branch3_block(x7, base=12, factors=[7, 11, 13], normalization='instance',
                           before_shortcut_relu=before_shortcut_relu, after_shortcut_relu=after_shortcut_relu,
                           divisor_rank=divisor_rank)
    x7 = gcd_branch3_block(x7, base=12, factors=[13, 17, 19], normalization='instance',
                           before_shortcut_relu=before_shortcut_relu, after_shortcut_relu=after_shortcut_relu,
                           divisor_rank=divisor_rank)
    x7 = T.gcd_conv2d(x7, (3, 3), num_filters=221, strides=2, padding='same', activation=None, divisor_rank=1)
    x7 = T.InstanceNormalization()(x7)
    x7 = T.leaky_relu6(x7)

    x7 = gcd_branch3_block(x7, 10, factors=[13, 17, 19], normalization='instance',
                           before_shortcut_relu=before_shortcut_relu, after_shortcut_relu=after_shortcut_relu,
                           divisor_rank=divisor_rank)
    x7 = gcd_branch3_block(x7, 10, factors=[19, 23, 29], normalization='instance',
                           before_shortcut_relu=before_shortcut_relu, after_shortcut_relu=after_shortcut_relu,
                           divisor_rank=divisor_rank)
    x7 = gcd_branch3_block(x7, 10, factors=[29, 31, 37], normalization='instance',
                           before_shortcut_relu=before_shortcut_relu, after_shortcut_relu=after_shortcut_relu,
                           divisor_rank=divisor_rank)
    x7 = T.gcd_conv2d(x7, (3, 3), num_filters=155, strides=2, padding='same', activation=None, divisor_rank=1)
    x7 = T.InstanceNormalization()(x7)
    x7 = T.leaky_relu6(x7)
    x7 = T.conv2d(x7, (1, 1), num_filters=100, activation=None, strides=1, padding='same')
    x7 = C.squeeze(C.layers.GlobalAveragePooling()(x7))
    z7 = T.sigmoid(x7)
    return z7


z_challenger1 = challengerNet(input_var, divisor_rank=0, before_shortcut_relu=False, after_shortcut_relu=False)
z_challenger2 = challengerNet(input_var, divisor_rank=1, before_shortcut_relu=False, after_shortcut_relu=False)
z_challenger3 = challengerNet(input_var, divisor_rank=0, before_shortcut_relu=True, after_shortcut_relu=False)
z_challenger4 = challengerNet(input_var, divisor_rank=0, before_shortcut_relu=False, after_shortcut_relu=True)

z_baseline = baselineNet(input_var)

f = codecs.open('model_log_cifar_baseline.txt', 'a', encoding='utf-8-sig')
# model = Function.load('Models/model5_cifar100.model')

dict = {'challenger1': z_challenger1, 'challenger2': z_challenger2, 'challenger3': z_challenger3,
        'challenger4': z_challenger4, 'baseline': z_baseline}

for k, z in dict.items():

    flops_baseline = calculate_flops(z)
    print('flops_{0}:{1}'.format(k, flops_baseline))

    loss = C.cross_entropy_with_softmax(z, label_var)
    err = 1 - C.classification_error(z, label_var)

    learning_rate = 2e-3
    lr = learning_rate
    C.logging.log_number_of_parameters(z)
    progress_printer = C.logging.ProgressPrinter(freq=5, first=5, tag='Training', num_epochs=3)
    learner = C.adam(z.parameters, lr=C.learning_rate_schedule([lr], C.UnitType.minibatch),
                     momentum=C.momentum_schedule(0.75))
    # learner =cntkx.learners.RAdam(z_class.parameters, learning_rate, 0.912, beta2=0.999,l1_regularization_weight=1e-3,
    # l2_regularization_weight=5e-4, epoch_size=300)
    trainer = C.Trainer(z, (loss, err), learner, progress_printer)
    tot_loss = 0
    tot_metrics = 0
    for epoch in range(3):
        mbs = 0

        print('epoch {0}'.format(epoch))
        while mbs < 1000:
            try:
                # 定義數據如何對應變數
                raw_imgs, raw_labels = dataset.next_bach(32)

                # if mbs==100:
                #
                #     for p in z.parameters:
                #
                #         #print('{0}   {1}'.format(p.owner.root_function.name, node.owner.op_name))
                #         print('{0}   {1}'.format(p.uid, p.value.shape))
                #
                #         print('max: {0} min: {1} mean:{2}'.format(p.value.max(), p.value.min(), p.value.mean()))

                trainer.train_minibatch({input_var: raw_imgs, label_var: raw_labels})
                tot_loss += trainer.previous_minibatch_loss_average
                tot_metrics += trainer.previous_minibatch_evaluation_average
                if (mbs + 1) % 10 == 0:
                    f.writelines(['model_{0}_1  epoch {1}  {2}/1000 loss: {3} metrics:{4} \n'.format(k, epoch, mbs + 1,
                                                                                                     tot_loss / 10.,
                                                                                                     tot_metrics / 10.)])
                    tot_loss = 0
                    tot_metrics = 0
                if (mbs + 1) % 250 == 0:
                    lr = learning_rate / (1 + 0.01 * 4 * (epoch + (mbs / 250.)))
                    print('learning rate:{0}'.format(lr))
                    z.save('Models/model_cifar100_{0}_1.model'.format(k))
            except Exception as e:
                PrintException()
                print(e)
            mbs += 1
        trainer.summarize_training_progress()
