import os
import sys
import codecs
os.environ['TRIDENT_BACKEND']='cntk'
from trident import backend as T

import cntk as C
from cntk.layers import *
from cntk.ops.functions import *
from cntk.ops import *
from cntk.learners import *
import numpy as np
import linecache

#C.debugging.set_computation_network_trace_level(1000)
#C.debugging.set_checked_mode(True)
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def calculate_flops(x):
    flops=0
    for p in x.parameters:
        flops+=p.value.size
    return flops





data=T.load_cifar('cifar100','train',is_flatten=False)
dataset=T.Dataset('cifar100')
dataset.mapping(data=data[0],labels=data[1],scenario='train')

input_var = C.input_variable((3, 32,32), dtype=np.float32)
label_var = C.input_variable((100), dtype=np.float32)



x1=C.layers.Convolution2D((3, 3), 32, T.leaky_relu6, strides=1, pad=True)(input_var)
x1 = T.gcd_conv2d(x1, (3, 3), num_filters=56, strides=1, padding='same', activation=None, divisor_rank=0)
x1=T.InstanceNormalization()(x1)
x1=T.leaky_relu6(x1)
branch1 = T.gcd_conv2d(x1, (3, 3), num_filters=77, strides=1, padding='same', activation=None, divisor_rank=0)
branch1=T.InstanceNormalization()(branch1)
branch1=T.leaky_relu6(branch1)
branch1 = T.gcd_conv2d(branch1, (3, 3), num_filters=56, strides=1, padding='same', activation=None, divisor_rank=0)
branch1=T.InstanceNormalization()(branch1)
branch1=T.leaky_relu6(branch1)
x1=x1+branch1

x1 = T.gcd_conv2d(x1, (3, 3), num_filters=104, strides=2, padding='same', activation=None, divisor_rank=0)
x1=T.InstanceNormalization()(x1)
x1=T.leaky_relu6(x1)
branch1 = T.gcd_conv2d(x1, (3, 3), num_filters=152, strides=1, padding='same', activation=None, divisor_rank=0)
branch1=T.InstanceNormalization()(branch1)
branch1=T.leaky_relu6(branch1)
branch1 = T.gcd_conv2d(branch1, (3, 3), num_filters=104, strides=1, padding='same', activation=None, divisor_rank=0)
branch1=T.InstanceNormalization()(branch1)
branch1=T.leaky_relu6(branch1)
x1=x1+branch1
x1=dropout(x1,0.2)
x1 = T.gcd_conv2d(x1, (3, 3), num_filters=152, strides=2, padding='same', activation=None, divisor_rank=0)
x1=T.InstanceNormalization()(x1)
x1=T.leaky_relu6(x1)
branch1 = T.gcd_conv2d(x1, (3, 3), num_filters=386, strides=1, padding='same', activation=None, divisor_rank=0)
branch1=T.InstanceNormalization()(branch1)
branch1=T.leaky_relu6(branch1)
branch1 = T.gcd_conv2d(branch1, (3, 3), num_filters=152, strides=1, padding='same', activation=None, divisor_rank=0)
branch1=T.InstanceNormalization()(branch1)
branch1=T.leaky_relu6(branch1)
x1=x1+branch1
x1=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x1)
x1=C.squeeze(C.layers.GlobalAveragePooling()(x1))
z1=C.layers.Dense((10),activation=T.sigmoid)(x1)


def gcd_block(inp,base=16,factors=None,divisor_rank=0):
    if factors is None:
        factors=[3,5,7]
    input_shape=inp.shape[0]
    sc = T.gcd_conv2d(inp, (3, 3), num_filters=int(base*factors[0]), strides=1, padding='same', activation=None, divisor_rank=divisor_rank,name='gcd_block2_{0}_{1}'.format(base,factors[0]))
    sc = T.leaky_relu6(sc)
    sc =T.gcd_conv2d(sc, (3, 3), num_filters=int(base*factors[1]), strides=1, padding='same', activation=None, divisor_rank=divisor_rank,name='gcd_block3_{0}_{1}'.format(base,factors[1]))
    sc = T.leaky_relu6(sc)
    sc = T.gcd_conv2d(sc, (3, 3), num_filters=input_shape, strides=1, padding='same', activation=None, divisor_rank=divisor_rank,name='gcd_block4_{0}'.format(input_shape))
    inp=inp+sc
    inp= T.gcd_conv2d(inp, (3, 3), num_filters=int(base*factors[2]), strides=1, padding='same', activation=None, divisor_rank=0,name='gcd_block5_{0}_{1}'.format(base,factors[2]))
    inp = T.InstanceNormalization()(inp)
    inp = T.leaky_relu6(inp)
    return inp



x2=C.layers.Convolution2D((3, 3), 24, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
x2 =gcd_block(x2,8,[3,5,7])
x2 =gcd_block(x2,16,[3,5,7])
#x2=dropout(x2,0.2)
x2 = T.gcd_conv2d(x2, (3, 3), num_filters=128, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x2=T.InstanceNormalization()(x2)
x2=T.leaky_relu6(x2)
x2 =gcd_block(x2,16,[3,5,7])
x2 =gcd_block(x2,24,[3,5,7])
x2 = T.gcd_conv2d(x2, (3, 3), num_filters=256, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x2=T.InstanceNormalization()(x2)
x2=T.leaky_relu6(x2)
x2 =gcd_block(x2,24,[13,5,7])
x2 =gcd_block(x2,32,[3,5,7])
x2 = T.gcd_conv2d(x2, (3, 3), num_filters=384, strides=1, padding='same', activation=None, divisor_rank=0)#8*13
x2=T.InstanceNormalization()(x2)
x2=T.leaky_relu6(x2)
x2=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x2)
x2=C.squeeze(C.layers.GlobalAveragePooling()(x2))
z2=C.layers.Dense((10),activation=T.sigmoid)(x2)





x3=C.layers.Convolution2D((3, 3), 32, T.leaky_relu6, strides=1, pad=True)(input_var)
x3 = T.gcd_conv2d(x3, (3, 3), num_filters=56, strides=1, padding='same', activation=None, divisor_rank=0) #8*7
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3 = T.gcd_conv2d(x3, (3, 3), num_filters=88, strides=1, padding='same', activation=None, divisor_rank=0)#8*11
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3 = T.gcd_conv2d(x3, (3, 3), num_filters=104, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3 = T.gcd_conv2d(x3, (3, 3), num_filters=136, strides=1, padding='same', activation=None, divisor_rank=0)#8*17
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3=dropout(x3,0.2)
x3 = T.gcd_conv2d(x3, (3, 3), num_filters=152, strides=2, padding='same', activation=None, divisor_rank=0)##8*19
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3 = T.gcd_conv2d(x3, (3, 3), num_filters=184, strides=1, padding='same', activation=None, divisor_rank=0)#8*23
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3 = T.gcd_conv2d(x3, (3, 3), num_filters=232, strides=2, padding='valid', activation=None, divisor_rank=0)#8*29
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x3)
x3=C.squeeze(C.layers.GlobalAveragePooling()(x3))
z3=C.layers.Dense((10),activation=T.sigmoid)(x3)



x4=C.layers.Convolution2D((3, 3), 32, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
x4 =gcd_block(x4,8,[3,5,7])
x4 =gcd_block(x4,12,[3,5,7])
#x4=dropout(x4,0.2)
x4 = T.gcd_conv2d(x4, (3, 3), num_filters=128, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x4=T.InstanceNormalization()(x4)
x4=T.leaky_relu6(x4)
x4 =gcd_block(x4,8,[11,13,17])
x4 =gcd_block(x4,12,[11,13,17])
x4 = T.gcd_conv2d(x4, (3, 3), num_filters=256, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x4=T.InstanceNormalization()(x4)
x4=T.leaky_relu6(x4)
x4 =gcd_block(x4,6,[19,23,29])
x4 =gcd_block(x4,12,[19,23,29])
x4 = T.gcd_conv2d(x4, (3, 3), num_filters=368, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x4=T.InstanceNormalization()(x4)
x4=T.leaky_relu6(x4)
x4=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x4)
x4=C.squeeze(C.layers.GlobalAveragePooling()(x4))
z4=C.layers.Dense((100),activation=T.sigmoid)(x4)


def gcd_double_block(inp, base=16, factors=None):
    if factors is None:
        factors = [3, 5, 7,11]
    inp = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[2]), strides=1, padding='same', activation=None,divisor_rank=0)
    inp = T.InstanceNormalization()(inp)
    inp = T.leaky_relu6(inp)

    sc1 = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[0]), strides=1, padding='same', activation=None,divisor_rank=0)
    sc1 = T.leaky_relu6(sc1)
    sc1 = T.gcd_conv2d(sc1, (3, 3), num_filters=int(base * factors[1]), strides=1, padding='same', activation=None, divisor_rank=0)
    sc1 = T.leaky_relu6(sc1)
    sc1 = T.gcd_conv2d(sc1, (3, 3), num_filters=int(base * factors[2]), strides=1, padding='same', activation=None,divisor_rank=0)

    sc2 = T.gcd_conv2d(inp, (3, 3), num_filters=int(base * factors[1]), strides=1, padding='same', activation=None,divisor_rank=0)
    sc2 = T.leaky_relu6(sc2)
    sc2 = T.gcd_conv2d(sc2, (3, 3), num_filters=int(base * factors[3]), strides=1, padding='same', activation=None,divisor_rank=0)
    sc2 = T.leaky_relu6(sc2)
    sc2 = T.gcd_conv2d(sc2, (3, 3), num_filters=int(base * factors[2]), strides=1, padding='same', activation=None,divisor_rank=0)
    inp = inp + 0.5*sc1+0.5*sc2
    return inp


x5=C.layers.Convolution2D((3, 3), 32, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
x5 =gcd_double_block(x5,8,[3,5,7,11])

x5 = T.gcd_conv2d(x5, (3, 3), num_filters=7*13*2, strides=2, padding='same', activation=None, divisor_rank=0)#7*11*2
x5=T.InstanceNormalization()(x5)
x5=T.leaky_relu6(x5)
x5 =gcd_double_block(x5,8,[11,13,17,19])
x5 = T.gcd_conv2d(x5, (3, 3), num_filters=17*23*2, strides=2, padding='same', activation=None, divisor_rank=0)#71*11*2
x5=T.InstanceNormalization()(x5)
x5=T.leaky_relu6(x5)
x5 =gcd_double_block(x5,8,[19,23,29,31])
x5 = T.gcd_conv2d(x5, (3, 3), num_filters=29*9, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x5=T.InstanceNormalization()(x5)
x5=T.leaky_relu6(x5)
x5=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x5)
x5=C.squeeze(C.layers.GlobalAveragePooling()(x5))
z5=C.layers.Dense((10),activation=T.sigmoid)(x5)




x6=C.layers.Convolution2D((3, 3), 32, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
x6 =gcd_block(x6,8,[3,5,7],divisor_rank=1)
x6 =gcd_block(x6,12,[3,5,7],divisor_rank=1)
#x6=dropout(x6,0.2)
x6 = T.gcd_conv2d(x6, (3, 3), num_filters=128, strides=2, padding='same', activation=None, divisor_rank=1)#8*13
x6=T.InstanceNormalization()(x6)
x6=T.leaky_relu6(x6)
x6 =gcd_block(x6,8,[11,13,17],divisor_rank=1)
x6 =gcd_block(x6,12,[11,13,17],divisor_rank=1)
x6 = T.gcd_conv2d(x6, (3, 3), num_filters=256, strides=2, padding='same', activation=None, divisor_rank=1)#8*13
x6=T.InstanceNormalization()(x6)
x6=T.leaky_relu6(x6)
x6 =gcd_block(x6,8,[19,23,29],divisor_rank=1)
x6 =gcd_block(x6,12,[19,23,29],divisor_rank=1)
x6=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x6)
x6=C.squeeze(C.layers.GlobalAveragePooling()(x6))
z6=C.layers.Dense((10),activation=T.sigmoid)(x6)



x7=C.layers.Convolution2D((3, 3), 24, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
x7 =gcd_block(x7,8,[3,5,7])
x7 =gcd_block(x7,8,[3,5,7])
#x7=dropout(x7,0.2)
x7 = T.gcd_conv2d(x7, (3, 3), num_filters=128, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x7=T.InstanceNormalization()(x7)
x7=T.leaky_relu6(x7)
x7 =gcd_block(x7,8,[11,13,17])
x7 =gcd_block(x7,8,[11,13,17])
x7 = T.gcd_conv2d(x7, (3, 3), num_filters=256, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x7=T.InstanceNormalization()(x7)
x7=T.leaky_relu6(x7)
x7 =gcd_block(x7,8,[19,23,29])
x7 =gcd_block(x7,8,[19,23,29])
x7 = T.gcd_conv2d(x7, (3, 3), num_filters=368, strides=2, padding='same', activation=None, divisor_rank=0)#8*13
x7=T.InstanceNormalization()(x7)
x7=T.leaky_relu6(x7)
x7=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x7)
x7=C.squeeze(C.layers.GlobalAveragePooling()(x7))
z7=C.layers.Dense((10),activation=T.sigmoid)(x7)


x8=C.layers.Convolution2D((3, 3), 24, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
x8 =gcd_block(x8,8,[3,5,7], divisor_rank=1)
x8 =gcd_block(x8,12,[3,5,7], divisor_rank=1)
#x8=dropout(x8,0.2)
x8 = T.gcd_conv2d(x8, (3, 3), num_filters=12*8, strides=2, padding='same', activation=None, divisor_rank=1)#8*13
x8=T.InstanceNormalization()(x8)
x8=T.leaky_relu6(x8)
x8 =gcd_block(x8,8,[9,11,13], divisor_rank=1)
x8 =gcd_block(x8,12,[9,11,13], divisor_rank=1)
x8 = T.gcd_conv2d(x8, (3, 3), num_filters=12*16, strides=2, padding='same', activation=None, divisor_rank=1)#8*13
x8=T.InstanceNormalization()(x8)
x8=T.leaky_relu6(x8)
x8 =gcd_block(x8,8,[17,19,23], divisor_rank=1)
x8 =gcd_block(x8,12,[17,19,23], divisor_rank=1)
x8=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x8)
x8=C.squeeze(C.layers.GlobalAveragePooling()(x8))
z8=C.layers.Dense((10),activation=T.sigmoid)(x8)


flops_x1 = calculate_flops(z1)
print('flops_x1:{0}'.format(flops_x1))
flops_x2 = calculate_flops(z2(input_var))
print('flops_x2:{0}'.format(flops_x2))
flops_x3 = calculate_flops(z3)
print('flops_x3:{0}'.format(flops_x3))
flops_x4 = calculate_flops(z4)
print('flops_x4:{0}'.format(flops_x4))
flops_x5 = calculate_flops(z5)
print('flops_x5:{0}'.format(flops_x5))
flops_x6 = calculate_flops(z6)
print('flops_x6:{0}'.format(flops_x6))
flops_x7 = calculate_flops(z7)
print('flops_x7:{0}'.format(flops_x7))
flops_x8 = calculate_flops(z8)
print('flops_x7:{0}'.format(flops_x8))




f=codecs.open('model_log_cifar100.txt','a',encoding='utf-8-sig')

z=z4
loss=C.cross_entropy_with_softmax(z,label_var)
err=1 - C.classification_error(z, label_var)

learning_rate=1e-3
lr=learning_rate
C.logging.log_number_of_parameters(z)
progress_printer = C.logging.ProgressPrinter(freq=5,first=5, tag='Training', num_epochs=50)
learner = C.adam(z.parameters, lr=C.learning_rate_schedule([lr], C.UnitType.minibatch),
                       momentum=C.momentum_schedule(0.95))
#learner =cntkx.learners.RAdam(z_class.parameters, learning_rate, 0.912, beta2=0.999,l1_regularization_weight=1e-3,l2_regularization_weight=5e-4, epoch_size=300)
trainer = C.Trainer(z, (loss,err), learner, progress_printer)
tot_loss=0
tot_metrics=0
for epoch in range(50):
    mbs = 0
    print('epoch {0}'.format(epoch))
    while mbs < 500:
        try:
            # 定義數據如何對應變數
           raw_imgs, raw_labels= dataset.next_bach(32)
           trainer.train_minibatch({input_var:(np.expand_dims(raw_imgs,1)-127.5)/127.5,label_var:raw_labels})
           tot_loss+=trainer.previous_minibatch_loss_average
           tot_metrics+=trainer.previous_minibatch_evaluation_average
           if (mbs+1)%10==0:
                f.writelines(['model {0}  epoch {1}  {2}/1000 loss: {3} metrics:{4} \n'.format('cifar100',epoch,mbs+1,tot_loss/10.,tot_metrics/10.)])
                tot_loss = 0
                tot_metrics = 0
        except Exception as e:
            PrintException()
            print(e)
        mbs+=1
    trainer.summarize_training_progress()
    z.save('Models/model4__cifar.model')
    learning_rate = learning_rate / (1 + 0.01 * epoch)

