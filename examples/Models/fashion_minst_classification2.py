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



data=T.load_mnist('fashion-mnist','train',is_flatten=False)
dataset=T.Dataset('fashion-mnist')
dataset.mapping(data=data[0],labels=data[1],scenario='train')


input_var = C.input_variable((1, 28, 28), dtype=np.float32)
label_var = C.input_variable((10), dtype=np.float32)

x1=C.layers.Convolution2D((3, 3), 32, T.leaky_relu6, strides=1, pad=True)(input_var)
x1 = T.gcd_conv2d(x1, (3, 3), num_filters=56, strides=1, padding='same', activation=None, divisor_rank=0)
x1=T.InstanceNormalization()(x1)
x1=T.leaky_relu6(x1)
x1 = T.gcd_conv2d(x1, (3, 3), num_filters=104, strides=2, padding='same', activation=None, divisor_rank=0)
x1=T.InstanceNormalization()(x1)
x1=T.leaky_relu6(x1)
#x1=dropout(x1,0.2)
x1 = T.gcd_conv2d(x1, (3, 3), num_filters=152, strides=2, padding='same', activation=None, divisor_rank=0)
x1=T.InstanceNormalization()(x1)
x1=T.leaky_relu6(x1)
x1 = T.gcd_conv2d(x1, (3, 3), num_filters=368, strides=2, padding='same', activation=None, divisor_rank=0)
x1=T.InstanceNormalization()(x1)
x1=T.leaky_relu6(x1)
x1=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x1)
x1=C.squeeze(C.layers.GlobalAveragePooling()(x1))
z1=C.layers.Dense((10),activation=T.sigmoid)(x1)



x6=C.layers.Convolution2D((3, 3), 32, T.leaky_relu6, strides=1, pad=True)(input_var)
x6 = T.gcd_conv2d(x6, (3, 3), num_filters=56, strides=1, padding='same', activation=None, divisor_rank=0)
x6=T.InstanceNormalization()(x6)
x6=T.leaky_relu6(x6)

x6 = T.gcd_conv2d(x6, (3, 3), num_filters=104, strides=2, padding='same', activation=None, divisor_rank=0)
x6=T.InstanceNormalization()(x6)
x6=T.leaky_relu6(x6)

branch1 = T.gcd_conv2d(x6, (3, 3), num_filters=160, strides=1, padding='same', activation=None, divisor_rank=1)
branch1=T.InstanceNormalization()(branch1)
branch1=T.leaky_relu6(branch1)
branch1=C.layers.Convolution2D((1, 1), 104, T.leaky_relu6, strides=1, pad=True)(branch1)
x6=x6+branch1

#x6=dropout(x6,0.2)
x6 = T.gcd_conv2d(x6, (3, 3), num_filters=152, strides=2, padding='same', activation=None, divisor_rank=0)
x6=T.InstanceNormalization()(x6)
x6=T.leaky_relu6(x6)

branch2 = T.gcd_conv2d(x6, (3, 3), num_filters=209, strides=1, padding='same', activation=None, divisor_rank=1)
branch2=T.InstanceNormalization()(branch2)
branch2=T.leaky_relu6(branch2)
branch2=C.layers.Convolution2D((1, 1), 152, T.leaky_relu6, strides=1, pad=True)(branch2)
x6=x6+branch2


x6 = T.gcd_conv2d(x6, (3, 3), num_filters=368, strides=2, padding='same', activation=None, divisor_rank=0)
x6=T.InstanceNormalization()(x6)
x6=T.leaky_relu6(x6)
x6=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x6)
x6=C.squeeze(C.layers.GlobalAveragePooling()(x6))
z6=C.layers.Dense((10),activation=T.sigmoid)(x6)


x2=C.layers.Convolution2D((3, 3), 32, T.leaky_relu6, strides=1, pad=True)(input_var)
x2 = T.gcd_conv2d1(x2, (3, 3), num_filters=56, strides=1, padding='same', activation=None, divisor_rank=0)
x2=T.InstanceNormalization()(x2)
x2=T.leaky_relu6(x2)
x2 = T.gcd_conv2d1(x2, (3, 3), num_filters=104, strides=2, padding='same', activation=None, divisor_rank=0)
x2=T.InstanceNormalization()(x2)
x2=T.leaky_relu6(x2)
#x2=dropout(x2,0.2)
x2 = T.gcd_conv2d1(x2, (3, 3), num_filters=152, strides=2, padding='same', activation=None, divisor_rank=0)
x2=T.InstanceNormalization()(x2)
x2=T.leaky_relu6(x2)
x2 = T.gcd_conv2d1(x2, (3, 3), num_filters=368, strides=2, padding='same', activation=None, divisor_rank=0)
x2=T.InstanceNormalization()(x2)
x2=T.leaky_relu6(x2)
x2=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x2)
x2=C.squeeze(C.layers.GlobalAveragePooling()(x2))
z2=C.layers.Dense((10),activation=T.sigmoid)(x2)

x3=C.layers.Convolution2D((3, 3), 32, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
x3 = C.layers.Convolution2D((3, 3), 56,activation=None, strides=1, pad=True)(x3)
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3 = C.layers.Convolution2D((3, 3), 104,activation=None, strides=2, pad=True)(x3)
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
#x3=dropout(x3,0.2)
x3 = C.layers.Convolution2D((3, 3), 152, activation=None, strides=2, pad=True)(x3)
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3 = C.layers.Convolution2D((3, 3), 368, activation=None, strides=2, pad=True)(x3)
x3=T.InstanceNormalization()(x3)
x3=T.leaky_relu6(x3)
x3=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x3)
x3=C.squeeze(C.layers.GlobalAveragePooling()(x3))
z3=C.layers.Dense((10),activation=T.sigmoid)(x3)



x4=C.layers.Convolution2D((3, 3), 32, activation=T.leaky_relu6, strides=1, pad=True)(input_var)
x4 = C.layers.Convolution2D((3, 3), 64,activation=None, strides=1, pad=True)(x4)
x4=T.InstanceNormalization()(x4)
x4=T.leaky_relu6(x4)
x4 = C.layers.Convolution2D((3, 3), 128,activation=None, strides=2, pad=True)(x4)
x4=T.InstanceNormalization()(x4)
x4=T.leaky_relu6(x4)
#x4=dropout(x4,0.2)
x4 = C.layers.Convolution2D((3, 3), 192, activation=None, strides=2, pad=True)(x4)
x4=T.InstanceNormalization()(x4)
x4=T.leaky_relu6(x4)
x4 = C.layers.Convolution2D((3, 3), 256, activation=None, strides=2, pad=True)(x4)
x4=T.InstanceNormalization()(x4)
x4=T.leaky_relu6(x4)
x4=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x4)
x4=C.squeeze(C.layers.GlobalAveragePooling()(x4))
z4=C.layers.Dense((10),activation=T.sigmoid)(x4)


x5=C.layers.Convolution2D((3, 3), 32, T.leaky_relu6, strides=1, pad=True)(input_var)
x5 = T.gcd_conv2d(x5, (3, 3), num_filters=56, strides=1, padding='same', activation=None, divisor_rank=1)
x5=T.InstanceNormalization()(x5)
x5=T.leaky_relu6(x5)
x5 = T.gcd_conv2d(x5, (3, 3), num_filters=104, strides=2, padding='same', activation=None, divisor_rank=1)
x5=T.InstanceNormalization()(x5)
x5=T.leaky_relu6(x5)
#x5=dropout(x5,0.2)
x5 = T.gcd_conv2d(x5, (3, 3), num_filters=152, strides=2, padding='same', activation=None, divisor_rank=1)
x5=T.InstanceNormalization()(x5)
x5=T.leaky_relu6(x5)
x5 = T.gcd_conv2d(x5, (3, 3), num_filters=368, strides=2, padding='same', activation=None, divisor_rank=1)
x5=T.InstanceNormalization()(x5)
x5=T.leaky_relu6(x5)
x5=C.layers.Convolution2D((1, 1), 10, T.leaky_relu6, strides=1, pad=True)(x5)
x5=C.squeeze(C.layers.GlobalAveragePooling()(x5))
z5=C.layers.Dense((10),activation=T.sigmoid)(x5)

flops_x1 = calculate_flops(x1)
print('flops_x1:{0}'.format(flops_x1))
flops_x2 = calculate_flops(x2)
print('flops_x2:{0}'.format(flops_x2))
flops_x3 = calculate_flops(x3)
print('flops_x3:{0}'.format(flops_x3))
flops_x4 = calculate_flops(x4)
print('flops_x4:{0}'.format(flops_x4))
flops_x5 = calculate_flops(x5)
print('flops_x5:{0}'.format(flops_x5))

flops_x6 = calculate_flops(x6)
print('flops_x6:{0}'.format(flops_x6))


f=codecs.open('model_log_round1.txt','a',encoding='utf-8-sig')
models=[z1,z2,z3,z4,z5,z6]

for i in range(6):
    z=models[i]
    loss=C.cross_entropy_with_softmax(z,label_var)
    err=1 - C.classification_error(z, label_var)

    learning_rate=1e-3
    C.logging.log_number_of_parameters(z)
    progress_printer = C.logging.ProgressPrinter(freq=5,first=5, tag='Training', num_epochs=10)
    learner = C.adam(z.parameters, lr=C.learning_rate_schedule([learning_rate], C.UnitType.minibatch),
                           momentum=C.momentum_schedule(0.95))
    #learner =cntkx.learners.RAdam(z_class.parameters, learning_rate, 0.912, beta2=0.999,l1_regularization_weight=1e-3,l2_regularization_weight=5e-4, epoch_size=300)
    trainer = C.Trainer(z, (loss,err), learner, progress_printer)
    tot_loss=0
    tot_metrics=0
    for epoch in range(5):
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
                    f.writelines(['model {0}  epoch {1}  {2}/1000 loss: {3} metrics:{4} \n'.format(i+1,epoch,mbs+1,tot_loss/10.,tot_metrics/10.)])
                    tot_loss = 0
                    tot_metrics = 0
            except Exception as e:
                PrintException()
                print(e)
            mbs+=1
        trainer.summarize_training_progress()
        z.save('Models/model_round2_{0}.model'.format(i+1))

