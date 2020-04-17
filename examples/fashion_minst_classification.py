import os
import math
import sys
import codecs
os.environ['TRIDENT_BACKEND']='cntk'
import  trident as T
from trident.data import data_loaders
import cntk as C
from cntk.layers import *
from cntk.ops.functions import *
from cntk.ops import *
from cntk.learners import *
import numpy as np
import linecache

C.debugging.set_computation_network_trace_level(1)
C.debugging.set_checked_mode(False)

def _gcd(x, y):
    gcds=[]
    gcd = 1
    if x % y == 0:
        gcds.append(int(y))
    for k in range(int(y //2), 0, -1):
        if x % k == 0 and y % k == 0:
            gcd = k
            gcds.append(int(k))
    return gcds


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


def _gcd(x, y):
    gcds=[]
    gcd = 1
    if x % y == 0:
        gcds.append(int(y))
    for k in range(int(y //2), 0, -1):
        if x % k == 0 and y % k == 0:
            gcd = k
            gcds.append(int(k))
    return gcds

def _get_divisors(n):
    return  [d for d in range(2, int(math.sqrt(n))) if n % d == 0]

def _isprime(n):
    divisors = [d for d in range(2, int(math.sqrt(n))) if n % d == 0]
    return all( n % od != 0 for od in divisors if od != n )



def gcd_mb_block(x, size_out, expand=1, kernel=1, strides=1, se_ratio=0.25, dc_ratio=0.2, **kw):
    """ MobileNet Bottleneck Block. """
    input_shape = x.shape[0]
    expand_shape = int(expand)

    se_shape = int(input_shape * se_ratio)
    activation_fn = 'leaky_relu6' if size_out == 24 else None
    # T.H_Swish if  input_shape == size_out and strides==1 else T.Relu6
    #1 = T.GcdConv2d(kernel_size=(1, 1), num_filters=expand_shape, strides=1, padding='same', activation='leaky_relu6')(x)
    x1 = T.GcdConv2d(kernel_size=(3, 3), num_filters=expand_shape, strides=strides, padding='same', activation=activation_fn)(x)

    se = GlobalAveragePooling()(x1)
    se = T.Conv2d(kernel_size=(1, 1), num_filters=input_shape // 4, strides=1, activation='leaky_relu6',padding='same')(se)
    se = T.Conv2d(kernel_size=(1, 1), num_filters=expand_shape, strides=1, activation=T.Sigmoid, padding='same')(se)
    x1 = x1 * se
    x1 = T.GcdConv2d(kernel_size=(1, 1), num_filters=input_shape, strides=1, padding='same', activation='leaky_relu6')( x1)
    if strides == 1:
        x1 = x1 + x
    else:
        x1 = plus(x1, C.layers.MaxPooling((2, 2), (2, 2), True)(x))

    x1 = T.GcdConv2d(kernel_size=(3, 3), num_filters=size_out, strides=1, padding='same', activation=activation_fn)(x1)
    return x1

# gcd_spec_b0 = ((16, 16, 3, 1, 1, 0.25, 0.2),(24, 64, 3, 2, 1, 0.25, 0.2),(24, 72, 3, 1, 1, 0.25, 0.2),
#                (40, 72, 3, 2, 1, 0.25, 0.2), (40, 72, 3, 1, 1, 0.25, 0.2), (40, 120, 3, 1, 1, 0.25, 0.2), (80, 184, 3, 2, 1, 0.25, 0.2),)


gcd_spec_b0 = ((24, 64, 3, 2, 1, 0.25, 0.0), (40, 72, 3, 2, 1, 0.25, 0.1/7.), (80, 184, 3, 2, 1, 0.25, 0.2/7.),)

def get_gdc_efficientnet_b0(x):
    global  gcd_spec_b0
    x = T.Conv2d(kernel_size=(3, 3),num_filters=16, strides=1, activation='h_swish')(x)
    for out_size, expand, kernel, strides, repeat, se_ratio, dc_ratio in gcd_spec_b0:
        for i in range(repeat):
            x = gcd_mb_block(x, out_size, expand, kernel, strides, se_ratio, dc_ratio)
    x = T.GcdConv2d((3, 3), num_filters=112, activation='leaky_relu6')(x)
    x = T.GcdConv2d((3, 3), num_filters=160, activation=None)(x)
    x = T.GcdConv2d((3, 3), num_filters=256, activation='leaky_relu6')(x)
    x= T.GcdConv2d((1, 1), num_filters=150, strides=1, padding='same', activation=None)(x)
    x = T.Classifier(num_classes=10,is_multiselect=False,classifier_type='global_avgpool')(x)
    return x





def gcd_mb_block1(x, size_out, expand=1, kernel=1, strides=1, se_ratio=0.25, dc_ratio=0.2, **kw):
    """ MobileNet Bottleneck Block. """
    input_shape = x.shape[0]
    expand_shape =int( expand)

    activation_fn='leaky_relu6' if size_out==24 else None
    padding='valid' if strides==2 else 'same'

    if strides==2:
        x = T.GcdConv2d(kernel_size=(3,3), num_filters=expand_shape, strides=strides,padding=padding,activation=activation_fn)(x)
        x1 = T.GcdConv2d(kernel_size=(1, 1),num_filters=expand_shape*input_shape//64,strides=1,padding='same',activation='leaky_relu6')(x)
        x1 = T.GcdConv2d(kernel_size=(3, 3), num_filters=expand_shape, strides=1, padding='same', activation='leaky_relu6')(x1)
        if dc_ratio > 0:
            x1 = C.layers.Dropout(dc_ratio)(x1)
        x = x1 + x
    elif strides == 1:
        x1 = T.GcdConv2d(kernel_size=(3, 3), num_filters=expand_shape, strides=strides, padding=padding, activation=activation_fn)(x)
        x1 = T.GcdConv2d(kernel_size=(1, 1), num_filters=input_shape, strides=1, padding='same',activation='leaky_relu6')(x1)
        if dc_ratio > 0:
            x1 = C.layers.Dropout(dc_ratio)(x1)
        x = x1 + x

    x = T.GcdConv2d(kernel_size=(3, 3), num_filters=size_out, strides=1, padding='same', activation=activation_fn)(x)

    # if input_shape == size_out and strides==1:
    #     x1 =C.plus(x, x1)
        # x2 = T.GcdConv2d(kernel_size=(1, 1), num_filters=expand_shape, strides=1, padding='same', activation=T.Relu6)(x1)
        # x2 = T.GcdConv2d(kernel_size=(3, 3), num_filters=expand_shape, strides=1, padding='same',activation=None)(x2)
        # # se = GlobalAveragePooling()(x2)
        # # se = T.GcdConv2d(kernel_size=(1, 1), num_filters=se_shape, strides=1, activation=T.Relu6, padding='same')(se)
        # # se = T.GcdConv2d(kernel_size=(1, 1), num_filters=x2.shape[0], activation=T.Sigmoid, padding='same')(se)
        # # x2=x2+ x2 * se
        # x2 = T.GcdConv2d(kernel_size=(1, 1), num_filters=size_out, strides=1,activation=T.Relu6,padding='same')(x2)
        #
        # x1 =C.plus(x1, x2)

    return x
 #(80, 184, 3, 2, 1, 0.25, 0.2)
gcd_spec_b1 =((24, 56, 3, 2, 1, 0.25, 0.1/7.), (40, 88, 3, 2, 1, 0.25, 0.2/7.),(56, 104, 3, 1, 1, 0.25, 0.3/7.), (88, 136, 3, 2, 1, 0.25, 0.3/7.), (104, 152, 3, 1, 1, 0.25, 0.3/7.))

def get_gdc_efficientnet_b1(x):
    global  gcd_spec_b1
    x = T.Conv2d(kernel_size=(3, 3),num_filters=16, strides=1, activation='h_swish')(x)
    for out_size, expand, kernel, strides, repeat, se_ratio, dc_ratio in gcd_spec_b1:
        for i in range(repeat):
            x = gcd_mb_block1(x, out_size, expand, kernel, strides , se_ratio, dc_ratio)
    x = T.GcdConv2d((3, 3), num_filters=112, activation='leaky_relu6')(x)
    x = T.GcdConv2d((3, 3), num_filters=232, activation=None)(x)
    x = T.GcdConv2d((3, 3), num_filters=145, activation='leaky_relu6')(x)
    x = T.GcdConv2d((1, 1), num_filters=10, strides=1, padding='same', activation=None)(x)
    x = T.Classifier(num_classes=10,is_multiselect=False,classifier_type='global_avgpool')(x)
    return x


data=T.lload_mnist('fashin-mnist','train',is_flatten=False)


input_var = C.input_variable((1, 28, 28), dtype=np.float32)
label_var = C.input_variable((10), dtype=np.float32)





def baselineNet(input_var):
    global classifiertype
    return T.Sequential2(input_var,[
            T.Conv2d((3, 3), num_filters=16, strides=1, activation='h_swish', padding='same'),
            T.ShortCut({
                'left': [T.Conv2d((1, 1), num_filters=16, strides=1, activation=None, padding='same'),
                        T.Conv2d((3, 3), num_filters=16, strides=1, activation=None, padding='same'),
                         T.Conv2d((1, 1), num_filters=16, strides=1, activation=None, padding='same')]}),

                T.Conv2d_Block((1, 1), num_filters=64, strides=(1, 1), auto_pad=True, activation='leaky_relu6',normalization='instance'),
                T.Conv2d_Block((3, 3), num_filters=64, strides=(2, 2), auto_pad=True, activation=None,normalization=None),
                T.Conv2d_Block((1, 1), num_filters=24, strides=(1, 1), auto_pad=True, activation='leaky_relu6', normalization='instance'),

                 T.ShortCut({
                    'left': [T.Conv2d((1, 1), num_filters=72, strides=1, activation=None, padding='same'),
                            T.Conv2d((3, 3), num_filters=72, strides=1, activation=None, padding='same'),
                             T.Conv2d((1, 1), num_filters=24, strides=1, activation=None, padding='same')]}),

                T.Conv2d_Block((1, 1), num_filters=72, strides=(1, 1), auto_pad=True, activation='leaky_relu6',normalization='instance'),
                T.Conv2d_Block((3, 3), num_filters=72, strides=(2, 2), auto_pad=True, activation=None,normalization=None),
                T.Conv2d_Block((1, 1), num_filters=40, strides=(1, 1), auto_pad=True, activation='leaky_relu6', normalization='instance'),

                T.ShortCut({'left': [
                            T.Conv2d((1, 1), num_filters=72, strides=1, activation=None, padding='same'),
                             T.Conv2d((3, 3), num_filters=72, strides=1, activation=None, padding='same'),
                             T.Conv2d((1, 1), num_filters=40, strides=1, activation=None, padding='same')]}),
                T.ShortCut({'left': [
                                    T.Conv2d((1, 1), num_filters=120, strides=1, activation=None, padding='same'),
                                     T.Conv2d((3, 3), num_filters=120, strides=1, activation=None, padding='same'),
                                     T.Conv2d((1, 1), num_filters=40, strides=1, activation=None, padding='same')]}),

                T.Conv2d_Block((1, 1),num_filters= 184,strides=(1,1), auto_pad=True, activation='leaky_relu6', normalization='instance'),
                T.Conv2d_Block((3, 3), num_filters=184, strides=(2,2), auto_pad=True, activation=None,normalization=None),
                T.Conv2d_Block((1, 1), num_filters=80,strides= (1,1), auto_pad=True, activation='leaky_relu6', normalization='instance'),

            T.Conv2d_Block((3, 3), num_filters=112,strides= 1, auto_pad=True, activation='leaky_relu6', normalization=None),
            T.Conv2d_Block((3, 3), num_filters=160, strides=1, auto_pad=True, activation='leaky_relu6', normalization=None),
            T.Conv2d_Block((3, 3), num_filters=256, strides=1, auto_pad=True, activation='leaky_relu6', normalization=None),
            T.Conv2d((1, 1), num_filters=150, strides=1, padding='same', activation=None),#這個活化函數必須拿掉，讓梯度好順利傳遞
        T.Classifier(num_classes=10,is_multiselect=False,classifier_type='global_avgpool')
        ],name='')


def gcd_challengerNet(input_var,self_norm=True):
    global classifiertype
    norm= 'None'
    if self_norm:
        norm=None

    return T.Sequential2(input_var,[
            T.Conv2d((3, 3), num_filters=16, strides=1, activation='h_swish', padding='same'),
            # T.ShortCut({
            #     'left': [T.GcdConv2d((1, 1), num_filters=16, strides=1, activation='h_swish', padding='same',self_norm=self_norm),
            #             T.GcdConv2d((3, 3), num_filters=16, strides=1, activation=None, padding='same',self_norm=self_norm),
            #              T.GcdConv2d((1, 1), num_filters=16, strides=1, activation='h_swish', padding='same',self_norm=self_norm)]}),

                T.GcdConv2d_Block((1, 1), num_filters=64, strides=(1, 1), auto_pad=True, activation='leaky_relu6',normalization=norm,self_norm=self_norm),
                T.GcdConv2d_Block((3, 3), num_filters=64, strides=(2, 2), auto_pad=True, activation='leaky_relu6', normalization=norm,self_norm=self_norm),
                T.GcdConv2d_Block((1, 1), num_filters=24, strides=(1, 1), auto_pad=True, activation='leaky_relu6', normalization=norm,self_norm=self_norm),

                 # T.ShortCut({
                 #    'left': [T.GcdConv2d((1, 1), num_filters=72, strides=1, activation='h_swish', padding='same',self_norm=self_norm),
                 #            T.GcdConv2d((3, 3), num_filters=72, strides=1, activation=None, padding='same',self_norm=self_norm),
                 #             T.GcdConv2d((1, 1), num_filters=24, strides=1, activation='h_swish', padding='same',self_norm=self_norm)]}),

                T.GcdConv2d_Block((1, 1), num_filters=72, strides=(1, 1), auto_pad=True, activation='leaky_relu6',normalization=norm,self_norm=self_norm),
                T.GcdConv2d_Block((3, 3), num_filters=72, strides=(2, 2), auto_pad=True, activation=None,normalization=norm,self_norm=self_norm),
                T.GcdConv2d_Block((1, 1), num_filters=40, strides=(1, 1), auto_pad=True, activation='leaky_relu6', normalization=norm,self_norm=self_norm),

                # T.ShortCut({'left': [
                #             T.GcdConv2d((1, 1), num_filters=72, strides=1, activation='h_swish', padding='same',self_norm=self_norm),
                #              T.GcdConv2d((3, 3), num_filters=72, strides=1, activation=None, padding='same',self_norm=self_norm),
                #              T.GcdConv2d((1, 1), num_filters=40, strides=1, activation='h_swish', padding='same',self_norm=self_norm)]}),
                # T.ShortCut({'left': [
                #                     T.GcdConv2d((1, 1), num_filters=120, strides=1, activation='h_swish', padding='same',self_norm=self_norm),
                #                      T.GcdConv2d((3, 3), num_filters=120, strides=1, activation=None, padding='same',self_norm=self_norm),
                #                      T.GcdConv2d((1, 1), num_filters=40, strides=1, activation='h_swish', padding='same',self_norm=self_norm)]}),

                T.GcdConv2d_Block((1, 1),num_filters= 184,strides=(1,1), auto_pad=True, activation='leaky_relu6', normalization=norm,self_norm=self_norm),
                T.GcdConv2d_Block((3, 3), num_filters=184, strides=(2,2), auto_pad=True, activation=None, normalization=norm,self_norm=self_norm),
                T.GcdConv2d_Block((1, 1), num_filters=80,strides= (1,1), auto_pad=True, activation='leaky_relu6', normalization=norm),

            T.GcdConv2d_Block((3, 3), num_filters=112,strides= 1, auto_pad=True, activation='leaky_relu6', normalization=norm,self_norm=self_norm),
            T.GcdConv2d_Block((3, 3), num_filters=160, strides=1, auto_pad=True, activation=None, normalization=norm,self_norm=self_norm),
            T.GcdConv2d_Block((3, 3), num_filters=256, strides=1, auto_pad=True, activation='leaky_relu6', normalization=norm,self_norm=self_norm),
            T.GcdConv2d((1, 1), num_filters=150, strides=1, padding='same', activation=None),#這個活化函數必須拿掉，讓梯度好順利傳遞
        T.Classifier(num_classes=10,is_multiselect=False,classifier_type='global_avgpool')
        ],name='')


def depthwise_challengerNet(input_var):
    global classifiertype
    return T.Sequential2(input_var, [T.Conv2d((3, 3), num_filters=16, strides=1, activation='h_swish', padding='same'),
        T.ShortCut({
            'left': [T.DepthwiseConv2d((1, 1), depth_multiplier=1, strides=1, activation='leaky_relu6', padding='same'),
                     T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, activation='leaky_relu6',
                                       padding='same')]}),

        T.DepthwiseConv2d((1, 1), depth_multiplier=3, strides=1, padding='same', activation='leaky_relu6'),
        T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, padding='same', activation='leaky_relu6'),
        T.Conv2d((3, 3), num_filters=24, strides=2, padding='same', activation='leaky_relu6'),

        T.ShortCut({'left': [T.DepthwiseConv2d((1, 1), depth_multiplier=3, strides=1, activation=None, padding='same'),
                             T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, activation=None, padding='same'),
                             T.Conv2d((1, 1), num_filters=24, strides=1, activation=None, padding='same')]}),

        T.DepthwiseConv2d((1, 1), depth_multiplier=3, strides=1, padding='same', activation='leaky_relu6'),
        T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, padding='same', activation='leaky_relu6'),
        T.Conv2d((3, 3), num_filters=40, strides=2, padding='same', activation='leaky_relu6'),

        T.ShortCut({'left': [T.DepthwiseConv2d((1, 1), depth_multiplier=3, strides=1, activation=None, padding='same'),
            T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, activation=None, padding='same'),
            T.Conv2d((1, 1), num_filters=40, strides=1, activation=None, padding='same')]}),
            T.ShortCut({'left': [
                            T.DepthwiseConv2d((1, 1),depth_multiplier=3, strides=1, activation=None, padding='same'),
                             T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, activation=None, padding='same'),
                             T.Conv2d((1, 1), num_filters=40, strides=1, activation=None, padding='same')]}),

        T.DepthwiseConv2d((1, 1), depth_multiplier=3, strides=1, padding='same', activation='h_swish'),
        T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, padding='same', activation='h_swish'),
        T.Conv2d((3, 3), num_filters=80, strides=1, padding='same', activation='h_swish'),

        T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, padding='same', activation='h_swish'),
        T.DepthwiseConv2d((3, 3), depth_multiplier=1, strides=1, padding='same', activation='h_swish'),
        T.Conv2d((3, 3), num_filters=256, strides=1, padding='same', activation='h_swish'),
        T.Conv2d((1, 1), num_filters=150, strides=1, padding='same', activation=None),  # 這個活化函數必須拿掉，讓梯度好順利傳遞
        T.Classifier(num_classes=10, is_multiselect=False, classifier_type='global_avgpool')], name='')

print('')
print('')
print('--------------------z1-------------------')
z1=gcd_challengerNet(input_var)
flops_x1 = calculate_flops(z1)
print('flops_x1:{0}'.format(flops_x1))

print('')
print('')
print('--------------------z2-------------------')
z2=get_gdc_efficientnet_b0(input_var)
flops_x2 = calculate_flops(z2)
print('flops_x2:{0}'.format(flops_x2))
for p in z2.parameters:
    print(p.value.mean())



print('')
print('')
print('--------------------z3-------------------')
z3=get_gdc_efficientnet_b1(input_var)
flops_x3 = calculate_flops(z3)
print('flops_x3:{0}'.format(flops_x3))
for p in z3.parameters:
    print(p.value.mean())
#z3=depthwise_challengerNet(input_var)
#z3=get_gdc_efficientnet_b0(input_var)



# flops_x4 = calculate_flops(x4)
# print('flops_x4:{0}'.format(flops_x4))
# flops_x5 = calculate_flops(x5)
# print('flops_x5:{0}'.format(flops_x5))

#flops_x6 = calculate_flops(x6)
#print('flops_x6:{0}'.format(flops_x6))


f=codecs.open('model_log_round1.txt','a',encoding='utf-8-sig')






for cls in ['global_avgpool']:
    classifiertype=cls
    for learning_rate in [1e-3]:

        k2 = ' gcd_challenger'
        #z2 = gcd_challengerNet(input_var)

        k1='baseline'
        #z1=get_efficientnet_b0(input_var)

        k3 = 'depthwise_challenger'
        #z3 =gcd_challengerNet(input_var)


        prefix1='fmnist_model_{0}_{1}_{2}'.format(k1,cls,5e-2).replace('.','_')
        prefix2 = 'fmnist_model_{0}_{1}_{2}'.format(k2, cls, 5e-2).replace('.', '_')
        prefix3 = 'fmnist_model_{0}_{1}_{2}'.format(k3, cls, 5e-2).replace('.', '_')


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


        progress_printer1 = C.logging.ProgressPrinter(freq=50, first=1, tag='Training', num_epochs=10)
        progress_printer2 = C.logging.ProgressPrinter(freq=50, first=1, tag='Training', num_epochs=10)
        progress_printer3 = C.logging.ProgressPrinter(freq=50, first=1, tag='Training', num_epochs=10)
        learner1 = C.adam(z1.parameters, lr=C.learning_rate_schedule([lr], C.UnitType.minibatch),
                        momentum=C.momentum_schedule(0.75))


        trainer1 = C.Trainer(z1, (loss1, err1), learner1, progress_printer1)

        learner2 = C.adam(z2.parameters, lr=C.learning_rate_schedule([lr], C.UnitType.minibatch),
                          momentum=C.momentum_schedule(0.75))


        trainer2 = C.Trainer(z2, (loss2, err2), learner2, progress_printer2)
        learner3 = C.adam(z3.parameters, lr=C.learning_rate_schedule([lr], C.UnitType.minibatch),
                         momentum=C.momentum_schedule(0.75),l2_regularization_weight=1e-3 )
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

                    if  (mbs+1)%250==0:
                        lr = learning_rate / (1 + 0.05 * 4 * (epoch + (mbs / 250.)))
                        print('learning rate:{0}'.format(lr))
                        # print(prefix1)
                        # for p in z1.parameters:
                        #     #print('{0}   {1}'.format(p.owner.root_function.name, node.owner.op_name))
                        #     print('{0}   {1}'.format(p.uid, p.value.shape))
                        #     print('max: {0:.4f} min: {1:.4f} mean:{2:.4f}'.format(p.value.max(), p.value.min(), p.value.mean()))
                        # print('')
                        #
                        # print(prefix2)
                        # for p in z2.parameters:
                        #     # print('{0}   {1}'.format(p.owner.root_function.name, node.owner.op_name))
                        #     print('{0}   {1}'.format(p.uid, p.value.shape))
                        #     print('max: {0:.4f} min: {1:.4f} mean:{2:.4f}'.format(p.value.max(), p.value.min(), p.value.mean()))
                        #
                        # print(prefix3)
                        # for p in z3.parameters:
                        #     # print('{0}   {1}'.format(p.owner.root_function.name, node.owner.op_name))
                        #     print('{0}   {1}'.format(p.uid, p.value.shape))
                        #     print('max: {0:.4f} min: {1:.4f} mean:{2:.4f}'.format(p.value.max(), p.value.min(), p.value.mean()))
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
                    if(mbs + 1) % 500 == 0 or  mbs==1  or (mbs + 1) == 50  or (mbs + 1) == 100 or  (mbs + 1) == 200 or  (mbs + 1) == 300 or  (mbs + 1) == 400:

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
                            print('challenger1 zero ratio {0:.3%} max: {1:.4f}  min: {2:.4f}'.format(tot_zero / tot_p, max_p, min_p))

                            tot_p = 0
                            tot_zero = 0
                            max_p = 0
                            min_p = 0
                            for p in z3.parameters:
                                tot_p += p.value.size
                                tot_zero += np.equal(p.value, 0).astype(np.float32).sum()
                                max_p = max(max_p, p.value.max())
                                min_p = min(min_p, p.value.min())
                            print('challenger2 zero ratio {0:.3%} max: {1:.4f}  min: {2:.4f}'.format(tot_zero / tot_p, max_p, min_p))

                    if (mbs + 1) % 50 == 0:
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
