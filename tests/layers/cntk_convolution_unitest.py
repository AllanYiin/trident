import unittest
import os
os.environ['TRIDENT_BACKEND']='cntk'
from trident import get_backend as T
import cntk as C
import numpy as np


def calculate_flops(x):
    flops=0
    for p in x.parameters:
        flops+=p.value.size
    return flops



class TestConvolution(unittest.TestCase):


    def test_strides1(self):
        x = C.input_variable((3,128,256),dtype=np.float32)
        x2 = C.layers.Convolution2D((3, 3), 128,C.relu,strides= (2, 2),pad=True)(x)
        x1=T.Conv2d((3, 3), 128, (2, 2), 'same', activation=T.lecun_tanh)(x)

        print(x1.shape)
        self.assertEquals(x1.shape, (128,64,128))

    def test_strides2(self):
        x=C.input_variable((3,128,256),dtype=np.float32)
        x2 = C.layers.Convolution2D((3, 3), 128, C.relu, strides=(2, 2), pad=False)(x)
        x1=T.Conv2d((3, 3), 128, 2, 'valid', T.leaky_relu)(x)
        print(x1.shape)
        self.assertEquals(x1.shape, (128,62,126))

    def test_strides3(self):
        x = C.input_variable((3, 128, 256), dtype=np.float32)
        x2 = C.layers.Convolution2D((3, 3), 128, C.relu, strides=(1, 2), pad=True)(x)
        x3= T.Conv2d((3, 3), 128, (1, 2), 'same', T.relu6)(x)
        print(x3.shape)
        self.assertEquals(x3.shape, (128, 128, 128))

    def test_depthwise1(self):
        x = C.input_variable((3, 128, 256), dtype=np.float32)
        x2 = C.layers.Convolution2D((3, 3), 128, C.relu, strides=1, pad=True)(x)
        x3= T.depthwise_conv2d(x,(3, 3),1, 'same', T.relu6,depth_multiplier=2)
        print(x3.shape)
        self.assertEquals(x3.shape, (128, 128, 128))

    def test_separatable(self):
        x = C.input_variable((64, 128,128 ), dtype=np.float32)
        x2 = C.layers.Convolution2D((3, 3),128,C.relu, strides=1, pad=True)(x)
        x3_1 = T.depthwise_conv2d(x, (3, 3), 1, 'same', T.relu6, depth_multiplier=2)
        x3= T.sepatable_conv2d(x,(3, 3),256,strides=1,padding= 'same',activation= T.relu6,depth_multiplier=2)
        print(x3.shape)
        self.assertEquals(x3.shape, (256, 128, 128))

    def test_gcd(self):
        x = C.input_variable((168, 128, 128), dtype=np.float32)
        # x2 = C.layers.Convolution2D((3, 3), 112, C.relu, strides=1, pad=True)(x)
        # x3_1 = T.depthwise_conv2d(x, (3, 3), 1, 'same', T.relu6, depth_multiplier=2)
        x3 = T.gcd_conv2d(x, (3, 3), num_filters=112, strides=1, padding='same', activation=T.relu6)
        print(x3.shape)
        self.assertEquals(x3.shape, (112, 128, 128))

    def test_gcd2(self):
        x = C.input_variable((168, 128, 128), dtype=np.float32)
        # x2 = C.layers.Convolution2D((3, 3), 112, C.relu, strides=1, pad=True)(x)
        # x3_1 = T.depthwise_conv2d(x, (3, 3), 1, 'same', T.relu6, depth_multiplier=2)
        x3 = T.gcd_conv2d(x, (3, 3), num_filters=112, strides=1, padding='same', activation=T.relu6,divisor_rank=1)
        print(x3.shape)
        self.assertEquals(x3.shape, (112, 128, 128))
    def test_gcd3(self):
        x = C.input_variable((7068, 128, 128), dtype=np.float32)
        # x2 = C.layers.Convolution2D((3, 3), 112, C.relu, strides=1, pad=True)(x)
        # x3_1 = T.depthwise_conv2d(x, (3, 3), 1, 'same', T.relu6, depth_multiplier=2)
        x3 = T.gcd_conv2d(x, (3, 3), num_filters=13312, strides=1, padding='same', activation=T.relu6,divisor_rank=0)
        print(x3.shape)
        self.assertEquals(x3.shape, (13312, 128, 128))

    def test_gcd4(self):

        x = C.input_variable((168, 128, 128), dtype=np.float32)
        x5 = T.Conv2d_Block((3, 3), num_filters=112, strides=1, padding='same', activation=T.relu6)(x)
        x1 = T.gcd_conv2d(x, (3, 3), num_filters=112, strides=1, padding='same', activation=T.relu6,divisor_rank=0)
        x2 = T.gcd_conv2d1(x, (3, 3), num_filters=112, strides=1, padding='same', activation=T.relu6, divisor_rank=0)
        x3 = C.layers.Convolution2D((3, 3), 112, C.relu, strides=1, pad=True)(x)
        flops_x1=calculate_flops(x1)
        flops_x2 =calculate_flops(x2)
        flops_x3 = calculate_flops(x3)

        f_x=np.random.uniform(-1,1,(168, 128, 128))
        value1=x1(f_x)
        value2 = x2(f_x)
        value3 = x3(f_x)
        mean1=  value1.mean()
        mean2=value2.mean()
        mean3 = value3.mean()
        check=np.array_equal(value1,value2)
        self.assertEquals(check,True)

if __name__ == '__main__':
    unittest.main()
