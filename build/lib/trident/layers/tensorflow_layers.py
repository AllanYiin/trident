from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import collections
import itertools
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn_ops
_tf_data_format= 'channels_last'

def _preprocess_padding(padding):
  """Convert keras' padding to TensorFlow's padding.

  Arguments:
      padding: string, one of 'same' , 'valid'

  Returns:
      a string, one of 'SAME', 'VALID'.

  Raises:
      ValueError: if invalid `padding'`
  """
  if padding == 'same':
    padding = 'SAME'
  elif padding == 'valid':
    padding = 'VALID'
  else:
    raise ValueError('Invalid padding: ' + str(padding))
  return padding




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



class GCD_Conv2D(tf.keras.layers.Layer):
    def __init__(self,
            kernel_size,
            num_filters =None,
            strides=(1, 1),
            padding='valid',
            activation = None,
            init = 'glorot_uniform',
            use_bias = True,
            init_bias = 'zeros',
            divisor_rank = 0,
            dilation = (1, 1),
            self_norm=True,
            weights_contraint = None ,**kwargs):
        super(GCD_Conv2D, self).__init__(**kwargs)
        padding=_preprocess_padding(padding)
        self.rank =2
        self.num_filters = num_filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format =  'channels_last'
        self.dilation_rate = conv_utils.normalize_tuple(dilation, 2, 'dilation_rate')
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(init)
        self.bias_initializer =tf.keras.initializers.get(init_bias)
        self.kernel_regularizer = tf.keras.regularizers.get(None)
        self.bias_regularizer = tf.keras.regularizers.get(None)
        self.activity_regularizer = tf.keras.regularizers.get(None)
        self.kernel_constraint = tf.keras.constraints.get(weights_contraint)
        self.bias_constraint = tf.keras.constraints.get(None)
        self.input_spec = tf.keras.InputSpec(ndim=self.rank + 2)
        self.divisor_rank=divisor_rank
        self.self_norm=self_norm

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' + str(4) + '; Received input shape:', str(input_shape))

        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1

        # gcd_list = []
        # gcd_list.extend(_gcd(c, self.num_filters))
        # self.gcd = gcd_list[0]
        # # divisor_rank=min(divisor_rank,len(gcd_list))
        # groups = gcd_list[int(self.divisor_rank)]
        # input_num_filters = c
        # num_filters_1 = gcd_list[0]
        # num_filters_2 = self.num_filters
        # if c == self.gcd or self.num_filter == self.gcd:
        #     groups = 1
        # else:
        #     groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]
        # print('input:{0}   output:{1}  gcd:{2}'.format(c, num_filters_2, groups))
        # strides = [1] + self.strides + [1]
        # dilations = [1] + self.dilations + [1]

        input_num_filters =input_shape[0]

        if self.num_filters is None:
            self.num_filters = input_num_filters
        self.num_filters_1 = input_num_filters
        self.num_filters_2 = self.num_filters
        gcd_list = []
        gcd_list.extend(_gcd(input_num_filters, self.num_filters))

        gcd = 1
        if len(gcd_list) == 0:
            self.groups = input_num_filters
            num_filters_1 = input_num_filters
        else:
            gcd = gcd_list[0]
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]
            num_filters_1 = gcd

        self.num_filters_2 = self.num_filters
        factors = _get_divisors(self.num_filters // gcd)
        if input_num_filters == gcd:
            self.groups = input_num_filters
            self.num_filters_1 = self.num_filters

        elif self.num_filters == gcd:
            self.groups = gcd
            self.num_filters_1 = self.num_filters
        elif len(factors) == 0:
            self.groups = gcd
            self.num_filters_1 = gcd
        elif len(factors) == 1:
            self. num_filters_1 = gcd * factors[0]
        else:
            self.num_filters_1 = gcd * factors[1]
        num_filters_per_group = (int(input_num_filters // self.groups),)

        self.kernel = self.add_weight(name='kernel',
            shape=[self.kernel_size[0], self.kernel_size[1], input_num_filters, self.num_filters_1],
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint,
            trainable=True, dtype=self.dtype)
        self.built = True
    def call(self, x, *args, **kwargs):
        input_num_filters=K.int_shape(x)[-1]
        inputs=nn_ops.conv2d(
            x,
            filter=self.kernel ,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilations,
            data_format=_tf_data_format)
        if self.self_norm:
            x = K.reshape(x, (self.groups, self.num_filters_1 // self.groups, x.shape[1], x.shape[2]))
            group_mean = K.mean(x, axis=[1, 2, 3])
            group_variance = K.mean(K.square(x - K.stop_gradient(group_mean)), axis=[1, 2, 3])
            group_std = K.sqrt(group_variance)
            x = (x - group_mean) / group_std
            x = K.reshape(x, (self.num_filters_1, x.shape[-2], x.shape[-1]))
        x=tf.keras.layers.Conv2D(self.num_filters,(1,1),(1,1),'valid',_tf_data_format,use_bias=False)(x)
        return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space +[self.num_filters])


    def get_config(self):
        config = {'rank':self.rank,'filters':self.filters,
                  'kernel_size':self.kernel_size ,'strides':self.strides,
                  'padding':self.padding ,'data_format':self.data_format ,
                  'dilation_rate':self.dilation_rate,'activation':self.activation ,
                  'use_biass':self.use_bias ,'kernel_initializer': self.kernel_initializer ,
                  'bias_initializer':self.bias_initializer ,'kernel_regularizer':self.kernel_regularizer ,
                  'bias_regularizer':self.bias_regularizer ,'activity_regularizer':self.activity_regularizer,
                  'kernel_constraint': self.kernel_constraint,'bias_constraint':self.bias_constraint,
                  'input_spec':self.input_spec,'divisor_rank':self.divisor_rank,'gcd':self.gcd}
        base_config = super(GCD_Conv2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class GCD_Conv2D_1(tf.keras.layers.Layer):
    def __init__(self,
            kernel_size,
            num_filters =None,
            strides=(1, 1),
            padding='valid',
            activation = None,
            init = 'glorot_uniform',
            use_bias = True,
            init_bias = 'zeros',
            divisor_rank = 0,
            dilation = (1, 1),
            self_norm=True,
            weights_contraint = None ,**kwargs):
        super(GCD_Conv2D_1, self).__init__(**kwargs)
        padding=_preprocess_padding(padding)
        self.rank =2
        self.num_filters = num_filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format =  'channels_last'
        self.dilation_rate = conv_utils.normalize_tuple(dilation, 2, 'dilation_rate')
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(init)
        self.bias_initializer =tf.keras.initializers.get(init_bias)
        self.kernel_regularizer = tf.keras.regularizers.get(None)
        self.bias_regularizer = tf.keras.regularizers.get(None)
        self.activity_regularizer = tf.keras.regularizers.get(None)
        self.kernel_constraint = tf.keras.constraints.get(weights_contraint)
        self.bias_constraint = tf.keras.constraints.get(None)
        self.input_spec = tf.keras.InputSpec(ndim=self.rank + 2)
        self.divisor_rank=divisor_rank
        self.self_norm=self_norm

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' + str(4) + '; Received input shape:', str(input_shape))

        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1

        # gcd_list = []
        # gcd_list.extend(_gcd(c, self.num_filters))
        # self.gcd = gcd_list[0]
        # # divisor_rank=min(divisor_rank,len(gcd_list))
        # groups = gcd_list[int(self.divisor_rank)]
        # input_num_filters = c
        # num_filters_1 = gcd_list[0]
        # num_filters_2 = self.num_filters
        # if c == self.gcd or self.num_filter == self.gcd:
        #     groups = 1
        # else:
        #     groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]
        # print('input:{0}   output:{1}  gcd:{2}'.format(c, num_filters_2, groups))
        # strides = [1] + self.strides + [1]
        # dilations = [1] + self.dilations + [1]

        input_num_filters =input_shape[0]

        if self.num_filters is None:
            self.num_filters = input_num_filters
        self.num_filters_1 = input_num_filters
        self.num_filters_2 = self.num_filters
        gcd_list = []
        gcd_list.extend(_gcd(input_num_filters, self.num_filters))

        gcd = 1
        if len(gcd_list) == 0:
            self.groups = input_num_filters
            num_filters_1 = input_num_filters
        else:
            gcd = gcd_list[0]
            self.groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]
            num_filters_1 = gcd

        self.num_filters_2 = self.num_filters
        factors = _get_divisors(self.num_filters // gcd)
        if input_num_filters == gcd:
            self.groups = input_num_filters
            self.num_filters_1 = self.num_filters

        elif self.num_filters == gcd:
            self.groups = gcd
            self.num_filters_1 = self.num_filters
        elif len(factors) == 0:
            self.groups = gcd
            self.num_filters_1 = gcd
        elif len(factors) == 1:
            self. num_filters_1 = gcd * factors[0]
        else:
            self.num_filters_1 = gcd * factors[1]
        num_filters_per_group = (int(input_num_filters // self.groups),)

        self.kernel = self.add_weight(name='kernel',
            shape=[self.kernel_size[0], self.kernel_size[1], input_num_filters, self.num_filters_1],
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint,
            trainable=True, dtype=self.dtype)
        self.built = True
    def call(self, x, *args, **kwargs):
        input_num_filters=K.int_shape(x)[-1]
        inputs=nn_ops.conv2d(
            x,
            filter=self.kernel ,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilations,
            data_format=_tf_data_format)
        if self.self_norm:
            x = K.reshape(x, (self.groups, self.num_filters_1 // self.groups, x.shape[1], x.shape[2]))
            group_mean = K.mean(x, axis=[1, 2, 3])
            group_variance = K.mean(K.square(x - K.stop_gradient(group_mean)), axis=[1, 2, 3])
            group_std = K.sqrt(group_variance)
            x = (x - group_mean) / group_std
            x = K.reshape(x, (self.num_filters_1, x.shape[-2], x.shape[-1]))
        x=tf.keras.layers.Conv2D(self.num_filters,(1,1),(1,1),'valid',_tf_data_format,use_bias=False)(x)
        return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space +[self.num_filters])


    def get_config(self):
        config = {'rank':self.rank,'filters':self.filters,
                  'kernel_size':self.kernel_size ,'strides':self.strides,
                  'padding':self.padding ,'data_format':self.data_format ,
                  'dilation_rate':self.dilation_rate,'activation':self.activation ,
                  'use_biass':self.use_bias ,'kernel_initializer': self.kernel_initializer ,
                  'bias_initializer':self.bias_initializer ,'kernel_regularizer':self.kernel_regularizer ,
                  'bias_regularizer':self.bias_regularizer ,'activity_regularizer':self.activity_regularizer,
                  'kernel_constraint': self.kernel_constraint,'bias_constraint':self.bias_constraint,
                  'input_spec':self.input_spec,'divisor_rank':self.divisor_rank,'gcd':self.gcd}
        base_config = super(GCD_Conv2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

class GCD_Conv2D1(tf.keras.layers.Layer):
    def __init__(self,
            kernel_size,
            num_filters =None,
            strides=(1, 1),
            padding='valid',
            activation = None,
            init = 'glorot_uniform',
            use_bias = True,
            init_bias = 'zeros',
            divisor_rank = 0,
            dilation = (1, 1),
            weights_contraint = None ,**kwargs):
        super(GCD_Conv2D1, self).__init__(**kwargs)
        padding=_preprocess_padding(padding)
        self.rank =2
        self.num_filters = num_filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format =  'channels_last'
        self.dilation_rate = conv_utils.normalize_tuple(dilation, 2, 'dilation_rate')
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(init)
        self.bias_initializer =tf.keras.initializers.get(init_bias)
        self.kernel_regularizer = tf.keras.regularizers.get(None)
        self.bias_regularizer = tf.keras.regularizers.get(None)
        self.activity_regularizer = tf.keras.regularizers.get(None)
        self.kernel_constraint = tf.keras.constraints.get(weights_contraint)
        self.bias_constraint = tf.keras.constraints.get(None)
        self.input_spec = tf.keras.InputSpec(ndim=self.rank + 2)
        self.divisor_rank=divisor_rank

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' + str(4) + '; Received input shape:', str(input_shape))

        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1

        gcd_list = []
        gcd_list.extend(_gcd(c, self.num_filters))
        self.gcd = gcd_list[0]
        # divisor_rank=min(divisor_rank,len(gcd_list))
        self.groups = gcd_list[int(self.divisor_rank)]
        input_num_filters = c
        num_filters_1 = gcd_list[0]
        num_filters_2 = self.num_filters
        if c == self.gcd or self.num_filter == self.gcd:
            groups = 1
        else:
            groups = gcd_list[min(int(self.divisor_rank), len(gcd_list))]
        print('input:{0}   output:{1}  gcd:{2}'.format(c, num_filters_2, groups))
        strides = [1] + self.strides + [1]
        dilations = [1] + self.dilations + [1]
        self.kernel = self.add_weight(name='kernel',
            shape=[self.kernel_size[0], self.kernel_size[1], input_num_filters, num_filters_1],
            initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint,
            trainable=True, dtype=self.dtype)
        self.built = True
    def call(self, inputs, *args, **kwargs):
        def MakeConv2d(inputs,num_filters):
            return tf.keras.layers.Conv2D(num_filters,self.kernel_size,self.strides,'valid',_tf_data_format,use_bias=False,dilation_rate=self.dilation_rate)(inputs)
        input_num_filters=K.int_shape(inputs)[-1]
        if input_num_filters != self.gcd and self.num_filters != self.gcd:
            t1_splits = array_ops.split(inputs, self.groups, axis=3)
            inputs = array_ops.concat([MakeConv2d(t1s,self.gcd//self.groups) for t1s in  t1_splits],axis=1 )
        inputs=tf.keras.layers.Conv2D(self.num_filters,(1,1),(1,1),'valid',_tf_data_format,use_bias=False)(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0]] + new_space +[self.num_filters])


    def get_config(self):
        config = {'rank':self.rank,'filters':self.filters,
                  'kernel_size':self.kernel_size ,'strides':self.strides,
                  'padding':self.padding ,'data_format':self.data_format ,
                  'dilation_rate':self.dilation_rate,'activation':self.activation ,
                  'use_biass':self.use_bias ,'kernel_initializer': self.kernel_initializer ,
                  'bias_initializer':self.bias_initializer ,'kernel_regularizer':self.kernel_regularizer ,
                  'bias_regularizer':self.bias_regularizer ,'activity_regularizer':self.activity_regularizer,
                  'kernel_constraint': self.kernel_constraint,'bias_constraint':self.bias_constraint,
                  'input_spec':self.input_spec,'divisor_rank':self.divisor_rank,'gcd':self.gcd}
        base_config = super(GCD_Conv2D1, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))








