import tensorflow as tf

from ..backend.common import get_session, addindent, enforce_singleton, unpack_singleton, get_time_suffix, get_class, \
    format_time, get_terminal_size, snake2camel, camel2snake
from ..backend.tensorflow_backend import Layer, Sequential,to_tensor
from ..backend.tensorflow_ops import *
__all__ = ['BatchNorm','BatchNorm2d','BatchNorm3d','get_normalization']

_session = get_session()





_epsilon = _session.epsilon





class BatchNorm(Layer):
    def __init__(self, momentum=0.1, affine=True, track_running_stats=True, axis=-1,renorm=False,eps=1e-5,name=None, **kwargs):
        """
        http://pytorch.org/docs/stable/nn.html#batchnorm1d

        Args:
            dim: 1d, 2d, or 3d BatchNorm
         eps: nn.BatchNorm parameter
            momentum: nn.BatchNorm parameter
            affine: nn.BatchNorm parameter
            track_running_stats: nn.BatchNorm parameter
        """
        super().__init__(name=name)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)
        self.eps = _epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.weight=None
        self.bias=None
        self.renorm=renorm
        self.running_mean = None
        self.running_var =None
        self.num_batches_tracked =None

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.assign(tf.zeros(shape=[self.input_filters]))
            self.running_var.assign(tf.ones(shape=[self.input_filters]))
            self.num_batches_tracked.assign(0)
        if self.affine :
            self.weight.assign(tf.ones(shape=[self.input_filters]))
            self.bias.assign(tf.zeros(shape=[self.input_filters]))

    def assign_moving_average(self, variable, value, momentum, inputs_size):
        with tf.name_scope('AssignMovingAvg') as scope:
            decay = to_tensor(1.0 - momentum)
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = tf.where(inputs_size > 0, update_delta, tf.zeros_like(update_delta))
            return variable.assign_sub(update_delta, name=scope)


    def build(self, input_shape):
        if self._built == False:
            self.input_filters= input_shape.as_list()[-1]

            ndims = len(input_shape.dims)

            # Convert axis to list and resolve negatives
            if isinstance(self.axis, int):
                self.axis = [self.axis]


            if self.affine:
                self.weight = tf.Variable(tf.ones(shape=[self.input_filters]), name='weight') #gamma//scale
                self.bias = tf.Variable(tf.zeros(shape=[self.input_filters]), name='bias') #beta/ offset

            if self.track_running_stats:
                self.running_mean = tf.Variable(tf.zeros(shape=[self.input_filters]), name='running_mean')
                self.running_var = tf.Variable(tf.ones(shape=[self.input_filters]), name='running_var')
                self.num_batches_tracked =0

            self._built = True

    def forward(self, *x):
        x = enforce_singleton(x)
        input_shape = x.shape
        ndims= len(x.shape)
        reduction_axes = [i for i in range(len(x.shape)) if i not in self.axis]
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
        def _broadcast(v):
            if v is not None and len(v.shape) != ndims and reduction_axes != list(range(ndims - 1)):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.weight), _broadcast(self.bias)

        mean, variance = tf.nn.moments(x, axes=reduction_axes, keepdims=False)
        running_mean = self.running_mean
        running_var = self.running_var

        if not self.training:
            mean=to_tensor(running_mean)
            variance=to_tensor(running_var)

        new_mean, new_variance = mean, variance
        def _do_update(var, value):
            """Compute the updates for mean and variance."""
            return self.assign_moving_average(var, value, self.momentum, None)

        def mean_update():
            true_branch = lambda: _do_update(self.running_mean, new_mean)
            false_branch = lambda: self.running_mean
            if  self.training:
                return true_branch
            else:
                return false_branch

        def variance_update():
            """Update the moving variance."""

            def true_branch_renorm():
                # We apply epsilon as part of the moving_stddev to mirror the training
                # code path.
                running_stddev = _do_update(sqrt(self.running_var), sqrt(new_variance + self.epsilon))
                self.running_var.assign(tf.nn.relu(running_stddev * running_stddev - self.epsilon), name='AssignNewValue')
                return self.running_var

            if self.renorm:
                true_branch = true_branch_renorm
            else:
                true_branch = lambda: _do_update(self.running_var, new_variance)

            false_branch = lambda: self.running_var
            if self.training:
                return true_branch
            else:
                return false_branch

        mean_update()
        variance_update()

        return tf.nn.batch_normalization(x, self.running_mean, self.running_var, offset,scale, self.eps)
    def extra_repr(self):
        return '{input_filters}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


BatchNorm2d=BatchNorm
BatchNorm3d=BatchNorm

#
# class GroupNorm(tf.keras.layers.Layer):
#     def __init__(self,  num_groups=32, affine=True, **kwargs):
#         super(GroupNorm, self).__init__(**kwargs)
#         self.supports_masking = True
#         self.groups = num_groups
#         self.axis = -1
#         self.epsilon = _epsilon
#         self.center = affine
#         self.scale = affine
#         self.affine=affine
#         self.beta_initializer = tf.keras.initializers.get(kwargs.get('beta_initializer'))
#         self.gamma_initializer =  tf.keras.initializers.get(kwargs.get('gamma_initializer'))
#         self.beta_regularizer = tf.keras.regularizers.get(kwargs.get('beta_regularizer'))
#         self.gamma_regularizer = tf.keras.regularizers.get(kwargs.get('gamma_regularizer'))
#         self.beta_constraint = tf.keras.constraints.get(kwargs.get('beta_constraint'))
#         self.gamma_constraint = tf.keras.constraints.get(kwargs.get('gamma_constraint'))
#
#     def build(self, input_shape):
#         dim = input_shape[self.axis]
#
#         if dim is None:
#             raise ValueError('Axis ' + str(self.axis) + ' of '
#                              'input tensor should have a defined dimension '
#                              'but the layer received an input with shape ' +
#                              str(input_shape) + '.')
#
#         if dim < self.groups:
#             raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
#                              'more than the number of channels (' +
#                              str(dim) + ').')
#
#         if dim % self.groups != 0:
#             raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
#                              'multiple of the number of channels (' +
#                              str(dim) + ').')
#
#         self.input_spec = tf.keras.Input(ndim=len(input_shape),axes={self.axis: dim})
#         shape = (dim,)
#
#         if self.scale:
#             self.gamma = self.add_weight(shape=shape,
#                                          name='gamma',
#                                          initializer=self.gamma_initializer,
#                                          regularizer=self.gamma_regularizer,
#                                          constraint=self.gamma_constraint)
#         else:
#             self.gamma = None
#         if self.center:
#             self.beta = self.add_weight(shape=shape,
#                                         name='beta',
#                                         initializer=self.beta_initializer,
#                                         regularizer=self.beta_regularizer,
#                                         constraint=self.beta_constraint)
#         else:
#             self.beta = None
#         self.built = True
#
#     def call(self, inputs, **kwargs):
#         input_shape = K.int_shape(inputs)
#         tensor_input_shape = K.shape(inputs)
#
#         # Prepare broadcasting shape.
#         reduction_axes = list(range(len(input_shape)))
#         del reduction_axes[self.axis]
#         broadcast_shape = [1] * len(input_shape)
#         broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
#         broadcast_shape.insert(1, self.groups)
#
#         reshape_group_shape = K.shape(inputs)
#         group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
#         group_axes[self.axis] = input_shape[self.axis] // self.groups
#         group_axes.insert(1, self.groups)
#
#         # reshape inputs to new group shape
#         group_shape = [group_axes[0], self.groups] + group_axes[2:]
#         group_shape = K.stack(group_shape)
#         inputs = K.reshape(inputs, group_shape)
#
#         group_reduction_axes = list(range(len(group_axes)))
#         group_reduction_axes = group_reduction_axes[2:]
#
#         mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
#         variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)
#
#         inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
#
#         # prepare broadcast shape
#         inputs = K.reshape(inputs, group_shape)
#         outputs = inputs
#
#         # In this case we must explicitly broadcast all parameters.
#         if self.scale:
#             broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
#             outputs = outputs * broadcast_gamma
#
#         if self.center:
#             broadcast_beta = K.reshape(self.beta, broadcast_shape)
#             outputs = outputs + broadcast_beta
#
#         outputs = K.reshape(outputs, tensor_input_shape)
#
#         return outputs
#
#     def get_config(self):
#         config = {
#             'groups': self.groups,
#             'axis': self.axis,
#             'epsilon': self.epsilon,
#             'center': self.center,
#             'scale': self.scale,
#             'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
#             'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
#             'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
#             'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
#             'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
#             'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
#         }
#         base_config = super(GroupNorm, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
# GroupNorm2d=GroupNorm
# GroupNorm3d=GroupNorm
#
# class InstanceNorm(tf.keras.layers.Layer):
#     """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
#     Normalize the activations of the previous layer at each step,
#     i.e. applies a transformation that maintains the mean activation
#     close to 0 and the activation standard deviation close to 1.
#     # Arguments
#         axis: Integer, the axis that should be normalized
#             (typically the features axis).
#             For instance, after a `Conv2D` layer with
#             `data_format="channels_first"`,
#             set `axis=1` in `InstanceNormalization`.
#             Setting `axis=None` will normalize all values in each instance of the batch.
#             Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
#         epsilon: Small float added to variance to avoid dividing by zero.
#         center: If True, add offset of `beta` to normalized tensor.
#             If False, `beta` is ignored.
#         scale: If True, multiply by `gamma`.
#             If False, `gamma` is not used.
#             When the next layer is linear (also e.g. `nn.relu`),
#             this can be disabled since the scaling
#             will be done by the next layer.
#         beta_initializer: Initializer for the beta weight.
#         gamma_initializer: Initializer for the gamma weight.
#         beta_regularizer: Optional regularizer for the beta weight.
#         gamma_regularizer: Optional regularizer for the gamma weight.
#         beta_constraint: Optional constraint for the beta weight.
#         gamma_constraint: Optional constraint for the gamma weight.
#     # Input shape
#         Arbitrary. Use the keyword argument `input_shape`
#         (tuple of integers, does not include the samples axis)
#         when using this layer as the first layer in a model.
#     # Output shape
#         Same shape as input.
#     # References
#         - [Layer Normalization](https://arxiv.org/abs/1607.06450)
#         - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
#     """
#     def __init__(self,momentum=0.1, affine=True, **kwargs):
#         super(InstanceNorm, self).__init__(**kwargs)
#         self.supports_masking = True
#         self.axis = -1
#         self.epsilon = _epsilon
#         self.center = affine
#         self.scale = affine
#         self.affine = affine
#         self.beta_initializer = tf.keras.initializers.get(kwargs.get('beta_initializer'))
#         self.gamma_initializer = tf.keras.initializers.get(kwargs.get('gamma_initializer'))
#         self.beta_regularizer = tf.keras.regularizers.get(kwargs.get('beta_regularizer'))
#         self.gamma_regularizer = tf.keras.regularizers.get(kwargs.get('gamma_regularizer'))
#         self.beta_constraint = tf.keras.constraints.get(kwargs.get('beta_constraint'))
#         self.gamma_constraint = tf.keras.constraints.get(kwargs.get('gamma_constraint'))
#
#     def build(self, input_shape):
#         ndim = len(input_shape)
#         if self.axis == 0:
#             raise ValueError('Axis cannot be zero')
#
#         if (self.axis is not None) and (ndim == 2):
#             raise ValueError('Cannot specify axis for rank 1 tensor')
#
#         self.input_spec = tf.keras.Input(ndim=ndim)
#         if self.axis is None:
#             shape = (1,)
#         else:
#             shape = (input_shape[self.axis],)
#
#         if self.scale:
#             self.gamma = self.add_weight(shape=shape,
#                                          name='gamma',
#                                          initializer=self.gamma_initializer,
#                                          regularizer=self.gamma_regularizer,
#                                          constraint=self.gamma_constraint)
#         else:
#             self.gamma = None
#         if self.center:
#             self.beta = self.add_weight(shape=shape,
#                                         name='beta',
#                                         initializer=self.beta_initializer,
#                                         regularizer=self.beta_regularizer,
#                                         constraint=self.beta_constraint)
#         else:
#             self.beta = None
#         self.built = True
#
#     def call(self, inputs, training=None, **kwargs):
#         input_shape = K.int_shape(inputs)
#         reduction_axes = list(range(0, len(input_shape)))
#
#         if (self.axis is not None):
#             del reduction_axes[self.axis]
#
#         del reduction_axes[0]
#
#         mean = K.mean(inputs, reduction_axes, keepdims=True)
#         stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
#         normed = (inputs - mean) / stddev
#
#         broadcast_shape = [1] * len(input_shape)
#         if self.axis is not None:
#             broadcast_shape[self.axis] = input_shape[self.axis]
#
#         if self.scale:
#             broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
#             normed = normed * broadcast_gamma
#         if self.center:
#             broadcast_beta = K.reshape(self.beta, broadcast_shape)
#             normed = normed + broadcast_beta
#         return normed
#
#     def get_config(self):
#         config = {
#             'axis': self.axis,
#             'epsilon': self.epsilon,
#             'center': self.center,
#             'scale': self.scale,
#             'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
#             'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
#             'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
#             'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
#             'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
#             'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
#         }
#         base_config = super(InstanceNorm, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def get_normalization(fn_name):
    if fn_name is None:
        return None
    if isinstance(fn_name, str):
        if fn_name.lower().strip() in ['instance','in','i']:
            return None
            #return InstanceNorm()
        elif  fn_name.lower().strip() in ['batch','b']:
            return BatchNorm()
        elif  fn_name.lower().strip() in ['group','g']:
            return None
            #return GroupNorm(num_groups=16)
    fn_modules = ['trident.layers.tensorflow_normalizations']
    normalization_fn_ = get_class(fn_name, fn_modules)
    normalization_fn = normalization_fn_
    return normalization_fn
