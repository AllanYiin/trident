from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import math
import numbers
import copy
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variables as tf_variables
from trident.backend.common import get_session, addindent, enforce_singleton, unpack_singleton, get_time_suffix, get_class, \
    format_time, get_terminal_size, snake2camel, camel2snake,Signature,epsilon,TensorShape,dtype
from trident.backend.tensorflow_backend import Layer, Sequential, Parameter
from trident.backend.tensorflow_ops import *
from trident.layers.tensorflow_initializers import *


__all__ = ['InstanceNorm','InstanceNorm2d','InstanceNorm3d','BatchNorm','BatchNorm2d','BatchNorm3d','GroupNorm','GroupNorm2d','GroupNorm3d','LayerNorm','LayerNorm2d','LayerNorm3d','L2Norm','PixelNorm','EvoNormB0','EvoNormS0','get_normalization']

_session = get_session()
_epsilon = _session.epsilon


def instance_std(x, eps=1e-5):
    reduce_shape=range(len(x.shape))
    _, var = tf.nn.moments(x, axes=reduce_shape[1:-1], keepdims=True)
    return tf.sqrt(var + eps)



def group_std(x, groups, eps = 1e-5):
    rank = len(x.shape) - 2
    spaceshape=x.shape[1:-1]
    N=x.shape[0]
    C=x.shape[-1]
    x1 = x.reshape(N,groups,-1)
    var = (x1.var(dim=-1, keepdim = True)+eps).reshape(N,groups,-1)
    return (x1 / var.sqrt()).reshape((N,C,)+spaceshape)





class BatchNorm(Layer):
    """Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Shape:
        - Input: :math:`(N, H, W, C)`
        - Output: :math:`(N, H, W, C)` (same shape as input)

    References:
    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167

    """
    def __init__(self, momentum=0.1, affine=True, track_running_stats=True, axis=-1,renorm=False,eps=1e-8,name=None, **kwargs):
        """
         Args:
         eps: a value added to the denominator for numerical stability.
             Default: 1e-5
         momentum: the value used for the running_mean and running_var
             computation. Can be set to ``None`` for cumulative moving average
             (i.e. simple average). Default: 0.1
         affine: a boolean value that when set to ``True``, this module has
             learnable affine parameters. Default: ``True``
         track_running_stats: a boolean value that when set to ``True``, this
             module tracks the running mean and variance, and when set to ``False``,
             this module does not track such statistics and always uses batch
             statistics in both training and eval modes. Default: ``True``

         Examples:
             >>> bn=BatchNorm2d(affine=False)
             >>> input = to_tensor(np.random.standard_normal((2, 128, 128, 64)))
             >>> print(int_shape(bn(input)))
             (2, 64, 128, 128)

         """
        super().__init__(name=name)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = [axis]
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.register_parameter("weight",None) # gamma//scale
        self.register_parameter("bias",None) # beta/ offset

        self.renorm=renorm


    def reset_running_stats(self):
        if self.track_running_stats:
            self.register_buffer("running_mean", zeros(shape=[self.input_filters]))
            self.register_buffer("running_var", ones(shape=[self.input_filters]))
            self.register_buffer("num_batches_tracked", to_tensor(0, dtype=tf.int64), persistent=False)


        if self.affine :
            self.register_parameter("weight",tf.Variable(tf.ones(shape=[self.input_filters]), trainable=True, name='weight'))  # gamma//scale
            self.register_parameter("bias",tf.Variable(tf.zeros(shape=[self.input_filters]), trainable=True, name='bias'))  # beta/ offset


    def assign_moving_average(self, variable, value, momentum, inputs_size):
        with tf.name_scope('AssignMovingAvg') as scope:
            decay = to_tensor(1.0 - momentum,device=self.get_root().device,dtype=variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = where(inputs_size > 0, update_delta, zeros_like(update_delta))
            variable = tf.math.subtract(variable, update_delta, name=scope)
            return variable

    def build(self, input_shape:TensorShape):
        if self._built == False:
            self.input_filters= input_shape[self.filter_index]
            ndims = len(input_shape)
            # Convert axis to list and resolve negatives
            if isinstance(self.axis, int):
                self.axis = [self.axis]
            elif isinstance(self.axis, tuple):
                self.axis = list(self.axis)
            for idx, x in enumerate(self.axis):
                if x < 0:
                    self.axis[idx] = ndims + x

                # Validate axes
            for x in self.axis:
                if x < 0 or x >= ndims:
                    raise ValueError('Invalid axis: %d' % x)
            if len(self.axis) != len(set(self.axis)):
                raise ValueError('Duplicate axis: {}'.format(tuple(self.axis)))
            param_shape = [input_shape[dim] for dim in self.axis]

            if self.affine:
                self.weight=tf.Variable(tf.ones(shape=param_shape),trainable=True, name='weight') #gamma//scale
                self.bias=tf.Variable(tf.zeros(shape=param_shape),trainable=True, name='bias') #beta/ offset

            if self.track_running_stats:
                self.register_buffer('running_mean', zeros(shape=param_shape))
                self.register_buffer('running_var', ones(shape=param_shape))
                self.register_buffer('num_batches_tracked', to_tensor(0, dtype=tf.int64), persistent=False)

            self._built = True


    def forward(self, x, **kwargs) :

        input_shape = x.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        broadcast_shape = [1] * ndims
        for dim in self.axis:
            broadcast_shape[dim] = input_shape.dims[dim].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and self.axis != [ndims - 1]):
                return reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.weight), _broadcast(self.bias)

        mean, variance = moments(x, axis=reduction_axes, keepdims=True)

        running_mean = self.running_mean
        running_var = self.running_var

        if not self.training:
            mean, variance = self.running_mean, self.running_var

        new_mean, new_variance = mean, variance
        def _do_update(var, value):
            """Compute the updates for mean and variance."""
            return self.assign_moving_average(var, value, self.momentum, self.input_shape[0])

        def mean_update():
            """Update the moving variance."""
            true_branch = lambda: _do_update(self.running_mean, new_mean)
            false_branch = lambda: self.running_mean
            if  self.training:
                return true_branch
            else:
                return false_branch

        def variance_update():
            """Update the moving variance."""

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




class GroupNorm(Layer):
    """Applies Group Normalization over a mini-batch of inputs as described in the paper `Group Normalization`_ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Shape:
        - Input: :math:`(N, *, C)` where :math:`C=\text{num_channels}`
        - Output: :math:`(N, *, C)` (same shape as input)

    References:
    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494

    """
    def __init__(self, num_groups=16,affine=True,axis=-1, eps=1e-5, **kwargs):
        """
        Args:
            num_groups (int): number of groups to separate the channels into
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            affine: a boolean value that when set to ``True``, this module
                has learnable per-channel affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.

        Examples:
            >>> gn=GroupNorm(affine=False)
            >>> input = to_tensor(np.random.standard_normal((2,  128, 128, 64)))
            >>> print(int_shape(gn(input)))
            (2, 64, 128, 128)

        """
        super().__init__()
        self.affine=affine
        self.num_groups = num_groups
        self.eps = eps
        self.axis=axis


    def build(self, input_shape:TensorShape):
        if self._built == False :
            assert  self.input_filters % self.num_groups == 0, 'number of groups {} must divide number of channels {}'.format(self.num_groups,  self.input_filters)
            if self.affine:
                self.weight = tf.Variable(ones((self.input_filters)))
                self.bias =  tf.Variable(zeros((self.input_filters)))

            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

                self._built = True


    def forward(self, x, **kwargs) :

        # Prepare broadcasting shape.
        origin_shape=list(int_shape(x))
        group_shape =list(int_shape(x))
        last_dim=group_shape[self.axis]

        group_shape[self.axis]=last_dim//self.num_groups
        group_shape.insert(self.axis, self.groups)
        x=reshape(x,group_shape)
        x_mean,x_variance=moments(x,axis=self.axis,keepdims=True)
        x=(x-x_mean)/(sqrt(x_variance)+self.eps)
        x = reshape(x,origin_shape)
        if self.affine:
            x=x*self.weight+self.bias
        return x
GroupNorm2d=GroupNorm
GroupNorm3d=GroupNorm




class InstanceNorm(GroupNorm):
    """Applies Instance Normalization

    `Instance Normalization: The Missing Ingredient for Fast Stylization`_ .

    .. math::

        'y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta'

    The mean and standard-deviation are calculated per-dimension separately
    for each object in a mini-batch. :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size) if :attr:`affine` is ``True``.

    Instance Normalization is an specific case of ```GroupNormalization```since
    it normalizes all features of one channel. The Groupsize is equal to the
    channel size. Empirically, its accuracy is more stable than batch norm in a
    wide range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    By default, this layer uses instance statistics computed from input data in
    both training and evaluation modes.

    If :attr:`track_running_stats` is set to ``True``, during training this
    layer keeps running estimates of its computed mean and variance, which are
    then used for normalization during evaluation. The running estimates are
    kept with a default :attr:`momentum` of 0.1.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    .. note::
        :class:`InstanceNorm2d` and :class:`LayerNorm` are very similar, but
        have some subtle differences. :class:`InstanceNorm2d` is applied
        on each channel of channeled data like RGB images, but
        :class:`LayerNorm` is usually applied on entire sample and often in NLP
        tasks. Additionally, :class:`LayerNorm` applies elementwise affine
        transform, while :class:`InstanceNorm2d` usually don't apply affine
        transform.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    References:
    .. _`Instance Normalization: The Missing Ingredient for Fast Stylization`:
        https://arxiv.org/abs/1607.08022
    """

    def __init__(self, momentum=0.1, affine=True, track_running_stats=True, eps=1e-5, in_sequence=False, axis=1, name=None, **kwargs):
        """
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, H, W)`
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            momentum: the value used for the running_mean and running_var computation. Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters, initialized the same way as done for batch normalization.
                Default: ``False``.
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``False``

        Examples:
            >>> innorm=InstanceNorm(affine=False)
            >>> input = torch.randn(2, 64, 128, 128)
            >>> print(int_shape(innorm(input)))
            (2, 64, 128, 128)

        """
        super().__init__(name=name)
        self.in_sequence = in_sequence
        self.eps = _epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.axis = axis


    def reset_running_stats(self):
        if self.track_running_stats:
            fill_zeros(self.running_mean)
            fill_ones(self.running_var)
            self.num_batches_tracked=to_tensor(0,dtype=dtype.long)
        if self.affine :
            fill_ones(self.weight)
            fill_zeros(self.bias)
    def build(self, input_shape:TensorShape):
        if not self._built:
            if self.affine:
                self.weight = Parameter(random_normal(self.input_filters,std=0.02))
                self.bias = Parameter(random_normal(self.input_filters,std=0.02))
                fill_ones(self.weight)
                fill_zeros(self.bias)
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

            if self.track_running_stats:
                self.register_buffer('running_mean', zeros(self.input_filters))
                self.register_buffer('running_var', ones(self.input_filters))
                self.register_buffer('num_batches_tracked', to_tensor(0,dtype=dtype.long))
            else:
                self.register_parameter('running_mean', None)
                self.register_parameter('running_var', None)
                self.register_parameter('num_batches_tracked', None)
            self.reset_running_stats()

            self._built = True
    def forward(self, x, **kwargs):
        x= tf.nn.batch_normalization(x, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)
        return x




InstanceNorm2d=InstanceNorm
InstanceNorm3d=InstanceNorm


class LayerNorm(Layer):
    """Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization`_ .

    .. math::
        'y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta'

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)



    References:
    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450

    """
    def __init__(self,  eps=1e-5, affine=True,name=None, **kwargs):
        """
    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Examples:

        >>> input = to_tensor(np.random.standard_normal((2, 128, 128,64)))
        >>> # With Learnable Parameters
        >>> m = LayerNorm(int_shape(input)[1:])
        >>> # Without Learnable Parameters
        >>> m = LayerNorm(int_shape(input)[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

        """
        super().__init__(name=name)

        self.eps = eps
        self.affine = affine


    def build(self, input_shape:TensorShape):
        if self._built == False :
            if self.affine:
                self.weight = tf.Variable(tf.ones(shape=self.input_filters), name='weight') #gamma//scale
                self.bias = tf.Variable(tf.zeros(shape=self.input_filters), name='bias') #beta/ offset
            self._built=True


    def forward(self, x, **kwargs) :

        mean = x.mean(dim=self.axis, keepdim=True).detach()
        std = x.std(dim=self.axis, keepdim=True).detach()
        return self.weight * (x - mean) / (std + self._eps) +self.bias


LayerNorm2d=LayerNorm
LayerNorm3d=LayerNorm



class L2Norm(Layer):
    def __init__(self,in_sequence=False, axis=-1,name=None, **kwargs):
        super().__init__(in_sequence=in_sequence,name=name)
        self.eps=epsilon()
        self.axis=axis

    def build(self, input_shape:TensorShape):
        if self._built == False :
            self._built = True
    def forward(self, x, **kwargs):
        x= l2_normalize(x,axis=self.axis,keepdims=True)
        return x




class PixelNorm(Layer):
    def __init__(self,eps=1e-5, axis=-1, name=None,**kwargs):
        super(PixelNorm, self).__init__(name=name)
        self.eps=eps
        self.axis=axis


    def forward(self, x, **kwargs):
        return x /sqrt(mean(x ** 2, axis=self.axis, keepdims=True) + self.eps)


class SpectralNorm(Layer):
    """
    Attributes:
       layer: tensorflow keras layers (with kernel attribute)
    """

    def __init__(self, module, name='weight', power_iterations=1, in_sequence=False, **kwargs):
        super(SpectralNorm, self).__init__(in_sequence=in_sequence, name=name)
        self.module = module
        self.name = name
        self.power_iterations = power_iterations

    def build(self, input_shape:TensorShape):
        """Build `Layer`"""

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('`SpectralNormalization` must wrap a layer that'
                                 ' contains a `kernel` for weights')

            self.w = self.layer.kernel
            self.w_shape = self.w.shape.as_list()

            self.u = tf.Variable(zeros(tuple([1, self.w_shape[2]])), trainable=False)
            self.v =  tf.Variable(zeros(tuple([1, self.w_shape[1]])), trainable=False)
            kaiming_normal( self.u,a=math.sqrt(5))
            kaiming_normal(self.v, a=math.sqrt(5))

            self.u.assign(l2_normalize( self.u .value().detach()))
            self.v.assign(l2_normalize( self.v .value().detach()))
            self.w_bar = tf.Variable(self.w .value().detach(), requires_grad=False)

    def forward(self, x, **kwargs) :

        if self.training == True:
            # Recompute weights for each forward pass
            self._compute_weights()

        output = self.layer(x)
        return output


    def _compute_weights(self):
        """Generate normalized weights.
        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        w_reshaped = reshape(self.w, [-1, self.w_shape[-1]])
        eps = 1e-12
        _u = copy.deepcopy(self.u)
        _v =matmul(_u, w_reshaped,transpose_b=True)
        _v = _v / maximum(reduce_sum(_v ** 2) ** 0.5, eps)
        _u = matmul(_v, w_reshaped)
        _u = _u / maximum(reduce_sum(_u ** 2) ** 0.5, eps)
        self.u.assign(_u)
        sigma =matmul(matmul(_v, w_reshaped), _u,transpose_b=True)

        self.layer.kernel.assign(self.w / sigma)


class EvoNormB0(Layer):
    def __init__(self,rank=2,nonlinear=True,momentum=0.9,eps = 1e-5):
        super(EvoNormB0, self).__init__()
        self.rank=rank
        self.nonlinear = nonlinear
        self.momentum = momentum
        self.eps = eps


    def build(self, input_shape:TensorShape):
        if self._built == False :
            newshape=np.ones(self.rank+2)
            newshape[-1]=self.input_filters
            newshape=tuple(newshape.astype(np.int32).tolist())
            self.weight = tf.Variable(ones(newshape))
            self.bias = tf.Variable(zeros(newshape))
            if self.nonlinear:
                self.v = tf.Variable(ones(newshape))
            self.register_buffer('running_var', ones(newshape))
            self._built=True

    def forward(self, x, **kwargs):
        if self.training:
            permute_pattern=np.arange(0,self.rank+2)
            permute_pattern[0]=1
            permute_pattern[1]=0

            x1 = x.permute(tuple(permute_pattern)).reshape(self.input_filters, -1)
            var = x1.var(dim=1).reshape(self.weight.shape)
            self.running_var.assign(self.momentum * self.running_var + (1 - self.momentum) * var)
        else:
            var = self.running_var
        if self.nonlinear:
            den = max((var+self.eps).sqrt(), self.v * x + instance_std(x))
            return x / den * self.weight + self.bias
        else:
            return x * self.weight + self.bias


class EvoNormS0(Layer):
    def __init__(self,rank=2,groups=8,nonlinear=True):
        super(EvoNormS0, self).__init__()
        self.nonlinear = nonlinear
        self.groups = groups


    def build(self, input_shape:TensorShape):
        if self._built == False :
            newshape = np.ones(self.rank + 2)
            newshape[-1] = self.input_filters
            newshape = tuple(newshape.astype(np.int32).tolist())
            self.weight = tf.Variable(ones(newshape))
            self.bias = tf.Variable(zeros(newshape))
            if self.nonlinear:
                self.v = tf.Variable(ones(newshape))
            self.register_buffer('running_var', ones(newshape))
            self._built=True

    def forward(self, x, **kwargs):
        if self.nonlinear:
            num = sigmoid(self.v * x)
            std = group_std(x,self.groups)
            return num * std * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta



def get_normalization(fn_name):
    if fn_name is None:
        return None
    elif isinstance(fn_name, Layer) and 'Norm' in fn_name.__class__.__name__:
        return fn_name
    elif inspect.isclass(fn_name) and fn_name.__class__.__name__ == type:
        return fn_name()
    elif inspect.isclass(fn_name):
        return fn_name
    elif isinstance(fn_name, str):
        if fn_name.lower().strip() in ['instance','in','i']:
            return None
            #return InstanceNorm()
        elif fn_name.lower().strip() in ['batch_norm', 'batch', 'bn', 'b']:
            return BatchNorm2d()
        elif  fn_name.lower().strip() in ['group','g']:
            return None
            #return GroupNorm(num_groups=16)
    elif inspect.isclass(fn_name):
        return fn_name
    fn_modules = ['trident.layers.tensorflow_normalizations']
    normalization_fn_ = get_class(fn_name, fn_modules)
    normalization_fn = normalization_fn_
    return normalization_fn
