import math
import warnings
import numpy as np
import cntk as C
from cntk.ops import *
from cntk.layers.blocks import _INFERRED
from cntk.default_options import default_override_or, get_default_override
from cntk.layers.blocks import identity, _initializer_for

from cntk.internal import _as_tuple
from cntk.variables import Record

from .cntk_blocks import *
from .cntk_normalizations import *
from .cntk_activations import *

dev = C.device.use_default_device()
if dev.type() == 0:
    warnings.warn(
        'CNTK backend warning: GPU is not detected. '
        'CNTK\'s CPU version is not fully optimized,'
        'please run with GPU to get better performance.')


def _window(x, axis, begin, end, step, stride, initial_state=None):
    '''
    helper to expand a sequence into a window, splicing them along the given axis (which must already exist)
    '''
    shifted = [
        C.sequence.past_value(x, initial_state=initial_state, time_step=-t) if t < 0 else x                                                        if t == 0 else
        C.sequence.future_value(x, initial_state=initial_state, time_step=t)
        for t in range(begin, end, step)
    ]
    r = C.splice(*shifted, axis=axis)
    if stride != 1:
        raise NotImplementedError('windowed convolution with stride not yet implemented')
    return r


# helper to expand options that can be specified as a single value
def _pad_to_shape(filter_shape, param, what):
    param = _as_tuple(param)
    if len(param) == 1: # broadcast
        while len(param) < len(filter_shape):
            param = (param[0],) + param
    if len(param) != len(filter_shape):
        raise ValueError("{} parameter ({}) must be a scalar or have same number of elements as the filter_shape parameter ({})".format(what, param, filter_shape))
    return param




def _Convolution(filter_shape,  # shape of receptive field, e.g. (3,3)
                 num_filters=None,  # e.g. 64 or None (which means 1 channel and don't add a dimension)
                 sequential=False,  # time convolution if True (filter_shape[0] corresponds to dynamic axis)
                 activation=default_override_or(identity),
                 init=default_override_or(C.glorot_uniform()),
                 pad=default_override_or(False),
                 strides=1,
                 sharing=True,  # (must be True currently)
                 bias=default_override_or(True),
                 init_bias=default_override_or(0),
                 reduction_rank=1,  # (0 means input has no depth dimension, e.g. audio signal or B&W image)  --TODO: call it item_rank?
                 transpose_weight=False,  # (must be False currently)
                 dilation=1,
                 groups=1,
                 input_num_filters=None,
                 max_temp_mem_size_in_samples=0,
                 op_name='Convolution',
                 name=''):
    '''
    Convolution(filter_shape, num_filters=None, sequential=False, activation=identity, init=glorot_uniform(), pad=False, strides=1, sharing=True, bias=True, init_bias=0, reduction_rank=1, transpose_weight=False, dilation=1, groups=1, max_temp_mem_size_in_samples=0, op_name='Convolution', name='')

    This implementation allows for (1) group convolution and (2) initialisation with reduction rank = 1, both of which
    was not possible in the original cntk implementayion.

    More details please refer to the original cntk documentation.

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     sequential (bool, defaults to `False`): if `True`, also convolve along the dynamic axis. ``filter_shape[0]`` corresponds to dynamic axis.
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     sharing (bool, defaults to `True`): When `True`, every position uses the same Convolution kernel.  When `False`, you can have a different Convolution kernel per position, but `False` is not supported.
     bias (bool, optional, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     transpose_weight (bool, defaults to `False`): When this is `True` this is convolution, otherwise this is correlation (which is common for most toolkits)
     dilation (tuple, optional): the dilation value along each axis, default 1 mean no dilation.
     groups (`int`, default 1): number of groups during convolution, that controls the connections between input and output channels. Deafult value is 1,
      which means that all input channels are convolved to produce all output channels. A value of N would mean that the input (and output) channels are
      divided into N groups with the input channels in one group (say i-th input group) contributing to output channels in only one group (i-th output group).
      Number of input and output channels must be divisble by value of groups argument. Also, value of this argument must be strictly positive, i.e. groups > 0.
     max_temp_mem_size_in_samples (int, defaults to 0): Limits the amount of memory for intermediate convolution results.  A value of 0 means, memory is automatically managed.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the convolution operation to it
    '''

    activation = C.get_default_override(_Convolution, activation=activation)
    init       = C.get_default_override(_Convolution, init=init)
    pad        = C.get_default_override(_Convolution, pad=pad)
    bias       = C.get_default_override(_Convolution, bias=bias)
    init_bias  = C.get_default_override(_Convolution, init_bias=init_bias)

    # tuplify all tuple inputs that can also be given as scalars if rank 1
    filter_shape = _as_tuple(filter_shape)
    num_filters  = _as_tuple(num_filters or ())
    filter_rank  = len(filter_shape)
    strides      = _pad_to_shape(filter_shape, strides, 'strides')
    sharing      = _pad_to_shape(filter_shape, sharing, 'sharing')
    pad          = _pad_to_shape(filter_shape, pad, 'pad')
    dilation     = _pad_to_shape(filter_shape, dilation, 'dilation')

    num_filters_per_group = None
    depth_multiplier=1

    if depth_multiplier==1 and num_filters[0]>input_num_filters and  num_filters[0]% input_num_filters == 0:
        depth_multiplier=num_filters[0]//input_num_filters
        num_filters_per_group = (int(input_num_filters / groups),)

    if  groups>0 and num_filters[0] < input_num_filters and  input_num_filters%num_filters[0]  == 0:

        depth_multiplier = num_filters[0] // groups
        num_filters_per_group = (int(input_num_filters / groups),)

    if (reduction_rank != 0) and (reduction_rank != 1):
        raise NotImplementedError("Convolution: reduction_rank must be 0 or 1")
    if transpose_weight:
        raise NotImplementedError("Convolution: transpose_weight option currently not supported")
    if not sharing:
        raise NotImplementedError("Convolution: sharing option currently must be True")
    if groups <= 0:
        raise ValueError("Convolution: groups must be strictly positive, i.e. groups > 0.")
    if input_num_filters and input_num_filters % groups != 0:
        raise ValueError('input_num_filters must be divisible by number of groups')
    if groups > 1 and num_filters[0] % groups != 0:
        raise ValueError('num_filters must be divisible by number of groups')
    if groups > 1 and reduction_rank == 0:
        raise ValueError('reduction_rank cannot be zero in group convolution i.e. there must be incoming channels')
    if sequential:
        raise ValueError("Use cntk.layers.SequentialConvolution instead")



        # TODO: work on groups, understand how reduction==0 and init=np might affect group which doesn't have inferred

    # The convolution() function currently requires exactly one input and one output depth axis.
    # So we emulate those dimensions on this level
    emulating_output_depth = num_filters == ()
    emulating_input_depth = reduction_rank == 0

    actual_output_channels_shape = num_filters if not emulating_output_depth else (1,)
    actual_reduction_shape = _INFERRED if num_filters_per_group is None else num_filters_per_group
    actual_filter_shape = filter_shape

    # add the dimension to the options as well
    num_emulated_axes = emulating_input_depth
    strides = (1,) * num_emulated_axes + strides
    sharing = (True,) * num_emulated_axes + sharing
    pad = (False,) * num_emulated_axes + pad

    kernel_shape = actual_reduction_shape + actual_filter_shape  # kernel := filter plus reductionDims

    # init can be an np.array, which must have the correct dimensions subject to faking depth
    # Once we no longer fake depth at this outer level, we can remove this.
    if isinstance(init, np.ndarray):

        if init.shape[-len(filter_shape):] != filter_shape and init.shape[0] != num_filters[0]:
            raise ValueError("a constant initializer was passed that is of wrong shape")

        init_kernel = init

        # with no inferred axis in W and no emulated axis,
        # padding must be explicit for all axes (channel, static axes)
        # typically, with inferred axis, pad omits channel pad, which is taken to be False. (static axes)
        # with emulate axis, the extra pad would have been supplied already
        pad = (False,) * reduction_rank + pad

    elif num_filters_per_group:

        # with no inferred axis in W and no emulated axis,
        # padding must be explicit for all axes (channel, seq, static axes)
        # typically, with inferred axis, pad omits channel pad, which is taken to be False. (seq, static axes)
        # with emulate axis, the extra pad would have been supplied already
        pad = (False,) * reduction_rank + pad  # assume pad[0] is seq axis, pad[1:] is static axes
        init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-len(actual_output_channels_shape)))

    else:
        init_kernel = _initializer_for(init, Record(filter_rank=filter_rank, output_rank=-len(actual_output_channels_shape)))

    # parameters bound to this Function
    W = C.Parameter(actual_output_channels_shape + kernel_shape,                    init=init_kernel, name='W')                    # (K, C, H, W) aka [ W x H x C x K ]
    b = C.Parameter(actual_output_channels_shape + (1,) * len(actual_filter_shape), init=init_bias,   name='b') if bias else None  # (K,    1, 1) aka [ 1 x 1 x     K ]

    # expression
    @C.BlockFunction(op_name, name)
    def convolve(x):
        # insert additional axis to emulate depth
        if reduction_rank == 0:
            # x: (spatial_shape)
            x = C.expand_dims(x, axis=C.Axis.new_leading_axis())  # e.g. (480, 640) -> (1, 480, 640)
            # x: (in_depth or emulated_in_depth, spatial_shape)

        # actual convolution
        r = C.convolution(W, x,
                          strides=strides,
                          sharing=sharing,
                          auto_padding=pad,
                          dilation=dilation,
                          groups=groups,
                          max_temp_mem_size_in_samples=max_temp_mem_size_in_samples)

        if bias:
            r = r + b

        # if no output dimension is desired, then strip it
        # also need to strip the fake singleton axes, since they are not reduced away
        num_axes_to_remove = emulating_output_depth
        if num_axes_to_remove > 0:
            # (out_depth, emulated axes, spatial_shape)
            r = C.squeeze(r)  # e.g. (2000, 1, 480, 640) -> (2000, 480, 640)
            # (out_depth, spatial_shape)

        if activation is not None:
            r = activation(r)
        return r

    return convolve





def Conv2D(filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
           num_filters=None,
           strides=(1, 1),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=default_override_or(C.glorot_uniform()),
           bias=default_override_or(True),
           init_bias=default_override_or(0),
           reduction_rank=default_override_or(1),
           dilation=1,
           groups=1,
           op_name='Conv2D',
           input_num_filters=None,
           name=''):

    reduction_rank = 1
    if len(_as_tuple(filter_shape)) > 2:
        raise ValueError('Convolution2D: filter_shape must be a scalar or a 2D tuple, e.g. 3 or (3,3)')


    padding        = C.get_default_override(Conv2D, padding=padding)
    pad = False
    sharing=True
    if padding.lower() == 'valid':
        pad = False
    elif padding.lower() == 'same':
        pad = True
    filter_shape = _as_tuple(filter_shape)
    num_filters = _as_tuple(num_filters or ())
    filter_rank = len(filter_shape)
    strides = _pad_to_shape(filter_shape, strides, 'strides')
    sharing = _pad_to_shape(filter_shape, sharing, 'sharing')
    pad = _pad_to_shape(filter_shape, pad, 'pad')
    dilation = _pad_to_shape(filter_shape, dilation, 'dilation')

    reduction_rank =  C.get_default_override(Conv2D, reduction_rank=reduction_rank)
    activation = C.get_default_override(Conv2D, activation=activation)
    init = C.get_default_override(Conv2D, init=init)
    bias       = C.get_default_override(Conv2D, bias=bias)
    init_bias  = C.get_default_override(Conv2D, init_bias=init_bias)
    groups = C.get_default_override(Conv2D, groups=groups)


    if dev.type() == 0 and dilation != (1, 1):
        raise ValueError(
            'Dilated convolution on CPU is not supported by CNTK backend. '
            'Please set `dilation_rate` to (1, 1). '
            'You passed: %s' % (dilation,))
    return _Convolution(filter_shape, num_filters=num_filters, activation=activation, init=init, pad=pad,
                        sequential=False, strides=strides, sharing=sharing, bias=bias, init_bias=init_bias,
                        reduction_rank=reduction_rank, dilation=dilation, groups=groups, input_num_filters=input_num_filters,
                        transpose_weight=False, max_temp_mem_size_in_samples=0, op_name=op_name, name = name)

def conv2d(x,
           filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
           num_filters=None,
           strides=(1, 1),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=default_override_or(C.glorot_uniform()),
           bias=default_override_or(True),
           init_bias=default_override_or(0),
           dilation=1,
           groups=1,
           name=''):
    filter_shape = _as_tuple(filter_shape)
    num_filters = _as_tuple(num_filters or ())
    reduction_rank = 1
    if len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[0] == 1):
        reduction_rank = 0
    return Conv2D(filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
           num_filters=num_filters,
           strides=strides,
           padding=padding,
           activation=activation,
           init=init,
           bias=bias,
           init_bias=init_bias,
           dilation=dilation,
           groups=1,
           input_num_filters=x.shape[0],
           op_name='Conv2D',
           name=name)(x)



def depthwise_conv2d(x,
           filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
           strides=(1, 1),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=default_override_or(C.glorot_uniform()),
           bias=default_override_or(True),
           init_bias=default_override_or(0),
           depth_multiplier=1,
           dilation=1,
           groups=None,
           name=''):

    reduction_rank = 1
    if len(list(x.shape)) == 2 or (len(list(x.shape)) == 3 and x.shape[0] == 1):
        reduction_rank = 0
    activation = C.get_default_override(depthwise_conv2d, activation=activation)
    init = C.get_default_override(depthwise_conv2d, init=init)
    bias = C.get_default_override(depthwise_conv2d, bias=bias)
    init_bias = C.get_default_override(depthwise_conv2d, init_bias=init_bias)
    groups = C.get_default_override(depthwise_conv2d, groups=groups)

    input_num_filters = x.shape[0]
    num_filters_per_group=None
    if groups is None:
        groups=input_num_filters

    num_filters =input_num_filters* depth_multiplier
    num_filters = _as_tuple(num_filters or ())

    if input_num_filters and num_filters[0] % groups == 0 and input_num_filters % groups == 0:
        num_filters_per_group = (int(
            input_num_filters / groups),)  # TODO: work on groups, understand how reduction==0 and init=np might
        # affect group which doesn't have inferred
    if input_num_filters and input_num_filters % groups != 0:
        raise ValueError('input_num_filters must be divisible by number of groups')
    if groups > 1 and num_filters[0] % groups != 0:
        raise ValueError('num_filters must be divisible by number of groups')
    if groups > 1 and reduction_rank == 0:
        raise ValueError('reduction_rank cannot be zero in group convolution i.e. there must be incoming channels')
    return Conv2D(filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
           num_filters=num_filters,
           strides=strides,
           padding=padding,
           activation=activation,
           init=init,
           bias=bias,
           init_bias=init_bias,
           dilation=dilation,
           groups=groups,
           input_num_filters=x.shape[0],
           op_name='DepthwiseConv2D',
           name=name)(x)



def sepatable_conv2d(x,
           filter_shape,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
           num_filters=None,
           strides=(1, 1),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=default_override_or(C.glorot_uniform()),
           bias=default_override_or(True),
           init_bias=default_override_or(0),
           depth_multiplier=1,
           groups=None,
           dilation=1,
           op_name='DepthwiseConv2D',
           name=''):
    activation = C.get_default_override(sepatable_conv2d, activation=activation)
    init = C.get_default_override(sepatable_conv2d, init=init)
    bias = C.get_default_override(sepatable_conv2d, bias=bias)
    init_bias = C.get_default_override(sepatable_conv2d, init_bias=init_bias)
    groups = C.get_default_override(sepatable_conv2d, groups=groups)

    input_num_filters = x.shape[0]
    num_filters_per_group = None
    num_filters = input_num_filters * depth_multiplier
    num_filters = _as_tuple(num_filters or ())

    if input_num_filters and num_filters[0] % groups == 0 and input_num_filters % groups == 0:
        num_filters_per_group = (
        int(input_num_filters / groups),)
    # TODO: work on groups, understand how reduction==0 and init=np might
    #depthwise_kernel
    x=depthwise_conv2d(x,filter_shape=filter_shape,
                       strides=strides,
                       padding='same',
                       activation=activation,
                       init=init,
                       bias=bias,
                       init_bias=init_bias,
                       depth_multiplier=depth_multiplier,
                       dilation=dilation,
                       groups=groups,
                       name='depthwise_kernel')
    x = conv2d(x, filter_shape=(1,1),num_filters=num_filters, strides=strides, padding='valid', activation=activation,
                         init=init, bias=bias, init_bias=init_bias, dilation=1,
                         groups=1, name='pointwise_kernel')
    return x



def _gcd(x, y):
    gcds=[]
    gcd = 1
    if x % y == 0:
        gcds.append(y)
        return y
    for k in range(int(y / 2), 0, -1):
        if x % k == 0 and y % k == 0:
            gcd = k
            gcds.append(k)
    return gcds


def gcd_conv2d(x,
           filter_shape,
           num_filters=None,
           strides=(1, 1),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=default_override_or(C.glorot_uniform()),
           bias=default_override_or(True),
           init_bias=default_override_or(0),
           divisor_rank=default_override_or(0),
           dilation=1,
           groups=None,
           name=''):

    reduction_rank = 1
    if len(list(x.shape)) == 2 or (len(list(x.shape)) == 3 and x.shape[0] == 1):
        reduction_rank = 0

    activation = C.get_default_override(depthwise_conv2d, activation=activation)
    init = C.get_default_override(depthwise_conv2d, init=init)
    bias = C.get_default_override(depthwise_conv2d, bias=bias)
    init_bias = C.get_default_override(depthwise_conv2d, init_bias=init_bias)
    groups = C.get_default_override(depthwise_conv2d, groups=groups)

    input_num_filters = x.shape[0]
    if num_filters is None:
        num_filters=input_num_filters

    gcd_list=_gcd(input_num_filters,num_filters)
    divisor_rank=min(divisor_rank,len(gcd_list))
    groups=gcd_list[divisor_rank]
    num_filters_1=gcd_list[0]
    num_filters_2=num_filters
    #: [56, 28, 14, 8, 7, 4, 2, 1]
    #168=>121
    #divisor_rank=0   output filters 56   groups是  56  從每組3個像素選1個再縮小回2
    # divisor_rank=1   output filters 56   groups是 28 從每組6個像素選2個再縮小回2
    # divisor_rank=2  output filters 56   groups是 14 從每組12個像素選4個

    x = Conv2D(filter_shape,
           num_filters=num_filters_1,
           strides=strides,
           padding='same',
           activation=activation,
           init=init,
           bias=bias,
           init_bias=init_bias,
           dilation=dilation,
           groups=groups,
           input_num_filters=x.shape[0],
           op_name='GCD_Conv2D',
           name=name)(x)
    x = conv2d(x, filter_shape=(1, 1), num_filters=num_filters_2, strides=strides, padding='valid', activation=activation,
               init=init, bias=bias, init_bias=init_bias, dilation=1, groups=1, name='pointwise_kernel')
    return x








