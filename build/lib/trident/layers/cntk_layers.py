import math
import warnings
import six
import numpy as np
import cntk as C
from cntk.ops import *
from cntk.layers.blocks import _INFERRED
from cntk.default_options import default_override_or, get_default_override
from cntk.layers.blocks import identity, _initializer_for

from cntk.internal import _as_tuple
from cntk.variables import Record

from .cntk_constraints import *
from .cntk_activations import *
import sys

import linecache


def _PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


#prines=[２．３．５．７．１１．１３．１７．１９．２３．２９．３１．　３７．４１．４３．４７．５３．５９．６１．６７．７１．　７３．７９．８３．８９．９７]

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
def _pad_to_shape(kernel_size, param, what):
    param = _as_tuple(param)
    if len(param) == 1: # broadcast
        while len(param) < len(kernel_size):
            param = (param[0],) + param
    if len(param) != len(kernel_size):
        raise ValueError("{} parameter ({}) must be a scalar or have same number of elements as the kernel_size parameter ({})".format(what, param, kernel_size))
    return param




def ConvolutionBase(kernel_size=default_override_or((3,3)),  # shape of receptive field, e.g. (3,3)
                    num_filters=default_override_or(False), # e.g. 64 or None (which means 1 channel and don't add a dimension)
                    sequential=default_override_or(False),# time convolution if True (kernel_size[0] corresponds to dynamic axis)
                    activation=default_override_or(identity),
                    init=C.he_normal(0.02),
                    pad=default_override_or(False),
                    strides=default_override_or((1,1)),
                    sharing=True,  # (must be True currently)
                    use_bias=default_override_or(False),
                    init_bias=default_override_or(0),
                    reduction_rank=default_override_or(1) ,  # (0 means input has no depth dimension, e.g. audio signal or B&W image)  --TODO: call it item_rank?
                    transpose_weight=default_override_or(False),  # (must be False currently)
                    dilation=default_override_or(1) ,
                    groups=default_override_or(1) , 
                    input_num_filters=default_override_or(1) , 
                    max_temp_mem_size_in_samples=0,
                    op_name=default_override_or('Convolution'),
                    weights_contraint=default_override_or(default_constrains),
                    gcd=default_override_or(1) ,
                    name=''):
    '''
    Convolution(kernel_size, num_filters=None, sequential=False, activation=identity, init=glorot_uniform(), pad=False, strides=1, sharing=True, bias=True, init_bias=0, reduction_rank=1, transpose_weight=False, dilation=1, groups=1, max_temp_mem_size_in_samples=0, op_name='Convolution', name='')

    This implementation allows for (1) group convolution and (2) initialisation with reduction rank = 1, both of which
    was not possible in the original cntk implementayion.

    More details please refer to the original cntk documentation.

    Args:
     kernel_size (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     sequential (bool, defaults to `False`): if `True`, also convolve along the dynamic axis. ``kernel_size[0]`` corresponds to dynamic axis.
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
    kernel_size = C.get_default_override(ConvolutionBase, kernel_size=kernel_size)
    num_filters = C.get_default_override(ConvolutionBase, num_filters=num_filters)
    strides = C.get_default_override(ConvolutionBase, strides=strides)
    activation = C.get_default_override(ConvolutionBase, activation=activation)

    init       = C.get_default_override(ConvolutionBase, init=init)
    pad        = C.get_default_override(ConvolutionBase, pad=pad)
    use_bias       = C.get_default_override(ConvolutionBase, use_bias=use_bias)
    init_bias  = C.get_default_override(ConvolutionBase, init_bias=init_bias)
    weights_contraint= C.get_default_override(ConvolutionBase, weights_contraint=weights_contraint)
    groups = C.get_default_override(ConvolutionBase, groups=groups)
    dilation = C.get_default_override(ConvolutionBase, dilation=dilation)
    input_num_filters = C.get_default_override(ConvolutionBase, input_num_filters=input_num_filters)
    op_name = C.get_default_override(ConvolutionBase, op_name=op_name)
    reduction_rank = C.get_default_override(ConvolutionBase, reduction_rank=reduction_rank)
    transpose_weight = C.get_default_override(ConvolutionBase, transpose_weight=transpose_weight)
    gcd = C.get_default_override(ConvolutionBase, gcd=gcd)



    kernel_size = _as_tuple(kernel_size)
    num_filters  = _as_tuple(num_filters or ())

    filter_rank  = len(kernel_size)
    strides      = _pad_to_shape(kernel_size, strides, 'strides')
    sharing      = _pad_to_shape(kernel_size, sharing, 'sharing')
    pad          = _pad_to_shape(kernel_size, pad, 'pad')
    dilation     = _pad_to_shape(kernel_size, dilation, 'dilation')

    num_filters_per_group = None
    depth_multiplier=1
    if input_num_filters is not None and  groups>=1:
        depth_multiplier=num_filters[0]//groups
        num_filters_per_group=_as_tuple(input_num_filters//groups or ())


    if (reduction_rank != 0) and (reduction_rank != 1):
        raise NotImplementedError("Convolution: reduction_rank must be 0 or 1")
    if transpose_weight:
        raise NotImplementedError("Convolution: transpose_weight option currently not supported")
    if not sharing:
        raise NotImplementedError("Convolution: sharing option currently must be True")
    if groups <= 0:
        raise ValueError("Convolution: groups must be strictly positive, i.e. groups > 0.")
    if groups>1 and  input_num_filters  and input_num_filters  % groups != 0:
        raise ValueError('input_num_filters must be divisible by number of groups')

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
    actual_kernel_size = kernel_size

    # add the dimension to the options as well
    num_emulated_axes = emulating_input_depth
    strides = (1,) * num_emulated_axes + strides
    sharing = (True,) * num_emulated_axes + sharing
    pad = (False,) * num_emulated_axes + pad

    #
    # if groups>1 and  gcd==1 and depth_multiplier==1 and num_filters[0]>input_num_filters[0] and  num_filters[0]% input_num_filters[0] == 0:
    #     depth_multiplier=num_filters[0]//input_num_filters[0]
    #
    #
    # if  gcd==1 and groups>1 and num_filters[0] < input_num_filters and  input_num_filters%num_filters[0]  == 0:
    #     depth_multiplier = num_filters[0] // groups
    #     num_filters_per_group = (int(input_num_filters  //groups),)

    kernel_shape = actual_reduction_shape + actual_kernel_size  # kernel := filter plus reductionDims

    # init can be an np.array, which must have the correct dimensions subject to faking depth
    # Once we no longer fake depth at this outer level, we can remove this.
    if isinstance(init, np.ndarray):

        if init.shape[-len(kernel_size):] != kernel_size and init.shape[0] != num_filters[0]:
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
    b = C.Parameter(actual_output_channels_shape + (1,) * len(actual_kernel_size), init=init_bias,   name='b') if use_bias else None  # (K,    1, 1) aka [ 1 x 1 x     K ]

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

        if use_bias:
            r = r + b

        #r =r *C.constant(1-np.isnan(W.value).astype(np.float32))
        # if no output dimension is desired, then strip it
        # also need to strip the fake singleton axes, since they are not reduced away
        num_axes_to_remove = emulating_output_depth
        if num_axes_to_remove > 0:
            # (out_depth, emulated axes, spatial_shape)
            r = C.squeeze(r)  # e.g. (2000, 1, 480, 640) -> (2000, 480, 640)
            # (out_depth, spatial_shape)
        #r=C.clip(r,-1e3,1e3)

        cons=min_max_norm
        if activation is not None  and  activation !=identity:
            r = activation()(r)
            # r=cons(r)
        return r

    return convolve




def Conv2d(kernel_size=default_override_or((3, 3)),  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
           num_filters=default_override_or(None),
           strides=default_override_or((1,1)),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=C.he_normal(0.02),
           use_bias=default_override_or(False),
           init_bias=default_override_or(0),
           reduction_rank=default_override_or(1),
           dilation=default_override_or(1),
           groups=default_override_or(1),
           op_name='Conv2d',
           weights_contraint=default_override_or(default_constrains),
           input_filters=default_override_or(None),
           gcd=default_override_or(1),
           name=''):

    reduction_rank = 1
    if len(_as_tuple(kernel_size)) > 2:
        raise ValueError('Convolution2D: kernel_size must be a scalar or a 2D tuple, e.g. 3 or (3,3)')
    kernel_size = C.get_default_override(Conv2d, kernel_size=kernel_size)
    num_filters    = C.get_default_override(Conv2d, num_filters=num_filters)
    strides = C.get_default_override(Conv2d, strides=strides)
    padding        = C.get_default_override(Conv2d, padding=padding)
    reduction_rank = C.get_default_override(Conv2d, reduction_rank=reduction_rank)
    activation = C.get_default_override(Conv2d, activation=activation)
    use_bias = C.get_default_override(Conv2d, use_bias=use_bias)
    init_bias = C.get_default_override(Conv2d, init_bias=init_bias)
    groups = C.get_default_override(Conv2d, groups=groups)
    dilation = C.get_default_override(Conv2d, dilation=dilation)
    weights_contraint = C.get_default_override(Conv2d, weights_contraint=weights_contraint)
    input_filters= C.get_default_override(Conv2d, input_filters=input_filters)
    gcd = C.get_default_override(Conv2d, gcd=gcd)
    pad = False
    sharing=True

    if padding.lower() == 'valid':
        pad = False
    elif padding.lower() == 'same':
        pad = True
    kernel_size = _as_tuple(kernel_size)
    num_filters = _as_tuple(num_filters or ())
    filter_rank = len(kernel_size)
    strides = _pad_to_shape(kernel_size, strides, 'strides')
    sharing = _pad_to_shape(kernel_size, sharing, 'sharing')
    pad = _pad_to_shape(kernel_size, pad, 'pad')
    dilation = _pad_to_shape(kernel_size, dilation, 'dilation')
    if  activation is not None and  isinstance(activation,str):
        activation= get_activation(activation)


    if dev.type() == 0 and dilation != (1, 1):
        raise ValueError(
            'Dilated convolution on CPU is not supported by CNTK backend. '
            'Please set `dilation_rate` to (1, 1). '
            'You passed: %s' % (dilation,))

    return ConvolutionBase(kernel_size, num_filters=num_filters, activation=activation, init=init, pad=pad,
                           sequential=False, strides=strides, sharing=sharing, use_bias=use_bias, init_bias=init_bias,
                           reduction_rank=reduction_rank, dilation=dilation, groups=groups, input_num_filters=input_filters,
                           transpose_weight=False, max_temp_mem_size_in_samples=0, op_name=op_name, weights_contraint=weights_contraint,
                           gcd=gcd, name = name)



def Conv3d(kernel_size,     # shape of receptive field, e.g. (3,3,3). Must be a 3-element tuple.
                  num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                  strides=1,
                   padding=default_override_or('same'),
                  activation=default_override_or(identity),
                  init=default_override_or(C.he_normal()),
                  use_bias=default_override_or(False),
                   init_bias=default_override_or(0),

                  reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                  dilation=1,
                  groups=1,
                  op_name = 'Conv3d',
                  weights_contraint=default_override_or(default_constrains),
                  input_num_filters=default_override_or(None),
                  gcd=default_override_or(1),
                  name=''):
    '''
    Convolution3D(filter_shape, num_filters=None, activation=identity, init=glorot_uniform(), pad=False, strides=1, bias=True, init_bias=0, reduction_rank=1, name='')

    Layer factory function to create a 3D convolution layer with optional non-linearity.
    Same as `Convolution()` except that filter_shape is verified to be 3-dimensional.
    See `Convolution()` for extensive documentation.

    Args:
     filter_shape (`int` or `tuple` of `ints`): shape (spatial extent) of the receptive field, *not* including the input feature-map depth. E.g. (3,3) for a 2D convolution.
     num_filters (int, defaults to `None`): number of filters (output feature-map depth), or ``()`` to denote scalar output items (output shape will have no depth axis).
     activation (:class:`~cntk.ops.functions.Function`, defaults to `identity`): optional function to apply at the end, e.g. `relu`
     init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to :func:`~cntk.initializer.glorot_uniform` ): initial value of weights `W`
     pad (`bool` or `tuple` of `bools`, defaults to `False`): if `False`, then the filter will be shifted over the "valid"
      area of input, that is, no value outside the area is used. If ``pad=True`` on the other hand,
      the filter will be applied to all input positions, and positions outside the valid region will be considered containing zero.
      Use a `tuple` to specify a per-axis value.
     strides (`int` or `tuple` of `ints`, defaults to 1): stride of the convolution (increment when sliding the filter over the input). Use a `tuple` to specify a per-axis value.
     bias (bool, defaults to `True`): the layer will have no bias if `False` is passed here
     init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
     reduction_rank (`int`, defaults to 1): set to 0 if input items are scalars (input has no depth axis), e.g. an audio signal or a black-and-white image
      that is stored with tensor shape (H,W) instead of (1,H,W)
     dilation (tuple, optional): the dilation value along each axis, default 1 mean no dilation.
     groups (`int`, default 1): number of groups during convolution, that controls the connections between input and output channels. Deafult value is 1,
      which means that all input channels are convolved to produce all output channels. A value of N would mean that the input (and output) channels are
      divided into N groups with the input channels in one group (say i-th input group) contributing to output channels in only one group (i-th output group).
      Number of input and output channels must be divisble by value of groups argument. Also, value of this argument must be strictly positive, i.e. groups > 0.
     name (str, defaults to ''): the name of the function instance in the network

    Returns:
        cntk.ops.functions.Function:
        A function that accepts one argument and applies the convolution operation to it

    '''

    activation = get_default_override(Conv3d, activation=activation)
    init       = get_default_override(Conv3d, init=init)
    padding        = get_default_override(Conv3d, padding=padding)
    weights_contraint = C.get_default_override(Conv3d, weights_contraint=weights_contraint)
    input_num_filters = C.get_default_override(Conv3d, input_num_filters=input_num_filters)
    gcd = C.get_default_override(Conv3d, gcd=gcd)
    pad = False
    sharing = True
    if padding.lower() == 'valid':
        pad = False
    elif padding.lower() == 'same':
        pad = True
    use_bias       = get_default_override(Conv3d, use_bias=use_bias)
    init_bias  = get_default_override(Conv3d, init_bias=init_bias)

    strides = _pad_to_shape(kernel_size, strides, 'strides') if  not isinstance(strides,tuple) else strides
    sharing = _pad_to_shape(kernel_size, sharing, 'sharing')
    pad = _pad_to_shape(kernel_size, pad, 'pad')
    dilation = _pad_to_shape(kernel_size, dilation, 'dilation')if  not isinstance(dilation,tuple) else dilation
    if activation is not None and isinstance(activation, str):
        activation = get_activation(activation)


    if len(_as_tuple(kernel_size)) > 3:
         raise ValueError('Convolution3D: filter_shape must be a scalar or a 3D tuple, e.g. 3 or (3,3,3)')
    kernel_size = _pad_to_shape((0,0,0), kernel_size, 'filter_shape')
    return ConvolutionBase(kernel_size, num_filters=num_filters, activation=activation, init=init, pad=pad,
                               sequential=False, strides=strides, sharing=sharing, use_bias=use_bias, init_bias=init_bias,
                               reduction_rank=reduction_rank, dilation=dilation, groups=groups, input_num_filters=input_num_filters,
                               transpose_weight=False, max_temp_mem_size_in_samples=0, op_name=op_name, weights_contraint=weights_contraint,
                               gcd=gcd,name = name)

# def conv2d(x, kernel_size, strides=(1, 1), padding='valid',
#
#            data_format=None, dilation_rate=(1, 1)):
#     kernel_size = _as_tuple(kernel_size)
#     num_filters = _as_tuple(num_filters or ())
#
#     filter_rank = len(kernel_size)
#     strides = _pad_to_shape(kernel_size, strides, 'strides')
#     pad = _pad_to_shape(kernel_size, pad, 'pad')
#     dilation = _pad_to_shape(kernel_size, dilation, 'dilation')
#     if dev.type() == 0 and dilation_rate != (1, 1):
#         raise ValueError(
#             'Dilated convolution on CPU is not supported by CNTK backend. '
#             'Please set `dilation_rate` to (1, 1). '
#             'You passed: %s' % (dilation_rate,))
#     dilation_rate = (1,) + dilation_rate
#     x = C.convolution(kernel,
#                       x,
#                       strides,
#                       auto_padding=[False, padding, padding],
#                       dilation=dilation_rate)
#     return x




def conv2d(x,
           kernel_size=default_override_or((3, 3)),  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
           num_filters=default_override_or(None),
           strides=default_override_or((1, 1)),
           padding=default_override_or('same'),
           activation=default_override_or('identity'),
           init=C.he_normal(0.02),
           use_bias=default_override_or(True),
           init_bias=default_override_or(0),
           dilation=default_override_or(1),
           groups=default_override_or(1),
           weights_contraint=default_override_or(default_constrains),
           gcd=default_override_or(1),
           name=''):
    kernel_size = C.get_default_override(conv2d, kernel_size=kernel_size)
    num_filters = C.get_default_override(conv2d, num_filters=num_filters)
    strides = C.get_default_override(conv2d, strides=strides)
    padding = C.get_default_override(conv2d, padding=padding)
    activation = C.get_default_override(conv2d, activation=activation)
    init = C.get_default_override(conv2d, init=init)
    use_bias = C.get_default_override(conv2d, use_bias=use_bias)
    init_bias = C.get_default_override(conv2d, init_bias=init_bias)
    groups = C.get_default_override(conv2d, groups=groups)
    dilation = C.get_default_override(conv2d, dilation=dilation)
    weights_contraint = C.get_default_override(conv2d, weights_contraint=weights_contraint)
    gcd = C.get_default_override(conv2d, gcd=gcd)
    activation_fn=get_activation(activation)
    input_shape=x.shape
    kernel_size = _as_tuple(kernel_size)
    reduction_rank = 1 
    if len(input_shape) == 2 or (len(input_shape) == 3 and input_shape[0] == 1):
        reduction_rank = 0
    return Conv2d(kernel_size,  # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
                  num_filters=num_filters,
                  strides=strides,
                  padding=padding,
                  activation=activation_fn,
                  init=init,
                  use_bias=use_bias,
                  init_bias=init_bias,
                  dilation=dilation,
                  groups=groups,
                  input_filters=input_shape[0],
                  op_name='Conv2d',
                  weights_contraint=weights_contraint,
                  reduction_rank=reduction_rank,
                  gcd=gcd,
                  name=name)(x)



def depthwise_conv2d(x,
                     kernel_size=default_override_or((3, 3)),
                     depth_multiplier=default_override_or(1),
                     strides=default_override_or((1, 1)),
                     padding=default_override_or('same'),
                     activation=default_override_or('identity'),
                     init=C.he_normal(0.02),
                     use_bias=default_override_or(False),
                     init_bias=default_override_or(0),
                     dilation=default_override_or(1),
                     weights_contraint=default_override_or(default_constrains),
                     groups=default_override_or(1),
                     name=''):
    try:
        kernel_size = C.get_default_override(sepatable_conv2d, kernel_size=kernel_size)
        strides = C.get_default_override(sepatable_conv2d, strides=strides)
        padding = C.get_default_override(sepatable_conv2d, padding=padding)
        activation = C.get_default_override(sepatable_conv2d, activation=activation)
        use_bias = C.get_default_override(sepatable_conv2d, use_bias=use_bias)
        init = C.get_default_override(sepatable_conv2d, init=init)
        bias = C.get_default_override(sepatable_conv2d, use_bias=use_bias)
        init_bias = C.get_default_override(sepatable_conv2d, init_bias=init_bias)
        dilation = C.get_default_override(sepatable_conv2d, dilation=dilation)
        depth_multiplier = C.get_default_override(sepatable_conv2d, depth_multiplier=depth_multiplier)
        weights_contraint = C.get_default_override(sepatable_conv2d, weights_contraint=weights_contraint)

        groups = x.shape[0]
        input_num_filters = x.shape[0]
        num_filters =  x.shape[0] * depth_multiplier
        num_filters = _as_tuple(num_filters or ())

        num_filters_per_group = depth_multiplier
        # TODO: work on groups, understand how reduction==0 and init=np might
        print('input:{0}   output:{1}->{2}  gcd:{3} group:{4}   放大因子:{5} '.format(input_num_filters, input_num_filters,  num_filters, '--', groups,
                                                                                  num_filters[0] / input_num_filters))
        #depthwise_kernel
        x = Conv2d(kernel_size, num_filters=num_filters,
                   strides=strides, padding=padding, activation=activation,
                   init=init, use_bias=use_bias, init_bias=init_bias, dilation=dilation, groups=groups,
                   input_filters= input_num_filters, op_name='DepthwiseConv2d', weights_contraint=weights_contraint, name=name)(x)
        return x
    except:
        _PrintException()






def DepthwiseConv2d(
            kernel_size=default_override_or((3, 3)),
            depth_multiplier=default_override_or(1),
            strides=default_override_or((1, 1)),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=C.he_normal(0.5),
           use_bias=default_override_or(False),
           init_bias=default_override_or(0),
           dilation=default_override_or(1),
           weights_contraint=default_override_or(default_constrains),
           name=''):

    def internal_op(x):
        return depthwise_conv2d(x,
            kernel_size,
            depth_multiplier=depth_multiplier,
            strides=strides,
           padding=padding,
           activation=activation,
           init=init,
           use_bias=use_bias,
           init_bias=init_bias,
           dilation=dilation,
           weights_contraint=weights_contraint,
                                groups=x.shape[0],
           name=name)
    return internal_op




def sepatable_conv2d(x,
                     kernel_size=default_override_or((3, 3)),
                     # shape of receptive field, e.g. (3,3). Must be a 2-element tuple.
                     # num_filters=default_override_or(None),
                     depth_multiplier=default_override_or(1),
                     strides=default_override_or((1, 1)),
                     padding=default_override_or('same'),
                     activation=default_override_or('identity'),
                     init=C.he_normal(0.02),
                     use_bias=default_override_or(False),
                     init_bias=default_override_or(0),
                     dilation=default_override_or(1),
                     weights_contraint=default_override_or(default_constrains),

                     name=''):
    kernel_size = C.get_default_override(sepatable_conv2d, kernel_size=kernel_size)
    strides = C.get_default_override(sepatable_conv2d, strides=strides)
    padding = C.get_default_override(sepatable_conv2d, padding=padding)
    activation = C.get_default_override(sepatable_conv2d, activation=activation)
    use_bias = C.get_default_override(sepatable_conv2d, use_bias=use_bias)
    init = C.get_default_override(sepatable_conv2d, init=init)
    bias = C.get_default_override(sepatable_conv2d, use_bias=use_bias)
    init_bias = C.get_default_override(sepatable_conv2d, init_bias=init_bias)
    dilation = C.get_default_override(sepatable_conv2d, dilation=dilation)
    depth_multiplier = C.get_default_override(sepatable_conv2d, depth_multiplier=depth_multiplier)
    weights_contraint = C.get_default_override(sepatable_conv2d, weights_contraint=weights_contraint)

    groups =x.shape[0]
    input_num_filters=x.shape[0]
    num_filters = groups * depth_multiplier
    num_filters = _as_tuple(num_filters or ())

    num_filters_per_group =depth_multiplier
    # TODO: work on groups, understand how reduction==0 and init=np might
    print('input:{0}   output:{1}->{2}  gcd:{3} group:{4}   放大因子:{5} '.format(input_num_filters, input_num_filters,  num_filters, '--', groups,
                                                                              num_filters[0] / input_num_filters))
    #depthwise_kernel
    x = Conv2d(kernel_size, num_filters=x.shape[0],
               strides=strides, padding='same', activation=activation,
               init=init, use_bias=False, init_bias=init_bias, dilation=dilation, groups=input_num_filters,
               input_filters=1, op_name='GCD_Conv2d', weights_contraint=weights_contraint,
               gcd=1, name=name)(x)
    x = conv2d(x, kernel_size=(1,1),num_filters=num_filters, strides=1, padding='valid', activation=None,
                         init=init, use_bias='False', init_bias=init_bias, dilation=1,groups=1, weights_contraint=weights_contraint,name='pointwise_kernel')
    return x






def SeparableConv2d(
            kernel_size=default_override_or((3, 3)),
            depth_multiplier=default_override_or(1),
            strides=default_override_or((1, 1)),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=C.he_normal(0.5),
           use_bias=default_override_or(False),
           init_bias=default_override_or(0),
           dilation=default_override_or(1),
           weights_contraint=default_override_or(default_constrains),
          input_num_filters=default_override_or(None),
           name=''):
    def internal_op(x):
        return sepatable_conv2d(x,
            kernel_size,
            depth_multiplier,
            strides,
           padding,
           activation,
           init,
           use_bias,
           init_bias,
           dilation,
           weights_contraint,

           name)
    return internal_op



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

def GcdConv2d(
            kernel_size=default_override_or((3, 3)),
            num_filters=default_override_or(None),
            strides=default_override_or((1, 1)),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=C.he_normal(0.02),
           use_bias=default_override_or(False),
           init_bias=default_override_or(0),
           divisor_rank=default_override_or(0),
           dilation=default_override_or(1),
           weights_contraint=default_override_or(default_constrains),
           self_norm=default_override_or(True),
           name=''):
    def internal_op(x):
        return gcd_conv2d(x,
            kernel_size,
            num_filters,
            strides,
           padding,
           activation,
           init,
           use_bias,
           init_bias,
           divisor_rank,
           dilation,
           weights_contraint,
                          self_norm,
           name)
    return internal_op

def gcd_conv2d(x,
            kernel_size=default_override_or((3, 3)),
            num_filters=default_override_or(None),
            strides=default_override_or((1, 1)),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=C.he_normal(0.02),
           use_bias=default_override_or(False),
           init_bias=default_override_or(0),
           divisor_rank=default_override_or(0),
           dilation=default_override_or(1),
           weights_contraint=default_override_or(default_constrains),
           self_norm=default_override_or(True),
           name=''):

    kernel_size = C.get_default_override(conv2d, kernel_size=kernel_size)
    num_filters = C.get_default_override(conv2d, num_filters=num_filters)

    strides= C.get_default_override(gcd_conv2d, strides=strides)
    padding = C.get_default_override(gcd_conv2d, padding=padding)
    activation = C.get_default_override(gcd_conv2d, activation=activation)
    init = C.get_default_override(gcd_conv2d, init=init)
    use_bias = C.get_default_override(gcd_conv2d, use_bias=use_bias)
    init_bias = C.get_default_override(gcd_conv2d, init_bias=init_bias)
    dilation = C.get_default_override(gcd_conv2d, dilation=dilation)
    divisor_rank = C.get_default_override(gcd_conv2d, divisor_rank=divisor_rank)
    weights_contraint = C.get_default_override(gcd_conv2d, weights_contraint=weights_contraint)
    self_norm= C.get_default_override(gcd_conv2d, self_norm=self_norm)

    input_num_filters = x.shape[0]
    reduction_rank = 1
    if len(list(x.shape)) == 2 or (len(list(x.shape)) == 3 and x.shape[0] == 1):
        reduction_rank = 0

    if num_filters is None:
        num_filters=input_num_filters
    num_filters_1 = input_num_filters
    num_filters_2 = num_filters
    gcd_list=[]
    gcd_list.extend(_gcd(input_num_filters,num_filters))

    gcd =1
    if len(gcd_list)==0:
        groups=input_num_filters
        num_filters_1=input_num_filters
    else:
        gcd = gcd_list[0]
        groups = gcd_list[min(int(divisor_rank), len(gcd_list))]
        num_filters_1 = gcd

    num_filters_2 = num_filters
    factors = _get_divisors(num_filters // gcd)
    if input_num_filters == num_filters:
        factors = _get_divisors(num_filters)

    if input_num_filters == gcd:
        groups = input_num_filters
        num_filters_1 = num_filters
    elif num_filters == gcd:
        groups = gcd
        num_filters_1 = num_filters
    elif len(factors) == 0:
        groups = gcd
        num_filters_1 = gcd
    else:
        num_filters_1 = gcd * factors[-1]



    num_filters_per_group = (int(input_num_filters // groups),)

    #: [56, 28, 14, 8, 7, 4, 2, 1]
    # 168=>121
    # divisor_rank=0   output filters 56   groups是  56  從每組3個像素選1個再縮小回2
    # divisor_rank=1   output filters 56   groups是 28 從每組6個像素選2個再縮小回2
    # divisor_rank=2  output filters 56   groups是 14 從每組12個像素選4個

    print('input:{0}   output:{1}->{2}  gcd:{3} group:{4}   放大因子:{5} '.format(input_num_filters, num_filters_2,
                                                                              num_filters_2, gcd, groups,
                                                                              num_filters_2 // groups))

    x = Conv2d(kernel_size, num_filters=num_filters_1, strides=strides, padding='same', activation=None,
               init=init, use_bias=use_bias, init_bias=init_bias, dilation=dilation, groups=groups,
               input_filters=input_num_filters, op_name='GCD_Conv2d', weights_contraint=weights_contraint,
               gcd=gcd, name=name)(x)
    x = Conv2d(kernel_size, num_filters=num_filters_2, strides=1, padding='same', activation=None, init=init,
               use_bias=use_bias, init_bias=init_bias, dilation=dilation, groups=groups,
               input_filters=num_filters_1, op_name='GCD_Conv2d', weights_contraint=weights_contraint, gcd=gcd,
               name=name)(x)

    if self_norm:
        input_sgape=x.shape
        x = C.reshape(x, (groups, num_filters_2 // groups, x.shape[1], x.shape[2]))
        group_mean = C.reduce_mean(x, axis=[1, 2, 3])
        group_variance = C.reduce_mean(C.square(x - C.stop_gradient(group_mean)), axis=[1, 2, 3])
        group_std = C.sqrt(group_variance) + C.constant(1e-8)
        normed = (x - group_mean) / group_std
        x = C.reshape(normed,input_sgape)

    x = conv2d(x, kernel_size=(1, 1), num_filters=num_filters_2, strides=1, padding='valid', activation=None, init=init, use_bias=use_bias, init_bias=init_bias, dilation=1, groups=1, name='pointwise_kernel')
    activation_fn = get_activation(activation)
    if activation_fn is not None:

        x = activation_fn()(x)
    return x



def GcdConv2d_1(
            kernel_size=default_override_or((3, 3)),
            num_filters=default_override_or(None),
            strides=default_override_or((1, 1)),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=C.he_normal(0.02),
           use_bias=default_override_or(False),
           init_bias=default_override_or(0),
           divisor_rank=default_override_or(0),
           dilation=default_override_or(1),
           weights_contraint=default_override_or(default_constrains),
           self_norm=default_override_or(True),
           name=''):
    def internal_op(x):
        return gcd_conv2d_1(x,
            kernel_size,
            num_filters,
            strides,
           padding,
           activation,
           init,
           use_bias,
           init_bias,
           divisor_rank,
           dilation,
           weights_contraint,
                          self_norm,
           name)
    return internal_op

def gcd_conv2d_1(x,
            kernel_size=default_override_or((3, 3)),
            num_filters=default_override_or(None),
            strides=default_override_or((1, 1)),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=C.he_normal(0.02),
           use_bias=default_override_or(False),
           init_bias=default_override_or(0),
           divisor_rank=default_override_or(0),
           dilation=default_override_or(1),
           weights_contraint=default_override_or(default_constrains),
           self_norm=default_override_or(True),
           name=''):

    kernel_size = C.get_default_override(conv2d, kernel_size=kernel_size)
    num_filters = C.get_default_override(conv2d, num_filters=num_filters)

    strides= C.get_default_override(gcd_conv2d, strides=strides)
    padding = C.get_default_override(gcd_conv2d, padding=padding)
    activation = C.get_default_override(gcd_conv2d, activation=activation)
    init = C.get_default_override(gcd_conv2d, init=init)
    use_bias = C.get_default_override(gcd_conv2d, use_bias=use_bias)
    init_bias = C.get_default_override(gcd_conv2d, init_bias=init_bias)
    dilation = C.get_default_override(gcd_conv2d, dilation=dilation)
    divisor_rank = C.get_default_override(gcd_conv2d, divisor_rank=divisor_rank)
    weights_contraint = C.get_default_override(gcd_conv2d, weights_contraint=weights_contraint)
    self_norm= C.get_default_override(gcd_conv2d, self_norm=self_norm)

    input_num_filters = x.shape[0]
    reduction_rank = 1
    if len(list(x.shape)) == 2 or (len(list(x.shape)) == 3 and x.shape[0] == 1):
        reduction_rank = 0

    if num_filters is None:
        num_filters=input_num_filters
    num_filters_1 = input_num_filters
    num_filters_2 = num_filters
    gcd_list=[]
    gcd_list.extend(_gcd(input_num_filters,num_filters))

    gcd =1
    if len(gcd_list)==0:
        groups=input_num_filters
        num_filters_1=input_num_filters
    else:
        gcd = gcd_list[0]
        groups = gcd_list[min(int(divisor_rank), len(gcd_list))]
        num_filters_1 = gcd

    num_filters_2 = num_filters
    factors = _get_divisors(num_filters // gcd)
    if input_num_filters == num_filters:
        factors = _get_divisors(num_filters)

    if input_num_filters == gcd:
        groups = input_num_filters
        num_filters_1 = num_filters
    elif num_filters == gcd:
        groups = gcd
        num_filters_1 = num_filters
    elif len(factors) == 0:
        groups = gcd
        num_filters_1 = gcd
    else:
        num_filters_1 = gcd * factors[-1]



    num_filters_per_group = (int(input_num_filters // groups),)

    #: [56, 28, 14, 8, 7, 4, 2, 1]
    # 168=>121
    # divisor_rank=0   output filters 56   groups是  56  從每組3個像素選1個再縮小回2
    # divisor_rank=1   output filters 56   groups是 28 從每組6個像素選2個再縮小回2
    # divisor_rank=2  output filters 56   groups是 14 從每組12個像素選4個

    print('input:{0}   output:{1}->{2}  gcd:{3} group:{4}   放大因子:{5} '.format(input_num_filters, num_filters_2,
                                                                              num_filters_2, gcd, groups,
                                                                              num_filters_2 // groups))
    input_shape = x.shape
    x = C.reshape(x, (input_shape[0]// groups,groups, x.shape[1], x.shape[2]))
    if isinstance(kernel_size,tuple):
        if len(kernel_size)==2:
            kernel_size=(1,)+kernel_size
    else:
        kernel_size=(1,) + _as_tuple(kernel_size)+ _as_tuple(kernel_size)
    if isinstance(strides, tuple):
        if len(strides) == 2:
            strides = (1,) + strides
    else:
        strides = (1,) + _as_tuple(strides)+ _as_tuple(strides)
    if isinstance(dilation, tuple):
        if len(dilation) == 2:
            dilation = (1,) + dilation
    else:
        dilation = (1,) + _as_tuple(dilation)+ _as_tuple(dilation)

    x = Conv3d(kernel_size, num_filters=num_filters//groups, strides=strides, padding='same', activation=None,
               init=init, use_bias=use_bias, init_bias=init_bias, dilation=dilation, groups=1,
               input_num_filters=input_shape[0]//groups, op_name='GCD_Conv2d_1', weights_contraint=weights_contraint,
               gcd=gcd, name=name)(x)

    if self_norm:
        group_mean = C.reduce_mean(x, axis=[1, 2, 3])
        group_variance = C.reduce_mean(C.square(x - C.stop_gradient(group_mean)), axis=[1, 2, 3])
        group_std = C.sqrt(group_variance) + C.constant(1e-8)
        normed = (x - group_mean) / group_std
        x = C.reshape(normed,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))

    #x = conv2d(x, kernel_size=(1, 1), num_filters=num_filters_2, strides=1, padding='valid', activation=None, init=init, use_bias=use_bias, init_bias=init_bias, dilation=1, groups=1, name='pointwise_kernel')
    activation_fn = get_activation(activation)
    if activation_fn is not None:

        x = activation_fn()(x)
    return x



def gcd_conv2d1(x,
            kernel_size=default_override_or((3, 3)),
            num_filters=default_override_or(None),
            strides=default_override_or((1, 1)),
           padding=default_override_or('same'),
           activation=default_override_or(identity),
           init=C.xavier(0.1),
           use_bias=default_override_or(True),
           init_bias=default_override_or(0),
           divisor_rank=default_override_or(0),
           dilation=default_override_or(1),
           weights_contraint=default_override_or(default_constrains),
           name=''):
    groups = None
    reduction_rank = 1
    if len(list(x.shape)) == 2 or (len(list(x.shape)) == 3 and x.shape[0] == 1):
        reduction_rank = 0

    kernel_size = C.get_default_override(conv2d, kernel_size=kernel_size)
    num_filters = C.get_default_override(conv2d, num_filters=num_filters)
    strides = C.get_default_override(conv2d, strides=strides)
    padding = C.get_default_override(gcd_conv2d, padding=padding)
    activation = C.get_default_override(gcd_conv2d, activation=activation)
    init = C.get_default_override(gcd_conv2d, init=init)
    use_bias = C.get_default_override(gcd_conv2d, use_bias=use_bias)
    init_bias = C.get_default_override(gcd_conv2d, init_bias=init_bias)
    dilation = C.get_default_override(gcd_conv2d, dilation=dilation)
    divisor_rank = C.get_default_override(gcd_conv2d, divisor_rank=divisor_rank)
    weights_contraint = C.get_default_override(gcd_conv2d, weights_contraint=weights_contraint)

    input_num_filters = x.shape[0]
    if num_filters is None:
        num_filters=input_num_filters

    gcd_list=[]
    gcd_list.extend(_gcd(input_num_filters,num_filters))
   # divisor_rank=min(divisor_rank,len(gcd_list))
    gcd=gcd_list[0]
    num_filters_1=gcd_list[0]
    num_filters_2=num_filters
    if input_num_filters==gcd or num_filters==gcd :
        groups=1
        num_filters_1 = num_filters
    else:
        groups=gcd_list[min(int(divisor_rank),len(gcd_list))]

    print('input:{0}   output:{1}  gcd:{2}'.format(input_num_filters,num_filters_2,groups))
    #: [56, 28, 14, 8, 7, 4, 2, 1]
    #168=>121
    #divisor_rank=0   output filters 56   groups是  56  從每組3個像素選1個再縮小回2
    # divisor_rank=1   output filters 56   groups是 28 從每組6個像素選2個再縮小回2
    # divisor_rank=2  output filters 56   groups是 14 從每組12個像素選4個
    num_filters_per_group=input_num_filters//groups
    slice_list=[]
    for i in range(groups):
        slice_x=C.slice(x,0,i*num_filters_per_group,(i+1)*num_filters_per_group)
        slice_x = Conv2d(kernel_size,
                         num_filters=num_filters_1//groups,
                         strides=strides,
                         padding='same',
                         activation=activation,
                         init=init,
                         use_bias=use_bias,
                         init_bias=init_bias,
                         dilation=dilation,
                         groups=1,
                         input_filters=num_filters_per_group,
                         op_name='GCD_Conv2d_slice',
                         weights_contraint=weights_contraint,
                         name=name)(slice_x)
        slice_list.append(slice_x)
    x=splice(*slice_list,axis=0)
    x = conv2d(x, kernel_size=(1, 1), num_filters=num_filters_2, strides=1, padding='valid', activation=activation,
               init=init, use_bias=use_bias, init_bias=init_bias, dilation=1, groups=1, name='pointwise_kernel')
    return x









