from .load_backend import epsilon
from .load_backend import set_epsilon
from .load_backend import floatx
from .load_backend import set_floatx
from .load_backend import cast_to_floatx
from .load_backend import image_data_format
from .load_backend import set_image_data_format
# from .load_backend import reset_uids
# from .load_backend import get_uid
# from .load_backend import learning_phase
# from .load_backend import set_learning_phase
# from .load_backend import is_sparse
# from .load_backend import to_dense
# from .load_backend import variable
# from .load_backend import is_variable
# from .load_backend import constant
# from .load_backend import is_keras_tensor
# from .load_backend import is_tensor
# from .load_backend import placeholder
# from .load_backend import is_placeholder
# from .load_backend import shape
# from .load_backend import int_shape
# from .load_backend import ndim
# from .load_backend import dtype
# from .load_backend import eval
# from .load_backend import zeros
# from .load_backend import ones
# from .load_backend import eye
# from .load_backend import zeros_like
# from .load_backend import ones_like
# from .load_backend import identity
# from .load_backend import random_uniform_variable
# from .load_backend import random_normal_variable
# from .load_backend import count_params
# from .load_backend import cast
# from .load_backend import update
# from .load_backend import update_add
# from .load_backend import update_sub
# from .load_backend import moving_average_update
# from .load_backend import dot
# from .load_backend import batch_dot
# from .load_backend import transpose
# from .load_backend import gather
# from .load_backend import max
# from .load_backend import min
# from .load_backend import sum
# from .load_backend import prod
# from .load_backend import cumsum
# from .load_backend import cumprod
# from .load_backend import var
# from .load_backend import std
# from .load_backend import mean
# from .load_backend import any
# from .load_backend import all
# from .load_backend import argmax
# from .load_backend import argmin
# from .load_backend import square
# from .load_backend import abs
# from .load_backend import sqrt
# from .load_backend import exp
# from .load_backend import log
# from .load_backend import logsumexp
# from .load_backend import round
# from .load_backend import sign
# from .load_backend import pow
# from .load_backend import clip
# from .load_backend import equal
# from .load_backend import not_equal
# from .load_backend import greater
# from .load_backend import greater_equal
# from .load_backend import less
# from .load_backend import less_equal
# from .load_backend import maximum
# from .load_backend import minimum
# from .load_backend import sin
# from .load_backend import cos
# from .load_backend import normalize_batch_in_training
# from .load_backend import batch_normalization
# from .load_backend import concatenate
# from .load_backend import reshape
# from .load_backend import permute_dimensions
# from .load_backend import resize_images
# from .load_backend import resize_volumes
# from .load_backend import repeat_elements
# from .load_backend import repeat
# from .load_backend import arange
# from .load_backend import tile
# from .load_backend import flatten
# from .load_backend import batch_flatten
# from .load_backend import expand_dims
# from .load_backend import squeeze
# from .load_backend import temporal_padding
# from .load_backend import spatial_2d_padding
# from .load_backend import spatial_3d_padding
# from .load_backend import stack
# from .load_backend import one_hot
# from .load_backend import reverse
# from .load_backend import slice
# from .load_backend import get_value
# from .load_backend import batch_get_value
# from .load_backend import set_value
# from .load_backend import batch_set_value
# from .load_backend import print_tensor
# from .load_backend import function
# from .load_backend import gradients
# from .load_backend import stop_gradient
# from .load_backend import rnn
# from .load_backend import switch
# from .load_backend import in_train_phase
# from .load_backend import in_test_phase
# from .load_backend import relu
# from .load_backend import elu
# from .load_backend import softmax
# from .load_backend import softplus
# from .load_backend import softsign
# from .load_backend import categorical_crossentropy
# from .load_backend import sparse_categorical_crossentropy
# from .load_backend import binary_crossentropy
# from .load_backend import sigmoid
# from .load_backend import hard_sigmoid
# from .load_backend import tanh
# from .load_backend import dropout
# from .load_backend import l2_normalize
# from .load_backend import in_top_k
# from .load_backend import conv1d
# from .load_backend import separable_conv1d
# from .load_backend import conv2d
# from .load_backend import separable_conv2d
# from .load_backend import conv2d_transpose
# from .load_backend import depthwise_conv2d
# from .load_backend import conv3d
# from .load_backend import conv3d_transpose
# from .load_backend import pool2d
# from .load_backend import pool3d
# from .load_backend import bias_add
# from .load_backend import random_normal
# from .load_backend import random_uniform
# from .load_backend import random_binomial
# from .load_backend import truncated_normal
# from .load_backend import ctc_label_dense_to_sparse
# from .load_backend import ctc_batch_cost
# from .load_backend import ctc_decode
# from .load_backend import map_fn
# from .load_backend import foldl
# from .load_backend import foldr
# from .load_backend import local_conv1d
# from .load_backend import local_conv2d
from .load_backend import backend
from .load_backend import normalize_data_format
from .load_backend import name_scope



if backend() == 'pytorch':
    from .load_backend import pattern_broadcast
elif backend() == 'tensorflow':
    from .load_backend import clear_session
    from .load_backend import manual_variable_initialization
    from .load_backend import get_session
    from .load_backend import set_session
elif backend() == 'cntk':
    from .load_backend import clear_session