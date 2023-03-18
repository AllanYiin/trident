"""Activation Layers"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import builtins
import inspect

import string
from functools import partial
from pydoc import locate

import six
import jax
import jax.numpy as jnp
import jaxlib

from trident.backend.common import get_function, get_class, camel2snake,snake2camel, enforce_singleton,TensorShape
from trident.backend.jax_backend import Layer,Sequential,Parameter
from trident.backend.jax_ops import *
from trident.layers.jax_activations import get_activation

__all__ = ['Dense']


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
def get_layer_repr(layer):
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    if hasattr(layer, 'extra_repr') and callable(layer.extra_repr):
        extra_repr = layer.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
    child_lines = []
    if isinstance(layer,Layer) and layer.layers is not None:
        for module in layer.layers:
            mod_str = repr(module)
            mod_str = addindent(mod_str, 2)
            child_lines.append('(' + module.name + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = layer.__class__.__name__ + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str
class Dense(Layer):
    def __init__(self, num_filters, use_bias=True, activation=None,kernel_regularizer=None, keep_output=False, name=None, **kwargs):
        super(Dense, self).__init__(name=name,keep_output=keep_output)
        self.rank = 0
        if isinstance(num_filters, int):
            self.num_filters = num_filters
        elif isinstance(num_filters, tuple):
            self.num_filters = unpack_singleton(num_filters)
        else:
            raise ValueError('output_shape should be integer, list of integer or tuple of integer...')

        self.use_bias = use_bias
        if kernel_regularizer == 'l2':
            self.kernel_regularizer = l2_normalize
        else:
            self.kernel_regularizer = None

        self.activation = get_activation(activation)


    def build(self, input_shape:TensorShape):
        if not self._built:
            if len(input_shape.dims) == 1:
                self.input_filters = input_shape.dims[0]
            else:
                self.input_filters = input_shape[self.filter_index]
            self.register_parameter('weight',Parameter(data=random_normal(shape=(self.input_filters,self.num_filters), mean=0., std=0.2) , name='weight'))
            kaiming_uniform(self.weight, a=math.sqrt(5))
            if self.use_bias:
                self.register_parameter('bias',Parameter(data=random_normal(shape=(self.num_filters), mean=0., std=0.002) , name='bias'))
            self._built = True

    def forward(self, x, **kwargs) :

        if hasattr(self, 'kernel_regularizer') and self.kernel_regularizer is not None:
            x = self.kernel_regularizer(self.weight)@x
        else:
            x =self.weight@x
        if self.use_bias:
            x=x+ self.bias

        if self.activation is not None:
            x = self.activation(x)
        return x