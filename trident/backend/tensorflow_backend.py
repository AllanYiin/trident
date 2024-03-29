from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import builtins
import copy
from copy import deepcopy
import functools
import gc
import inspect
import itertools
import numbers
import operator
import os
import sys
import typing
import uuid
import weakref
from collections import abc
from collections import defaultdict, namedtuple
from distutils.version import LooseVersion
from functools import partial
from itertools import islice
from types import MethodType
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List, \
    Iterable

import numpy as np
import tensorflow as tf

# from tensorflow.python import enable_eager_execution

tf.executing_eagerly()

from tensorflow.python.framework import ops

from tensorflow.python.module import module
from tensorflow.python.util import object_identity
from trident.backend import iteration_tools
from trident.backend.common import camel2snake, to_list, unpack_singleton, enforce_singleton, OrderedDict, Signature, \
    PrintException, TensorShape, \
    get_args_spec, is_instance
from trident.backend.tensorflow_ops import *
from trident.backend import tensorflow_ops as tops
from trident.backend import dtype as Dtype

_FUN_NAMES = [
    ('float', tops.float),
    ('long', tops.long),
    ('equal', tops.equal),
    ('int', tops.int),
    ('to', tops.to)]
for target_fun_name, source_fun in _FUN_NAMES:
    setattr(Tensor, target_fun_name, source_fun)

from trident.backend.tensorspec import *
from trident.data.utils import pickle_it
from trident.backend import tensorflow_serialization as serialization
from trident import context

__all__ = ['set_device', 'DTYPE_MAPPING', 'Layer', 'get_device', 'Parameter', 'Sequential', 'ModuleList', 'ModuleDict',
           'summary', 'normalize_padding', 'load', 'save', 'try_map_args_and_call',
           'fix_layer', 'fix_keras_module']

ctx = context._context()

DTYPE_MAPPING = {
    tf.bool: Dtype.bool,
    tf.int8: Dtype.int8,
    tf.int16: Dtype.int16,
    tf.int32: Dtype.int32,
    tf.int64: Dtype.int64,
    tf.uint8: Dtype.uint8,
    tf.float16: Dtype.float16,
    tf.float32: Dtype.float32,
    tf.float64: Dtype.float64,
    tf.complex64: Dtype.complex64,
    tf.complex128: Dtype.complex128,

}


def get_device():
    """get current device

    Returns: device string ('cpu', 'cuda)

    """
    if ctx.device is None or ctx.device == 'cuda':
        set_device('/gpu:0' if len(tf.config.list_physical_devices('GPU')) > 0 else "/cpu:0")
    return ctx.device


def set_device(device='/cpu:0'):
    if device.lower() == 'cuda' or device.lower() == 'gpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        ctx.device = '/gpu:0'
    if device.lower() == 'cpu':
        os.environ.pop("CUDA_VISIBLE_DEVICES")
        ctx.device = '/cpu:0'
    if 'gpu' in device and len(tf.config.list_physical_devices('GPU')) == 0:
        raise ValueError('Gpu is not available...')
    try:

        gcitems = gc.get_objects()
        for i in range(len(gcitems)):
            item = gcitems[i]
            try:
                if not inspect.isclass(item) and is_tensor(item):
                    with tf.device(ctx.device):
                        item = tf.identity(item)
                elif not inspect.isclass(item) and is_instance(item, 'trident.backend.tensorflow_backend.Layer'):
                    item.to(device)
                elif not inspect.isclass(item) and is_instance(item, 'Layer'):
                    with tf.device(ctx.device):
                        item = tf.identity(item)
            except Exception as e:
                print(e)

    except Exception as e:
        print(e)


version = tf.version
sys.stdout.write('Tensorflow version:{0}.\n'.format(version.VERSION))

tf_version = LooseVersion(vstring=version.VERSION)
base_version = LooseVersion(vstring='2.2.0-rc0')

if tf_version.version < base_version.version:
    raise ValueError('trident only support Tensorflow 2.2.0-rc0 or newer.\n')

try:

    tf.executing_eagerly()
    sys.stdout.write('executing_eagerly\n')
except Exception as e:
    sys.stdout.write('executing_eagerly fail. {0}\n'.format(e))

sys.stdout.write('use device:{0}.\n'.format(get_device()))


def load(path):
    """load model from *.pth or *.pth.tar

    Args:
        path (str):

    Returns:

    """
    with tf.device(get_device()):
        if '.tar' in path:
            return serialization.load_pthtar(path)
        else:
            return serialization.load(path)


def save(obj, path, is_compressed=False):
    serialization.save(obj, path, is_compressed=is_compressed)
    return True


class RemovableHandle(object):
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (dict): An additional dictionary whose keys will be deleted
            when the same keys are removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref = (
            weakref.ref(extra_dict)
            if extra_dict is not None
            else None
        )

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        if self.extra_dict_ref is not None:
            extra_dict = self.extra_dict_ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        return (
            (self.hooks_dict_ref(), self.id)
            if self.extra_dict_ref is None
            else (self.hooks_dict_ref(), self.id, self.extra_dict_ref())
        )

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        self.extra_dict_ref = (
            None
            if len(state) < 3
            else weakref.ref(OrderedDict() if state[2] is None else state[2])
        )

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()


def _is_not_trainable_variable(obj):
    return module._is_variable(obj) and not getattr(obj, "trainable", False)


# The internal graph maintained by Keras and used by the symbolic Keras APIs
# while executing eagerly (such as the functional API for model-building).
# This is thread-local to allow building separate models in different threads
# concurrently, but comes at the cost of not being able to build one model
# across threads.


# A global dictionary mapping graph objects to an index of counters used
# for various layer/optimizer names in each graph.
# Allows to give unique autogenerated names to layers, in a graph-specific way.
PER_GRAPH_OBJECT_NAME_UIDS = weakref.WeakKeyDictionary()


def reset_name(module: tf.Module, prefix_dict=None):
    def get_uid(prefix, seq):
        if prefix not in module._uid_prefixs or seq < module._uid_prefixs[prefix]:
            module._uid_prefixs[prefix] = seq
        return module._uid_prefixs[prefix]

    if not hasattr(module, '_uid_prefixs') or prefix_dict is not None:
        module._uid_prefixs = prefix_dict
    if not hasattr(module, 'default_name'):
        module.default_name = camel2snake(module.__class__.__name__) + '_' + str(
            get_global_uid(camel2snake(module.__class__.__name__)))
    prefix, seq = module.default_name.rsplit('_', 1)  # if '_' in module.default_name else
    seq = int(seq)
    module.default_name = prefix + '_' + str(seq - get_uid(prefix, seq) + 1)
    if module._name is None:
        module._name = module.default_name
    module.__name__ = module._name
    module.update_name_scope(module._name)


_UID_PREFIX = defaultdict(int)


def get_global_uid(prefix=''):
    _UID_PREFIX[prefix] += 1
    return _UID_PREFIX[prefix]


class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


r"""This tracks hooks common to all modules that are executed before/after
calling forward and backward. This is global state used for debugging/profiling
purposes"""
_global_backward_hooks = OrderedDict()
_global_forward_pre_hooks = OrderedDict()
_global_forward_hooks = OrderedDict()

_global_backward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_is_full_backward_hook: Optional[bool] = None

_grad_t = Union[Tuple[Tensor, ...], Tensor]
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar('T', bound='Layer')

_EXTRA_STATE_KEY_SUFFIX = '_extra_state'


def register_module_buffer_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a buffer registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_buffer` is invoked.
    It should have the following signature::

        hook(module, name, buffer) -> None or new buffer

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.id] = hook
    return handle


def register_module_module_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a module registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_module` is invoked.
    It should have the following signature::

        hook(module, name, submodule) -> None or new submodule

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle


def register_module_parameter_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a parameter registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_parameter` is invoked.
    It should have the following signature::

        hook(module, name, param) -> None or new parameter

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_parameter_registration_hooks)
    _global_parameter_registration_hooks[handle.id] = hook
    return handle


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a forward pre-hook common to all modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Registers a global forward hook for all the modules

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the output. It can modify the input inplace but
    it will not have effect on forward since this is called after
    :func:`forward` is called.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    handle = RemovableHandle(_global_forward_hooks)
    _global_forward_hooks[handle.id] = hook
    return handle


def register_module_backward_hook(
        hook: Callable[['Layer', _grad_t, _grad_t], Union[None, _grad_t]]
) -> RemovableHandle:
    r"""Registers a backward hook common to all the modules.

    This function is deprecated in favor of
    :func:`torch.nn.modules.module.register_module_full_backward_hook`
    and the behavior of this function will change in future versions.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is True:
        raise RuntimeError("Cannot use both regular backward hooks and full backward hooks as a "
                           "global Module hook. Please use only one of them.")

    _global_is_full_backward_hook = False

    handle = RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


def register_module_full_backward_pre_hook(
        hook: Callable[['Layer', _grad_t], Union[None, _grad_t]]
) -> RemovableHandle:
    r"""Registers a backward pre-hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time the gradients for the module are computed.
    The hook should have the following signature::

        hook(module, grad_output) -> Tensor or None

    The :attr:`grad_output` is a tuple. The hook should
    not modify its arguments, but it can optionally return a new gradient with
    respect to the output that will be used in place of :attr:`grad_output` in
    subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
    all non-Tensor arguments.

    For technical reasons, when this hook is applied to a Module, its forward function will
    receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
    of each Tensor returned by the Module's forward function.

    Global hooks are called before hooks registered with `register_backward_pre_hook`

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    handle = RemovableHandle(_global_backward_pre_hooks)
    _global_backward_pre_hooks[handle.id] = hook
    return handle


def register_module_full_backward_hook(
        hook: Callable[['Layer', _grad_t, _grad_t], Union[None, _grad_t]]
) -> RemovableHandle:
    r"""Registers a backward hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time the gradients with respect to a module
    are computed, i.e. the hook will execute if and only if the gradients with
    respect to module outputs are computed. The hook should have the following
    signature::

        hook(module, grad_input, grad_output) -> Tensor or None

    The :attr:`grad_input` and :attr:`grad_output` are tuples. The hook should
    not modify its arguments, but it can optionally return a new gradient with
    respect to the input that will be used in place of :attr:`grad_input` in
    subsequent computations. :attr:`grad_input` will only correspond to the inputs given
    as positional arguments and all kwarg arguments will not appear in the hook. Entries
    in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
    arguments.

    For technical reasons, when this hook is applied to a Module, its forward function will
    receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
    of each Tensor returned by the Module's forward function.

    Global hooks are called before hooks registered with `register_backward_hook`

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is False:
        raise RuntimeError("Cannot use both regular backward hooks and full backward hooks as a "
                           "global Module hook. Please use only one of them.")

    _global_is_full_backward_hook = True

    handle = RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.  See also
# https://github.com/python/mypy/issues/8795
def _forward_unimplemented(self, *input: Any) -> None:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")


# def Parameter(data, trainable=True, dtype=None, name=None, **kwargs):
#     if dtype is None:
#         dtype = tf.float32
#     return tf.Variable(initial_value=cast(data, dtype), trainable=trainable, name=name)


def Parameter(data, trainable=True, name=None):
    return tf.Variable(initial_value=data, trainable=trainable, dtype=data.dtype, name=name)


class Layer(tf.Module):
    """Trident extened tf.Module as base layer class.

   Your models should also subclass of this class.
    Layer contains :
        modules: another child layer(module) in it.
        parameters: the trainable parameters in the layer.
        buffers: the other non_trainable tensor in the layer.


    Attributes :
        training (bool): If True, means in the training phase, else in the infer or evaluation phase.

        rank (int): The number of the spatial related axes.

        _modules (OrderedDict) : storage of all the sub-modules.

        _parameters (OrderedDict) : storage of all the tranable weights.

        _buffers (OrderedDict) : storage of all the non-trainable tensor.

        _forward_hooks (OrderedDict) : storage of all the hooks triggered before the forward execution.

        _forward_pre_hooks (OrderedDict) : storage of all the hooks triggered  after the forward execution.

        _state_dict_hooks (OrderedDict) : storage of all the hooks triggered  when state_dict generating  execution.

        _load_state_dict_pre_hooks (OrderedDict) : storage of all the hooks triggered  when loading state_dict   execution.

        input_filters (int): input channels

        signature (int): the function signature of this layer.

        default_name: default_name is the same concept as in keras, it comes from class name with sequence number.

        relative_name:relative_name is the same concept as named_modules in pytorch. But in pytorch, you need to get the name from generator enumeration. In trident, you can access the relative name  with this attribute.

    References:
        https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/module/module.py#L35-L291


    """
    _version = 1
    _built: bool
    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: Set[str]
    _backward_pre_hooks: Dict[int, Callable]
    _backward_hooks: Dict[int, Callable]
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support Set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_hooks_with_kwargs: Dict[int, bool]
    _forward_pre_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support Set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_pre_hooks_with_kwargs: Dict[int, bool]
    _state_dict_hooks: Dict[int, Callable]
    _load_state_dict_pre_hooks: Dict[int, Callable]
    _state_dict_pre_hooks: Dict[int, Callable]
    _load_state_dict_post_hooks: Dict[int, Callable]
    _modules: Dict[str, Optional['Module']]

    def __init__(self,
                 name=None,
                 keep_output=False,
                 device=None,
                 dtype=None,
                 **kwargs):
        """
        Args:
            name (str) :name of the layer.
            keep_output (bool) :whether you need to kept output tensor in execution time.


        """
        object.__setattr__(self, 'uuid', uuid.uuid4().node)
        super().__setattr__('training', True)
        super().__setattr__('_built', False)

        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_buffers', OrderedDict())
        super().__setattr__('_non_persistent_buffers_set', set())
        super().__setattr__('_backward_pre_hooks', OrderedDict())
        super().__setattr__('_backward_hooks', OrderedDict())
        super().__setattr__('_is_full_backward_hook', None)
        super().__setattr__('_forward_hooks', OrderedDict())
        super().__setattr__('_forward_hooks_with_kwargs', OrderedDict())
        super().__setattr__('_forward_pre_hooks', OrderedDict())
        super().__setattr__('_forward_pre_hooks_with_kwargs', OrderedDict())
        super().__setattr__('_state_dict_hooks', OrderedDict())
        super().__setattr__('_state_dict_pre_hooks', OrderedDict())
        super().__setattr__('_load_state_dict_pre_hooks', OrderedDict())
        super().__setattr__('_load_state_dict_post_hooks', OrderedDict())
        super().__setattr__('_modules', OrderedDict())
        super().__setattr__('factory_kwargs', {'device': get_device(), 'dtype': Dtype.float32})


        prefix = self.__class__.__name__
        self._uid_prefixs = {}
        self._name = name
        self.is_root = True
        self.default_name = camel2snake(prefix) + '_' + str(get_global_uid(camel2snake(prefix)))
        self.relative_name = ''
        reset_name(self, self._uid_prefixs)

        super(Layer, self).__init__()

        self.batch_index = 0
        self.filter_index = -1


        self.rank = kwargs.get('rank', None)

        self._nodes = OrderedDict()
        self._input_shape: Optional[None, TensorShape, List[TensorShape]] = None
        self._output_shape: Optional[None, TensorShape, List[TensorShape]] = None

        with self.name_scope:

            self._input_shape = None
            self._output_shape = None

            self.input_filters = None
            self.input_spec = None

            self.keep_output = keep_output
            self._output_tensor = None
            self._signature = None

            self.dump_patches = True



    # Trick mypy into not applying contravariance rules to inputs by defining
    # forward as a value, rather than a function.  See also
    # https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward_unimplemented

    # def forward(self, *input, **kwargs):
    #     raise NotImplementedError

    def get_root(self):
        if not hasattr(self, '_nodes') or self._nodes is None:
            self.is_root = True
            return self
        elif len(self._nodes) > 0 and self._nodes.value_list[0].is_root:
            return self._nodes.value_list[0]
        else:
            for name, node in self._nodes.item_list:
                if node.is_root:
                    return node
            return self

    @property
    def name(self):
        """Name of the layer (string), set in the constructor."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.update_name_scope(self.name)

    # def _set_name_scope(self):
    def update_name_scope(self, name):
        self._name = name
        with ops.name_scope_v2(name) as scope_name:
            self._name_scope = ops.name_scope_v2(scope_name)

    @property
    def nodes(self):
        """The whole tree structured OrderedDict { uuid : module } , for module to access any node in this structures, ex. Shortcut"""
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        if self._nodes != value:
            self._nodes = value
            for mod in self.modules():
                mod._nodes = value

    def add_module(self,  name: str, module: Optional['Module']) -> None:
        """Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.
        1) add module as child
        2) generate default_name and relative_name
        3) update the nodes

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.

        """
        if name is None or len(name) == 0:
            name = module._name

        if module is None:
            raise KeyError("module  can't be None")
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(type(name).__name__))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")

        self._modules[name] = module
        self.nodes = OrderedDict([(mod.uuid, mod) for mod in list(self.modules()) if isinstance(mod, Layer)])
        if isinstance(module, Layer):
            for mod in module.modules():
                mod.nodes = self.nodes
                mod.is_root = False
                reset_name(mod, self._uid_prefixs)
                mod.relative_name = name if mod.relative_name == '' else name + '.' + mod.relative_name

        # elif inspect.isfunction(module) or callable(module):
        #     module.__name__ = name
        #     self._modules[name] = module

    def add(self, module):
        """Simplified 'add_module'

        Use the count of child modules as the default name.

        Args:
            module (Module): child module to be added to the module.

        """
        if module is None:
            raise KeyError("module  can't be None")
        elif isinstance(module, Layer):
            self.add_module(str(len(self._modules)),
                            module)  # self.nodes = nodes  # for mod in self.modules():  #     mod.nodes = nodes

        else:
            raise ValueError('Not valid module')

    def build(self, *input_shape: TensorShape):
        """ Do the shape inference and initialize weights and bias.

        `build' is a key method in trident, you can use  property `built' to check whether the layer do the build process.
        In build' , we need to put all the logics about  how to comfirm the shape of outputs, weights and bias according to the coming input tensor.

        Args:
            input_shape (tensor):  the shape representation exclude the batch axis.

        """
        pass

    def rebuild(self, *input_shape: TensorShape):
        """ Do the shape inference and initialize weights and bias.

        `build' is a key method in trident, you can use  property `built' to check whether the layer do the build process.
        In build' , we need to put all the logics about  how to comfirm the shape of outputs, weights and bias according to the coming input tensor.

        Args:
            input_shape (tensor):  the shape representation exclude the batch axis.

        """
        print(
            'Your model will start to rebuild, it will cause lost all existing trainable parameters, will you want to rebuild it?')
        ans = input('(Y/N) << ').lower()
        if ans in ['yes', 'y']:
            for name, module in self.named_modules():
                if module.trainable:
                    module._input_shape = None
                    module._output_shape = None
                    module._built = False
                    module._parameters = OrderedDict()
            dummay_input = to_tensor(input_shape.get_dummy_tensor()).to(get_device())
            out = self.forward(dummay_input)

    def register_backward_hook(self, hook):
        r"""Registers a backward hook on the module.

        The hook will be called every time the gradients with respect to module
        inputs are computed. The hook should have the following signature::

            hook(module, grad_input, grad_output) -> Tensor or None

        The :attr:`grad_input` and :attr:`grad_output` may be tuples if the
        module has multiple inputs or outputs. The hook should not modify its
        arguments, but it can optionally return a new gradient with respect to
        input that will be used in place of :attr:`grad_input` in subsequent
        computations.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        .. warning ::

            The current implementation will not have the presented behavior
            for complex :class:`Module` that perform many operations.
            In some failure cases, :attr:`grad_input` and :attr:`grad_output` will only
            contain the gradients for a subset of the inputs and outputs.
            For such :class:`Module`, you should use :func:`torch.Tensor.register_hook`
            directly on a specific input or output to get the required gradients.

        """
        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle


    def register_forward_pre_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...]], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.


        If ``with_kwargs`` is false or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        input. User can either return a tuple or a single modified value in the
        hook. We will wrap the value into a tuple if a single value is returned
        (unless that value is already a tuple). The hook should have the
        following signature::

            hook(module, args) -> None or modified input

        If ``with_kwargs`` is true, the forward pre-hook will be passed the
        kwargs given to the forward function. And if the hook modifies the
        input, both the args and kwargs should be returned. The hook should have
        the following signature::

            hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``forward_pre`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward_pre`` hooks
                on this :class:`torch.nn.modules.Module`. Note that global
                ``forward_pre`` hooks registered with
                :func:`register_module_forward_pre_hook` will fire before all
                hooks registered by this method.
                Default: ``False``
            with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
                given to the forward function.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(
            self._forward_pre_hooks,
            extra_dict=self._forward_pre_hooks_with_kwargs
        )
        self._forward_pre_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def register_forward_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.

        If ``with_kwargs`` is ``False`` or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        output. It can modify the input inplace but it will not have effect on
        forward since this is called after :func:`forward` is called. The hook
        should have the following signature::

            hook(module, args, output) -> None or modified output

        If ``with_kwargs`` is ``True``, the forward hook will be passed the
        ``kwargs`` given to the forward function and be expected to return the
        output possibly modified. The hook should have the following signature::

            hook(module, args, kwargs, output) -> None or modified output

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If ``True``, the provided ``hook`` will be fired
                before all existing ``forward`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward`` hooks on
                this :class:`torch.nn.modules.Module`. Note that global
                ``forward`` hooks registered with
                :func:`register_module_forward_hook` will fire before all hooks
                registered by this method.
                Default: ``False``
            with_kwargs (bool): If ``True``, the ``hook`` will be passed the
                kwargs given to the forward function.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(
            self._forward_hooks,
            extra_dict=self._forward_hooks_with_kwargs
        )
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def _get_name(self):
        return self.__class__.__name__

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""Adds a persistent buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the persistent state.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor): buffer to be registered.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Examples:

            >>> self.register_buffer('running_mean', tf.zeros([5]))

        """
        if '_buffers' not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(type(name).__name__))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, tf.Tensor) and not is_tensor(tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(tensorflow Tensor or None required)".format(type(tensor).__name__, name))
        else:
            with self.name_scope:
                self._buffers[name] = tensor
                if persistent:
                    self._non_persistent_buffers_set.discard(name)
                else:
                    self._non_persistent_buffers_set.add(name)

    def register_parameter(self,  name: str, param: Optional[tf.Variable]) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter): parameter to be added to the module.

        """

        if '_parameters' not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(type(name).__name__))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, tf.Variable):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(tf.Variable or None required)".format(type(param).__name__, name))
        else:
            self._parameters[name] = param

    def to(self: T, *args) -> T:
        device = None
        dtype = None
        non_blocking = None
        for arg in args:
            if 'cpu' in arg.lower() or 'gpu' in arg.lower() or 'cudu' in arg.lower():
                device = arg
            elif arg.lower() in ['float', 'int', 'bool', 'half', 'long']:
                dtype = arg
            elif isinstance(arg, bool):
                non_blocking = arg

        if device is None:
            device = get_device()
        if dtype is None and len(self.weights) > 0:
            dtype = self.weights[0].dtype
        if 'cpu' in device:
            self.cpu()
        elif 'gpu' in device or 'cuda' in device:
            self.cuda()
        for module in self.modules():
            try:
                if module._parameters is not None and len(module._parameters) > 0:
                    for name, para in module._parameters.items():
                        if para is None:
                            module._parameters[name] = None
                        else:
                            module._parameters[name].assign(tf.identity(cast(para.value(), dtype)))
                if module._buffers is not None and len(module._buffers) > 0:
                    for name, buff in module._buffers.items():
                        if buff is None:
                            module._buffers[name] = None
                        else:
                            module._buffers[name] = tf.identity(cast(buff, dtype))
                module.device = self.device
            except Exception as e:
                print(e)
                PrintException()

    def cuda(self: T, device: Optional[Union[int, str]] = None) -> T:
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """

        if tf.test.is_gpu_available:
            if self.get_root().device != '/gpu:0':
                self.get_root().device = '/gpu:0'
                with tf.device(self.device):
                    return self._apply(lambda t: tf.identity(t))

        else:
            sys.stderr.write('GPU is not available in this machone./n')

    def cpu(self: T) -> T:
        r"""Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        if self.get_root().device != '/cpu:0':
            with tf.device('/cpu:0'):
                self.get_root().device = '/cpu:0'
                return self._apply(lambda t: tf.identity(t))

    def gpu(self: T, device: Optional[Union[int, str]] = None) -> T:
        r"""Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """

        if tf.test.is_gpu_available:
            if self.get_root().device != '/gpu:0':
                self.get_root().device = '/gpu:0'
                with tf.device(self.device):
                    return self._apply(lambda t: tf.identity(t))

        else:
            sys.stderr.write('GPU is not available in this machone./n')

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for key, param in self._parameters.items():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't want to
                # track autograd history of `param_applied`, so we have to use
                # `with torch.no_grad():`
                param_applied = fn(param)
                if isinstance(param, tf.Variable):
                    self._parameters[key] = Parameter(param_applied, trainable=param.trainable)
                else:
                    param = param_applied
                #
                # if param.grad is not None:
                #     grad_applied = fn(param.grad)
                #     should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                #     if should_use_set_data:
                #         param.grad.data = grad_applied
                #     else:
                #         assert param.grad.is_leaf
                #         self._parameters[key].grad = grad_applied.requires_grad_(param.grad.requires_grad)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        r"""Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[ 1.,  1.],
                    [ 1.,  1.]])
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[ 1.,  1.],
                    [ 1.,  1.]])
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    #
    # def cuda(self: T, device: Optional[Union[int, str]] = None) -> T:
    #     r"""Moves all model parameters and buffers to the GPU.
    #
    #     This also makes associated parameters and buffers different objects. So
    #     it should be called before constructing optimizer if the module will
    #     live on GPU while being optimized.
    #
    #     Arguments:
    #         device (int, optional): if specified, all parameters will be
    #             copied to that device
    #
    #     Returns:
    #         Module: self
    #     """
    #     with tf.device('/gpu:0'):
    #         return self._apply(lambda t: t)
    #     #return self._apply(lambda t: t.cuda(device))

    @property
    def device(self):
        _root = self.get_root()
        if self.is_root or _root is None:
            if len(self.weights) == 0:
                return self.factory_kwargs['device']
            else:
                return self.weights[0].device.type
        else:
            return _root.device

    @device.setter
    def device(self, value):
        # if isinstance(value, tf.device):
        value =value.__str__()
        self.factory_kwargs['device'] = value
        self.to(value)




    @property
    def built(self):
        return self._built

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):

        if isinstance(value, tf.TensorShape):
            value = TensorShape(value.as_list())
        elif is_tensor(value) and value.ndim == 1 and value.dtype == Dtype.int32:
            value = TensorShape([None, ] + to_list(to_numpy(value)))
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all(
                [isinstance(item, numbers.Integral) for item in value]):
            value = TensorShape((None,) + value)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and all(
                [is_tensor(item) and ndim(item) == 1 and item.dtype == tf.int32 for item in value]):
            value = [TensorShape(to_list(to_numpy(sh))) for sh in value]
        elif isinstance(value, TensorShape):
            pass
        else:
            value = TensorShape(list(value))

        if self._built == False or self._input_shape is None or self.input_filters is None:
            self._input_shape = value
            self.input_filters = self._input_shape[self.filter_index]
            self.build(value)
            self._built = True

            if self.is_root:
                if self._signature is None:
                    self._signature = Signature(name=self.name)
                self._signature.inputs = OrderedDict()
                if isinstance(self._input_shape, TensorShape):
                    self._signature.inputs['input'] = TensorSpec(shape=self._input_shape, name='input')

                elif isinstance(self._input_shape, list):
                    for k in range(len(self._input_shape)):
                        self._signature.inputs['input_{0}'.format(k)] = TensorSpec(shape=self._input_shape[k],
                                                                                   name='input_{0}'.format(k))

    @property
    def output_shape(self):
        return self._output_shape

    @output_shape.setter
    def output_shape(self, value):
        if value is None:
            self._output_shape = value
            self._signature = None
        else:
            if is_tensor(value) and value.ndim == 1 and value.dtype == Dtype.int32:
                value = TensorShape([None, ] + to_list(to_numpy(value)))
            elif isinstance(value, tf.TensorShape):
                value = TensorShape(value.as_list())
            elif isinstance(value, (list, tuple)) and len(value) > 0 and all(
                    [isinstance(item, numbers.Integral) for item in value]):
                value = TensorShape((None,) + value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and all(
                    [is_tensor(item) and ndim(item) == 1 and item.dtype == tf.int32 for item in value]):
                value = [TensorShape(to_list(to_numpy(sh))) for sh in value]
            elif isinstance(value, TensorShape):
                pass
            else:
                value = TensorShape(list(value))

            self._output_shape = value

            if self.is_root:
                if self._signature is None:
                    self._signature = Signature(name=self.name)
                self._signature.outputs = OrderedDict()
                if isinstance(self._output_shape, TensorShape):
                    self._signature.outputs['output'] = TensorSpec(shape=self._output_shape, name='output')

                elif is_tensor(self._output_shape):
                    self._signature.outputs['output'] = TensorSpec(
                        shape=TensorShape(to_list(to_numpy(self._output_shape))), name='output')
                else:
                    for k in range(len(self._output_shape)):
                        self._signature.outputs['output_{0}'.format(k)] = TensorSpec(shape=self._output_shape[k],
                                                                                     name='output_{0}'.format(k))

    @property
    def signature(self):
        """

        Returns:

        """

        arg_spec = get_args_spec(self.forward)
        inspect_args = [arg for arg in list(arg_spec.args) if arg not in ['self', 'kwargs']]
        if isinstance(arg_spec.varargs, str):
            inspect_args.append(arg_spec.varargs)
        inspect_args = unpack_singleton(inspect_args)

        if self._signature is None or len(self._signature) == 0 or len(self._signature.inputs) == 0:
            self._signature = Signature(name=self.name)

            if self._input_shape is not None:
                if isinstance(self._input_shape, TensorShape) and isinstance(inspect_args, str):
                    self._signature.inputs[inspect_args] = TensorSpec(shape=TensorShape(self._input_shape),
                                                                      name=inspect_args)

                elif isinstance(self._input_shape, tuple):
                    for i in range(len(self._input_shape)):
                        self._signature.inputs["input_{0}".format(i)] = TensorSpec(
                            shape=TensorShape(self._input_shape[i]), name="input_{0}".format(i))
            else:
                for arg in inspect_args:
                    self._signature.inputs[arg] = TensorSpec(shape=None)

            if self._output_shape is not None:
                if isinstance(self._output_shape, TensorShape):
                    self._signature.outputs["output"] = TensorSpec(shape=TensorShape(self._output_shape), name="output")
                elif isinstance(self._output_shape, tuple):
                    for i in range(len(self._output_shape)):
                        self._signature.outputs["output_{0}".format(i)] = TensorSpec(
                            shape=to_tensor(self._output_shape[i]), name="output_{0}".format(i))
            else:
                self._signature.outputs["output"] = TensorSpec(shape=None)
        if isinstance(inspect_args, str) and len(self._signature.inputs) == 1 and self._signature.inputs.key_list[0] != inspect_args:
            self._signature.inputs[inspect_args] = self._signature.inputs.value_list[0]
            self._signature.inputs.pop(self._signature.inputs.key_list[0])
        elif isinstance(inspect_args, list) and len(self._signature.inputs) == len(inspect_args):
            for k1, k2 in zip(inspect_args, self._signature.inputs.key_list.copy()):
                if k1 != k2:
                    self._signature.inputs[k1] = self._signature.inputs[k2]
                    self._signature.inputs.pop(k2)
        return self._signature

    @signature.setter
    def signature(self, value):
        self._signature = value

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.
            for memory saving issue, we don'tb prefer to keep every input/output
            tensor in every layer.You should set self.keep_output flag to True, and then
            retrive the output tensor when the calll() is executing.
        Returns
                Output tensor or list of output tensors.
     Raises
                AttributeError: if the layer is connected to
                more than one incoming layers.
        """
        if self.keep_output == False:
            raise ValueError('Layer {0} has not set self.keep_output  to True, cannot access output '.format(self.name))
        return list(self._output_tensor) if isinstance(self._output_tensor, tuple) else self._output_tensor

    def reset_parameters(self):
        pass

    def copy(self):
        """Create a new FrozenDict with additional or replaced entries."""
        sig = get_signature(type(self).__init__)
        _args = OrderedDict()
        for inp in sig.inputs.key_list:
            if inp in self.__dict__:
                _args[inp] = self.__dict__[inp]
        shadow = type(self)(**_args)
        shadow.build(self.input_shape)
        for k, v in self.__dict__.items():
            if k not in _args and k not in ['_modules', '_parameters', '_buffers']:
                if is_tensor(v):
                    shadow.__dict__[k] = to_tensor(to_numpy(v), dtype=v.dtype, device=v.device,
                                                   requires_grad=v.requires_grad)
                elif isinstance(v, (str, bool, numbers.Number)) or k in ['_nodes']:
                    setattr(shadow, k, v)
                else:
                    setattr(shadow, k, deepcopy(v))
        shadow.load_state_dict(self.state_dict())
        shadow.to(self.device)
        return shadow

        # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
        # back that same object. But if they pass nothing, an `OrederedDict` is created and returned.



    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`Layer.state_dict`.
        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.
        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.value().detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.copy().detach()

    T_destination = TypeVar('T_destination', bound=Dict[str, Any])
    @overload
    def state_dict(self,  destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    # TODO: annotate with OrderedDict not Dict, but there is a problem:
    # https://docs.python.org/3/library/typing.html#typing.OrderedDict
    @overload
    def state_dict(self,  *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
        ...

    def state_dict(self,  destination=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        with tf.device(self.get_root().device):
            if destination is None:
                destination = OrderedDict()
                destination._metadata = OrderedDict()
            destination._metadata[prefix[:-1]] = local_metadata = dict(version=self._version)

            self._save_to_state_dict(destination, prefix, keep_vars)
            for name, module in self._modules.items():
                if module is not None:
                    module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
            for hook in self._state_dict_hooks.values():
                hook_result = hook(self, destination, prefix, local_metadata)
                if hook_result is not None:
                    destination = hook_result
            return destination

    def _register_load_state_dict_pre_hook(self, hook):
        r"""These hooks will be called with arguments: `state_dict`, `prefix`,
        `local_metadata`, `strict`, `missing_keys`, `unexpected_keys`,
        `error_msgs`, before loading `state_dict` into `self`. These arguments
        are exactly the same as those of `_load_from_state_dict`.
        """
        handle = RemovableHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.id] = hook
        return handle

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~tensorflow.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~tensorflow.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~tensorflow.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                try:
                    param.assign(to_tensor(input_param))

                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occured : {}.'
                                      .format(key, numel(param), numel(input_param), ex.args))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(self, state_dict, strict=True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~tensorflow.nn.Module.state_dict` function.
        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~tensorflow.nn.Module.state_dict` function. Default: ``True``
        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        with tf.device(self.get_root().device):
            missing_keys = []
            unexpected_keys = []
            error_msgs = []

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys,
                                             error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            load(self)
            load = None  # break load->load reference cycle

            if strict:
                if len(unexpected_keys) > 0:
                    error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
                if len(missing_keys) > 0:
                    error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

            # if len(error_msgs) > 0:
            #     raise RuntimeError(
            #         'Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs)))
            return _IncompatibleKeys(missing_keys, unexpected_keys)

    def save(self, file_path=''):
        # save({'state_dict': self.state_dict()}, file_path)
        pickle_it(file_path, {'state_dict': self.state_dict()})

    def save_onnx(self, file_path=''):
        pass

    def save_weight(self, file_path=''):
        pass

    # def forward(self, *input, **kwargs):
    #     r"""Defines the computation performed at every call.
    #
    #     Should be overridden by all subclasses.
    #
    #     .. note::
    #         Although the recipe for forward pass needs to be defined within
    #         this function, one should call the :class:`Module` instance afterwards
    #         instead of this since the former takes care of running the
    #         registered hooks while the latter silently ignores them.
    #     """
    #     raise NotImplementedError

    def _slow_forward(self, *input, **kwargs):
        return self._call_impl(*input, **kwargs)

    @tf.Module.with_name_scope
    def _call_impl(self, *args, **kwargs):
        forward_call = self.forward

        is_all_numpy = False
        is_built = self._built

        # only do in the root
        if self.is_root:

            if isinstance(args, tuple):
                is_all_numpy = all([isinstance(inp, np.ndarray) for inp in args])
                args = tuple([to_tensor(inp, device=get_device()) for inp in args])
            else:
                if isinstance(args, np.ndarray):
                    is_all_numpy = True
                args = to_tensor(args, device=get_device())
                args = (args,)

        if not self._built:
            inp = unpack_singleton(args)
            if is_tensor(inp):
                shp = tensor_to_shape(inp, need_exclude_batch_axis=True)
                self.input_shape = shp
                self.input_filters = shp[self.filter_index]

                if self.is_root:
                    if self._signature is None:
                        self._signature = get_signature(self)
                    if self._signature is not None and len(self._signature.inputs) > 0:
                        self._signature.inputs[self._signature.inputs.key_list[0]].shape = tensor_to_shape(inp,
                                                                                                           need_exclude_batch_axis=True,
                                                                                                           is_singleton=False)
                del inp
            elif isinstance(inp, (tuple, list)):
                if isinstance(inp[0], numbers.Number):
                    self.input_shape = TensorShape(list(inp))
                else:
                    out = self.forward(*inp, **kwargs)
                    # shp =[ tensor_to_shape(i, need_exclude_batch_axis=True,is_singleton=False) for i in inp]
                    # self.build(*shp)
            else:
                self.input_shape = TensorShape(list(inp))
                print('input shou be tensor or tuple of tensor')

            self._built = True

        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
            result = forward_call(*args, **kwargs)
            result = unpack_singleton(result)

            if hasattr(self, 'keep_output') and self.keep_output == True:
                # make a op
                self._output_tensor = result
            if self._output_shape is None or is_built == False:
                output = result
                if is_tensor(output):  # one output
                    self._output_shape = tensor_to_shape(output)
                elif isinstance(output, (list, tuple)):
                    output_shape = tuple([tensor_to_shape(item) for item in output if
                                          item is not None and not isinstance(item, (list, tuple))])
                    # if not isinstance(item, (list,tuple)) lstm
                    self._output_shape = unpack_singleton(output_shape)
            if is_all_numpy == True and self.training == False and self.is_root == True:
                if is_tensor(result):
                    return to_numpy(result)
                elif isinstance(result, (list, tuple)):
                    result = list(result)
                    return tuple([to_numpy(res) if is_tensor(res) else res for res in result])
            return result

        full_backward_hooks, non_full_backward_hooks = [], []
        backward_pre_hooks = []
        if self._backward_pre_hooks or _global_backward_pre_hooks:
            backward_pre_hooks = self._get_backward_pre_hooks()

        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook_id, hook in (
                    *_global_forward_pre_hooks.items(),
                    *self._forward_pre_hooks.items(),
            ):
                if hook_id in self._forward_pre_hooks_with_kwargs:
                    result = hook(self, args, kwargs)  # type: ignore[misc]
                    if result is not None:
                        if isinstance(result, tuple) and len(result) == 2:
                            args, kwargs = result
                        else:
                            raise RuntimeError(
                                "forward pre-hook must return None or a tuple "
                                f"of (new_args, new_kwargs), but got {result}."
                            )
                else:
                    result = hook(self, args)
                    if result is not None:
                        if not isinstance(result, tuple):
                            result = (result,)
                        args = result

        # bw_hook = None
        # if full_backward_hooks:
        #     bw_hook = hooks.BackwardHook(self, full_backward_hooks)
        #     input = full_backward_hooks(input)

        # don't use result = self.forward(i*nput, **kwargs) because EagerTensor will splited as a tuple....
        try:
            with tf.device(get_device()):

                result = forward_call(*args, **kwargs)
                result = unpack_singleton(result)

                if hasattr(self, 'keep_output') and self.keep_output == True:
                    # make a op
                    self._output_tensor = result
                if self._output_shape is None or is_built == False:
                    output = result
                    if is_tensor(output):  # one output
                        self._output_shape = tensor_to_shape(output)
                    elif isinstance(output, (list, tuple)):
                        output_shape = tuple([tensor_to_shape(item) for item in output if
                                              item is not None and not isinstance(item, (list, tuple))])
                        # if not isinstance(item, (list,tuple)) lstm
                        self._output_shape = unpack_singleton(output_shape)

                if _global_forward_hooks or self._forward_hooks:
                    for hook_id, hook in (
                            *_global_forward_hooks.items(),
                            *self._forward_hooks.items(),
                    ):
                        if hook_id in self._forward_hooks_with_kwargs:
                            hook_result = hook(self, args, kwargs, result)
                        else:
                            hook_result = hook(self, args, result)

                        if hook_result is not None:
                            result = hook_result

                # if bw_hook:
                #     result = bw_hook.setup_output_hook(result)

            if is_all_numpy == True and self.training == False and self.is_root == True:
                if is_tensor(result):
                    return to_numpy(result)
                elif isinstance(result, (list, tuple)):
                    result = list(result)
                    return tuple([to_numpy(res) if is_tensor(res) else res for res in result])
            return result
        except Exception as e:
            print('{0} ({1} call failed.)'.format(self.name, self.default_name))
            print(e)
            PrintException()
            raise e

    __call__: Callable[..., Any] = _call_impl

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Support loading old checkpoints that don't have the following attrs:
        if '_forward_pre_hooks' not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if '_state_dict_hooks' not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if '_load_state_dict_pre_hooks' not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, tf.Variable):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(value, name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, tf.Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(value, name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, tf.Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(value, name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                istensor = is_tensor(v)
                key = v.ref() if istensor else v
                if v is None or key in memo:
                    continue
                memo.add(key)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse=True):
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Examples:

            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse=True):
        r"""Returns an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Examples:

            >>> for buf in model.buffers():
            >>>     print(type(buf.data), buf.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        """
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix='', recurse=True):
        r"""Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            (string, torch.Tensor): Tuple containing the name and buffer

        Examples:

            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())

        """
        gen = self._named_members(lambda module: module._buffers.items(), prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self):
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self):
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Examples:

            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self):
        r"""Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Examples:

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
                    print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Examples:

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def train(self: T, mode: bool = True) -> T:
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train()
        return self

    def eval(self: T) -> T:
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        Returns:
            Module: self
        """
        return self.train(False)

    @property
    def trainable_weights(self) -> List[tf.Variable]:
        r"""The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.

        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.

        """
        return [x for x in self.parameters() if x.trainable]

    @property
    def non_trainable_weights(self) -> List[tf.Variable]:
        r"""The list of trainable variables (parameters) of the module.
        Parameters of this module and all its submodules are included.
        .. note::
            The list returned may contain duplicate parameters (e.g. output
            layer shares parameters with embeddings). For most usages, it's not
            necessary to ensure uniqueness.
        """
        return [x for x in self.parameters() if not x.trainable]

    @property
    def weights(self):
        return list(self.parameters())

    def get_weights(self):
        return [p.numpy() for p in list(self.parameters())]

    def set_weights(self, weights):
        params = self.weights
        if not params:
            return
        weight_value_tuples = []
        param_values = [w for w in params]
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Layer weight shape ' + str(pv.shape) + ' not compatible with '
                                                                         'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        for p, w in weight_value_tuples:
            p.assign(w.value())

    @property
    def trainable(self):
        """

        Returns: Is the layer trainable?

        """
        if len(self.weights) == 0:
            return False
        elif len(self.weights) > 0:
            for k, v in self._parameters.items():
                if v is not None and v.trainable == False:
                    return False
            else:
                return True

    @trainable.setter
    def trainable(self, value: bool):
        """

        Args:
            value (bool):  new value for the property "Trsainable"

        """
        n = 0
        need_update = False
        for name, para in self.named_parameters():
            if para is not None and para.trainable != value:
                para._trainable = value
                n += np.prod(to_numpy(int_shape(para)))
        if n > 0:
            if value:
                print('{0} parameters have set trainable'.format(n))
            else:
                print('{0} parameters have set untrainable'.format(n))

    def _get_name(self):
        return self.__class__.__name__
    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()

        # replicas do not have parameters themselves, the replicas reference the original
        # module.
        replica._parameters = OrderedDict()
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True

        return replica





class Sequential(Layer):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    _modules: Dict[str, tf.Module]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: tf.Module) -> None:
        ...

    @overload
    def __init__(self, arg: Dict[str, tf.Module]) -> None:
        ...

    def __init__(self, *args, name=None, keep_output=False):
        super().__setattr__('_modules', OrderedDict())
        super(Sequential, self).__init__(name=name,keep_output=keep_output)
        self._built = False
        args = unpack_singleton(args)
        if isinstance(args, OrderedDict):
            for key, module in args.items():
                self.add_module(key, module)
        elif len(args) == 1 and isinstance(args, OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        # self.to(self.device)

    # @property
    # def output_shape(self):
    #     if len(self)>0:
    #         return self[-1]._output_shape
    #     else:
    #         return None
    #
    # @output_shape.setter
    # def output_shape(self, value):
    #     if len(self) > 0:
    #         if isinstance(value, tf.TensorShape):
    #             value = to_tensor(value.as_list()).to('int')
    #         elif isinstance(value, (list, tuple)) and len(value) > 0:
    #             value = tuple(
    #                 [to_tensor(tensor_shape.as_list()).to('int') if isinstance(tensor_shape, tf.TensorShape) else to_tensor(tensor_shape).to('int') for tensor_shape in value])
    #
    #         else:
    #             value = to_tensor(value).to('int')
    #         self[-1]._output_shape = value
    #         self._signature=None

    def build(self, input_shape: TensorShape):
        if self._built == False and len(self._modules) > 0:
            self.__getitem__(0).input_shape = input_shape
            self._built = True

    def add_module(self, name, module):
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """

        if len(self._modules) > 0 and self._input_shape is not None and self[-1].built and self[-1]._output_shape is not None:
            last_output = self[-1]._output_shape
            dummay_input = to_tensor(last_output.get_dummy_tensor()).to(self.device)
            out = module(dummay_input)
            self._modules[name] = module
            if isinstance(out, OrderedDict):
                self._output_shape = tuple(
                    [tensor_to_shape(o, need_exclude_batch_axis=True, is_singleton=False) for o in out.value_list])
                self.get_root().signature.outputs = OrderedDict()
                for k, v in out.item_list:
                    self.get_root().signature.outputs[k] = tensor_to_shape(v, need_exclude_batch_axis=True,
                                                                           is_singleton=False)
            else:
                out = enforce_singleton(out)
                self._output_shape = tensor_to_shape(out, need_exclude_batch_axis=True, is_singleton=False)
                self._signature.outputs[self._signature.outputs.key_list[0]].shape = self._output_shape
                # if len(self.get_root().signature.outputs) > 0:
                #     self.get_root().signature=get_signature(self)
                # else:
                #     self.get_root().signature.outputs['output'] = self._output_shape.copy()

        else:
            if not hasattr(module, '_signature') or module._signature is None:
                module._signature = get_signature(module)
            sig =get_signature(module)
            super(Sequential, self).add_module(name, module)
            if len(self) == 1 or self._signature is None:
                self._signature = sig
            elif len(self) > 1:
                self._signature.outputs = deepcopy(sig.outputs)

    def remove_at(self, idx):
        self.__delitem__(idx)
        if len(self._modules) > 0:
            self._output_shape = self[-1]._output_shape
            if isinstance(self._signature, Signature):
                self._signature.outputs[self._signature.outputs.key_list[0]].shape = self[-1]._output_shape

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = idx.__index__()
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)


    def __setitem__(self, idx, module):
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))


    def __len__(self) -> int:
        return len(self._modules)


    def __add__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError('add operator supports only objects '
                             'of Sequential class, but {} is given.'.format(
                                 str(type(other))))

    def pop(self, key: Union[int, slice]) -> tf.Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError('add operator supports only objects '
                             'of Sequential class, but {} is given.'.format(
                                 str(type(other))))

    def __mul__(self, other: int) -> 'Sequential':
        if not isinstance(other, int):
            raise TypeError(f"unsupported operand type(s) for *: {type(self)} and {type(other)}")
        elif (other <= 0):
            raise ValueError(f"Non-positive multiplication factor {other} for {type(self)}")
        else:
            combined = Sequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> 'Sequential':
        return self.__mul__(other)

    def __imul__(self, other: int) -> 'Sequential':
        if not isinstance(other, int):
            raise TypeError(f"unsupported operand type(s) for *: {type(self)} and {type(other)}")
        elif (other <= 0):
            raise ValueError(f"Non-positive multiplication factor {other} for {type(self)}")
        else:
            len_original = len(self)
            offset = len(self)
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self


    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys





    def __iter__(self) -> Iterator[tf.Module]:
        return iter(self._modules.values())


    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    # def forward(self, input):
    #     for module in self:
    #         input = module(input)
    #     return input

    def forward(self, *x, **kwargs):
        x = unpack_singleton(x)
        for module in self._modules.values():
            # x = enforce_singleton(x)
            if isinstance(x, tuple):
                if len(module.signature.inputs) == len(x):  # self,x
                    x = module(*x, **kwargs)
                else:
                    x = enforce_singleton(x)
                    x = module(x, **kwargs)
            else:
                x = module(x, **kwargs)
            # class_name=module.__class__.__name__.lower()
            # if 'lstm' in class_name or 'gru' in class_name:
            #     if isinstance(x,tuple):
            #         x,hx=x
            #         kwargs['hx']=hx

        return x


    def append(self, module: tf.Module) -> 'Sequential':
        r"""Appends a given module to the end.

        Args:
            module (tf.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: tf.Module) -> 'Sequential':
        if not isinstance(module, Module):
            raise AssertionError(
                'module should be of type: {}'.format(Module))
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError(
                'Index out of range: {}'.format(index))
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> 'Sequential':
        for layer in sequential:
            self.append(layer)
        return self





class ModuleList(Layer):
    """Holds submodules in a list.

    :class:`~trident.backend.tensorflow_backend.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~trident.backend.tensorflow_backend..Layer` methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    """

    def __init__(self, modules: Optional[Iterable[Layer]] = None, name=None, keep_output=False, **kwargs) -> None:
        super(ModuleList, self).__init__(name=None, keep_output=False, **kwargs)
        name = self._name

        start_idx = len(self._modules)
        for module in modules:
            module.is_root = False
            if module.uuid != self.uuid:
                self.add_module(str(start_idx), module)
                reset_name(module, self._uid_prefixs)
                module.relative_name = name if not hasattr(module,
                                                           'relative_name') or module.relative_name == '' else name + '.' + module.relative_name

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)


    def __getitem__(self, idx: int) -> Layer:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Layer) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))


    def __len__(self) -> int:
        return len(self._modules)


    def __iter__(self) -> typing.Iterator[Layer]:
        return iter(self._modules.values())

    def __iadd__(self: T, modules: Iterable[Layer]) -> T:
        return self.extend(modules)


    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Layer) -> None:
        r"""Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (tf.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self: T, module: Layer) -> T:
        r"""Appends a given module to the end of the list.

        Arguments:
            module (tf.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self: T, modules: Iterable[Layer]) -> T:
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, abc.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def forward(self):
        raise NotImplementedError()


class ModuleDict(Layer):
    r"""Holds submodules in a dictionary.

    :class:`~torch.tf.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~torch.tf.Module` methods.

    :class:`~torch.tf.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.tf.ModuleDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.tf.ModuleDict` (the argument to :meth:`~torch.tf.ModuleDict.update`).

    Note that :meth:`~torch.tf.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(tf.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = tf.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = tf.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules: Optional[Mapping[str, Layer]] = None, name=None, keep_output=False,
                 is_multicasting=False, **kwargs) -> None:
        super(ModuleDict, self).__init__(name=None, keep_output=False, **kwargs)
        self.is_multicasting = is_multicasting
        if modules is not None:
            if len(modules) > 0:
                self.update(modules)


    def __getitem__(self, key: str) -> Layer:
        return self._modules[key]

    def __setitem__(self, key: str, module: Layer) -> None:
        self.add_module(key, module)
        if self._input_shape is not None:
            module.input_shape = self.input_shape

    def __delitem__(self, key: str) -> None:
        del self._modules[key]


    def __len__(self) -> int:
        return len(self._modules)


    def __iter__(self):
        return iter(self._modules)


    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()

    def pop(self, key: str) -> Layer:
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v


    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()


    def items(self) -> Iterable[Tuple[str, Layer]]:
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()


    def values(self) -> Iterable[Layer]:
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules: Mapping[str, Layer]) -> None:
        r"""Update the :class:`~torch.tf.ModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~torch.tf.ModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            modules (iterable): a mapping (dictionary) from string to :class:`~torch.tf.Module`,
                or an iterable of key-value pairs of type (string, :class:`~torch.tf.Module`)
        """
        if not isinstance(modules, abc.Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, (OrderedDict, ModuleDict)):
            for key, module in modules.items():
                self[key] = module
        elif isinstance(modules, abc.Mapping):
            for key, module in sorted(modules.items()):
                self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, abc.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]

    def build(self, input_shape: TensorShape):
        """

        Args:
            input_shape (torch.Size, tensor, list(int), tuple(int)): The input_shape information, not including batch axis.

        Returns:

        """
        if self._built == False and len(self._modules) > 0:
            self._input_shape = TensorShape(input_shape)
            input_shape = tuple(to_numpy(input_shape))
            dummay_input = to_tensor(self._input_shape.get_dummy_tensor()).to(self.device)

            for name, module in self.items():
                out = module(dummay_input)
                module.input_shape = input_shape
                module.output_shape = tensor_to_shape(out)
            self._built = True

    def forward(self, x, **kwargs):
        if self.is_multicasting:

            # x = enforce_singleton(x)
            results = OrderedDict()
            for name, module in self.items():
                out = module(x, **kwargs)
                results[name] = out
            return results
        else:
            raise NotImplementedError()


class Combine(Layer):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args, name=None):
        super(Combine, self).__init__()
        self._name = name
        self._built = False

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        elif len(args) == 1 and isinstance(args[0], (list)):
            for idx, module in enumerate(args[0]):
                self.add_module(str(idx), module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        self.to(self.device)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = idx.__index__()
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Combine, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, *x):
        outputs = []
        for module in self._modules.values():
            outputs.append(module(*x))
        return tuple(outputs)


def count_params(weights):
    """Count the total number of scalars composing the weights.

    Args:
        weights: An iterable containing the weights on which to compute params

    Returns:
        The total number of scalars composing the weights
    """
    return int(sum(np.prod(p.shape.as_list()) for p in object_identity.ObjectIdentitySet(weights)))


def calculate_flops(gen: Layer):
    """
    Calculate the flops given a generator of pytorch model.
    It only compute the flops of forward pass.


    """
    flops = 0
    mods = gen.named_modules()
    mods = list(mods)[1:]
    param_nums = []
    param_sizes = []
    for mod_name, mod in mods:
        p = list(mod.parameters())
        modsz = []
        all_params = 0
        for j in range(len(p)):
            modsz.append(np.array(p[j].size()))
            all_params += np.prod(p[j].size())

        param_nums.append(all_params)
        param_sizes.append(modsz)

    return np.array(param_nums).sum()


def summary(model, input_specs, batch_size=1, inputs=None, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            # class_name =module.re    module.name   # str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = module.relative_name if hasattr(module, 'relative_name') else module.name
            summary[m_key] = OrderedDict()
            summary[m_key]["class_name"] = module.__class__.__name__
            if hasattr(module, 'keep_output'):
                summary[m_key]["keep_output"] = module.keep_output
            else:
                summary[m_key]["keep_output"] = False
            input = iteration_tools.flatten([input], iterable_types=(list, tuple))
            input = unpack_singleton([item for item in input if item is not None])
            if isinstance(input, (list, tuple)):
                summary[m_key]["input_shape"] = list(int_shape(input[0]))
            elif is_tensor(input):
                summary[m_key]["input_shape"] = list(int_shape(input))
            summary[m_key]["input_shape"][0] = batch_size

            output = iteration_tools.flatten([output], iterable_types=(list, tuple))
            output = unpack_singleton([item for item in output if item is not None])
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = list(int_shape(output[0]))
            elif is_tensor(output):
                summary[m_key]["output_shape"] = list(int_shape(output))
            summary[m_key]["output_shape"][0] = batch_size

            params = 0
            summary[m_key]["flops"] = np.array([0], dtype=np.float64)
            summary[m_key]["macc"] = np.array([0], dtype=np.float64)
            summary[m_key]["trainable"] = np.array([0], dtype=np.float64)
            summary[m_key]["weight"] = OrderedDict()
            summary[m_key]["bias"] = OrderedDict()
            for name, para in module._parameters.items():
                if para is not None:
                    para_type = "weight"
                    if 'bias' in name or 'beta' in name:
                        para_type = "bias"

                    summary[m_key][para_type][name] = list(int_shape(para))
                    num_params = np.prod(np.array(list(int_shape(para)), dtype=np.float64))
                    spatial_dims = np.prod(np.array(summary[m_key]["output_shape"][2:]).astype(np.float64))
                    params += num_params
                    if para.trainable:
                        summary[m_key]["trainable"] += num_params

                    summary[m_key]["flops"] += (2 * num_params - 1) * spatial_dims
                    summary[m_key]["macc"] += num_params * spatial_dims

            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, (Sequential, ModuleList, ModuleDict))
                and not (module == model and len(module._parameters) == 0)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    for name, module in model.named_modules():
        module.relative_name = name
    # multiple inputs to the network

    # prevent pytorch 'ValueError: Expected more than 1 value per channel when training, got input size ....
    model.to(get_device())
    model.eval()

    inps = OrderedDict()
    for v in input_specs:
        k = v.name
        if v.shape is not None and v.shape._dims != [None]:
            inps[k] = to_tensor(v.get_dummy_tensor(), device=get_device())
        elif v.optional:
            inps[k] = v.default
        elif v.shape is None:
            inps[k] = None
        else:
            inps[k] = None
    # p    rint(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []
    try:
        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        if inputs is not None:
            if isinstance(inputs, OrderedDict):
                model(*list(inps.values()))
            elif isinstance(inputs, (Tensor, np.ndarray)):
                model(inputs)
            else:
                model(*inputs)
        else:
            model(*inps.value_list)

        # remove these hooks
        for h in hooks:
            h.remove()
        max_name_len = 0
        max_weight_len = 0
        for layer in summary:
            max_name_len = builtins.max(max_name_len, len(layer + "  [" + summary[layer]["class_name"] + "]") + 5)
            max_weight_len = builtins.max(max_weight_len, builtins.max(
                [len(str(item).replace('(', '').replace(')', '')) for item in
                 summary[layer]["weight"].items()]) + 5 if len(
                summary[layer]["weight"]) > 0 else 5)

        print(
            "--------------------------------------------------------------------------------------------------------------------------------")
        line_new = "{0:^50s} {1:<25s}  {2:<35s} {3:<8s}  {4:<8s}  {5:<25s}".replace('50s',
                                                                                    str(max_name_len) + 's').replace(
            '35s', str(max_weight_len) + 's').format("Layer (type)",
                                                     "Output Shape",
                                                     "Weight ", "Bias",
                                                     "Param #",
                                                     "FLOPS #")

        print(line_new)
        print("==============================================================================")
        total_params = np.array([0], dtype=np.float64)
        total_output = 0
        trainable_params = np.array([0], dtype=np.float64)
        flops = np.array([0], dtype=np.float64)
        macc = np.array([0], dtype=np.float64)
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            is_keep = '★' if summary[layer]["keep_output"] else ''
            class_name = summary[layer]["class_name"]
            # line_new = "{0:<50s} {1:<20s}  {2:<20s} {3:<8s}  {4:<8}  {5:<12}".format(
            #     layer+"  "+class_name,
            #     is_keep + str(summary[layer]["output_shape"]),
            #     str(summary[layer]["weight"] if 'weight' in summary[layer] else ''),
            #     str(summary[layer]["bias"] if 'bias' in summary[layer] else ''),
            #     summary[layer]["nb_params"],
            #     summary[layer]["flops"][0]
            # )

            line_new = "{0:<50s} {1:<25s}  {2:<35s} {3:<8s}  {4:,.0f}  {5:,.0f}  ".replace('50s',
                                                                                           str(max_name_len) + 's').replace(
                '35s', str(max_weight_len) + 's').format(
                (layer + "  [" + class_name + "]").ljust(max_name_len, ' '),
                (is_keep + str([None] + summary[layer]["output_shape"][1:])).ljust(25, ' '),
                str(summary[layer]["weight"].item_list[0] if 'weight' in summary[layer] and len(
                    summary[layer]["weight"]) > 0 else ' ').replace('(', '').replace(')', '').ljust(
                    max_weight_len, ' '),
                str(summary[layer]["bias"].item_list[0] if 'bias' in summary[layer] and len(
                    summary[layer]["bias"]) > 0 else ' ').replace('(', '').replace(')', '').ljust(8, ' '),
                summary[layer]["nb_params"],
                summary[layer]["flops"].sum()
            )
            if len(summary[layer]["weight"]) > 1:
                for n in range(1, len(summary[layer]["weight"])):
                    line_new_add = "{0:<50s} {1:<25s}  {2:<35s} {3:<8s}  {4}  {5}  ".replace('50s',
                                                                                             str(max_name_len) + 's').replace(
                        '35s', str(max_weight_len) + 's').format(
                        " ".ljust(max_name_len + len(layer + "  [" + class_name + "]") // 2, " "),
                        " ".ljust(25 + len(is_keep + str([None] + summary[layer]["output_shape"][1:])) // 2, " "),
                        str(summary[layer]["weight"].item_list[n] if n < len(
                            summary[layer]["weight"]) else ' ').replace('(', '').replace(')', '').ljust(max_weight_len,
                                                                                                        " "),
                        str(summary[layer]["bias"].item_list[n] if n < len(summary[layer]["bias"]) else ' ').replace(
                            '(', '').replace(')', '').ljust(8, " "), " ", " ")
                    line_new = line_new + '\n' + line_new_add

            total_params += summary[layer]["nb_params"]
            flops += float(summary[layer]["flops"])
            macc += float(summary[layer]["macc"].sum())
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                trainable_params += summary[layer]["trainable"]
            print(line_new)

        # assume 4 bytes/number (float on cuda).
        total_input_size = np.asarray(
            [np.abs(np.prod(to_numpy(spec.shape.dims[1:])) * batch_size * 4. / (1024 ** 2.)) for spec in input_specs if
             spec.optional == False and spec.shape is not None]).sum()
        total_output_size = np.abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = np.abs(total_params * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        print("================================================================")
        print("Total params: {0:,.0f}".format(total_params[0]))
        print("Trainable params: {0:,.0f}".format(trainable_params[0]))
        print("Non-trainable params: {0:,.0f}".format(total_params[0] - trainable_params[0]))
        print("Total MACC: {0:,.0f}".format(macc[0]))
        print("Total FLOPs: {0:.5f} GFLOPs".format(np.round(flops / 10. ** 9, 5)[0]))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")
        # return summary
        del hooks
    except Exception as e:
        del hooks
        print(e)
        PrintException()


def normalize_padding(padding, rank):
    """
    normalize different padding format to most comlete representaion.
    Tensorflow and Pytorch have totally different padding representation format and padding strategy.
    In Pytorch the complete format is 2* rank ex.: conv2d rank=2 NCHW , we only need to pad the last 2 dimentions
    and the most terrible  is  padding in rank format and 2*rank format are perfectly "REVERSE"!!
    Conv2d((3,3), 32, strides=1)   padding=(1, 1) ==>(height , width)
    Conv2d((3,3), 32, strides=1)   full padding=(1, 1, 1, 1) ==>(left right , top , down)

    In tensorflow the complete format is 2* (rank+2) ex.: conv2d rank=2 NHWC , we only need to pad the middle 2 dimentions and empty head and tail dimentions
    Conv2d((3,3), 32, strides=1)   padding=(1, 1) ==>(height , width)
    Conv2d((3,3), 32, strides=1)   full padding=((0, 0), (1, 1), (1, 1), (0, 0)) ==>((batch), (height), (width), (channel)

    for clear and no-confused definition, trident will always define padding by the nature ordinal of the dimention.
    ex. Conv2d((3,3), 32, strides=1)
    No matter in pytorch or tensorflow backend
    padding=(1, 1) ==>(height , width)
    padding=((1, 1),(1,1)) ==>(height , width)

    Args:
        padding (None, int, tuple):
        rank (int):

    Returns:
        the normalized format of padding
    Examples
    >>> normalize_padding(((1,0),(1,0)),2)
    ((1, 0), (1, 0))
    >>> normalize_padding((1,0),2)
    ((1, 1), (0, 0))
    >>> normalize_padding(1,2)
    ((1, 1), (1, 1))
    >>> normalize_padding((1),2)
    ((1, 1), (1, 1))
    >>> normalize_padding((1,0),2)
    ((1, 1), (0, 0))
   >>> normalize_padding((0,1 ,0, 1),2)
   ((0, 1), (0, 1))

    """
    if padding is None:
        # None=>((0,0),(0,0))
        padding = ((0, 0),) * rank
    elif isinstance(padding, int):
        # 1=>((1,1),(1,1))
        padding = ((padding, padding),) * rank
    elif isinstance(padding, (list, tuple)) and len(padding) == 1 and padding[0] == int:
        # (1)=>((1,1),(1,1))
        padding = ((padding[0], padding[0]),) * rank
    elif isinstance(padding, (list, tuple)) and len(padding) == rank and isinstance(padding[0], int):
        # rank=2 (1,1)=>((1,1,), (1,1))   (1,0)=>((1,1),(0,0))
        padding = list(padding)
        return_padding = []
        for i in range(rank):
            return_padding.append((padding[i], padding[i]))
        padding = tuple(return_padding)
    elif isinstance(padding, (list, tuple)) and len(padding) == rank and isinstance(padding[0], (list, tuple)):
        # rank=2  ((1,0),(1,0)=>(1,0,1,0)
        pass  # padding= tuple(list(itertools.chain(*list(padding))))
    elif isinstance(padding, (list, tuple)) and len(padding) == 2 * rank and isinstance(padding[0], int):
        padding = ((padding[0], padding[1]), (padding[2], padding[3]))
    return padding


def try_map_args_and_call(fn, data: OrderedDict, data_feed: OrderedDict = None, is_autocast_enabled=False):
    """This function is the core function for mapping callable and argments

    Args:
        fn (callable): the callable, maybe functions or layers
        data (OrderedDict): The key-value pair for available data.
        data_feed (OrderedDict): The relation between callable argments (key) and data (value)
        is_autocast_enabled


    Returns:
        The result of the callable base on data_feed

    """

    if isinstance(fn, tf.Tensor) or 'EagerTensor' in fn.__class__.__name__:
        return fn
    else:

        arg_map = OrderedDict()
        if isinstance(fn, Layer):
            _device = fn.get_root().device
            _signature = fn._signature
            if None in fn._signature.inputs.value_list:
                _signature = get_signature(fn)
            for arg in fn.signature.inputs.key_list:
                is_optional = fn.signature.inputs[arg].optional if isinstance(fn.signature.inputs[arg],
                                                                              TensorSpec) else False
                default = fn.signature.inputs[arg].default if isinstance(fn.signature.inputs[arg], TensorSpec) else None
                is_input = arg.lower() in ['x', 'input']
                is_output = arg.lower() in ['y', 'output', 'y_pred']
                if arg in data_feed and data_feed[arg] in data:
                    arg_map[arg] = to_tensor(data[data_feed[arg]], device=_device)
                elif arg in data:
                    arg_map[arg] = to_tensor(data[arg], device=_device)
                elif is_input and 'input' in data:
                    arg_map[arg] = to_tensor(data['input'], device=_device)
                elif is_output and 'output' in data:
                    arg_map[arg] = to_tensor(data['output'], device=_device)
                elif is_optional:
                    arg_map[arg] = default
                else:
                    raise ValueError('arg :{0} cannot mapping correctly!'.format(arg))
            if ctx.amp_available == True and is_autocast_enabled == True and get_device() == 'cuda':
                # not support yet
                out = fn(*arg_map.value_list)
            else:
                out = fn(*arg_map.value_list)
            return out

        elif (hasattr(fn, 'signature') or hasattr(fn, '_signature')) and callable(fn):
            sig = fn._signature if hasattr(fn, '_signature') else fn.signature
            for arg in sig.inputs.key_list:
                is_optional = sig.inputs[arg].optional if isinstance(sig.inputs[arg], TensorSpec) else False
                default = sig.inputs[arg].default if isinstance(sig.inputs[arg], TensorSpec) else None
                is_input = arg.lower() in ['x', 'input']
                is_output = arg.lower() in ['y', 'output', 'y_pred']
                if arg in data_feed and data_feed[arg] in data:
                    arg_map[arg] = to_tensor(data[data_feed[arg]], device=get_device())
                elif arg in data:
                    arg_map[arg] = to_tensor(data[arg], device=get_device())
                elif is_input and 'input' in data_feed and data_feed['input'] in data:
                    arg_map[arg] = to_tensor(data[data_feed['input']], device=get_device())
                elif is_output and 'output' in data_feed and data_feed['output'] in data:
                    arg_map[arg] = to_tensor(data[data_feed['output']], device=get_device())
                elif isinstance(fn, functools.partial) and arg in fn.keywords:
                    arg_map[arg] = fn.keywords[arg]
                elif is_optional:
                    arg_map[arg] = default
                elif arg == 'y_pred' and 'output' in data:
                    arg_map[arg] = data['output']
                elif arg == 'y_true' and 'target' in data:
                    arg_map[arg] = data['target']
                elif arg == 'y_true' and 'label' in data:
                    arg_map[arg] = data['label']
                elif arg == 'label' and 'target' in data:
                    arg_map[arg] = data['target']
                else:
                    raise ValueError('arg :{0} cannot mapping correctly!'.format(arg))

            if ctx.amp_available == True and is_autocast_enabled == True and get_device() == 'cuda':
                # not support yet
                if isinstance(fn, partial):
                    out = fn.func(*arg_map.value_list)
                else:
                    out = fn(*arg_map.value_list)
            else:
                if isinstance(fn, partial):
                    out = fn.func(*arg_map.value_list)
                else:
                    out = fn(*arg_map.value_list)
            return out
        elif callable(fn):
            fn.signature = get_signature(fn)
            args = get_signature(fn).inputs.key_list
            for arg in args:
                is_optional = fn.signature.inputs[arg].optional if isinstance(fn.signature.inputs[arg],
                                                                              TensorSpec) else False
                default = fn.signature.inputs[arg].default if isinstance(fn.signature.inputs[arg], TensorSpec) else None
                is_input = arg.lower() in ['x', 'input']
                is_output = arg.lower() in ['y', 'output', 'y_pred']
                if arg in data_feed and data_feed[arg] in data:
                    arg_map[arg] = to_tensor(data[data_feed[arg]], device=get_device())
                elif arg in data:
                    arg_map[arg] = to_tensor(data[arg], device=get_device())
                elif is_input and 'input' in data_feed and data_feed['input'] in data:
                    arg_map[arg] = to_tensor(data[data_feed['input']], device=get_device())
                elif is_output and 'output' in data_feed and data_feed['output'] in data:
                    arg_map[arg] = to_tensor(data[data_feed['output']], device=get_device())
                elif is_optional:
                    arg_map[arg] = default

                else:
                    arg_map.pop(arg)
            if ctx.amp_available == True and is_autocast_enabled == True and get_device() == 'cuda':
                # not support yet
                out = fn(*arg_map.value_list)
            else:
                out = fn(*arg_map.value_list)
            return out
        else:
            print('uncomplete arg_map', arg_map.key_list)


def force_deterministic(seed):
    """ Force most of the computation nodes to run deterministically.

    Args:
        seed (int): set the random seed for all random ops in the graph and readers.

    """

    set_seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = True
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = False
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def fix_layer(layer: Layer):
    """fix existing out-of-date model compatibility

    Args:
        layer (trident Layer):

    Returns: fixed layer

    """
    if not hasattr(layer, 'is_root'):
        layer.is_root = True
    if not hasattr(layer, 'device'):
        layer.device = get_device()

    def get_root(self):
        if not hasattr(self, '_nodes') or self._nodes is None or len(self._nodes) < 2:
            return self
        if hasattr(list(self._nodes.values())[0], 'is_root') and list(self._nodes.values())[0].is_root == True:
            return list(self._nodes.values())[0]
        else:
            for name, node in self._nodes.items():
                if hasattr(node, 'default_name') and node.default_name == "sequential_1":
                    return node
            return self

    layer.to(get_device())
    if not hasattr(layer, '_nodes'):
        layer._nodes = OrderedDict()

    if not hasattr(layer, '_uid_prefixs'):
        layer._uid_prefixs = {}
    reset_name(layer, layer._uid_prefixs)

    if 'ssd' in layer.__class__.__base__.__name__.lower() or 'yolo' in layer.__class__.__base__.__name__.lower():
        if hasattr(layer, 'nms_threshold'):
            delattr(layer, 'nms_threshold')
        if hasattr(layer, 'detection_threshold'):
            delattr(layer, 'detection_threshold')

    if layer._input_shape is not None and isinstance(layer._input_shape, tuple) and all(
            [isinstance(d, numbers.Integral) for d in layer._input_shape]):
        dims = [d for d in layer._input_shape]
        dims.insert(0, None)
        layer._input_shape = TensorShape(dims)
    elif layer._input_shape is not None and isinstance(layer._input_shape, tuple):
        layer._input_shape = tuple([TensorShape(to_numpy(item)) for item in TensorShape(layer._input_shape)])
    elif layer._input_shape is not None and is_tensor(layer._input_shape):
        dims = [d.item() for d in layer._input_shape]
        dims.insert(0, None)
        buffers = layer.__dict__.get('_buffers')
        if '_input_shape' in buffers:
            del buffers['_input_shape']
        object.__setattr__(layer, '_input_shape', TensorShape(dims))
    elif layer._input_shape is not None and not isinstance(layer._input_shape, TensorShape):
        layer._input_shape = TensorShape(to_numpy(layer._input_shape))
    elif layer._input_shape is not None and isinstance(layer._input_shape, TensorShape) and is_tensor(
            layer._input_shape.dims[0]):
        layer._input_shape = TensorShape([d.item() for d in layer._input_shape.dims])

    if layer._output_shape is not None and isinstance(layer._output_shape, tuple) and all(
            [isinstance(d, numbers.Integral) for d in layer._output_shape]):
        dims = [d for d in layer._input_shape]
        dims.insert(0, None)
        layer._input_shape = TensorShape(dims)
    elif layer._output_shape is not None and isinstance(layer._output_shape, tuple):
        layer._output_shape = tuple([TensorShape(to_numpy(item)) for item in TensorShape(layer._output_shape)])
    elif layer._output_shape is not None and is_tensor(layer._output_shape):
        dims = [d.item() for d in layer._output_shape]
        dims.insert(0, None)
        buffers = layer.__dict__.get('_buffers')
        if '_input_shape' in buffers:
            del buffers['_output_shape']
        object.__setattr__(layer, '_output_shape', TensorShape(dims))
    elif layer._output_shape is not None and not isinstance(layer._output_shape, TensorShape):
        layer._output_shape = TensorShape(layer._output_shape)
    elif layer._output_shape is not None and isinstance(layer._output_shape, TensorShape) and isinstance(
            layer._output_shape.dims[0], tf.TensorShape) and len(layer._output_shape.dims[0]) == 1:
        layer._output_shape = TensorShape([None if d is None else d[0] for d in layer._output_shape.dims])
    elif layer._output_shape is not None and isinstance(layer._output_shape, TensorShape) and isinstance(
            layer._output_shape.dims[0], tf.TensorShape):
        layer._output_shape = tuple([TensorShape(to_numpy(d)) for d in layer._output_shape.dims])

    if not hasattr(layer, 'get_root'):
        setattr(layer, 'get_root', MethodType(get_root, layer))

    for module in layer.modules():
        if not hasattr(module, '__signature__'):
            module.__signature__ = inspect.signature(module.forward)
        class_name = module.__class__.__name__
        if not hasattr(layer, 'get_root'):
            setattr(layer, 'get_root', MethodType(get_root, layer))

        if not hasattr(module, '_state_dict_pre_hooks'):
            object.__setattr__(module, '_forward_hooks_with_kwargs', OrderedDict())
            object.__setattr__(module, '_forward_pre_hooks_with_kwargs', OrderedDict())
            object.__setattr__(module, '_state_dict_pre_hooks', OrderedDict())

        if not hasattr(module, 'uuid'):
            module.uuid = uuid.uuid4().node
        # check for root
        if module.uuid == layer.uuid:
            module.is_root = True
        else:
            module.is_root = False
        if not hasattr(module, 'relative_name'):
            module.relative_name = ''
        if not hasattr(module, '_uid_prefixs'):
            module._uid_prefixs = layer.get_root()._uid_prefixs

        if not hasattr(module, 'default_name') or (module.default_name is None or len(module.default_name) == 0):
            module_prefix = module.__class__.__name__
            module.default_name = camel2snake(module_prefix) + '_' + str(get_global_uid(camel2snake(module_prefix)))

        if not hasattr(module, '_name'):
            module._name = None
        reset_name(module, layer.get_root()._uid_prefixs)

        if not hasattr(module, 'name'):
            module.name = module._name if module._name is not None and len(module._name) > 0 else module.relative_name

        if not hasattr(module, '_built'):
            setattr(module, 'built', True)

        if hasattr(module, 'keepdim'):
            value = getattr(module, 'keepdim')
            delattr(module, 'keepdim')
            setattr(module, 'keepdims', value)

        if not hasattr(module, '_non_persistent_buffers_set'):
            module._non_persistent_buffers_set = set()

        # fix for shape definition
        # if not isinstance(module._input_shape, TensorShape):
        #     module._input_shape = TensorShape(to_numpy(module._input_shape))
        #
        # if not isinstance(module._output_shape, TensorShape):
        #     module._output_shape = TensorShape(to_numpy(module._output_shape))

        if not hasattr(module, 'batch_index'):
            setattr(module, 'batch_index', 0)
        if not hasattr(module, 'filter_index'):
            setattr(module, 'filter_index', 1)
        if not hasattr(module, 'in_sequence'):
            setattr(module, 'in_sequence', False)

        if not hasattr(module, 'in_sequence'):
            if 'lstm' in class_name.lower() or 'gru' in class_name.lower() or 'rnn' in class_name.lower():
                module.in_sequence = True
            else:
                module.in_sequence = False

        if 'Conv' in class_name and 'Block' in class_name:
            if not hasattr(module, 'sequence_rank'):
                module.sequence_rank = 'cna'

        if 'Conv' in class_name:
            if not hasattr(module, 'depth_multiplier'):
                if 'Depthwise' in class_name or 'Separable' in class_name:
                    module.depth_multiplier = 1
                else:
                    module.depth_multiplier = None
            if not hasattr(module, 'use_spectral'):
                module.use_spectral = False

    if layer.is_root == True and (
            not hasattr(layer, '_signature') or layer._signature is None or len(layer._signature.inputs) == 0):
        layer._signature = None
        # if layer._input_shape is not None:
        #     if isinstance(layer._input_shape, TensorSpec):
        #         layer._signature.inputs["input"] = TensorSpec(shape=layer._input_shape, name="input")
        #     elif isinstance(layer._input_shape, tuple):
        #         for i in range(len(layer._input_shape)):
        #             layer._signature.inputs["input_{0}".format(i)] = TensorSpec(shape=layer._input_shape[i], name="input_{0}".format(i))
        # if layer._output_shape is not None:
        #
        #     if isinstance(layer._output_shape, TensorSpec):
        #         layer._signature.outputs["output"] = TensorSpec(shape=layer._output_shape, name="output")
        #     elif isinstance(layer._output_shape, tuple):
        #         for i in range(len(layer._output_shape)):
        #             layer._signature.outputs["output_{0}".format(i)] = TensorSpec(shape=layer._output_shape[i], name="output_{0}".format(i))
        #

        sig = layer.signature

    return layer


def fix_keras_module(module: tf.Module, input_tensor: Tensor = None, input_shape: (tuple, TensorShape) = None):
    import tensorflow.python.keras.backend as K
    def named_modules(keras_model, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if keras_model not in memo:
            memo.add(keras_model)
            yield prefix, keras_model

            layers = [keras_model]
            if hasattr(keras_model, 'layers'):
                layers = keras_model.layers
            for module in layers:
                name = module.name
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in named_modules(module, memo, submodule_prefix):
                    yield m

    def named_parameters(keras_model, memo=None, prefix=''):
        return [(w.name, w) for w in keras_model.weights]

    def to(keras_model, *args) -> T:
        device = None
        dtype = None
        non_blocking = None
        for arg in args:
            if 'cpu' in arg.lower() or 'gpu' in arg.lower() or 'cudu' in arg.lower():
                device = arg
            elif arg.lower() in ['float', 'int', 'bool', 'half', 'long']:
                dtype = arg
            elif isinstance(arg, bool):
                non_blocking = arg

        if device is None:
            device = get_device()
        # if dtype is None and len(keras_model.weights) > 0:
        #     dtype = keras_model.weights[0].dtype
        if 'cpu' in device:
            with tf.device('/cpu:0'):
                for p in keras_model.weights:
                    if dtype is None or dtype == p.dtype:
                        p.assign(tf.identity(p.value()))
                    else:
                        p.assign(tf.identity(cast(p.value(), dtype)))
        elif 'gpu' in device or 'cuda' in device:
            if tf.test.is_gpu_available:
                with tf.device('/gpu:0'):
                    for p in keras_model.weights:
                        if dtype is None or dtype == p.dtype:
                            p.assign(tf.identity(p.value()))
                        else:
                            p.assign(tf.identity(cast(p.value(), dtype)))
        return keras_model

    def train(keras_model):
        # 1 = train
        K.set_learning_phase(1)
        return keras_model

    def eval(keras_model):
        # 0 = test, 1 = train
        K.set_learning_phase(0)
        return keras_model

    module.is_root = True

    module._nodes = OrderedDict()
    for name, mod in named_modules(module):
        module._nodes[id(mod)] = mod
        mod.uuid = id(mod)
        if not hasattr(mod, 'named_modules'):
            setattr(mod, 'named_modules', MethodType(named_modules, mod))

        if not hasattr(mod, 'named_parameters'):
            setattr(mod, 'named_parameters', MethodType(named_parameters, mod))

        if not hasattr(mod, 'train'):
            setattr(mod, 'train', MethodType(train, mod))

        if not hasattr(mod, 'eval'):
            setattr(mod, 'eval', MethodType(eval, mod))

        mod.relative_name = name
    module.nodes = module._nodes
    if not hasattr(module, 'to'):
        setattr(module, 'to', MethodType(to, module))

    module._uid_prefixs = defaultdict(int)
    sig = Signature(name=module.name)
    for inp in module.inputs:
        sig.inputs[inp.name] = TensorSpec(shape=TensorShape(int_shape(inp)), dtype=DTYPE_MAPPING[inp.dtype],
                                          name=inp.name)
    for k in range(len(module.outputs)):
        out = module.outputs[k]
        out_name = 'output' if len(module.outputs) == 1 else 'output_{0}'.format(k)
        sig.outputs[out_name] = TensorSpec(shape=TensorShape(int_shape(out)), dtype=DTYPE_MAPPING[out.dtype],
                                           name=out_name)

    module._signature = sig
    module.signature = sig

    # def get_uid(prefix=''):
    #     module._uid_prefixs[prefix] += 1
    #     return module._uid_prefixs[prefix]
    #
    # def get_root(module):
    #     if not hasattr(module, '_nodes') or module._nodes is None or len(module._nodes) < 2:
    #         return module
    #     if hasattr(list(module._nodes.values())[0], 'is_root') and list(module._nodes.values())[0].is_root == True:
    #         return list(module._nodes.values())[0]
    #     else:
    #         for name, node in module._nodes.items():
    #             if hasattr(node, 'default_name') and node.default_name == "sequential_1":
    #                 return node
    #         return module

    # for name, mod in module.named_modules():
    #     if mod != module:
    #         module.is_root = False
    #     mod._built = True
    #     mod.built = True
    #     mod.relative_name = name
    #     mod.batch_index = 0
    #     mod.filter_index = 1
    #     mod.in_sequence = False
    #     mod.uuid = uuid.uuid4().node
    #     prefix = mod.__class__.__name__
    #     mod.default_name = camel2snake(prefix) + '_' + str(get_uid(camel2snake(prefix)))
    #
    #     mod._input_shape = None
    #     mod._output_shape = None
    #     mod.keep_output = False
    #     mod.register_buffer('_output_tensor', None, persistent=False)
    #
    #     mod._device = get_device()
    #     #mod._signature = inspect.signature(mod.forward)
    #     mod.dump_patches = True
    #     #
    #     # def getsignature(mod):
    #     #     return mod._signature
    #     #
    #     # def setsignature(mod, value):
    #     #     mod._signature = value
    #     #
    #     # def delsignature(mod):
    #     #     del mod._signature
    #     #
    #     # mod.signature= property(getsignature, setsignature, delsignature, "signature")
    #
    #     if not hasattr(mod, 'get_root'):
    #         setattr(mod, 'get_root', MethodType(get_root, mod))
    #     if hasattr(mod, 'dims'):
    #         mod.axis = mod.dims
    #     if hasattr(module, 'keepdim'):
    #         value = getattr(module, 'keepdim')
    #         setattr(module, 'keepdims', value)
    #
    # def register_hook(module):
    #     def hook(module, input, output):
    #         # class_name =module.re    module.name   # str(module.__class__).split(".")[-1].split("'")[0]
    #         input = iteration_tools.flatten([input], iterable_types=(list, tuple))
    #         input = unpack_singleton([item for item in input if item is not None])
    #
    #         if isinstance(input, (list, tuple)):
    #             module._input_shape = tuple([tensor_to_shape(t, need_exclude_batch_axis=True, is_singleton=False) for t in input])
    #             module.input_shape = module._input_shape
    #         elif is_tensor(input):
    #             module._input_shape = tensor_to_shape(input, need_exclude_batch_axis=True, is_singleton=False)
    #             module.input_shape = module._input_shape
    #
    #         output = iteration_tools.flatten([output], iterable_types=(list, tuple))
    #         output = unpack_singleton([item for item in output if item is not None])
    #         if isinstance(output, (list, tuple)):
    #             module._output_shape = tuple([tensor_to_shape(t, need_exclude_batch_axis=True, is_singleton=False) for t in output])
    #             module.output_shape = module._output_shape
    #         elif is_tensor(output):
    #             module._output_shape = tensor_to_shape(output, need_exclude_batch_axis=True, is_singleton=False)
    #             module.output_shape = module._output_shape
    #
    #         hooks.append(module.register_forward_hook(hook))
    #
    # hooks = []
    #
    # # register hook
    # module.apply(register_hook)
    #
    # if input_tensor is None:
    #     input_tensor = to_tensor(TensorShape(input_shape).get_dummy_tensor())
    # # make a forward pass
    # # print(x.shape)
    # module(input_tensor)
    #
    # # remove these hooks
    # for h in hooks:
    #     h.remove()
    return module
